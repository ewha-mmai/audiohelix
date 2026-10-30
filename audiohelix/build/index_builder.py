from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from audiohelix.audio_io import ChunkRecord, UnitRecord, make_unit_wavs
from audiohelix.config import AudioConfig, ClusterConfig, EmbedConfig, PathsConfig
from audiohelix.build.graph_build import GraphInput, build_concept_chunk_edges, build_graph
from audiohelix.build.unit_splitter import split_into_units
from audiohelix.utils import log

CHUNK_FULL_TEXT_FALLBACK = "(no content)"


def chunk_full_text(transcription: str, caption: str) -> str:
    """Chunk content shown to the LLM for the final answer."""
    parts = []
    if transcription:
        parts.append(f"Transcription: {transcription}")
    if caption:
        parts.append(f"Caption: {caption}")
    return "\n".join(parts) if parts else CHUNK_FULL_TEXT_FALLBACK


def load_caption_chunks(captions_path: Path, chunk_sec: float = 45.0) -> dict | None:
    """captions JSON -> {chunk_id: {idx, start_sec, end_sec, transcription,
    caption, concepts}}, or None if missing. start_sec/end_sec must be
    preserved as-is since the last chunk can be shorter than chunk_sec."""
    if not captions_path.exists():
        return None
    import json
    from audiohelix.config import INDEX_MODEL
    doc = json.loads(captions_path.read_text(encoding="utf-8"))
    caption_chunks = {}
    for c in doc["chunks"]:
        if INDEX_MODEL not in c:
            continue
        out = c[INDEX_MODEL]
        caption_chunks[c["chunk_id"]] = {
            "idx": c["idx"],
            "start_sec": float(c.get("start_sec", c["idx"] * chunk_sec)),
            "end_sec": float(c.get("end_sec", (c["idx"] + 1) * chunk_sec)),
            "transcription": out.get("transcription", ""),
            "caption": out.get("caption", ""),
            "concepts": [e for e in out.get("concepts", []) if e],
        }
    return caption_chunks if caption_chunks else None


def try_load_embedding_cache(cache_path: Path) -> dict | None:
    """Loads a per-item embedding cache -> {"chunk": {cid: emb}, "audio_unit": {uid: emb}}.
    Returns None if missing/corrupt/malformed; the caller falls back to re-encoding."""
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path) as z:
            if not {"chunk_id", "chunk_embedding",
                    "audio_unit_id", "audio_unit_embedding"} <= set(z.files):
                return None
            cids = [str(x) for x in z["chunk_id"]]
            p_emb = np.asarray(z["chunk_embedding"], dtype=np.float32)
            uids = [str(x) for x in z["audio_unit_id"]]
            s_emb = np.asarray(z["audio_unit_embedding"], dtype=np.float32)
    except (ValueError, OSError):
        log(f"  {cache_path.name}: cache corrupt, re-encoding")
        return None
    if len(cids) != p_emb.shape[0] or len(uids) != s_emb.shape[0]:
        log(f"  {cache_path.name}: cache length mismatch, re-encoding")
        return None
    return {
        "chunk": dict(zip(cids, p_emb)),
        "audio_unit": dict(zip(uids, s_emb)),
    }


def save_embedding_cache(cache_path: Path, chunk_ids: list[str],
                         chunk_embeddings: np.ndarray, audio_unit_ids: list[str],
                         audio_unit_embeddings: np.ndarray) -> None:
    """Writes via tmp file + os.replace for atomicity under concurrent builds
    (content is deterministic, so whichever write wins is fine)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(cache_path.name + ".tmp.npz")
    np.savez(tmp,
             chunk_id=np.asarray(chunk_ids),
             chunk_embedding=chunk_embeddings,
             audio_unit_id=np.asarray(audio_unit_ids),
             audio_unit_embedding=audio_unit_embeddings)
    os.replace(tmp, cache_path)


# ---------------------------------------------------------------------------
# Per-benchmark adapters
# ---------------------------------------------------------------------------

@dataclass
class IndexPaths:
    """Wraps PathsConfig. chunk_sec=None means it's already baked into `paths`
    (AudioMarathon/MMAU-Pro); otherwise it's passed on every call (TraceAV)."""

    paths: PathsConfig
    chunk_sec: float | None = None

    def captions_dir(self) -> Path:
        if self.chunk_sec is None:
            return self.paths.captions_dir_for()
        return self.paths.captions_dir_for(self.chunk_sec)

    def chunk_wav_dir(self, item_id: str) -> Path:
        base = self.paths.chunks_dir if self.chunk_sec is None else self.paths.chunks_dir_for(self.chunk_sec)
        return base / item_id

    def embedding_cache_dir(self, unit_sec: float) -> Path:
        if self.chunk_sec is None:
            return self.paths.embedding_cache_dir_for(unit_sec)
        return self.paths.embedding_cache_dir_for(unit_sec, self.chunk_sec)

    def units_out_dir(self, unit_sec: float) -> Path:
        if self.chunk_sec is None:
            return self.paths.units_dir_for(unit_sec)
        return self.paths.units_dir_for(unit_sec, self.chunk_sec)

    def graph_out_dir(self, scope: str, cluster_tag: str, item_id: str) -> Path:
        return self.paths.index_dir_for(scope, cluster_tag) / item_id


class AudioSource(Protocol):
    """Checks the source audio exists and cuts audio-unit wavs from it."""

    def exists(self) -> bool: ...

    def make_units(self, chunks: list[ChunkRecord], unit_sec: float,
                       out_dir: Path) -> list[UnitRecord]: ...


@dataclass
class SingleFileAudioSource:
    """TraceAV: exactly one source audio file per video."""

    path: Path

    def exists(self) -> bool:
        return self.path.exists()

    def make_units(self, chunks: list[ChunkRecord], unit_sec: float,
                       out_dir: Path) -> list[UnitRecord]:
        return make_unit_wavs(
            source_audio_path=str(self.path), chunk_records=chunks,
            out_dir=str(out_dir), unit_sec=unit_sec,
        )


@dataclass
class MultiFileAudioSource:
    """AudioMarathon/MMAU-Pro: a unit may span multiple files (multi-clip), so
    chunks are regrouped back to their source file (group_fn) and cut per
    file to keep file-local timing correct.

    group_fn is each benchmark's own pipeline_util.group_chunks_by_file,
    injected as-is since the manifest format it depends on is benchmark-owned."""

    files: list[dict]
    group_fn: Callable[[list[dict], list[ChunkRecord]], list[tuple[str, list[ChunkRecord]]]]

    def exists(self) -> bool:
        return not any(not Path(f["path"]).exists() for f in self.files)

    def make_units(self, chunks: list[ChunkRecord], unit_sec: float,
                       out_dir: Path) -> list[UnitRecord]:
        records: list[UnitRecord] = []
        for file_path, seg in self.group_fn(self.files, chunks):
            records += make_unit_wavs(
                source_audio_path=file_path, chunk_records=seg,
                out_dir=str(out_dir), unit_sec=unit_sec,
            )
        return records


# ---------------------------------------------------------------------------
# Phase 1 -- per-item feature collection
# ---------------------------------------------------------------------------

@dataclass
class ItemData:
    item_id: str
    chunks: list[ChunkRecord]
    chunk_embeddings: np.ndarray
    caption_chunks: dict                              # load_caption_chunks() result (for the text side)
    audio_unit_ids: list[str]
    audio_unit_paths: list[str]
    audio_unit_embeddings: np.ndarray
    audio_unit_to_chunk: dict[str, str] = field(default_factory=dict)


def collect_item_data(peav, item_id: str, item_paths: IndexPaths,
                      audio_source: AudioSource, audio_cfg: AudioConfig,
                      embed_cfg: EmbedConfig) -> ItemData | None:
    """Collects chunk/audio-unit features from captions. Returns None
    if captions or source audio are missing. Rows with NaN embeddings
    (empty/corrupt wavs) are dropped -- letting NaN into the graph breaks
    clustering and poisons retrieval scores."""
    captions_path = item_paths.captions_dir() / f"{item_id}.json"
    caption_chunks = load_caption_chunks(captions_path, audio_cfg.chunk_sec)
    if caption_chunks is None:
        log(f"  {item_id}: no captions, skipping")
        return None

    if not audio_source.exists():
        log(f"  {item_id}: no source audio, skipping")
        return None

    chunks: list[ChunkRecord] = []
    for cid, c in sorted(caption_chunks.items(), key=lambda x: x[1]["idx"]):
        chunks.append(ChunkRecord(
            chunk_id=cid, video_id=item_id, idx=c["idx"],
            start_sec=float(c.get("start_sec", c["idx"] * audio_cfg.chunk_sec)),
            end_sec=float(c.get("end_sec", (c["idx"] + 1) * audio_cfg.chunk_sec)),
            audio_path=str(item_paths.chunk_wav_dir(item_id) / f"{cid}.wav"),
        ))
    if not chunks:
        log(f"  {item_id}: 0 chunks, skipping")
        return None

    # Embeddings only depend on (unit_sec, chunk_sec), so they're cached per
    # item to skip re-encoding when just the cluster config changes. A hit
    # requires an exact id-set match, not a subset, so a re-captioned item's
    # changed chunk set can't silently reuse stale embeddings.
    cache_path = item_paths.embedding_cache_dir(audio_cfg.unit_sec) / f"{item_id}.npz"
    units_out_dir = item_paths.units_out_dir(audio_cfg.unit_sec)
    cached = try_load_embedding_cache(cache_path)
    use_cache = False

    if cached is not None and set(cached["chunk"]) == {p.chunk_id for p in chunks}:
        unit_records = audio_source.make_units(chunks, audio_cfg.unit_sec, units_out_dir)
        hit_uids = [s.unit_id for s in unit_records]
        if set(hit_uids) == set(cached["audio_unit"]):
            chunk_embeddings = np.stack([cached["chunk"][p.chunk_id] for p in chunks])
            audio_unit_embeddings = np.stack([cached["audio_unit"][s] for s in hit_uids])
            use_cache = True
            log(f"  {item_id}: using embedding cache ({len(chunks)} chunks, "
                f"{len(hit_uids)} audio units)")
        else:
            log(f"  {item_id}: cached audio-unit set differs, re-encoding")
    elif cached is not None:
        log(f"  {item_id}: cached chunk set differs, re-encoding")

    if not use_cache:
        chunk_embeddings = np.asarray(
            peav.encode_audio([p.audio_path for p in chunks], batch_size=embed_cfg.chunk_batch_size),
            dtype=np.float32,
        )
        nan_chunks = ~np.isfinite(chunk_embeddings).all(axis=1)
        if nan_chunks.any():
            dropped = [p.chunk_id for p, k in zip(chunks, nan_chunks) if k]
            log(f"  {item_id}: dropping {len(dropped)} chunks with NaN embeddings: {dropped}")
            chunks = [p for p, k in zip(chunks, nan_chunks) if not k]
            chunk_embeddings = chunk_embeddings[~nan_chunks]
            if not chunks:
                log(f"  {item_id}: 0 chunks left after NaN drop, skipping")
                return None

        unit_records = audio_source.make_units(chunks, audio_cfg.unit_sec, units_out_dir)

        audio_unit_embeddings = np.asarray(
            peav.encode_audio([s.audio_path for s in unit_records],
                              batch_size=embed_cfg.unit_batch_size),
            dtype=np.float32,
        )
        nan_units = ~np.isfinite(audio_unit_embeddings).all(axis=1)
        if nan_units.any():
            dropped = [s.unit_id for s, k in zip(unit_records, nan_units) if k]
            log(f"  {item_id}: dropping {len(dropped)} audio units with NaN embeddings: {dropped}")
            keep = ~nan_units
            unit_records = [s for s, k in zip(unit_records, keep) if k]
            audio_unit_embeddings = audio_unit_embeddings[keep]

        save_embedding_cache(cache_path, [p.chunk_id for p in chunks],
                             chunk_embeddings,
                             [s.unit_id for s in unit_records],
                             audio_unit_embeddings)

    audio_unit_ids = [s.unit_id for s in unit_records]
    audio_unit_paths = [s.audio_path for s in unit_records]
    audio_unit_to_chunk = {s.unit_id: s.chunk_id for s in unit_records}

    return ItemData(
        item_id=item_id, chunks=chunks, chunk_embeddings=chunk_embeddings,
        caption_chunks=caption_chunks, audio_unit_ids=audio_unit_ids,
        audio_unit_paths=audio_unit_paths,
        audio_unit_embeddings=audio_unit_embeddings,
        audio_unit_to_chunk=audio_unit_to_chunk,
    )


# ---------------------------------------------------------------------------
# Phase 2 -- per-item graph assembly (clusters built from this item's audio
# units only)
# ---------------------------------------------------------------------------

def assemble_item_graph(peav, data: ItemData, clusters,
                        item_paths: IndexPaths, embed_cfg: EmbedConfig,
                        scope: str, cluster_tag: str) -> bool:
    """Builds nodes/edges and saves via audiohelix.build.graph_build.build_graph().
    scope/cluster_tag are precomputed by the caller (each benchmark formats
    them differently) and just forwarded as-is."""
    t0 = time.time()
    chunks = data.chunks
    caption_chunks = data.caption_chunks

    # text unit / text concept
    chunk_id_to_unit_ids: dict[str, list[str]] = {}
    unit_id_to_text: dict[str, str] = {}
    chunk_id_to_concepts: dict[str, list[str]] = {}
    chunk_id_to_text: dict[str, str] = {}

    for p in chunks:
        c = caption_chunks[p.chunk_id]
        chunk_id_to_text[p.chunk_id] = chunk_full_text(c["transcription"], c["caption"])
        chunk_id_to_concepts[p.chunk_id] = c["concepts"]

        unit_ids: list[str] = []
        for i, s in enumerate(split_into_units(c["transcription"])):
            uid = f"{p.chunk_id}_t{i}"
            unit_id_to_text[uid] = s
            unit_ids.append(uid)
        for i, s in enumerate(split_into_units(c["caption"])):
            uid = f"{p.chunk_id}_c{i}"
            unit_id_to_text[uid] = s
            unit_ids.append(uid)
        chunk_id_to_unit_ids[p.chunk_id] = unit_ids

    text_unit_ids = list(unit_id_to_text.keys())
    text_unit_texts = [unit_id_to_text[uid] for uid in text_unit_ids]

    all_concept_texts = sorted({e for concepts in chunk_id_to_concepts.values() for e in concepts})

    concept_to_unit_ids: dict[str, set[str]] = {}
    unit_to_concept_texts: dict[str, set[str]] = {}
    for chunk_id, concepts in chunk_id_to_concepts.items():
        for uid in chunk_id_to_unit_ids.get(chunk_id, []):
            unit_lower = unit_id_to_text[uid].lower()
            for concept in concepts:
                if concept.lower() in unit_lower:
                    concept_to_unit_ids.setdefault(concept, set()).add(uid)
                    unit_to_concept_texts.setdefault(uid, set()).add(concept)

    concept_chunk_edges = build_concept_chunk_edges(
        chunk_id_to_unit_ids, unit_id_to_text, chunk_id_to_concepts,
    )
    chunk_concept_weights: dict[str, dict[str, float]] = {}
    for (cid, concept), w in concept_chunk_edges.items():
        chunk_concept_weights.setdefault(cid, {})[concept] = w

    # centroid: clusters built from this item's audio units only
    centroid_concept_ids = [c.concept_id for c in clusters]
    centroid_embeddings = np.stack([c.centroid for c in clusters]) if clusters else np.zeros((0, 1024))
    centroid_to_unit_ids = {c.concept_id: list(c.member_unit_ids) for c in clusters}

    # chunk affinity = (members in this chunk) / (total cluster members)
    chunk_centroid_weights: dict[str, dict[str, float]] = {}
    for c in clusters:
        members_by_chunk: dict[str, int] = {}
        for uid in c.member_unit_ids:
            cid = data.audio_unit_to_chunk[uid]
            members_by_chunk[cid] = members_by_chunk.get(cid, 0) + 1
        for cid, cnt in members_by_chunk.items():
            w = cnt / len(c.member_unit_ids)
            if w > 0:
                chunk_centroid_weights.setdefault(cid, {})[c.concept_id] = w

    concept_cluster_info = {
        c.concept_id: {"n_members": len(c.member_unit_ids), "members": sorted(c.member_unit_ids)}
        for c in clusters
    }
    n_singleton = sum(1 for c in clusters if len(c.member_unit_ids) == 1)

    gi = GraphInput(
        video_id=data.item_id,
        chunk_ids=[p.chunk_id for p in chunks],
        chunk_contents=[chunk_id_to_text[p.chunk_id] for p in chunks],
        chunk_audio_paths=[p.audio_path for p in chunks],
        chunk_embeddings=data.chunk_embeddings,
        text_unit_ids=text_unit_ids,
        text_unit_texts=text_unit_texts,
        chunk_id_to_unit_ids=chunk_id_to_unit_ids,
        audio_unit_ids=data.audio_unit_ids,
        audio_unit_paths=data.audio_unit_paths,
        audio_unit_to_chunk=data.audio_unit_to_chunk,
        audio_unit_embeddings=data.audio_unit_embeddings,
        text_concept_texts=all_concept_texts,
        chunk_concept_weights=chunk_concept_weights,
        concept_to_unit_ids={e: sorted(s) for e, s in concept_to_unit_ids.items()},
        unit_to_concept_texts={uid: sorted(concepts) for uid, concepts in unit_to_concept_texts.items()},
        centroid_concept_ids=centroid_concept_ids,
        centroid_embeddings=centroid_embeddings,
        centroid_to_unit_ids=centroid_to_unit_ids,
        chunk_centroid_weights=chunk_centroid_weights,
        concept_cluster_info=concept_cluster_info,
        chunk_order=[(p.idx, p.chunk_id) for p in chunks],
    )

    out_dir = item_paths.graph_out_dir(scope, cluster_tag, data.item_id)
    build_graph(gi, peav, embed_cfg, out_dir)

    log(
        f"  {data.item_id}: {len(clusters)} clusters ({n_singleton} singleton), "
        f"{len(data.audio_unit_ids)} audio units, {time.time() - t0:.1f}s"
        f" -> {out_dir}"
    )
    return True
