from __future__ import annotations

import json
import os
import time
from pathlib import Path

from audiohelix.build.cluster import cluster_unit_features
from audiohelix.config import AudioConfig, EmbedConfig, INDEX_MODEL, LLMConfig
from audiohelix.build.index_builder import (
    IndexPaths, ItemData, MultiFileAudioSource, SingleFileAudioSource,
    assemble_item_graph, collect_item_data,
)
from audiohelix.models.llm_backend import build_omni_llm
from audiohelix.build.manifest import (
    assert_manifest_chunk_cfg, chunk_item_files, group_chunks_by_file,
    load_manifest, items_from_manifest,
)
from audiohelix.prompts import CAPTION_ASSISTANT_PREFIX, CAPTION_PROMPT, parse_raw_output
from audiohelix.utils import log, setup_logging


# ---------------------------------------------------------------------------
# shared entry points
# ---------------------------------------------------------------------------

def load_source(paths, args) -> object:
    """The benchmark's question/item source: a manifest dict (AudioMarathon/
    MMAU-Pro) or a flat questions list (TraceAV). A manifest's chunk settings
    are asserted here, so a stale manifest is rejected before anything is cut."""
    if paths.questions_file.suffix == ".json":
        doc = json.loads(Path(paths.questions_file).read_text(encoding="utf-8"))
    else:
        raise SystemExit(f"unsupported source file: {paths.questions_file}")
    if isinstance(doc, dict):
        assert_manifest_chunk_cfg(doc, args.chunk_sec, args.min_last_chunk_sec)
    return doc


def targets(bench, paths, args) -> list[dict]:
    """The items this stage should process, as plain dicts:
    {item_id, suite, task_type, files[{path, clip, dur_s, start_chunk,
    n_chunks}], n_chunks}."""
    return bench.data.items_for_index(load_source(paths, args), args)


def audio_source_for(bench, item: dict):
    """Single-file benchmarks (TraceAV) cut from one source wav; multi-file
    items regroup chunks by file so each file keeps its own local timing."""
    if bench.config.AUDIO_SOURCE == "single":
        return SingleFileAudioSource(path=Path(item["files"][0]["path"]))
    return MultiFileAudioSource(files=item["files"], group_fn=group_chunks_by_file)


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def run_manifest(bench, args) -> dict:
    setup_logging(str(bench.config.DATA_DIR / "logs" / "build_manifest.log"))
    return bench.data.build_manifest(args)


# ---------------------------------------------------------------------------
# captions
# ---------------------------------------------------------------------------

def load_or_init_captions_doc(captions_path: Path, bench, item: dict,
                              chunk_sec: float) -> dict:
    """Existing doc if present (the resume baseline), else a fresh one from the
    benchmark's own skeleton."""
    item_id = item["item_id"]
    if captions_path.exists():
        doc = json.loads(captions_path.read_text(encoding="utf-8"))
        if doc.get("video_id") == item_id and abs(doc.get("chunk_sec", -1) - chunk_sec) < 1e-6:
            return doc
    return bench.data.caption_doc_init(item, chunk_sec)


def caption_one_item(llm, bench, item: dict, paths, args,
                     chunk_sec: float, min_last_chunk_sec: float) -> bool:
    item_id = item["item_id"]
    files = item["files"]
    missing = [f["path"] for f in files if not Path(f["path"]).exists()]
    if missing:
        log(f"  {item_id}: {len(missing)} source audio file(s) missing "
            f"({missing[0]}), skipping")
        return False

    # One captioner (INDEX_MODEL), so one doc per item.
    captions_path = bench.config.captions_dir(paths, chunk_sec) / f"{item_id}.json"
    doc = load_or_init_captions_doc(captions_path, bench, item, chunk_sec)
    done_ids = {c["chunk_id"] for c in doc["chunks"] if c.get(INDEX_MODEL, {}).get("raw")}

    chunks = chunk_item_files(
        files, item_id, out_dir=str(bench.config.chunks_dir(paths, chunk_sec)),
        chunk_sec=chunk_sec, min_last_chunk_sec=min_last_chunk_sec,
        expect_n=item.get("n_chunks"),
    )
    if not chunks:
        log(f"  {item_id}: 0 chunks, skipping")
        return False

    doc["chunks"] = [c for c in doc["chunks"] if c["chunk_id"] in {p.chunk_id for p in chunks}]
    doc_chunks = {c["chunk_id"]: c for c in doc["chunks"]}

    t0 = time.time()
    n_done, n_skip = 0, 0
    for i, p in enumerate(chunks):
        if p.chunk_id in done_ids and not args.force:
            n_skip += 1
            continue
        raw = llm.caption_audio(
            p.audio_path, CAPTION_PROMPT, assistant_prefix=CAPTION_ASSISTANT_PREFIX,
        )
        parsed = parse_raw_output(raw)
        # Merge into the existing chunk dict instead of overwriting it wholesale,
        # since the chunk may already carry its geometry.
        chunk = doc_chunks.get(p.chunk_id, {
            "chunk_id": p.chunk_id,
            "idx": p.idx,
            "start_sec": p.start_sec,
            "end_sec": p.end_sec,
        })
        chunk[INDEX_MODEL] = {
            "raw": raw,
            "transcription": parsed["transcription"],
            "caption": parsed["caption"],
            "concepts": parsed["concepts"],
        }
        doc_chunks[p.chunk_id] = chunk
        n_done += 1
        # Save after every chunk, so an interrupted run resumes from here.
        doc["chunks"] = sorted(doc_chunks.values(), key=lambda c: c["idx"])
        captions_path.parent.mkdir(parents=True, exist_ok=True)
        captions_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        log(f"  {item_id} [{i + 1}/{len(chunks)}] {p.chunk_id} "
            f"({p.start_sec:.0f}-{p.end_sec:.0f}s) captioned")

    log(f"  {item_id}: {n_done} new / {n_skip} skipped out of {len(chunks)} chunks "
        f"({time.time() - t0:.1f}s) -> {captions_path}")
    return True


class _LazyCaptionLLM:
    """Caption backend, built on first use.

    An already-captioned corpus (every chunk carries INDEX_MODEL's `raw`) turns
    the caption stage into a pure no-op, and the weights should then never load:
    caption generation is not reproducible across execution stacks, so a re-run
    over existing captions must not regenerate anything -- and must not need the
    caption GPU at all.
    """

    def __init__(self, build):
        self._build = build
        self._llm = None

    def caption_audio(self, *args, **kwargs):
        if self._llm is None:
            t0 = time.time()
            self._llm = self._build()
            log(f"LLM loaded: {time.time() - t0:.1f}s")
        return self._llm.caption_audio(*args, **kwargs)


def run_captions(bench, args) -> dict:
    paths = bench.config.paths(args.chunk_sec)
    setup_logging(str(bench.config.DATA_DIR / "logs" / "extract_captions.log"))
    targets_ = targets(bench, paths, args)
    log(f"{len(targets_)} target items (chunk_sec={args.chunk_sec:g}, model={INDEX_MODEL})")

    llm_cfg = LLMConfig(model=INDEX_MODEL)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu or llm_cfg.gpu
    llm = _LazyCaptionLLM(lambda: build_omni_llm(INDEX_MODEL))

    done = 0
    for item in targets_:
        if caption_one_item(llm, bench, item, paths, args,
                            args.chunk_sec, args.min_last_chunk_sec):
            done += 1
    log(f"done: {done}/{len(targets_)} items")
    return {"stage": "captions", "done": done, "total": len(targets_)}


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------

def run_graph(bench, args) -> dict:
    from audiohelix.models.embedder import build_chunk_embedder

    paths = bench.config.paths(args.chunk_sec)
    setup_logging(str(bench.config.DATA_DIR / "logs" / "build_index.log"))
    item_paths: IndexPaths = bench.config.index_paths(paths, args.chunk_sec)

    audio_cfg = AudioConfig(chunk_sec=args.chunk_sec, unit_sec=args.unit_sec,
                            min_last_chunk_sec=args.min_last_chunk_sec)
    cluster_cfg = bench.config.cluster_config(args)
    embed_cfg = EmbedConfig()

    targets_ = targets(bench, paths, args)
    log(f"{len(targets_)} target items (chunk_sec={args.chunk_sec:g}, "
        f"unit_sec={args.unit_sec:g}, cluster={cluster_cfg.tag(args.unit_sec)})")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu or LLMConfig(model=INDEX_MODEL).gpu
    t0 = time.time()
    peav = build_chunk_embedder("pe_av")
    log(f"PE-AV loaded: {time.time() - t0:.1f}s")

    # Phase 1 -- collect features for every item
    t1 = time.time()
    collected: list[tuple[dict, ItemData]] = []
    for item in targets_:
        try:
            source = audio_source_for(bench, item)
            data = collect_item_data(peav, item["item_id"], item_paths,
                                     source, audio_cfg, embed_cfg)
        except Exception as e:                      # noqa: BLE001 -- one bad item shouldn't kill the build
            log(f"  {item['item_id']}: error collecting, skipping "
                f"({type(e).__name__}: {e})")
            continue
        if data is not None:
            collected.append((item, data))
    log(f"collection done: {len(collected)}/{len(targets_)} items ({time.time() - t1:.1f}s)")

    # Phase 2 -- per-item clustering + graph assembly (each item's centroids
    # come from that item's own audio units only)
    done = 0
    for item, data in collected:
        t2 = time.time()
        try:
            clusters = cluster_unit_features(
                data.audio_unit_embeddings, data.audio_unit_ids, cfg=cluster_cfg,
            )
            log(f"  {data.item_id}: clustered {len(data.audio_unit_ids)} units -> "
                f"{len(clusters)} clusters ({time.time() - t2:.1f}s)")
            scope, tag = bench.config.index_location(
                item, args.chunk_sec, cluster_cfg.tag(args.unit_sec))
            if assemble_item_graph(peav, data, clusters, item_paths, embed_cfg,
                                   scope=scope, cluster_tag=tag):
                done += 1
        except Exception as e:                      # noqa: BLE001
            log(f"  {data.item_id}: error during clustering/graph assembly, skipping "
                f"({len(data.audio_unit_ids)} audio units, {type(e).__name__}: {e})")
    log(f"done: {done}/{len(collected)} items")
    return {"stage": "graph", "done": done, "total": len(targets_)}


def run_index(bench, args) -> dict:
    """captions + graph (the index-build pipeline, in one command)."""
    out = {}
    if args.stage in ("all", "captions"):
        out["captions"] = run_captions(bench, args)
    if args.stage in ("all", "graph"):
        out["graph"] = run_graph(bench, args)
    return out
