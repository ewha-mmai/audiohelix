from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import igraph as ig
import numpy as np

from audiohelix.models.embed_store import EmbeddingStore
from audiohelix.models.embedder import as_audio_encoder, as_text_encoder
from audiohelix.utils import log


@dataclass
class GraphInput:
    """Node-construction spec filled in by a per-benchmark script and passed
    to build_graph()."""

    video_id: str

    # chunk
    chunk_ids: List[str] = field(default_factory=list)            # order preserved (adjacency edges)
    chunk_contents: List[str] = field(default_factory=list)       # text shown to the LLM
    chunk_audio_paths: List[str] = field(default_factory=list)
    chunk_embeddings: np.ndarray | None = None                    # precomputed, to avoid re-encoding

    # text unit
    text_unit_ids: List[str] = field(default_factory=list)
    text_unit_texts: List[str] = field(default_factory=list)
    chunk_id_to_unit_ids: Dict[str, List[str]] = field(default_factory=dict)

    # audio unit
    audio_unit_ids: List[str] = field(default_factory=list)
    audio_unit_paths: List[str] = field(default_factory=list)
    audio_unit_to_chunk: Dict[str, str] = field(default_factory=dict)
    audio_unit_embeddings: np.ndarray | None = None             # precomputed, from clustering

    # text concept
    text_concept_texts: List[str] = field(default_factory=list)      # deduplicated
    chunk_concept_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)  # mention ratio
    concept_to_unit_ids: Dict[str, List[str]] = field(default_factory=dict)
    unit_to_concept_texts: Dict[str, List[str]] = field(default_factory=dict)

    # centroid concept
    centroid_concept_ids: List[str] = field(default_factory=list)
    centroid_embeddings: np.ndarray | None = None
    centroid_to_unit_ids: Dict[str, List[str]] = field(default_factory=dict)
    chunk_centroid_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)  # affinity
    concept_cluster_info: Dict[str, dict] = field(default_factory=dict)  # concept id -> {n_members, members}

    chunk_order: List[tuple] = field(default_factory=list)  # [(idx, chunk_id), ...], chronological


def build_concept_chunk_edges(
    chunk_id_to_unit_ids: Dict[str, List[str]],
    unit_id_to_text: Dict[str, str],
    chunk_id_to_concepts: Dict[str, List[str]],
) -> Dict[tuple, float]:
    """(chunk_id, concept_text) -> weight. Counts concept mentions across a
    chunk's text units, normalized by that chunk's total mention count."""
    from collections import defaultdict

    chunk_concept_count: Dict[tuple, int] = {}
    chunk_total_count: Dict[str, int] = defaultdict(int)

    for chunk_id, concepts in chunk_id_to_concepts.items():
        unit_ids = chunk_id_to_unit_ids.get(chunk_id, [])
        combined_text = " ".join(unit_id_to_text[uid] for uid in unit_ids if uid in unit_id_to_text)
        combined_text_lower = combined_text.lower()

        for concept in concepts:
            count = combined_text_lower.count(concept.lower())
            chunk_concept_count[(chunk_id, concept)] = count
            chunk_total_count[chunk_id] += count

    edges: Dict[tuple, float] = {}
    for (chunk_id, concept), count in chunk_concept_count.items():
        total = chunk_total_count[chunk_id]
        edges[(chunk_id, concept)] = (count / total) if total > 0 else 0.0
    return edges


def build_graph(gi: GraphInput, peav, embed_cfg, out_dir: Path) -> ig.Graph:
    """Writes GraphInput to 3 embedding stores + LinearRAG.graphml + graph_meta.json
    under out_dir/{video_id}/, and returns the igraph."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # chunk store: content = LLM text, embedding input = chunk wav (audio)
    chunk_store = EmbeddingStore(
        as_audio_encoder(peav), str(out_dir / "chunk_embedding.parquet"),
        batch_size=embed_cfg.chunk_batch_size, namespace="chunk",
    )
    if gi.chunk_embeddings is not None:
        chunk_store.insert(
            ids=gi.chunk_ids, contents=gi.chunk_contents,
            embeddings=gi.chunk_embeddings,
        )
    else:
        chunk_store.insert(
            ids=gi.chunk_ids, contents=gi.chunk_contents,
            embed_inputs=gi.chunk_audio_paths,
        )

    # unit store: text units (text) + audio units (wav), one store
    unit_store = EmbeddingStore(
        as_text_encoder(peav), str(out_dir / "unit_embedding.parquet"),
        batch_size=embed_cfg.unit_batch_size, namespace="unit",
    )
    if gi.text_unit_ids:
        unit_store.insert(ids=gi.text_unit_ids, contents=gi.text_unit_texts)
    if gi.audio_unit_ids:
        unit_store.insert(
            ids=gi.audio_unit_ids, contents=gi.audio_unit_paths,
            embed_inputs=gi.audio_unit_paths,
            embeddings=gi.audio_unit_embeddings,
        )

    # concept store: text concepts (text) + centroids (precomputed embedding), one store
    concept_store = EmbeddingStore(
        as_text_encoder(peav), str(out_dir / "concept_embedding.parquet"),
        batch_size=embed_cfg.concept_batch_size, namespace="concept",
    )
    if gi.text_concept_texts:
        concept_store.insert(ids=gi.text_concept_texts, contents=gi.text_concept_texts)
    if gi.centroid_concept_ids:
        assert gi.centroid_embeddings is not None
        concept_store.insert(
            ids=gi.centroid_concept_ids, contents=gi.centroid_concept_ids,
            embeddings=gi.centroid_embeddings,
        )

    # igraph: chunk + concept vertices
    graph = ig.Graph(directed=False)
    chunk_hash_id = {cid: chunk_store.hash_id_of(cid) for cid in gi.chunk_ids}
    text_concept_hash_id = {e: concept_store.hash_id_of(e) for e in gi.text_concept_texts}
    centroid_hash_id = {cpt: concept_store.hash_id_of(cpt) for cpt in gi.centroid_concept_ids}

    for cid in gi.chunk_ids:
        hid = chunk_hash_id[cid]
        graph.add_vertex(name=hid, content=chunk_store.hash_id_to_text[hid])
    for e in gi.text_concept_texts:
        hid = text_concept_hash_id[e]
        graph.add_vertex(name=hid, content=concept_store.hash_id_to_text[hid])
    for cpt in gi.centroid_concept_ids:
        hid = centroid_hash_id[cpt]
        graph.add_vertex(name=hid, content=concept_store.hash_id_to_text[hid])

    name_to_idx = {v["name"]: v.index for v in graph.vs}
    edges: list[tuple] = []
    weights: list[float] = []

    # chunk <-> text concept (mention ratio)
    for cid, concept_weights in gi.chunk_concept_weights.items():
        for concept, w in concept_weights.items():
            if w <= 0 or cid not in chunk_hash_id or concept not in text_concept_hash_id:
                continue
            edges.append((name_to_idx[chunk_hash_id[cid]], name_to_idx[text_concept_hash_id[concept]]))
            weights.append(w)

    # chunk <-> centroid (affinity)
    for cid, centroid_weights in gi.chunk_centroid_weights.items():
        for cpt, w in centroid_weights.items():
            if w <= 0 or cid not in chunk_hash_id or cpt not in centroid_hash_id:
                continue
            edges.append((name_to_idx[chunk_hash_id[cid]], name_to_idx[centroid_hash_id[cpt]]))
            weights.append(w)

    # adjacent chunks, chronological
    chunk_order = sorted(gi.chunk_order, key=lambda x: x[0])
    adjacent = [(a[1], b[1]) for a, b in zip(chunk_order, chunk_order[1:])]
    for cid_a, cid_b in adjacent:
        if cid_a in chunk_hash_id and cid_b in chunk_hash_id:
            edges.append((name_to_idx[chunk_hash_id[cid_a]], name_to_idx[chunk_hash_id[cid_b]]))
            weights.append(1.0)

    graph.add_edges(edges)
    graph.es["weight"] = weights
    graph.write_graphml(str(out_dir / "LinearRAG.graphml"))

    meta = {
        "video_id": gi.video_id,
        "chunk_id_to_text": {cid: t for cid, t in zip(gi.chunk_ids, gi.chunk_contents)},
        "chunk_id_to_audio_path": dict(zip(gi.chunk_ids, gi.chunk_audio_paths)),
        "chunk_id_to_unit_ids": gi.chunk_id_to_unit_ids,
        "unit_id_to_text": {uid: t for uid, t in zip(gi.text_unit_ids, gi.text_unit_texts)},
        "audio_unit_id_to_chunk": gi.audio_unit_to_chunk,
        "concept_to_unit_ids": {e: sorted(uids) for e, uids in gi.concept_to_unit_ids.items()},
        "unit_to_concept_texts": {uid: sorted(concepts) for uid, concepts in gi.unit_to_concept_texts.items()},
        "centroid_to_unit_ids": {cpt: sorted(uids) for cpt, uids in gi.centroid_to_unit_ids.items()},
        "chunk_concept_weights": gi.chunk_concept_weights,
        "chunk_centroid_weights": gi.chunk_centroid_weights,
        "concept_cluster_info": gi.concept_cluster_info,
        "chunk_order": sorted(gi.chunk_order, key=lambda x: x[0]),
    }
    (out_dir / "graph_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    n_text_concept = len(gi.text_concept_texts)
    n_centroid = len(gi.centroid_concept_ids)
    log(
        f"  {gi.video_id}: {graph.vcount()} nodes (chunk {len(gi.chunk_ids)}, "
        f"text concept {n_text_concept}, centroid {n_centroid}), "
        f"text unit {len(gi.text_unit_ids)}, audio unit {len(gi.audio_unit_ids)}, "
        f"{graph.ecount()} edges -> {out_dir}"
    )
    return graph
