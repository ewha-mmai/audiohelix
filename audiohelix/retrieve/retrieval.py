from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import igraph as ig
import numpy as np

from audiohelix.config import RetrieveConfig
from audiohelix.models.embed_store import EmbeddingStore
from audiohelix.models.embedder import as_audio_encoder, as_text_encoder
from audiohelix.utils import log


class AudioIndex:
    """Loaded view of a build_graph() output directory (graph + 3 embedding
    stores + meta), plus the centroid/audio-unit extensions."""

    def __init__(self, index_dir: Path, peav, retrieve_cfg: RetrieveConfig | None = None):
        self.peav = peav
        self.cfg = retrieve_cfg or RetrieveConfig()
        self.index_dir = Path(index_dir)

        self.graph = ig.Graph.Read_GraphML(str(self.index_dir / "LinearRAG.graphml"))

        self.chunk_store = EmbeddingStore(
            as_audio_encoder(self.peav), str(self.index_dir / "chunk_embedding.parquet"),
            batch_size=32, namespace="chunk",
        )
        self.concept_store = EmbeddingStore(
            as_text_encoder(self.peav), str(self.index_dir / "concept_embedding.parquet"),
            batch_size=64, namespace="concept",
        )
        self.unit_store = EmbeddingStore(
            as_text_encoder(self.peav), str(self.index_dir / "unit_embedding.parquet"),
            batch_size=64, namespace="unit",
        )

        meta = json.loads((self.index_dir / "graph_meta.json").read_text(encoding="utf-8"))
        self.chunk_id_to_text: dict[str, str] = meta["chunk_id_to_text"]
        self.chunk_id_to_audio_path: dict[str, str] = meta["chunk_id_to_audio_path"]
        self.chunk_id_to_unit_ids: dict[str, list[str]] = meta["chunk_id_to_unit_ids"]
        self.unit_id_to_text: dict[str, str] = meta["unit_id_to_text"]
        self.audio_unit_id_to_chunk: dict[str, str] = meta["audio_unit_id_to_chunk"]
        self.concept_to_unit_ids: dict[str, list[str]] = meta["concept_to_unit_ids"]
        self.unit_to_concept_texts: dict[str, list[str]] = meta["unit_to_concept_texts"]
        self.centroid_to_unit_ids: dict[str, list[str]] = meta["centroid_to_unit_ids"]
        self.chunk_concept_weights: dict[str, dict[str, float]] = meta["chunk_concept_weights"]
        self.chunk_centroid_weights: dict[str, dict[str, float]] = meta["chunk_centroid_weights"]
        self.concept_cluster_info: dict[str, dict] = meta.get("concept_cluster_info", {})

        # Merged mention-ratio (text concept) + affinity (centroid) weights, for calculate_chunk_scores.
        self.chunk_concept_weights_merged: dict[str, dict[str, float]] = defaultdict(dict)
        for cid, w in self.chunk_concept_weights.items():
            self.chunk_concept_weights_merged[cid].update(w)
        for cid, w in self.chunk_centroid_weights.items():
            self.chunk_concept_weights_merged[cid].update(w)

        self.chunk_hash_id = {cid: self.chunk_store.hash_id_of(cid) for cid in self.chunk_id_to_text}
        self.concept_hash_id = {t: h for h, t in self.concept_store.hash_id_to_text.items()}
        all_uids = set(self.unit_id_to_text) | set(self.audio_unit_id_to_chunk)
        self.unit_hash_id = {uid: self.unit_store.hash_id_of(uid) for uid in all_uids}

        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs}
        self.vertex_idx_to_node_name = {v.index: v["name"] for v in self.graph.vs}
        chunk_hash_id_set = set(self.chunk_hash_id.values())
        self.chunk_node_indices = [
            self.node_name_to_vertex_idx[h] for h in chunk_hash_id_set if h in self.node_name_to_vertex_idx
        ]

        self.chunk_embeddings = np.array(self.chunk_store.embeddings)
        self.concept_embeddings = np.array(self.concept_store.embeddings)
        self.unit_embeddings = np.array(self.unit_store.embeddings)

        # concept <-> unit links (text concept -> text unit, and centroid -> audio unit)
        self.concept_hash_id_to_unit_hash_ids: dict[str, list[str]] = defaultdict(list)
        for concept_key, uids in self.concept_to_unit_ids.items():
            eh = self.concept_hash_id.get(concept_key)
            if eh is None:
                continue
            for uid in uids:
                sh = self.unit_hash_id.get(uid)
                if sh is not None:
                    self.concept_hash_id_to_unit_hash_ids[eh].append(sh)
        for cpt, uids in self.centroid_to_unit_ids.items():
            eh = self.concept_hash_id.get(cpt)
            if eh is None:
                continue
            for uid in uids:
                sh = self.unit_hash_id.get(uid)
                if sh is not None:
                    self.concept_hash_id_to_unit_hash_ids[eh].append(sh)

        self.unit_hash_id_to_concept_hash_ids: dict[str, list[str]] = defaultdict(list)
        for concept_key, uids in self.concept_to_unit_ids.items():
            eh = self.concept_hash_id.get(concept_key)
            if eh is None:
                continue
            for uid in uids:
                sh = self.unit_hash_id.get(uid)
                if sh is not None:
                    self.unit_hash_id_to_concept_hash_ids[sh].append(eh)
        for cpt, uids in self.centroid_to_unit_ids.items():
            eh = self.concept_hash_id.get(cpt)
            if eh is None:
                continue
            for uid in uids:
                sh = self.unit_hash_id.get(uid)
                if sh is not None:
                    self.unit_hash_id_to_concept_hash_ids[sh].append(eh)


# Seeding: top-k per modality (text concept, audio centroid) from the whole-
# question embedding, each counted separately -- not top-2 overall.

def get_seed_concepts_topk(index: AudioIndex, question_embedding: np.ndarray, k_text: int = 1, k_audio: int = 1):
    """Returns (seed_concept_indices, seed_concept_texts, seed_concept_hash_ids,
    seed_concept_scores, seeds_by_modality). seeds_by_modality is
    {"text": [...], "audio": [...]}, each entry {"concept", "hash_id", "score"}
    (saved per-question by the caller)."""
    similarities = np.dot(index.concept_embeddings, question_embedding.reshape(-1, 1)).flatten()
    concept_hash_ids_list = list(index.concept_store.hash_ids)

    centroid_hash_ids = {
        index.concept_store.text_to_hash_id[cent_id]
        for cent_id in index.centroid_to_unit_ids
        if cent_id in index.concept_store.text_to_hash_id
    }
    group_indices = {
        "text": [i for i, h in enumerate(concept_hash_ids_list) if h not in centroid_hash_ids],
        "audio": [i for i, h in enumerate(concept_hash_ids_list) if h in centroid_hash_ids],
    }

    seed_concept_indices, seed_concept_texts, seed_concept_hash_ids, seed_concept_scores = [], [], [], []
    seeds_by_modality: dict[str, list[dict]] = {}
    per_modality_k = {"text": k_text, "audio": k_audio}
    for modality, idx_list in group_indices.items():
        seeds_by_modality[modality] = []
        if not idx_list:
            continue
        kk = min(per_modality_k[modality], len(idx_list))
        if kk <= 0:
            continue
        top_idx = sorted(idx_list, key=lambda i: similarities[i], reverse=True)[:kk]
        for idx in top_idx:
            idx = int(idx)
            concept_text = index.concept_store.hash_id_to_text[concept_hash_ids_list[idx]]
            seed_concept_indices.append(idx)
            seed_concept_hash_ids.append(concept_hash_ids_list[idx])
            seed_concept_texts.append(concept_text)
            seed_concept_scores.append(float(similarities[idx]))
            seeds_by_modality[modality].append({
                "concept": concept_text,
                "hash_id": concept_hash_ids_list[idx],
                "score": float(similarities[idx]),
            })
    log(f"  seed concept (whole-question embedding, text top-{k_text} + audio top-{k_audio}): {seed_concept_texts}")
    return (seed_concept_indices, seed_concept_texts, seed_concept_hash_ids, seed_concept_scores,
            seeds_by_modality)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def calculate_concept_scores(index: AudioIndex, question_embedding, seed_concept_hash_ids, seed_concept_scores):
    """BFS score propagation from seed concepts through concept<->unit links.

    Text concepts can propagate over multiple hops via text units. Centroid
    propagation converges after one hop, since hard clustering makes a
    centroid's unit links point right back to itself (a future soft-
    clustering configuration would let this keep propagating).
    """
    cfg = index.cfg
    actived_concepts: dict[str, tuple] = {}
    concept_weights = np.zeros(index.graph.vcount())

    for seed_hash_id, seed_score in zip(seed_concept_hash_ids, seed_concept_scores):
        if seed_hash_id not in index.node_name_to_vertex_idx:
            continue  # e.g. an NER concept absent from the graph
        node_idx = index.node_name_to_vertex_idx[seed_hash_id]
        actived_concepts[seed_hash_id] = (node_idx, seed_score, 1)
        concept_weights[node_idx] = seed_score

    used_unit_hash_ids: set[str] = set()
    current_concepts = actived_concepts.copy()
    iteration = 1
    q_emb = question_embedding.reshape(-1, 1) if question_embedding.ndim == 1 else question_embedding

    while current_concepts and iteration < cfg.max_iterations:
        new_concepts: dict[str, tuple] = {}
        for concept_hash_id, (_, concept_score, tier) in current_concepts.items():
            if concept_score < cfg.iteration_threshold:
                continue

            unit_hash_ids = [
                uid for uid in index.concept_hash_id_to_unit_hash_ids.get(concept_hash_id, [])
                if uid not in used_unit_hash_ids
            ]
            if not unit_hash_ids:
                continue

            unit_indices = [index.unit_store.hash_id_to_idx[uid] for uid in unit_hash_ids]
            unit_embeddings = index.unit_embeddings[unit_indices]
            unit_similarities = np.dot(unit_embeddings, q_emb).flatten()
            top_indices = np.argsort(unit_similarities)[::-1][:cfg.top_k_unit]
            for ti in top_indices:
                top_unit_hash_id = unit_hash_ids[ti]
                top_unit_score = unit_similarities[ti]
                used_unit_hash_ids.add(top_unit_hash_id)
                for next_concept_hash_id in index.unit_hash_id_to_concept_hash_ids.get(top_unit_hash_id, []):
                    # Score passed to a neighbor = (current concept score) * (unit-question similarity).
                    next_score = concept_score * top_unit_score
                    if next_score < cfg.iteration_threshold or next_concept_hash_id not in index.node_name_to_vertex_idx:
                        continue
                    next_idx = index.node_name_to_vertex_idx[next_concept_hash_id]
                    concept_weights[next_idx] += next_score  # accumulates across multiple paths
                    new_concepts[next_concept_hash_id] = (next_idx, next_score, tier + 1)
        actived_concepts.update(new_concepts)
        current_concepts = new_concepts.copy()
        iteration += 1

    return concept_weights, actived_concepts


def dense_chunk_retrieval(index: AudioIndex, question_embedding):
    """Question text embedding <-> chunk audio-feature embedding dot product
    (same PE-AV space)."""
    q_emb = question_embedding.reshape(1, -1)
    sims = np.dot(index.chunk_embeddings, q_emb.T).flatten()
    sorted_indices = np.argsort(sims)[::-1]
    return sorted_indices, sims[sorted_indices].tolist()


def run_ppr(index: AudioIndex, node_weights):
    cfg = index.cfg
    reset_prob = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)
    if reset_prob.sum() <= 0:
        # Every seed can end up with negative (zeroed) cosine similarity to the
        # question, leaving all weights at 0 and the reset vector summing to 0
        # -- igraph would raise on that. Fall back to a uniform reset (plain
        # PageRank) instead of crashing.
        reset_prob = np.full(index.graph.vcount(), 1.0 / index.graph.vcount())

    pagerank_scores = index.graph.personalized_pagerank(
        vertices=range(index.graph.vcount()), damping=cfg.damping, directed=False,
        weights="weight", reset=reset_prob, implementation="prpack",
    )

    doc_scores = np.array([pagerank_scores[idx] for idx in index.chunk_node_indices])
    sorted_local = np.argsort(doc_scores)[::-1]
    sorted_scores = doc_scores[sorted_local]
    sorted_chunk_hash_ids = [index.vertex_idx_to_node_name[index.chunk_node_indices[i]] for i in sorted_local]
    return sorted_chunk_hash_ids, sorted_scores.tolist()
