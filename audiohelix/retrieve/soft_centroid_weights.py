from __future__ import annotations

import json
import math
import statistics as _st
from collections import defaultdict

from audiohelix.utils import log

DEFAULT_NEAREST_CENTROIDS = 3
DEFAULT_CENTROID_SIM_THRESHOLD = 0.85

_CACHE_ATTR = "_soft_weight_stats"


def _hard_membership(index) -> dict[str, set[str]]:
    """graph_meta.json's **hard** centroid_to_unit_ids (build-time membership)."""
    meta = json.loads((index.index_dir / "graph_meta.json").read_text(encoding="utf-8"))
    return {cpt: set(uids) for cpt, uids in meta["centroid_to_unit_ids"].items()}


def _current_links(index) -> dict[str, set[str]]:
    """The centroid<->unit links currently active on the index (soft if
    soft is active, otherwise hard)."""
    return {cpt: set(uids) for cpt, uids in index.centroid_to_unit_ids.items()}


def ensure_soft_links(index, k: int = DEFAULT_NEAREST_CENTROIDS,
                      min_sim: float = DEFAULT_CENTROID_SIM_THRESHOLD) -> tuple[dict, str]:
    """Ensures the soft link is active, returning (links_dump, source).
    source is `"caller"` if already soft, else `"self"` (installed here --
    links_dump is only populated in that case)."""
    if not index.concept_cluster_info:
        log("  [soft-weights] WARN index has no centroids -- proceeding without soft weights")
        return {}, "none"
    hard = _hard_membership(index)
    current = _current_links(index)
    if current != hard:
        return {}, "caller"                       # already soft -- respect it
    from audiohelix.retrieve.retrieval_core import apply_topk_centroid_links
    dump = apply_topk_centroid_links(index, k, min_sim)
    if not dump:
        raise SystemExit(
            "soft weights: failed to install soft links (apply_topk_centroid_links returned an empty dump) -- "
            "the concept store may have no centroids, or unit links are empty. "
            "Scoring would silently fall back to hard membership instead.")
    return dump, "self"


def compute_soft_centroid_weights(index) -> tuple[dict[str, dict[str, float]], dict]:
    """Computes (w, stats) from the currently active soft link (doesn't modify the index).

    w(p, c) = (count of p's units linked to c) / (total links across all of p's units) -- sum_c w(p, .) = 1.
    """
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    n_links = 0
    n_missing_cid = 0
    n_units = 0
    for cpt, uids in index.centroid_to_unit_ids.items():
        for uid in uids:
            n_links += 1
            cid = index.audio_unit_id_to_chunk.get(uid)
            if cid is None:
                n_missing_cid += 1
                continue
            counts[cid][cpt] = counts[cid].get(cpt, 0) + 1
            n_units += 1

    weights: dict[str, dict[str, float]] = {}
    per_chunk_links: list[int] = []
    for cid, by_cpt in counts.items():
        total = sum(by_cpt.values())
        if total <= 0:
            continue
        weights[cid] = {cpt: cnt / total for cpt, cnt in by_cpt.items()}
        per_chunk_links.append(total)

    stats = {
        "soft_link_source": None,                 # filled in by install
        "n_soft_links": n_links,
        "n_links_without_chunk": n_missing_cid,
        "n_chunks_with_weights": len(weights),
        "n_chunks_total": len(index.chunk_id_to_text),
        "n_units_in_links": n_units,
        "links_per_chunk_mean": (_st.mean(per_chunk_links) if per_chunk_links else 0.0),
        "links_per_chunk_median": (_st.median(per_chunk_links) if per_chunk_links else 0.0),
        "n_centroids_seen": len({cpt for d in weights.values() for cpt in d}),
    }
    return weights, stats


def _verify_sum_one(weights: dict[str, dict[str, float]], tol: float = 1e-9) -> dict:
    """Checks the sum_c w(p, c) = 1 invariant for every chunk."""
    bad = []
    for cid, d in weights.items():
        s = math.fsum(d.values())
        if abs(s - 1.0) > tol:
            bad.append((cid, s))
    return {"sum_one_ok": not bad, "n_violations": len(bad), "examples": bad[:3]}


def _centroid_vertices(index) -> tuple[dict[str, int], int]:
    """cpt -> vertex index, from concept_cluster_info (the authoritative set --
    `centroid_to_unit_ids` would miss a centroid with zero links after a
    soft rewire)."""
    out: dict[str, int] = {}
    missing = 0
    for cpt in index.concept_cluster_info:
        h = index.concept_hash_id.get(cpt)
        idx = index.node_name_to_vertex_idx.get(h) if h else None
        if idx is None:
            missing += 1
            continue
        out[cpt] = idx
    return out, missing


def rewire_centroid_edges(index, weights: dict[str, dict[str, float]]) -> dict:
    """Replaces every chunk<->centroid edge with the (p, c, w>0) pairs
    from `weights`; other edges and vertices are untouched."""
    graph = index.graph
    cent_vidx, n_missing_centroid = _centroid_vertices(index)
    if not cent_vidx and not index.concept_cluster_info:
        return {"igraph_centroid_edges_before": 0, "igraph_centroid_edges_after": 0,
                "igraph_edges_total_after": graph.ecount(),
                "igraph_centroid_edges_match": True, "igraph_mismatch_examples": [],
                "n_centroids_without_vertex": 0, "n_centroids_isolated_after": 0}
    if not cent_vidx:
        raise SystemExit(
            "soft weights: found no centroid vertices in concept_cluster_info -- "
            "cannot rewire the graph (check the index/meta).")
    cent_set = set(cent_vidx.values())

    keep_pairs: list[tuple[int, int]] = []
    keep_w: list[float] = []
    n_dropped = 0
    for e in graph.es:
        u, v = e.tuple
        if u in cent_set or v in cent_set:
            n_dropped += 1                        # chunk<->centroid -- all rebuilt below
            continue
        keep_pairs.append((u, v))
        keep_w.append(float(e["weight"]))

    new_pairs: list[tuple[int, int]] = []
    new_w: list[float] = []
    for cid, by_cpt in weights.items():
        h = index.chunk_hash_id.get(cid)
        p_idx = index.node_name_to_vertex_idx.get(h) if h else None
        if p_idx is None:
            continue
        for cpt, w in by_cpt.items():
            c_idx = cent_vidx.get(cpt)
            if c_idx is None or w <= 0:
                continue
            new_pairs.append((p_idx, c_idx))
            new_w.append(float(w))

    graph.delete_edges(list(range(graph.ecount())))
    if keep_pairs:
        graph.add_edges(keep_pairs)
    if new_pairs:
        graph.add_edges(new_pairs)
    graph.es["weight"] = keep_w + new_w

    # Verification: the graph's chunk<->centroid edges must exactly match `weights`.
    got: dict[tuple[int, int], float] = {}
    for e in graph.es:
        u, v = e.tuple
        if u in cent_set or v in cent_set:
            got[e.tuple] = float(e["weight"])
    want = {(new_pairs[i][0], new_pairs[i][1]): new_w[i] for i in range(len(new_pairs))}
    mismatch = [k for k in set(got) | set(want)
                if abs(got.get(k, float("nan")) - want.get(k, float("nan"))) > 1e-12]
    return {
        "igraph_centroid_edges_before": n_dropped,
        "igraph_centroid_edges_after": len(got),
        "igraph_edges_total_after": graph.ecount(),
        "igraph_centroid_edges_match": not mismatch,
        "igraph_mismatch_examples": mismatch[:3],
        "n_centroids_without_vertex": n_missing_centroid,
        "n_centroids_isolated_after": sum(
            1 for c in cent_set if graph.degree(c) == 0),
    }


def install_soft_centroid_weights(index, k: int = DEFAULT_NEAREST_CENTROIDS,
                                  min_sim: float = DEFAULT_CENTROID_SIM_THRESHOLD,
                                  verbose: bool = True) -> dict:
    """Ensures the soft link is active, computes soft w, and applies it to
    the index (once per index). Returns stats."""
    cached = getattr(index, _CACHE_ATTR, None)
    if cached is not None:
        return cached

    hard_weights = dict(index.chunk_centroid_weights)     # for comparison/stats, before overwriting
    _, source = ensure_soft_links(index, k, min_sim)
    weights, stats = compute_soft_centroid_weights(index)
    stats["soft_link_source"] = source
    stats["nearest_centroids"] = k
    stats["centroid_sim_threshold"] = min_sim
    stats.update(_verify_sum_one(weights))

    # 1. chunk_centroid_weights
    index.chunk_centroid_weights = weights

    # 2. Rebuild the merged weights, same order as before (text first, then
    #    centroid). Key spaces don't collide (text keys are concept strings,
    #    centroid keys are cptNNNN).
    merged: dict[str, dict[str, float]] = defaultdict(dict)
    for cid, d in index.chunk_concept_weights.items():
        merged[cid].update(d)
    for cid, d in weights.items():
        merged[cid].update(d)
    index.chunk_concept_weights_merged = merged

    # 3. Rewire igraph
    stats.update(rewire_centroid_edges(index, weights))

    # Compare hard and soft weights for diagnostics (how much did
    # scoring change for the same chunk/centroid pair).
    hard_pairs = {(cid, cpt): w for cid, d in hard_weights.items() for cpt, w in d.items()}
    soft_pairs = {(cid, cpt): w for cid, d in weights.items() for cpt, w in d.items()}
    common = set(hard_pairs) & set(soft_pairs)
    deltas = [abs(hard_pairs[k] - soft_pairs[k]) for k in common]
    stats.update({
        "hard_edges": len(hard_pairs),
        "soft_edges": len(soft_pairs),
        "edges_common": len(common),
        "edges_hard_only": len(set(hard_pairs) - set(soft_pairs)),
        "edges_soft_only": len(set(soft_pairs) - set(hard_pairs)),
        "w_soft_mean": (_st.mean(soft_pairs.values()) if soft_pairs else 0.0),
        "w_soft_median": (_st.median(soft_pairs.values()) if soft_pairs else 0.0),
        "w_soft_max": (max(soft_pairs.values()) if soft_pairs else 0.0),
        "w_hard_mean": (_st.mean(hard_pairs.values()) if hard_pairs else 0.0),
        "w_hard_median": (_st.median(hard_pairs.values()) if hard_pairs else 0.0),
        "w_hard_max": (max(hard_pairs.values()) if hard_pairs else 0.0),
        "delta_common_mean": (_st.mean(deltas) if deltas else 0.0),
        "delta_common_median": (_st.median(deltas) if deltas else 0.0),
    })

    if not stats["sum_one_ok"] or not stats["igraph_centroid_edges_match"]:
        raise SystemExit(
            f"soft weights: verification failed -- sum_one_ok={stats['sum_one_ok']} "
            f"({stats['n_violations']} violations), igraph_centroid_edges_match="
            f"{stats['igraph_centroid_edges_match']} (e.g. {stats['igraph_mismatch_examples']}). "
            "Continuing with mismatched weights/graph would make the results uninterpretable.")

    setattr(index, _CACHE_ATTR, stats)
    if verbose:
        log(f"  [soft-weights] applied: link source={source}, {stats['n_soft_links']} links / "
            f"{stats['n_chunks_with_weights']}/{stats['n_chunks_total']} chunks, "
            f"w(soft) med {stats['w_soft_median']:.4f} max {stats['w_soft_max']:.4f} "
            f"(hard med {stats['w_hard_median']:.4f} max {stats['w_hard_max']:.4f}), "
            f"edges {stats['igraph_centroid_edges_before']}->"
            f"{stats['igraph_centroid_edges_after']} (soft_only "
            f"{stats['edges_soft_only']}, hard_only {stats['edges_hard_only']})")
    return stats


def weight_stats(index) -> dict | None:
    """The installed weight stats (None if not yet installed) -- a public
    accessor for summaries/verification."""
    return getattr(index, _CACHE_ATTR, None)


def weight_meta() -> dict:
    """The weight-metadata block logged in a run's summary.json."""
    return {
        "centroid_weight": "soft_topk_chunk_normalized",
        "centroid_weight_detail": (
            "w(p, c) = (#{u : chunk(u)=p and c is among u's top-N nearest} / "
            "#{total soft links across p's units}) -- chunk-normalized, sum_c w(p, c) = 1. "
            "Depends on the link config (N, t); actual values recorded under soft_weights/."),
        "centroid_weight_scope": ("activation + chunk bonus + PPR edges, all rewired to "
                                  "soft w. Text concept weights/edges and adjacent-chunk "
                                  "edges (1.0) are unchanged."),
        "centroid_weight_denominator": "chunk (that chunk's total soft link count) -- sum_c w = 1",
        "centroid_weight_links": "installed by `apply_topk_centroid_links` at retrieval time",
    }


def aggregate_weight_stats(stats_list: list[dict]) -> dict | None:
    """Aggregates a list of per-video install stats into one run-level summary
    (`centroid_weight_stats`); None if the list is empty.

    Takes install_soft_centroid_weights's return dicts (= weight_stats(index))
    directly. w values here are **means of per-video means** (videos differ
    in size; for an exact global median, aggregate soft_weights/ directly
    from the output folder instead).
    """
    w_stats_all = [s for s in stats_list if s]
    if not w_stats_all:
        return None
    nv = len(w_stats_all)
    return {
        "n_videos": nv,
        "n_soft_links": sum(s["n_soft_links"] for s in w_stats_all),
        "n_chunks_with_weights": sum(s["n_chunks_with_weights"] for s in w_stats_all),
        "n_chunks_total": sum(s["n_chunks_total"] for s in w_stats_all),
        "n_links_without_chunk": sum(s["n_links_without_chunk"] for s in w_stats_all),
        "edges_soft": sum(s["soft_edges"] for s in w_stats_all),
        "edges_hard": sum(s["hard_edges"] for s in w_stats_all),
        "edges_common": sum(s["edges_common"] for s in w_stats_all),
        "edges_soft_only": sum(s["edges_soft_only"] for s in w_stats_all),
        "edges_hard_only": sum(s["edges_hard_only"] for s in w_stats_all),
        "igraph_centroid_edges_before": sum(
            s["igraph_centroid_edges_before"] for s in w_stats_all),
        "igraph_centroid_edges_after": sum(
            s["igraph_centroid_edges_after"] for s in w_stats_all),
        "w_soft_mean_video_mean": sum(s["w_soft_mean"] for s in w_stats_all) / nv,
        "w_soft_max": max(s["w_soft_max"] for s in w_stats_all),
        "w_hard_mean_video_mean": sum(s["w_hard_mean"] for s in w_stats_all) / nv,
        "w_hard_max": max(s["w_hard_max"] for s in w_stats_all),
        "sum_one_ok_all": all(s["sum_one_ok"] for s in w_stats_all),
        "igraph_centroid_edges_match_all": all(
            s["igraph_centroid_edges_match"] for s in w_stats_all),
    }
