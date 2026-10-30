from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from audiohelix.config import RetrieveConfig
from audiohelix.retrieve.retrieval import (
    AudioIndex, calculate_concept_scores, dense_chunk_retrieval,
    get_seed_concepts_topk, run_ppr,
)
from audiohelix.utils import log

CHUNK_TEXT_FALLBACK = "(no content)"


def format_parsed_text(chunk: dict) -> str:
    """Reassembles a parsed caption-JSON chunk into a text block for the model."""
    parts = []
    if chunk.get("transcription"):
        parts.append(f"Transcription: {chunk['transcription']}")
    if chunk.get("caption"):
        parts.append(f"Caption: {chunk['caption']}")
    concepts = [e for e in chunk.get("concepts", []) if e]
    if concepts:
        parts.append(f"Entities: {', '.join(concepts)}")
    return "\n".join(parts) if parts else CHUNK_TEXT_FALLBACK


def make_query_seed_fn(retrieve_cfg: RetrieveConfig):
    """Wraps audiohelix.retrieve.retrieval.get_seed_concepts_topk as seed_fn(index, question,
    question_embedding). Stashes the question embedding in `state` so it can
    be reused later for audio-unit scoring without re-encoding."""
    state = {"question_embedding": None}

    def seed_fn(index, question, question_embedding):
        state["question_embedding"] = question_embedding
        return get_seed_concepts_topk(
            index, question_embedding,
            retrieve_cfg.seed_top_k_text, retrieve_cfg.seed_top_k_audio,
        )

    return seed_fn, state


def apply_topk_centroid_links(index: AudioIndex, k: int, min_sim: float) -> dict[str, list[dict]]:
    """Replaces audio-unit <-> centroid links with nearest-top-k soft
    links (in memory only, by cosine similarity), instead of the hard
    membership baked in at build time. The nearest centroid is always
    linked; the 2nd..kth are linked only if similarity >= min_sim (min_sim
    <= 0 disables the gate); N=1, min_sim=0 reproduces hard membership
    exactly. Rewrites only centroid_to_unit_ids and the two
    concept<->unit hash-id adjacency maps -- text links, chunk edges,
    the igraph, and concept_cluster_info are untouched; no index file is
    written.

    Returns {audio_unit_id: [{"concept_id", "score"}, ...]} (descending
    score, linked only) for inspection.
    """
    k = max(1, int(k))

    # Centroid matrix C. meta keys (cpt = cpt{seq}) are resolved via the concept
    # store's hash (content == concept id, so hash lookup works natively).
    centroid_cpts: list[str] = []
    centroid_hash_ids: list[str] = []
    for cpt in sorted(index.centroid_to_unit_ids):
        h = index.concept_hash_id.get(cpt)
        if h is None or h not in index.concept_store.hash_id_to_idx:
            continue
        centroid_cpts.append(cpt)
        centroid_hash_ids.append(h)
    if not centroid_cpts:
        log("  soft-link warning: no centroids in concept store, skipping")
        return {}
    c_idx = np.array([index.concept_store.hash_id_to_idx[h] for h in centroid_hash_ids], dtype=np.intp)
    C = index.concept_embeddings[c_idx]                        # (n_c, 1024) float32

    # Audio-unit matrix A. All keys of audio_unit_id_to_chunk are
    # "every audio unit in this video" (deterministic, independent of the
    # hard membership values).
    audio_pairs: list[tuple[str, str]] = []                   # (native uid, hash id)
    for uid in sorted(index.audio_unit_id_to_chunk):
        sh = index.unit_hash_id.get(uid)
        if sh is None or sh not in index.unit_store.hash_id_to_idx:
            log(f"  soft-link warning: {uid} not in unit store, skipping")
            continue
        audio_pairs.append((uid, sh))
    if not audio_pairs:
        return {}
    a_idx = np.array([index.unit_store.hash_id_to_idx[sh] for _, sh in audio_pairs], dtype=np.intp)
    A = index.unit_embeddings[a_idx]                      # (n_s, 1024) float32

    # Similarity + top-k. If fewer centroids than k exist, link to all of them.
    # Ties broken by stable argsort on concept index for determinism.
    n_c = len(centroid_cpts)
    kk = min(k, n_c)
    if kk < k:
        log(f"  soft-link warning: only {n_c} centroids < N={k} -- linking each audio unit to the nearest {kk}")
    S = A @ C.T                                               # (n_s, n_c) float32
    top_cols = np.argsort(-S, axis=1, kind="stable")[:, :kk]

    # Rebuild adjacency: copy text-side entries unchanged, replace only the
    # centroid<->audio-unit side with the gated top-k result.
    centroid_hash_set = set(centroid_hash_ids)
    audio_hash_set = {sh for _, sh in audio_pairs}
    concept_adj: dict[str, list[str]] = defaultdict(list)         # concept(hash) -> unit(hash)
    for eh, shs in index.concept_hash_id_to_unit_hash_ids.items():
        if eh not in centroid_hash_set:
            concept_adj[eh] = list(shs)
    unit_adj: dict[str, list[str]] = defaultdict(list)       # unit(hash) -> concept(hash)
    for sh, ehs in index.unit_hash_id_to_concept_hash_ids.items():
        if sh not in audio_hash_set:
            unit_adj[sh] = list(ehs)

    new_c2s: dict[str, list[str]] = defaultdict(list)
    dump: dict[str, list[dict]] = {}
    n_kept = 0
    n_gated = 0
    for i, (uid, sh) in enumerate(audio_pairs):
        kept: list[dict] = []
        for ci, col in enumerate(top_cols[i].tolist()):
            sim = float(S[i, col])
            if ci > 0 and min_sim > 0 and sim < min_sim:      # nearest is always linked
                n_gated += 1
                continue
            eh = centroid_hash_ids[col]
            concept_adj[eh].append(sh)
            unit_adj[sh].append(eh)
            cpt = centroid_cpts[col]
            new_c2s[cpt].append(uid)
            kept.append({"concept_id": cpt, "score": sim})
            n_kept += 1
        dump[uid] = kept

    index.centroid_to_unit_ids = {cpt: sorted(uids) for cpt, uids in new_c2s.items()}
    index.concept_hash_id_to_unit_hash_ids = concept_adj
    index.unit_hash_id_to_concept_hash_ids = unit_adj
    log(f"  soft links: {len(audio_pairs)} audio units x nearest-top-{kk} -> "
        f"{n_kept} links kept, {n_gated} dropped below sim {min_sim if min_sim > 0 else 0}")
    return dump


def _audio_unit_seq(uid: str) -> int:
    """Extracts seq (time position) from an audio-unit id "{chunk_id}_a{seq:02d}"."""
    return int(uid.rsplit("_a", 1)[1])


def chunk_unit_counts(index: AudioIndex) -> dict[str, int]:
    """chunk_id -> total number of units that chunk has (the label's
    `of {m}`), counted from the index rather than disk so it stays
    consistent even if wav files are deleted mid-run."""
    cnt: dict[str, int] = {}
    for cid in index.audio_unit_id_to_chunk.values():
        cnt[cid] = cnt.get(cid, 0) + 1
    return cnt


def pick_top_audio_units(
    index: AudioIndex, video_id: str, cids: list[str],
    question_embedding: np.ndarray, concept_embeddings: np.ndarray | None,
    m: int, units_dir: Path,
) -> tuple[list[list[dict]], dict]:
    """For each retrieved chunk, picks its top-m own audio units by
    query similarity (every caller passes concept_embeddings=None, so only
    that path is exercised), ranked by (-score, seq) then reordered
    chronologically. cids keeps retrieve_one's order. Returns (per_cid,
    stats): per_cid[i] is cids[i]'s chosen-unit list with fields
    {chunk_id, unit_id, seq, wav_path, query_sim, max_concept_sim, score}.
    """
    m = max(1, int(m))
    n_cids = len(cids)
    per_cid: list[list[dict]] = [[] for _ in range(n_cids)]

    cid_set = set(cids)
    if not cid_set:
        return per_cid, {"formula": "query_sim", "n_chunks": 0,
                         "n_candidates": 0, "n_chosen": 0,
                         "skipped_no_store": 0, "skipped_no_wav": 0}

    # Collect candidates, batching the store rows for one batched dot product.
    cand_rows: list[int] = []          # unit-store row indices
    cand_info: list[tuple[int, str, int]] = []   # (cid position, uid, seq)
    skipped_no_store = 0
    for pi, cid in enumerate(cids):
        uids = sorted(
            (uid for uid, p in index.audio_unit_id_to_chunk.items() if p == cid),
            key=_audio_unit_seq,
        )
        for uid in uids:
            sh = index.unit_hash_id.get(uid)
            if sh is None or sh not in index.unit_store.hash_id_to_idx:
                skipped_no_store += 1        # e.g. a NaN-dropped unit
                continue
            cand_rows.append(index.unit_store.hash_id_to_idx[sh])
            cand_info.append((pi, uid, _audio_unit_seq(uid)))

    stats = {"n_chunks": n_cids, "n_candidates": len(cand_rows),
             "n_chosen": 0, "skipped_no_store": skipped_no_store, "skipped_no_wav": 0}
    if not cand_rows:
        stats["formula"] = "mean(query_sim, max_concept_sim)" if concept_embeddings is not None else "query_sim"
        return per_cid, stats

    A = index.unit_embeddings[np.array(cand_rows, dtype=np.intp)]   # (N, 1024) float32
    sim_q = np.dot(A, question_embedding)                              # (N,)
    if concept_embeddings is not None:
        sim_e_max = np.dot(A, concept_embeddings.T).max(axis=1)         # (N,)
        score = 0.5 * (sim_q + sim_e_max)
        stats["formula"] = "mean(query_sim, max_concept_sim)"
    else:
        sim_e_max = np.full(sim_q.shape, np.nan)
        score = sim_q
        stats["formula"] = "query_sim"

    # Per-chunk top-m, ranked by (-score, seq) for determinism, then
    # reordered by seq (chronological).
    skipped_no_wav = 0
    n_chosen = 0
    by_cid: dict[int, list[int]] = defaultdict(list)                   # cid position -> candidate index
    for j, (pi, _, _) in enumerate(cand_info):
        by_cid[pi].append(j)

    for pi, cid in enumerate(cids):
        js = by_cid.get(pi, [])
        if not js:
            continue
        ranked = sorted(js, key=lambda j: (-float(score[j]), cand_info[j][2]))
        picked: list[dict] = []
        for j in ranked[:min(m, len(ranked))]:
            _, uid, seq = cand_info[j]
            wav = units_dir / video_id / f"{uid}.wav"
            if not wav.exists():          # e.g. deleted after the index was built
                skipped_no_wav += 1
                continue
            picked.append({
                "chunk_id": cid,
                "unit_id": uid,
                "seq": seq,
                "wav_path": str(wav),
                "query_sim": float(sim_q[j]),
                "max_concept_sim": None if concept_embeddings is None else float(sim_e_max[j]),
                "score": float(score[j]),
            })
        picked.sort(key=lambda r: r["seq"])
        per_cid[pi] = picked
        n_chosen += len(picked)
    stats["n_chosen"] = n_chosen
    stats["skipped_no_wav"] = skipped_no_wav
    return per_cid, stats


def calculate_chunk_scores(index: AudioIndex, actived_concepts: dict):
    """Chunk score = activated-concept bonus only (no dense term).

    Every chunk starts at 0; if an activated concept has mention-ratio /
    centroid-affinity weight w>0 within a chunk, that chunk gets

        chunk_score = ln(1 + sum_over_activated_concepts(concept_score(e) * ln(1+w) / tier(e)))

    A chunk with no concept bonus keeps a reset weight of 0 (reachable only
    through PPR diffusion). Scaled by cfg.chunk_node_weight.
    """
    cfg = index.cfg
    chunk_weights = np.zeros(index.graph.vcount())
    chunk_hash_id_to_native = {h: cid for cid, h in index.chunk_hash_id.items()}
    for chunk_hash_id in index.chunk_store.hash_ids:
        chunk_id = chunk_hash_id_to_native.get(chunk_hash_id)
        if chunk_id is None:
            continue
        weights_for_chunk = index.chunk_concept_weights_merged.get(chunk_id, {})
        total_concept_bonus = 0.0
        for concept_hash_id, (_, concept_score, tier) in actived_concepts.items():
            concept_key = index.concept_store.hash_id_to_text.get(concept_hash_id)
            if concept_key is None:
                continue
            w = weights_for_chunk.get(concept_key, 0.0)
            if w > 0:
                denom = tier if tier >= 1 else 1
                total_concept_bonus += concept_score * math.log(1 + w) / denom
        if total_concept_bonus > 0 and chunk_hash_id in index.node_name_to_vertex_idx:
            chunk_weights[index.node_name_to_vertex_idx[chunk_hash_id]] = (
                math.log(1 + total_concept_bonus) * cfg.chunk_node_weight)
    return chunk_weights


def retrieve_one(index: AudioIndex, question: str, seed_fn=None) -> tuple[list[str], list[str], dict]:
    """One question -> (top-k chunk texts, chunk wav paths, per-modality seed detail).

    seed (question embedding vs. concept embeddings, per-modality top-k, no
    NER) -> BFS activation (calculate_concept_scores) -> chunk score
    (bonus-only) -> PPR over concept+chunk reset weights -> top
    retrieval_top_k -> chronological reorder. Falls back to dense-only
    retrieval if the seed set is empty (no concepts to compute a bonus from).
    """
    t_total = time.time()
    question_embedding = index.peav.encode_text([question])[0]
    seed_fn = seed_fn or (lambda idx, q, q_emb: get_seed_concepts_topk(
        idx, q_emb, index.cfg.seed_top_k_text, index.cfg.seed_top_k_audio))
    (seed_indices, seed_texts, seed_hash_ids, seed_scores, seeds_by_modality) = seed_fn(
        index, question, question_embedding)

    if seed_texts:
        concept_weights, actived_concepts = calculate_concept_scores(
            index, question_embedding, seed_hash_ids, seed_scores)
        chunk_weights = calculate_chunk_scores(index, actived_concepts)
        node_weights = concept_weights + chunk_weights
        sorted_chunk_hash_ids, _ = run_ppr(index, node_weights)
        final_hash_ids = sorted_chunk_hash_ids[: index.cfg.retrieval_top_k]
    else:
        sorted_indices, _ = dense_chunk_retrieval(index, question_embedding)
        final_hash_ids = [index.chunk_store.hash_ids[i]
                          for i in sorted_indices[: index.cfg.retrieval_top_k]]

    hash_to_cid = {h: cid for cid, h in index.chunk_hash_id.items()}

    def _time_idx(h: str) -> int:
        cid = hash_to_cid.get(h)
        return int(cid.rsplit("_ch", 1)[-1]) if cid else 10 ** 9

    final_hash_ids = sorted(final_hash_ids, key=_time_idx)
    chunk_texts: list[str] = []
    chunk_audio_paths: list[str] = []
    for h in final_hash_ids:
        cid = hash_to_cid.get(h)
        if cid is None:
            continue
        audio_path = index.chunk_id_to_audio_path.get(cid, "")
        if not audio_path:
            continue
        chunk_texts.append(index.chunk_store.hash_id_to_text.get(h, ""))
        chunk_audio_paths.append(audio_path)
    log(f"  retrieve (seed=question-embedding top-k, chunk score=dense-free bonus): "
        f"{time.time() - t_total:.2f}s, {len(chunk_audio_paths)} chunks selected")
    return chunk_texts, chunk_audio_paths, seeds_by_modality


# ---------------------------------------------------------------------------
# AudioMarathon/MMAU-Pro only (global variant) -- TraceAV doesn't use these.
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, "
                  "Alibaba Group, capable of perceiving auditory and visual "
                  "inputs, as well as generating text and speech.")


def pick_top_audio_units_global(
    index: AudioIndex, video_id: str, cids: list[str],
    question_embedding: np.ndarray, budget: int, units_dir: Path,
) -> tuple[list[list[dict]], dict]:
    """Like pick_top_audio_units, but selects a global top-`budget` across
    the whole retrieved-chunk pool instead of a fixed top-m per chunk.
    Same candidate collection and scoring (query similarity only); ranking key
    is (-score, cid position, seq) for determinism."""
    budget = max(0, int(budget))
    n_cids = len(cids)
    per_cid: list[list[dict]] = [[] for _ in range(n_cids)]
    base_stats = {"formula": "query_sim", "n_chunks": n_cids, "n_candidates": 0,
                  "n_chosen": 0, "skipped_no_store": 0, "skipped_no_wav": 0,
                  "budget": budget, "kept": 0, "kept_chunks": 0}
    if n_cids == 0 or budget == 0:
        return per_cid, base_stats

    cand_rows: list[int] = []
    cand_info: list[tuple[int, str, int]] = []
    skipped_no_store = 0
    for pi, cid in enumerate(cids):
        uids = sorted(
            (uid for uid, p in index.audio_unit_id_to_chunk.items() if p == cid),
            key=_audio_unit_seq,
        )
        for uid in uids:
            sh = index.unit_hash_id.get(uid)
            if sh is None or sh not in index.unit_store.hash_id_to_idx:
                skipped_no_store += 1
                continue
            cand_rows.append(index.unit_store.hash_id_to_idx[sh])
            cand_info.append((pi, uid, _audio_unit_seq(uid)))
    stats = dict(base_stats)
    stats["n_candidates"] = len(cand_rows)
    stats["skipped_no_store"] = skipped_no_store
    if not cand_rows:
        return per_cid, stats

    pool_rows: list[int] = []
    pool_info: list[tuple[int, str, int, str]] = []
    skipped_no_wav = 0
    for j, (pi, uid, seq) in enumerate(cand_info):
        wav = units_dir / video_id / f"{uid}.wav"
        if not wav.exists():
            skipped_no_wav += 1
            continue
        pool_rows.append(cand_rows[j])
        pool_info.append((pi, uid, seq, str(wav)))
    stats["skipped_no_wav"] = skipped_no_wav
    if not pool_rows:
        return per_cid, stats

    A = index.unit_embeddings[np.array(pool_rows, dtype=np.intp)]
    sim_q = np.dot(A, question_embedding)

    ranked = sorted(
        range(len(pool_rows)),
        key=lambda j: (-float(sim_q[j]), pool_info[j][0], pool_info[j][2]),
    )
    chosen = ranked[: min(budget, len(pool_rows))]
    for j in chosen:
        pi, uid, seq, wav = pool_info[j]
        per_cid[pi].append({
            "chunk_id": cids[pi],
            "unit_id": uid,
            "seq": seq,
            "wav_path": wav,
            "query_sim": float(sim_q[j]),
            "max_concept_sim": None,
            "score": float(sim_q[j]),
        })
    for g in per_cid:
        g.sort(key=lambda r: r["seq"])
    stats["n_chosen"] = len(chosen)
    stats["kept"] = len(chosen)
    stats["kept_chunks"] = sum(1 for g in per_cid if g)
    return per_cid, stats


EVIDENCE_FRAMING = (
    "You are given audio extracted from a video. The audio is a concatenation of the "
    "important segments of the video selected in advance as the key evidence needed to "
    "answer the question."
)


def answer_one_audio_text_units(
    llm, chunk_unit_paths: list[list[str]], chunk_texts: list[str],
    question: str, options: str, prompt_fn, system_text: str | None = None,
    llm_cfg=None, concat_audio: bool = False,
    clip_of: list[int | None] | None = None,
    evidence_framing: bool = False,
) -> dict:
    """QA assembly: chunk texts (n) + per-chunk top-m selected audio
    units. If clip_of is given, multi-clip-unit chunk labels get a
    "(Clip {c})" suffix (same length as chunk_texts; None = single-file unit)."""
    from audiohelix.audio_io import load_wavs_concat
    from audiohelix.config import LLMConfig

    assert len(chunk_unit_paths) == len(chunk_texts), \
        f"audio ({len(chunk_unit_paths)}) and text ({len(chunk_texts)}) lengths differ."
    if clip_of is not None:
        assert len(clip_of) == len(chunk_texts), "clip_of length mismatch"

    llm_cfg = llm_cfg or LLMConfig()
    system_text = system_text or DEFAULT_SYSTEM
    t0 = time.time()

    prompt = prompt_fn(question, options)

    prompt_log_parts: list[str] = []
    user_content = []

    def _chunk_label(i: int) -> str:
        clip = clip_of[i - 1] if clip_of is not None else None
        return f"Passage {i} (Clip {clip})" if clip is not None else f"Passage {i}"

    tmp_concat_path: str | None = None
    if concat_audio:
        flat = [p for ps in chunk_unit_paths for p in ps]
        waveform = load_wavs_concat(flat) if flat else None
        if waveform is not None:
            # Some backends (Voxtral) only accept a file path, not an in-memory
            # array, so write a temp wav (harmless for the Qwen family, which
            # accepts either).
            import soundfile as sf
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, waveform, 16000)
            tmp_concat_path = tmp.name
            user_content.append({"type": "audio", "audio": tmp.name})
            prompt_log_parts.append(f"[AUDIO_CONCAT: {len(flat)} clips merged]\n")
        audio_mode = "audio_units_text_concat"
        for i, t in enumerate(chunk_texts, start=1):
            text_part = f"{_chunk_label(i)} text:\n{t}\n"
            user_content.append({"type": "text", "text": text_part})
            prompt_log_parts.append(text_part)
    else:
        audio_mode = "audio_units_text_separate"
        for i, (unit_paths, t) in enumerate(zip(chunk_unit_paths, chunk_texts), start=1):
            if unit_paths:
                audio_label = f"--- {_chunk_label(i)} Audio ---"
                user_content.append({"type": "text", "text": audio_label})
                prompt_log_parts.append(f"{audio_label}\n")
                for sp in unit_paths:
                    user_content.append({"type": "audio", "audio": sp})
                    prompt_log_parts.append(f"[AUDIO_INPUT: {sp}]\n")
            text_part = f"[{_chunk_label(i)} Text]:\n{t}\n"
            user_content.append({"type": "text", "text": text_part})
            prompt_log_parts.append(f"{text_part}\n")

    instruction_part = f"\n{prompt}"
    if evidence_framing and concat_audio:
        instruction_part = f"\n{EVIDENCE_FRAMING}\n{instruction_part}"
    user_content.append({"type": "text", "text": instruction_part})
    prompt_log_parts.append(instruction_part)

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": user_content},
    ]

    try:
        raw = llm._hf_generate(
            conversation, max_new_tokens=llm_cfg.qa_max_new_tokens,
            do_sample=False, no_repeat_ngram_size=None,
            min_new_tokens=llm_cfg.qa_min_new_tokens,
        )
    finally:
        if tmp_concat_path is not None:
            os.unlink(tmp_concat_path)
    n_audio = sum(len(ps) for ps in chunk_unit_paths)
    log(f"  generating QA answer ({audio_mode}, {n_audio} audio + text): {time.time() - t0:.2f}s")

    return {
        "raw": raw,
        "audio_mode": audio_mode,
        "prompt": "".join(prompt_log_parts),
    }
