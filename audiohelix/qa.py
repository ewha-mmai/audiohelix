from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from audiohelix.present import pos_emb_llm, pos_emb_time
from audiohelix.build import load_source
from audiohelix.config import LLMConfig, RetrieveConfig
from audiohelix.present.content_builder import format_parsed_text
from audiohelix.models.embedder import build_chunk_embedder
from audiohelix.build.index_builder import load_caption_chunks
from audiohelix.models.llm_backend import build_omni_llm
from audiohelix.retrieve.retrieval import AudioIndex
from audiohelix.retrieve.retrieval_core import (
    apply_topk_centroid_links, chunk_unit_counts, make_query_seed_fn,
    pick_top_audio_units, retrieve_one,
)
from audiohelix.retrieve.soft_centroid_weights import install_soft_centroid_weights, weight_stats
from audiohelix.utils import log, setup_logging


def _b(bench, name, default=None):
    """A per-bench constant/flag, with a faithful default when the bench's
    config.py does not set it."""
    return getattr(bench.config, name, default)


def _audio_mode(bench, args) -> str | None:
    """The run's presentation id, stored in each row so a later resume can tell
    whether the existing rows use the same configuration. AF3 gets its own variant
    of the id: one concatenated audio input reads differently from one input per
    segment."""
    mode = _b(bench, "AUDIO_MODE")
    if mode is None:
        return None
    if args.model == "audio_flamingo3":
        mode = _b(bench, "AUDIO_MODE_AF3", mode)
    return mode


def _dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


@contextmanager
def _pe_context(bench, args, llm, timemaps, stats: dict):
    """Time-aligned positions while generating this question.

    "llm_local_timemaps"       the LLM layer only, over one LOCAL time map per
                               audio input (TraceAV)
    "encoder_and_llm_absolute" the encoder layer, then the LLM layer, over the
                               one ABSOLUTE time map of the concatenated input
                               (AudioMarathon, MMAU-Pro)
    """
    mode = _b(bench, "PE_MODE", "llm_local_timemaps")
    if not timemaps:
        yield
        return
    if mode == "llm_local_timemaps":
        if args.pos_emb_llm != "on":
            yield
            return
        with pos_emb_llm.time_aligned_llm_pos(llm, timemaps) as mr_ctx:
            yield
        st = mr_ctx.stats.get("llm", {})
        stats.update({
            "applied": mr_ctx.n_applied, "skipped": mr_ctx.n_skipped,
            "n_audio": len(timemaps), "skip_reason": mr_ctx.skip_reason,
            "runs": st.get("runs", []), "n_jumps": st.get("n_jumps"),
            "max_jump_sec": st.get("max_jump_sec"),
        })
        return

    time_map = timemaps[0]
    pe_fps = getattr(args, "pos_emb_fps", None)
    if args.pos_emb_time == "on":
        with pos_emb_time.time_aligned_pos_emb(llm, time_map, pe_fps=pe_fps):
            if args.pos_emb_llm == "on":
                with pos_emb_llm.time_aligned_llm_pos(llm, time_map):
                    yield
            else:
                yield
    else:
        with pos_emb_llm.time_aligned_llm_pos(llm, time_map):
            yield


def _n_audio_inputs(pres: dict) -> int:
    """How many audio inputs reach the model: the one merged waveform when the
    presentation concatenates, else one per selected segment."""
    if pres.get("audio_item") is not None:
        return 1
    return len(pres.get("audio_sequence") or [])


def _build_conversation(bench, args, pres: dict, instruction: str) -> list[dict]:
    """The conversation: the bench's retrieved content, then the instruction as
    its own block, then the merged audio input when the presentation is
    concatenated."""
    content = list(pres["content"])
    content.append({"type": "text", "text": f"\n{instruction}"})
    if pres.get("audio_item") is not None:
        content.append(pres["audio_item"])
    msgs: list[dict] = []
    sys_text = bench.prompts.system(args.model)
    if sys_text:
        msgs.append({"role": "system", "content": [{"type": "text", "text": sys_text}]})
    msgs.append({"role": "user", "content": content})
    return msgs


def answer_one(bench, args, run: dict, record: dict, item: dict | None, qi: int) -> None:
    """Retrieve for one question, ask, and persist. Each failure kind follows
    its own policy: "raise" aborts the run, "skip" drops the question, "row"
    records it as an error row."""
    config = bench.config
    paths, out_dir = run["paths"], run["out_dir"]
    cfg = run["llm_cfg"]
    item_id = record[config.ID_FIELD]

    def fail(kind: str, policy: str | None, detail: str = "") -> None:
        """Route one failure kind through its bench policy: "raise" aborts the
        run (traceav -- its runner had no per-question rescue), "skip" drops the
        question, "row" records it as an error row. `kind` is the token the
        bench's error_row stores; `detail` only reaches the log."""
        log(f"  {item_id}: {kind}" + (f" ({detail})" if detail else ""))
        eng = bench.evaluate
        if policy == "row" and hasattr(eng, "error_row"):
            eng.error_row(out_dir, record, args, kind)
        if policy == "raise":
            raise SystemExit(f"{item_id}: {kind}")

    item_n_chunks = (item or {}).get("n_chunks")

    # --- the item's graph -------------------------------------------------
    if item is not None and item_n_chunks == 0:
        fail("no_chunks", _b(bench, "NO_CHUNKS_POLICY", "skip"))
        return
    index_dir = bench.data.index_dir_for(record, paths, run["cluster_tag"])
    if not index_dir.exists():
        fail("no_index", _b(bench, "MISSING_INDEX_POLICY", "skip"), detail=str(index_dir))
        return

    try:
        k = args.retrieval_top_k
        if _b(bench, "INDEX_CAP", "raw") == "min_n_chunks" and item_n_chunks:
            k = max(1, min(k, item_n_chunks))
        retrieve_cfg = RetrieveConfig(
            seed_top_k_text=run["seed_top_k_text"], seed_top_k_audio=run["seed_top_k_audio"],
            retrieval_top_k=k,
        )
        index = AudioIndex(index_dir, run["peav"], retrieve_cfg)

        # Audio-unit -> nearest-centroid links. Both dumps are diagnostics
        # only; the soft weights are what the traversal actually uses.
        nearest, sim = args.nearest_centroids, args.centroid_sim_threshold
        links = apply_topk_centroid_links(index, nearest, sim)
        if _b(bench, "DIAGNOSTICS", False) and links:
            _dump_json(out_dir / "centroid_links" / f"{item_id}.json", links)
        w_stats = install_soft_centroid_weights(index, nearest, sim)
        if _b(bench, "DIAGNOSTICS", False):
            _dump_json(out_dir / "soft_weights" / f"{item_id}.json", w_stats)
        # Once per ITEM (the aggregate uses one entry per video, not per
        # question) -- a item answering several questions must not be counted
        # several times.
        if item_id not in run["weight_stats"]:
            run["weight_stats"][item_id] = weight_stats(index)

        n_total_by_cid = chunk_unit_counts(index)
        run["n_total_cache"][item_id] = n_total_by_cid

        chunks = load_caption_chunks(
            config.captions_dir(paths, args.chunk_sec) / f"{item_id}.json",
            args.chunk_sec) or {}

        # Start the per-question clock here, after the index and
        # caption caches are in place, so `elapsed_sec` measures retrieval ->
        # answer and not the graph load.
        t_q = time.time()
        chunk_texts, chunk_audio_paths, seeds = retrieve_one(
            index, record["question"], seed_fn=run["seed_fn"])
        cids = [Path(ap).stem for ap in chunk_audio_paths]
        parsed_texts = [
            (format_parsed_text(chunks[cid]) if cid in chunks else chunk_texts[i])
            for i, cid in enumerate(cids)
        ]

        per_cid, pick_stats = pick_top_audio_units(
            index, item_id, cids, run["seed_state"]["question_embedding"], None,
            args.audio_top_m, config.units_dir(paths, args.unit_sec, args.chunk_sec))
        unit_paths = [[rec["wav_path"] for rec in per_cid[i]] for i in range(len(cids))]
    except SystemExit:
        raise
    except Exception as e:                                  # noqa: BLE001
        fail(f"retrieval_error: {str(e)[:500]}", _b(bench, "RETRIEVAL_ERROR_POLICY", "raise"))
        return

    if _b(bench, "DIAGNOSTICS", False):
        _diagnostics(bench, args, run, record, qi, item_id, seeds, per_cid, cids, pick_stats)

    # --- present + ask ----------------------------------------------------
    concat = bool(getattr(args, "concat_audio", False) or args.model == "audio_flamingo3")
    pe_on = args.pos_emb_time == "on" or args.pos_emb_llm == "on"
    if pe_on and not concat and _b(bench, "PE_MODE") == "encoder_and_llm_absolute":
        raise SystemExit("--pos-emb-* is on but --concat-audio is off, so the "
                         "concatenated audio order is unknown -- cannot time-align")

    clip_of = bench.data.clip_of(record, item, cids) if hasattr(bench.data, "clip_of") else None
    total_counts = bench.data.total_counts(record, item, n_total_by_cid, cids) \
        if hasattr(bench.data, "total_counts") else n_total_by_cid

    instruction = bench.prompts.instruction(
        record, args.model, concat=concat,
        evidence_framing=bool(getattr(args, "evidence_framing", False)))

    def present(pad_waves: bool = False) -> dict:
        return bench.present.present(
            record, unit_paths, parsed_texts, model=args.model, concat=concat,
            instruction=instruction, clip_of=clip_of, cids=cids, total_counts=total_counts,
            chunks=chunks, item=item, pad_waves=pad_waves, args=args)

    pres = present()
    for w in pres.get("warns", []):
        log(f"  [qa] WARN {w}")

    def logged_prompt(p: dict, prefix: str = "") -> str | None:
        """The `qa_prompt` column: the presentation's body, then the
        instruction, then any tail the bench logs after it (traceav's AF3
        `[AUDIO_CONCAT: ...]` note)."""
        if p.get("prompt_log_body") is None:
            return None
        return prefix + p["prompt_log_body"] + f"\n{instruction}" + (p.get("prompt_log_tail") or "")

    qa_prompt = logged_prompt(pres)

    pe_stats: dict = {}
    temp_files = list(pres.get("temp_files") or [])
    try:
        try:
            raw = _generate(bench, args, cfg, run, _build_conversation(bench, args, pres, instruction),
                            pres.get("timemaps") or [], pe_stats)
        except ValueError as e:
            # One retry for the case where the processor crashes on N waveforms
            # of different lengths: rebuild with every item zero-padded to the
            # longest, and say so in the prompt log. The condition is the
            # fewer than two audio inputs (a single merged AF3
            # waveform is one) means padding cannot help, so re-raise.
            if not _b(bench, "PAD_RETRY", False) or "axes don't match" not in str(e) \
                    or _n_audio_inputs(pres) < 2:
                raise
            log(f"  [qa] WARN processor pad error -- retrying once with zero-pad: {str(e)[:120]}")
            prefix = _b(bench, "PAD_RETRY_PREFIX",
                        "[ZERO-PAD RETRY: all unit waveforms padded to max length]\n")
            pres = present(pad_waves=True)
            temp_files += list(pres.get("temp_files") or [])
            qa_prompt = logged_prompt(pres, prefix)
            raw = _generate(bench, args, cfg, run, _build_conversation(bench, args, pres, instruction),
                            pres.get("timemaps") or [], pe_stats)
    finally:
        for tmp in temp_files:
            try:
                os.remove(tmp)
            except OSError:
                log(f"  [qa] WARN failed to delete temp file: {tmp}")

    # --- score + persist --------------------------------------------------
    parsed = bench.evaluate.parse(raw, record)
    correct = bench.evaluate.is_correct(record, raw, parsed)
    n_audio = _n_audio_inputs(pres)
    n_no_audio = len(parsed_texts) - len(pres.get("chunks_with_audio") or [])
    extra = {
        "qa_prompt": qa_prompt,
        "retrieved_chunks": chunk_texts,
        "parsed_texts": parsed_texts,
        "audio_units": [rec for group in per_cid for rec in group],
        "elapsed_sec": time.time() - t_q,
        "audio_mode": _audio_mode(bench, args),
        "retrieval_top_k": args.retrieval_top_k,
        "audio_top_m": args.audio_top_m,
        "chunk_segments": pres.get("chunk_segments"),
        "unit_segments": pres.get("unit_segments"),
        "n_total_by_cid": total_counts,
        "n_audio_inputs": n_audio,
        "n_text_only_chunks": n_no_audio,
        "n_no_audio_chunks": n_no_audio,
        "pos_emb_stats": pe_stats,
        "kept_cids": cids,
    }
    row = bench.evaluate.result_row(record, raw, parsed, correct, extra)
    bench.evaluate.emit(out_dir, record, raw, parsed, row)
    log(f"  total: {extra['elapsed_sec']:.2f}s -> {parsed!r}"
        + (f" (correct {correct})" if correct is not None else ""))


def _generate(bench, args, cfg, run: dict, conversation, timemaps, pe_stats) -> str:
    llm = run["llm"]
    with _pe_context(bench, args, llm, timemaps, pe_stats):
        return llm._hf_generate(
            conversation, max_new_tokens=cfg.qa_max_new_tokens,
            do_sample=False, no_repeat_ngram_size=None,
            min_new_tokens=cfg.qa_min_new_tokens,
        )


def _diagnostics(bench, args, run, record, qi, item_id, seeds, per_cid, cids,
                 pick_stats) -> None:
    """Write per-question retrieval diagnostics when enabled."""
    out_dir = run["out_dir"]
    key = bench.config.ID_FIELD
    base = {"question_idx": qi + 1, key: item_id,
            "task_type": record.get("task_type"), "question": record["question"]}
    _dump_json(out_dir / "seeds" / item_id / f"{qi + 1:03d}_seeds.json", {
        **base, "seed_method": _b(bench, "SEED_METHOD_NAME", "query_embedding_topk"),
        "seed_top_k_text": run["seed_top_k_text"], "seed_top_k_audio": run["seed_top_k_audio"],
        "text_seeds": (seeds or {}).get("text", []),
        "audio_seeds": (seeds or {}).get("audio", []),
    })
    totals = run["n_total_cache"].get(item_id, {})
    _dump_json(out_dir / "audio_units" / item_id / f"{qi + 1:03d}_units.json", {
        **base, "query_concepts": [], "retrieval_top_k": args.retrieval_top_k,
        "audio_top_m": args.audio_top_m, "stats": pick_stats,
        # restricted to the retrieved cids, exactly as the reference dump does
        # this diagnostic differ from the reference on every question.
        "n_total_by_cid": {cid: totals.get(cid) for cid in cids},
        "chunks": [{"chunk_id": cid, "units": per_cid[i], "n_total": totals.get(cid)}
                     for i, cid in enumerate(cids)],
    })


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run_qa(bench, args) -> dict:
    config = bench.config

    # Refuse unsupported flags/config *before* anything is created: a refused
    # run leaves no results directory behind.
    if _b(bench, "MULTI_CLIP_GRAPHS", False):
        raise SystemExit("MULTI_CLIP_GRAPHS=1 (per-clip indexes) is not part of this "
                         "configured pipeline -- unset it to use the default config.")
    if getattr(args, "only_multi", None) and not _b(bench, "ONLY_MULTI_FLAG", False):
        raise SystemExit(f"--only-multi is not supported by {config.NAME} -- only "
                         f"MMAU-Pro's questions span several clips")
    if getattr(args, "shard", None) and not _b(bench, "SHARD_FLAG", False):
        raise SystemExit(f"--shard is not supported by {config.NAME} -- only "
                         f"MMAU-Pro's runner takes i/n shards")

    paths = config.paths(args.chunk_sec)
    cluster_tag = _b(bench, "CLUSTER_TAG") or config.CLUSTER.tag(args.unit_sec, args.chunk_sec)
    model_tag = config.model_tag(args.model, args)
    base_out = config.out_dir(paths, args, model_tag, cluster_tag)
    out_dir = Path(getattr(args, "out", None) or (base_out / time.strftime("%Y%m%d_%H%M%S")))
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(out_dir / "run_qa.log"))
    log(f"{config.NAME}: index_model={args.index_model} qa_model={args.model} "
        f"cluster={cluster_tag} -> {out_dir}")

    source = load_source(paths, args)
    questions = bench.data.questions(source, args)
    items_by_id = {u["item_id"]: u for u in source["items"]} if isinstance(source, dict) else {}
    bench.evaluate.startup(out_dir, args, questions)

    # Set the device only when asked: an unset CUDA_VISIBLE_DEVICES leaves the
    # ambient environment alone, so a machine
    # with a single GPU needs no flags.
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    dry = bool(getattr(args, "dry_run", False))
    t0 = time.time()
    peav = build_chunk_embedder("pe_av")
    llm = None if dry else build_omni_llm(args.model)
    log(f"models loaded: {time.time() - t0:.1f}s"
        + (" (dry run: no QA model)" if dry else ""))
    if not dry:
        if args.pos_emb_time == "on":
            pos_emb_time.install(llm)
        if args.pos_emb_llm == "on" and pos_emb_llm.install_llm(llm) is None:
            raise SystemExit(f"--pos-emb-llm is on but {args.model} exposes no "
                             f"position-embedding hook to install")

    seed_t, seed_a = args.seed_top_k_text, args.seed_top_k_audio
    seed_fn, seed_state = make_query_seed_fn(RetrieveConfig(
        seed_top_k_text=seed_t, seed_top_k_audio=seed_a))
    run = {
        "paths": paths, "out_dir": out_dir, "llm": llm, "peav": peav,
        "llm_cfg": LLMConfig(model=args.model), "seed_fn": seed_fn, "seed_state": seed_state,
        "cluster_tag": cluster_tag, "seed_top_k_text": seed_t, "seed_top_k_audio": seed_a,
        "retrieval_top_k": args.retrieval_top_k, "n_total_cache": {}, "weight_stats": {},
    }

    done = bench.evaluate.load_done(out_dir, args)
    if not args.force:
        bench.evaluate.resume_guard(out_dir, args)

    # Numbering follows the FULL question list so
    # per-question dumps by that index and records it as `question_idx`, so a
    # resumed or sharded run keeps the same file names as a fresh one.
    todo = [(qi, q) for qi, q in enumerate(questions)
            if args.force or bench.evaluate.row_key(q) not in done]
    log(f"{len(questions)} questions, {len(questions) - len(todo)} already done, "
        f"{len(todo)} to run")

    if dry:
        return _dry_run(bench, args, run, todo, items_by_id)

    for qi, record in todo:
        log(f"=== question {qi + 1}/{len(questions)} ({record[config.ID_FIELD]}) ===")
        try:
            answer_one(bench, args, run, record, items_by_id.get(record[config.ID_FIELD]), qi)
        except SystemExit:
            raise
        except Exception as e:                              # noqa: BLE001 -- isolate per-question failures
            log(f"  QA failed -- {type(e).__name__}: {e}")
            policy = _b(bench, "RETRIEVAL_ERROR_POLICY", "raise")
            if policy == "raise":
                # TraceAV's runner has no per-question rescue: a crash there
                # ends the run instead of silently dropping a row.
                raise
            if policy == "row" and hasattr(bench.evaluate, "error_row"):
                bench.evaluate.error_row(out_dir, record, args, f"qa_error: {str(e)[:500]}")

    summary = bench.evaluate.finalize(out_dir, run, args)
    log(f"done -> {out_dir}")
    return summary


def _dry_run(bench, args, run: dict, todo: list[tuple[int, dict]], items_by_id: dict) -> dict:
    """Assemble every conversation (retrieval + presentation) without loading
    the QA model and dump them -- the byte-level check of the prompt path."""
    out = []
    nearest, sim = args.nearest_centroids, args.centroid_sim_threshold
    concat = bool(getattr(args, "concat_audio", False) or args.model == "audio_flamingo3")
    for qi, record in todo:
        item_id = record[bench.config.ID_FIELD]
        index_dir = bench.data.index_dir_for(record, run["paths"], run["cluster_tag"])
        if not index_dir.exists():
            continue
        item = items_by_id.get(item_id)
        index = AudioIndex(index_dir, run["peav"], RetrieveConfig(
            seed_top_k_text=run["seed_top_k_text"], seed_top_k_audio=run["seed_top_k_audio"],
            retrieval_top_k=args.retrieval_top_k))
        apply_topk_centroid_links(index, nearest, sim)
        install_soft_centroid_weights(index, nearest, sim)
        n_total_by_cid = chunk_unit_counts(index)
        run["n_total_cache"][item_id] = n_total_by_cid
        chunks = load_caption_chunks(
            bench.config.captions_dir(run["paths"], args.chunk_sec) / f"{item_id}.json",
            args.chunk_sec) or {}
        chunk_texts, chunk_audio_paths, _seeds = retrieve_one(
            index, record["question"], seed_fn=run["seed_fn"])
        cids = [Path(ap).stem for ap in chunk_audio_paths]
        parsed_texts = [(format_parsed_text(chunks[cid]) if cid in chunks else chunk_texts[i])
                        for i, cid in enumerate(cids)]
        per_cid, _stats = pick_top_audio_units(
            index, item_id, cids, run["seed_state"]["question_embedding"], None,
            args.audio_top_m, bench.config.units_dir(run["paths"], args.unit_sec, args.chunk_sec))
        unit_paths = [[rec["wav_path"] for rec in per_cid[i]] for i in range(len(cids))]
        instruction = bench.prompts.instruction(
            record, args.model, concat=concat,
            evidence_framing=bool(getattr(args, "evidence_framing", False)))
        pres = bench.present.present(
            record, unit_paths, parsed_texts, model=args.model, concat=concat,
            instruction=instruction,
            clip_of=bench.data.clip_of(record, item, cids) if hasattr(bench.data, "clip_of") else None,
            cids=cids, total_counts=bench.data.total_counts(record, item, n_total_by_cid, cids)
            if hasattr(bench.data, "total_counts") else n_total_by_cid,
            chunks=chunks, item=item, args=args)
        out.append({
            "question_idx": qi + 1, "question": record["question"],
            "retrieved_chunks": chunk_texts, "parsed_texts": parsed_texts,
            "unit_paths": unit_paths, "instruction": instruction,
            "conversation": _build_conversation(bench, args, pres, instruction),
            "prompt_log_body": pres.get("prompt_log_body"),
            "prompt_log_tail": pres.get("prompt_log_tail"),
            "timemaps": [getattr(t, "__dict__", str(t)) for t in (pres.get("timemaps") or [])],
        })
        for tmp in pres.get("temp_files") or []:
            try:
                os.remove(tmp)
            except OSError:
                pass
    dump = run["out_dir"] / "dry_run.json"
    dump.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"dry run: {len(out)} conversations -> {dump}")
    return {"stage": "qa-dry-run", "n": len(out), "out": str(dump)}
