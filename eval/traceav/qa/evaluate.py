from __future__ import annotations

import json
import time
from pathlib import Path

from audiohelix.prompts import (
    normalize_correct_options, normalize_option_letters, parse_response_to_prediction,
)
from audiohelix.retrieve.soft_centroid_weights import aggregate_weight_stats, weight_meta
from audiohelix.utils import log

from ..setup import config
from . import prompts


def parse(raw: str, record: dict):
    """Return deduplicated option letters found in the raw response."""
    return ",".join(parse_response_to_prediction(raw))


def is_correct(record: dict, raw: str, parsed) -> bool:
    """Return whether predicted and correct option sets match."""
    pred = normalize_option_letters(str(parsed).split(","))
    correct = normalize_correct_options(record["correct_options"])
    return sorted(pred) == sorted(correct)


def summarize(rows: list[dict]) -> dict:
    """Compute overall and per-task accuracy."""
    summary: dict = {"overall": {"n": 0, "correct": 0, "accuracy": 0.0}, "by_task_type": {}}
    correct = sum(1 for r in rows if is_correct(r, r.get("raw_answer", ""), r["pred_answer"]))
    summary["overall"] = {"n": len(rows), "correct": correct,
                          "accuracy": correct / len(rows) if rows else 0.0}

    by_task: dict[str, list[bool]] = {}
    for r in rows:
        by_task.setdefault(r["task_type"], []).append(
            is_correct(r, r.get("raw_answer", ""), r["pred_answer"]))
    for task, flags in sorted(by_task.items()):
        c = sum(flags)
        summary["by_task_type"][task] = {"n": len(flags), "correct": c,
                                         "accuracy": c / len(flags) if flags else 0.0}
    return summary


_RESULTS = "results.json"
_ROW_ORDER = (
    "qa_prompt", "retrieved_chunks", "parsed_texts", "audio_units",
    "elapsed_sec", "audio_mode", "retrieval_top_k", "audio_top_m",
    "chunk_segments", "unit_segments", "n_total_by_cid", "n_audio_inputs",
    "n_text_only_chunks", "pos_emb_stats", "kept_cids",
)

_GUARD_MSG = ("the existing results.json has a different audio_mode -- "
              "resuming under different conditions would mix results. "
              "Use a new folder (or --force).")

_RUN: dict = {}                          # {"out_dir", "model"}
_ROWS: dict[str, list[dict]] = {}        # out_dir -> the in-memory results list


def startup(out_dir, args, questions) -> None:
    """Initialize run state and create diagnostic folders."""
    out_dir = Path(out_dir)
    _RUN.clear()
    _RUN.update({"out_dir": out_dir, "model": args.model})
    subs = ("centroid_links", "soft_weights", "seeds", "audio_units")
    for sub in subs:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    log(f"{config.NAME}: {len(questions)} questions, diagnostics -> "
        + ", ".join(subs))


def row_key(record: dict) -> tuple:
    """Return the question key used for resume."""
    return (record[config.ID_FIELD], record["question"])


def load_done(out_dir, args) -> set:
    """Load completed question keys and the in-memory results list."""
    key = str(out_dir)
    out_path = Path(out_dir) / _RESULTS
    if getattr(args, "force", False) or not out_path.exists():
        _ROWS[key] = []
        return set()
    rows = json.loads(out_path.read_text(encoding="utf-8"))
    _ROWS[key] = rows
    return {row_key(r) for r in rows}


def resume_guard(out_dir, args) -> None:
    """Reject resume when the stored audio mode differs."""
    if getattr(args, "force", False):
        return
    key = str(out_dir)
    rows = _ROWS.get(key)
    if rows is None:
        out_path = Path(out_dir) / _RESULTS
        if not out_path.exists():
            return
        rows = json.loads(out_path.read_text(encoding="utf-8"))
        _ROWS[key] = rows
    if any(r.get("audio_mode") != config.AUDIO_MODE for r in rows):
        raise SystemExit(_GUARD_MSG)


def error_row(out_dir, record, args, kind) -> None:
    """TraceAV's runners never wrote an error row: a failing question was
    logged and the loop moved on. Kept as a log-only sink for the shared
    driver (kind is the same "no index at ..."/"no_chunks"/
    "retrieval_error: ..." string the other eval store)."""
    log(f"  {record[config.ID_FIELD]}: {kind}")


def result_row(record, raw, parsed, correct: bool, extra: dict) -> dict:
    """Build one TraceAV results row."""
    del correct
    row = {**record, "pred_answer": parsed, "raw_answer": raw}
    for k in _ROW_ORDER:
        if k not in extra:
            continue
        v = extra[k]
        if k == "pos_emb_stats":
            v = v or None
        row[k] = v
    return row


def emit(out_dir, record, raw, parsed, row) -> None:
    """Append a row and rewrite results.json."""
    rows = _ROWS.setdefault(str(out_dir), [])
    rows.append(row)
    (Path(out_dir) / _RESULTS).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def finalize(out_dir, run: dict, args) -> dict:
    """Evaluate saved rows, write summary files, and return the summary."""
    out_dir = Path(out_dir)
    rows = _read_rows(out_dir)
    summary = summarize(rows)
    summary["meta"] = _meta(rows, run, args, out_dir)
    _write_prompt_txt(out_dir, args)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"saved: {out_dir / _RESULTS} ({len(rows)} questions)")
    log(f"overall accuracy: {summary['overall']['accuracy']:.4f} "
        f"({summary['overall']['correct']}/{summary['overall']['n']})")
    for task, s in summary["by_task_type"].items():
        log(f"  {task}: {s['accuracy']:.4f} ({s['correct']}/{s['n']})")
    return summary


def _read_rows(out_dir: Path) -> list[dict]:
    out_path = out_dir / _RESULTS
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    return _ROWS.get(str(out_dir)) or []


def _write_prompt_txt(out_dir: Path, args) -> None:
    """Write the run's instruction template to prompt.txt."""
    prompt_file = out_dir / "prompt.txt"
    if prompt_file.exists():
        return
    try:
        text = prompts.instruction({"question": "<question>", "options": {"A": "..."}},
                                   args.model, concat=False)
        prompt_file.write_text(text + "\n", encoding="utf-8")
    except Exception:                                             # noqa: BLE001
        pass


def _cluster_dict(args) -> dict:
    c = config.cluster_config(args)
    return {"method": "spectral",
            "spectral_n_clusters": c.spectral_n_clusters}


def _nearest(args) -> int:
    return getattr(args, "nearest_centroids", None) or config.NEAREST_CENTROIDS


def _sim(args) -> float:
    v = getattr(args, "centroid_sim_threshold", None)
    return config.CENTROID_SIM_THRESHOLD if v is None else v


def _common(args, run, out_dir: Path) -> dict:
    """The meta block's head with consistent expressions and ordering.
    runner)."""
    return {
        "index_model": args.index_model,
        "qa_model": args.model,
        "model": args.model,             # backward-compat key for older summary readers
        "model_tag": config.model_tag(args.model, args),
        # The graph's captioner is fixed and is never a QA backbone, so this is
        # always True now; kept as run provenance rather than dropped.
        "cross": args.index_model != args.model,
    }


def _meta(rows: list[dict], run: dict, args, out_dir: Path) -> dict:
    pos_emb_on = args.pos_emb_llm == "on"
    meta = {
        **_common(args, run, out_dir),
        "presentation": "unit_separate_seg_of_total_no_entities",
        "prompt_source": prompts.prompt_source_of(args.model),
        "audio_mode": (config.AUDIO_MODE_AF3 if args.model == "audio_flamingo3"
                       else config.AUDIO_MODE),
        "audio_input": ("each selected unit as its own audio (never concatenated -- up to "
                        "audio_top_m per chunk), preceded by `audio chunk {i}'s segment "
                        "{n} of {m} (0:25-0:30):`"
                        + (" / AF3: single concatenated waveform + "
                           "`audio chunks 1-N's segments (concatenated):` label"
                           if args.model == "audio_flamingo3" else "")),
        "text_input": ("parsed_text once after each chunk's audio group, label "
                       "`audio chunk {i}'s textual information:` (Transcription/Caption "
                       "lines, no Entities line -- concepts are still used by retrieval's "
                       "graph nodes)"),
        "concepts_in_text": False,
        **weight_meta(),
        "text_scheme": "Transcription: .../ Caption: ...  (no `Entities:` line)",
        "label_scheme": ("audio chunk {i}'s segment {n} of {m} ({m}:{ss}–{m}:{ss}): "
                         "(i=1-based chunk, n=that chunk's 1-based unit "
                         "index, m=that chunk's total unit count -- a video's last chunk is "
                         "smaller since it's under 45s. Time range is position within that "
                         "chunk's own audio, not an absolute video timestamp) / "
                         "audio chunk {i}'s textual information:"),
        "seed_method": "query_embedding_topk_per_modality",
        "seed_top_k_text": run["seed_top_k_text"],
        "seed_top_k_audio": run["seed_top_k_audio"],
        "retrieval_top_k": args.retrieval_top_k,
        "audio_top_m": args.audio_top_m,
        "unit_score_formula": "query_sim (no NER concepts)",
        "nearest_centroids": _nearest(args),
        "centroid_sim_threshold": _sim(args),
        "centroid_link": "topk_nearest_retrieve_time",
        "cluster_tag": run["cluster_tag"],
        "unit_sec": args.unit_sec,
        "chunk_sec": args.chunk_sec,
        "cluster_config": _cluster_dict(args),
        "questions_file": str(config.MANIFEST_FILE),
        "limit": args.limit,
        "run_flags": {k: v for k, v in sorted(vars(args).items())},
        "pos_emb_llm": args.pos_emb_llm,
        "pos_emb_layer": "llm_only" if pos_emb_on else None,
        "pos_emb_input": ("local time axis (t_local = seq within chunk x unit_sec) -- "
                          "video timestamp is not used" if pos_emb_on
                          else "concat order (baseline -- no time information)"),
        "pos_emb_note": ("single-segment timemap per audio, so PE reassignment is identity "
                         "(rel = 0..n_tok-1, zero shift, zero jump) for qwen25/qwen3/voxtral. "
                         "The only case where it actually changes is AF3's single concat run. "
                         "Apply count per question should equal the audio count."
                         if pos_emb_on else None),
        "pos_emb_aligned_questions": sum(1 for r in rows if r.get("pos_emb_stats")),
        "pos_emb_unaligned_questions": sum(1 for r in rows if not r.get("pos_emb_stats")),
        "results_dir": str(out_dir),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # Add the run-level soft-weight aggregate after `created`.
    agg = aggregate_weight_stats(_run_weight_stats(run))
    if agg:
        agg["n_videos_in_index"] = len(run.get("n_total_cache") or {}) or agg["n_videos"]
        meta["centroid_weight_stats"] = agg
    return meta


def _run_weight_stats(run: dict) -> list[dict]:
    """The run's per-video weight stats: the driver's dict (unit_id -> stats)
    taken as-is, or a per-question list collapsed by identical dicts (the
    stats are deterministic per index, so repeats are the same video)."""
    ws = run.get("weight_stats")
    if isinstance(ws, dict):
        return [s for s in ws.values() if s]
    return _unique_stats(ws)


def _unique_stats(stats_list) -> list[dict]:
    """Per-question stats -> per-video stats (identical dicts collapsed; order
    of first appearance kept)."""
    out: list[dict] = []
    seen: set[str] = set()
    for s in stats_list or []:
        if not s:
            continue
        sig = json.dumps(s, sort_keys=True, default=str)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(s)
    return out
