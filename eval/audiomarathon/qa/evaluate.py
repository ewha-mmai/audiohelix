"""AudioMarathon scoring and result/summary output."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from audiohelix.present.content_builder import CHUNK_TEXT_FALLBACK  # noqa: F401
from audiohelix.retrieve.soft_centroid_weights import weight_meta
from audiohelix.utils import log

from ..setup import config as cfg
from . import prompts
from ..adapters import load_adapter
from ..setup.data import _as_suites, item_from_record

try:
    from sklearn.metrics import f1_score as _sk_f1_score
except ImportError:  # sklearn may be absent in some environments -- summary falls back to accuracy only
    _sk_f1_score = None


def parse(raw: str, record: dict):
    """Parse a model response with the suite adapter."""
    adapter = load_adapter(record["suite"])
    return adapter.parse_response(raw, item_from_record(record, None))


def is_correct(record: dict, raw: str, parsed) -> bool:
    """Apply the suite adapter's correctness rule."""
    adapter = load_adapter(record["suite"])
    return adapter.is_correct(item_from_record(record, None), raw, parsed)


def row_key(record: dict):
    """Return the resume key for a question."""
    return (record["item_id"], record["qid"])


def _read_results(out_dir) -> list[dict]:
    out_path = Path(out_dir) / "results.json"
    if not out_path.exists():
        return []
    return json.loads(out_path.read_text(encoding="utf-8"))


def load_done(out_dir, args) -> set:
    """Return keys already present in results.json, unless ``--force`` is set."""
    if getattr(args, "force", False):
        return set()
    results = _read_results(out_dir)
    if results:
        log(f"loaded {len(results)} existing results (resume) -> {out_dir}")
    return {(r["item_id"], r["qid"]) for r in results}


def resume_guard(out_dir, args) -> None:
    """Hook for the shared QA resume policy."""
    return None


def error_row(out_dir, record: dict, args, kind: str) -> None:
    """Log a skipped question without writing a result row."""
    qid = record.get("qid")
    if kind == "no_chunks":
        log(f"=== question ({qid}): item has 0 chunks, skipping ===")
    elif kind == "no_index":
        log(f"=== question ({qid}): no index, skipping ===")
    elif kind.startswith("retrieval_error"):
        detail = kind.split(":", 1)[1].strip() if ":" in kind else kind
        log(f"  {qid}: retrieval failed -- {detail}")
    else:
        log(f"  {qid}: {kind}")


def result_row(record: dict, raw: str, parsed, correct: bool, extra: dict) -> dict:
    """Build one results.json row."""
    extra = extra or {}
    row = {
        **record,
        "raw_answer": raw, "pred_answer": parsed, "is_correct": correct,
        "answer_text": record.get("answer_text", ""),   # for the official weighted-F1 rescorer
        "qa_prompt": extra.get("qa_prompt"),
        "retrieved_chunks": extra.get("retrieved_chunks"),
        "parsed_texts": extra.get("parsed_texts"),
        "elapsed_sec": extra.get("elapsed_sec"),
    }
    if extra.get("n_text_only_chunks") is not None:    # refined-engine diagnostic
        row["n_text_only_chunks"] = extra["n_text_only_chunks"]
    return row


def emit(out_dir, record: dict, raw: str, parsed, row: dict, error=None) -> None:
    """Append or replace a result row in results.json."""
    if error is not None or row is None:
        return
    out_dir = Path(out_dir)
    results = _read_results(out_dir)
    key = row_key(record)
    for i, r in enumerate(results):
        if (r.get("item_id"), r.get("qid")) == key:      # re-run of the same question
            results[i] = row
            break
    else:
        results.append(row)
    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def startup(out_dir, args, questions=None) -> None:
    """Create the run folder, diagnostic snapshot, and prompt template."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(_snapshot_source(), out_dir / "run_qa_snapshot.py")
    suites = _as_suites(getattr(args, "suite", None))
    model = getattr(args, "model", "") or ""
    (out_dir / "prompt.txt").write_text(prompts.prompt_txt(suites, model), encoding="utf-8")


def _snapshot_source() -> Path:
    """Return the module used for the per-run diagnostic snapshot."""
    try:
        import importlib
        return Path(importlib.import_module("audiohelix.qa").__file__)
    except Exception:  # noqa: BLE001 -- snapshot is diagnostic only
        return Path(__file__)


def summarize(rows: list[dict]) -> dict:
    """Compute per-suite and overall accuracy and weighted F1."""
    by_suite: dict[str, list[bool]] = {}
    by_suite_f1: dict[str, tuple[list, list]] = {}
    for r in rows:
        by_suite.setdefault(r["suite"], []).append(bool(r["is_correct"]))
        yt, yp = by_suite_f1.setdefault(r["suite"], ([], []))
        if r.get("answer_text") is not None and r.get("pred_answer") is not None:
            yt.append(str(r["answer_text"]))
            yp.append(str(r["pred_answer"]))
    summary: dict = {
        "by_suite": {s: {"accuracy": sum(v) / len(v), "correct": sum(v), "n": len(v)}
                     for s, v in by_suite.items()},
    }
    if _sk_f1_score is not None:
        f1_by_suite: dict[str, float] = {}
        for s, (yt, yp) in by_suite_f1.items():
            if yt:
                f1_by_suite[s] = _sk_f1_score(yt, yp, average="weighted", zero_division=0)
                summary["by_suite"][s]["f1_weighted"] = f1_by_suite[s]

        def _weighted_overall_f1(exclude: tuple[str, ...] = ()) -> dict | None:
            total_n = 0
            wsum = 0.0
            for s, f1 in f1_by_suite.items():
                if s in exclude:
                    continue
                n = summary["by_suite"][s]["n"]
                total_n += n
                wsum += f1 * n
            if not total_n:
                return None
            return {"f1_weighted": wsum / total_n, "n": total_n}

        overall_f1 = _weighted_overall_f1()
        overall_f1_ex_had_slue = _weighted_overall_f1(exclude=("had", "slue"))
        if overall_f1 is not None:
            summary["overall_f1_weighted"] = overall_f1
        if overall_f1_ex_had_slue is not None:
            summary["overall_f1_weighted_ex_had_slue"] = overall_f1_ex_had_slue
    all_correct = [bool(r["is_correct"]) for r in rows]
    if all_correct:
        summary["overall"] = {
            "accuracy": sum(all_correct) / len(all_correct),
            "correct": sum(all_correct), "n": len(all_correct),
        }
    return summary


def _run_flags(args, run) -> dict:
    """Return the run flags stored in summary metadata."""
    return {
        "model": getattr(args, "model", None),
        "index_model": getattr(args, "index_model", None),
        "suite": getattr(args, "suite", None),
        "suites": _as_suites(getattr(args, "suite", None)),
        "concat_audio": bool(getattr(args, "concat_audio", False)),
        "evidence_framing": bool(getattr(args, "evidence_framing", False)),
        "pos_emb_time": getattr(args, "pos_emb_time", "off"),
        "pos_emb_fps": getattr(args, "pos_emb_fps", None),
        "pos_emb_llm": getattr(args, "pos_emb_llm", "off"),
    }


def finalize(out_dir, run, args) -> dict:
    """Write summary.json and return the summary."""
    out_dir = Path(out_dir)
    rows = _read_results(out_dir)
    meta = {
        **_run_flags(args, run),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "centroid_weight": weight_meta(),
    }
    summary = {"meta": meta, **summarize(rows)}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"saved: {out_dir / 'results.json'} ({len(rows)} questions)")
    return summary
