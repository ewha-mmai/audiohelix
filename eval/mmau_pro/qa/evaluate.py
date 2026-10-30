from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from audiohelix.utils import log
from audiohelix.retrieve.soft_centroid_weights import weight_meta

from ..setup import config
from . import prompts
from ..adapters import load_adapter


def parse(raw: str, record: dict) -> str | None:
    """Return the stripped model response, or None when empty."""
    if not raw:
        return None
    return raw.strip()


def is_correct(record: dict, raw: str, parsed) -> None:
    """MMAU-Pro is scored separately by the NVEmbed scoring pass."""
    return None


def summarize(rows: list[dict]) -> dict:
    """Run-completion summary (counts only). MMAU-Pro's accuracy summary is
    written by the separate NVEmbed scoring pass -- it cannot be computed from
    these rows, since they carry no correctness.

    `rows` may be result-row dicts, per-question dicts carrying "error", or
    None entries (result_row() returns None for MMAU); only the error field is
    read."""
    errs = []
    for r in rows or []:
        if isinstance(r, dict):
            e = r.get("error")
            if not e and isinstance(r.get("extra"), dict):
                e = r["extra"].get("error")
            if e:
                errs.append(e)
    errors = Counter(errs)
    n = len(rows or [])
    return {
        "n": n,
        "n_answered": n - sum(errors.values()),
        "n_errors": sum(errors.values()),
        "errors": dict(sorted(errors.items())),
        "note": "scored separately by scoring/score_mmau_pro.py (NVEmbed, base env)",
    }


def result_row(record: dict, raw: str, parsed, correct: bool, extra: dict) -> dict | None:
    """MMAU keeps no in-run row dict: the only record is the pred.jsonl line
    written by emit(). Returning None is the contract for this bench (the old
    _append_mmau_row wrote {"id", "model_output", "error"} directly)."""
    return None


def row_key(record: dict):
    """Return the per-question resume key."""
    return record["meta"].get("id")


def load_done(out_dir, args=None) -> set:
    """Return IDs already present in pred.jsonl."""
    jsonl_path = Path(out_dir) / "pred.jsonl"
    done: set = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
        log(f"loaded {len(done)} existing results (resume) -> {Path(out_dir)}")
    return done


def resume_guard(out_dir, args=None) -> None:
    """No guard for MMAU: unlike the concat/pos-emb runs, nothing here depends
    on a stored audio_mode, so a resume is always safe (M had none)."""
    return None


def _append(out_dir, record: dict, model_output: str, error: str | None) -> None:
    """Appends one line to MMAU's pred.jsonl and flushes immediately (resume support)."""
    jsonl_path = Path(out_dir) / "pred.jsonl"
    rec = {"id": record["meta"].get("id"), "model_output": model_output, "error": error}
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def emit(out_dir, record: dict, raw: str, parsed, row, error: str | None = None) -> None:
    """Write this question's raw output to pred.jsonl."""
    _append(out_dir, record, "" if error else (raw or ""), error)


def error_row(out_dir, record: dict, args, kind: str) -> None:
    """Write a failure record to pred.jsonl."""
    _append(out_dir, record, "", kind)


_FLAG_KEYS = ("model", "index_model", "suite", "concat_audio", "evidence_framing",
              "pos_emb_time", "pos_emb_fps", "pos_emb_llm")


def _flag(args, name: str):
    """Return a CLI value, falling back to this benchmark's default."""
    v = getattr(args, name, None)
    if v is not None:
        return v
    return {
        "suite": config.SUITE,
        "index_model": config.INDEX_MODEL,
        "concat_audio": False,
        "evidence_framing": False,
        "pos_emb_time": config.POS_EMB_TIME_DEFAULT,
        "pos_emb_llm": config.POS_EMB_LLM_DEFAULT,
        "pos_emb_fps": None,
        "model": None,
    }[name]


def startup(out_dir, args, questions=None) -> None:
    """Create the run folder, metadata, prompt template, and diagnostics folder."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "centroid_links").mkdir(exist_ok=True)

    run_flags = {k: _flag(args, k) for k in _FLAG_KEYS}
    (out_dir / "meta.json").write_text(json.dumps({
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "flags": run_flags,
        "limit": getattr(args, "limit", None),
        "centroid_weight": weight_meta(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    model = getattr(args, "model", None)
    instr_tmpl = prompts.AF3_INSTRUCTION if model == "audio_flamingo3" else prompts.INSTRUCTION
    prompt_txt = f"[SYSTEM]\n{prompts.SYSTEM_PROMPT}\n\n[INSTRUCTION]\n{instr_tmpl}"
    if getattr(args, "evidence_framing", False) and model != "audio_flamingo3":
        prompt_txt += f"\n\n[EVIDENCE_FRAMING (concat only)]\n{prompts.EVIDENCE_FRAMING}"
    (out_dir / "prompt.txt").write_text(prompt_txt, encoding="utf-8")


def _run_questions(run: dict, args) -> list[dict]:
    """Return the manifest questions selected for this run."""
    from audiohelix.build import load_source
    from ..setup import data as bench_data
    paths = (run or {}).get("paths") or config.paths()
    return bench_data.questions(load_source(paths, args), args)


def finalize(out_dir, run, args) -> dict:
    """Writes MMAU's pred.parquet (base df's columns +
    model_output/error/audio_count/is_multi_audio), merging this run's question
    range with pred.jsonl's accumulated results across resumed runs, and returns
    the summary dict.

    Row set is the base dataframe filtered to this run's question range.
    Questions never run get the "not_run" marker."""
    out_dir = Path(out_dir)
    suite = config.SUITE
    adapter = load_adapter(suite)
    df = adapter.load_base_df()

    qids = {row_key(q) for q in _run_questions(run, args)}
    jsonl_path = out_dir / "pred.jsonl"
    df = df[df["id"].isin(qids)].reset_index(drop=True)

    df["model_output"] = ""
    df["error"] = None
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                m = df["id"] == rec["id"]
                if m.any():
                    df.loc[m, "model_output"] = rec.get("model_output", "")
                    df.loc[m, "error"] = rec.get("error")
            except (json.JSONDecodeError, KeyError):
                continue
    # Rows never run (e.g. no audio) get an explicit error marker so the
    # The scorer does not count them.
    df.loc[df["model_output"].eq("") & df["error"].isna(), "error"] = "not_run"
    df["audio_count"] = df["audio_path"].apply(
        lambda x: len(x) if hasattr(x, "__len__") else 1)
    df["is_multi_audio"] = df["audio_count"] > 1
    n_failed = int(df["error"].notna().sum())
    n_not_run = int(df["error"].eq("not_run").sum())
    parquet_path = out_dir / "pred.parquet"
    df.to_parquet(parquet_path, index=False)
    log(f"[{suite}] saved pred parquet: {parquet_path} "
        f"({len(df)} rows, {n_failed} errors -- scoring is done separately by "
        f"scoring/score_mmau_pro.py)")

    errors = Counter(df.loc[df["error"].notna(), "error"].tolist())
    return {
        "suite": suite,
        "n": len(df),
        "n_answered": len(df) - n_failed,
        "n_errors": n_failed,
        "not_run": n_not_run,
        "errors": dict(sorted(errors.items())),
        "pred_parquet": str(parquet_path),
        "note": "accuracy comes from scoring/score_mmau_pro.py (NVEmbed, base env)",
    }
