from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent          # .../mmau_pro/scoring
PKG_ROOT = _HERE.parent                          # .../mmau_pro
MMAU_BASE = Path(os.environ.get("MMAU_PRO_ROOT", "/path/to/MMAU-Pro"))
BP_DIR = MMAU_BASE / "base_perform"
PARQUET_PATH = MMAU_BASE / "test.parquet"
LENGTH_TYPES = ["long", "ultra-long", "ultra_long"]

# So the official scorer module can be imported regardless of cwd.
sys.path.insert(0, str(BP_DIR))


def main():
    parser = argparse.ArgumentParser(
        description="Scores MMAU-Pro's pred.parquet with the official NVEmbed scorer (base env only)")
    parser.add_argument("--pred", required=True,
                        help="pred.parquet (or pred.jsonl) written by stage3/run_qa")
    parser.add_argument("--out-dir", default=None,
                        help="where to save summary/detail (default: same folder as --pred)")
    parser.add_argument("--model-output-column", default="model_output")
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES (default: 4,6)")
    args = parser.parse_args()

    import os
    import torch
    import transformers
    if transformers.__version__.startswith("5."):
        raise SystemExit(
            f"[error] transformers {transformers.__version__} -- NV-Embed-v2 scoring only works "
            f"in your conda base env (transformers 4.51), not this one:\n"
            f"  conda run -n base python {sys.argv[0]} --pred {args.pred}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu or "4,6"

    pred_path = Path(args.pred)
    out_dir = Path(args.out_dir) if args.out_dir else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    assert pred_path.exists(), f"pred not found: {pred_path} (run stage3/run_qa first)"

    df = load_pred_df(pred_path)
    if "is_multi_audio" not in df.columns:
        df["audio_count"] = df["audio_path"].apply(
            lambda x: len(x) if hasattr(x, "__len__") else 1)
        df["is_multi_audio"] = df["audio_count"] > 1

    n_not_run = int(df["error"].eq("not_run").sum())
    n_err = int(df["error"].notna().sum()) - n_not_run
    df_eval = df[df["error"].ne("not_run")].copy()   # not_run = unfinished, excluded
    print(f"pred {len(df)} rows -- scoring {len(df_eval)} / errors {n_err} (scored wrong) "
          f"/ not_run {n_not_run} (excluded -- resume then rescore)")

    # ---- closed-ended: NVEmbed similarity matching ----
    closed_categories = [c for c in df_eval["category"].unique()
                         if c not in ("open", "instruction following")]
    closed_df = df_eval[df_eval["category"].isin(closed_categories)].copy()
    closed_df = closed_df[closed_df["choices"].notna()].copy()
    if closed_df.empty:
        raise SystemExit("No closed-ended questions to score.")

    print(f"[eval] closed-ended {len(closed_df)} -- NVEmbed")
    from evaluate_mmau_pro_comprehensive import (
        calculate_metrics, evaluate_closedended_with_nvembed, load_nvembed_model,
    )
    nvembed_model = load_nvembed_model()
    try:
        choices_list = [list(c) if hasattr(c, "__iter__") else [str(c)]
                        for c in closed_df["choices"].tolist()]
        gt = closed_df["answer"].tolist()
        preds, confidences = evaluate_closedended_with_nvembed(
            nvembed_model,
            closed_df["question"].tolist(),
            choices_list,
            gt,
            closed_df[args.model_output_column].fillna("").tolist(),
            closed_df["category"].tolist(),
        )
        is_multi = closed_df["is_multi_audio"].tolist()

        category_results: dict = {}
        overall = calculate_metrics(gt, preds)
        category_results["closed_overall"] = {"type": "closed", "count": len(closed_df),
                                              "metrics": overall}
        no_multi_mask = [not m for m in is_multi]
        if sum(no_multi_mask) > 0:
            category_results["closed_no_multi"] = {
                "type": "closed", "count": sum(no_multi_mask),
                "metrics": calculate_metrics(
                    [g for g, m in zip(gt, no_multi_mask) if m],
                    [p for p, m in zip(preds, no_multi_mask) if m])}
        if sum(is_multi) > 0:
            category_results["closed_multi_only"] = {
                "type": "closed", "count": sum(is_multi),
                "metrics": calculate_metrics(
                    [g for g, m in zip(gt, is_multi) if m],
                    [p for p, m in zip(preds, is_multi) if m])}

        for category in closed_categories:
            mask = (closed_df["category"] == category).tolist()
            if sum(mask) > 0:
                category_results[category] = {
                    "type": "closed", "count": sum(mask),
                    "metrics": calculate_metrics(
                        [g for g, m in zip(gt, mask) if m],
                        [p for p, m in zip(preds, mask) if m])}

        detail_path = out_dir / f"{pred_path.stem}_closed_detail.jsonl"
        with detail_path.open("w", encoding="utf-8") as f:
            for i, (g, p, c, m) in enumerate(zip(gt, preds, confidences, is_multi)):
                f.write(json.dumps({
                    "id": closed_df["id"].iloc[i],
                    "category": closed_df["category"].iloc[i],
                    "is_multi_audio": bool(m),
                    "ground_truth": g,
                    "prediction": p,
                    "confidence": float(c),
                    "correct": bool(g == p),
                }, ensure_ascii=False) + "\n")
        print(f"  sample detail saved: {detail_path}")
    finally:
        del nvembed_model
        gc.collect()
        torch.cuda.empty_cache()

    summary = {
        "scorer": "NVEmbed-v2 (official evaluate_mmau_pro_comprehensive)",
        "pred": str(pred_path),
        "total_samples": len(df),
        "evaluated_samples": len(df_eval),
        "inference_errors": n_err,
        "not_run_excluded": n_not_run,
        "category_results": category_results,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"summary saved: {summary_path}")

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY (closed-ended, NVEmbed)")
    print("=" * 70)
    for key, label in [("closed_overall", "closed overall (combined)"),
                       ("closed_no_multi", "closed, excluding multi"),
                       ("closed_multi_only", "closed, multi only")]:
        r = category_results.get(key)
        if r:
            print(f"  {label:<24} n={r['count']:<4} accuracy={r['metrics'].get('accuracy', 0):.4f}")
    for category in closed_categories:
        r = category_results.get(category)
        if r:
            print(f"  {category:<24} n={r['count']:<4} accuracy={r['metrics'].get('accuracy', 0):.4f}")
    overall_acc = overall["accuracy"]
    print(f"Overall accuracy (NVEmbed): {overall_acc:.4f} "
          f"({category_results['closed_overall']['count']})")
    if n_not_run:
        print(f"[note] {n_not_run} not_run excluded -- resume stage3 then rescore.")


def load_pred_df(pred_path: Path):
    """Load predictions joined by item id against the base dataframe."""
    import pandas as pd
    if pred_path.suffix == ".parquet":
        return pd.read_parquet(pred_path)
    assert pred_path.suffix == ".jsonl", "pred must be .parquet or .jsonl"
    df = pd.read_parquet(PARQUET_PATH)
    df = df[df["length_type"].isin(LENGTH_TYPES)].reset_index(drop=True)
    out = df.copy()
    out["model_output"] = ""
    out["error"] = None
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            m = out["id"] == rec["id"]
            if m.any():
                out.loc[m, "model_output"] = rec.get("model_output", "")
                out.loc[m, "error"] = rec.get("error")
    return out


if __name__ == "__main__":
    main()
