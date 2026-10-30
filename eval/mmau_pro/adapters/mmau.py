from __future__ import annotations

import os

import pandas as pd
from pathlib import Path

from eval.mmau_pro.adapters.common import Item

MMAU_BASE = Path(os.environ.get("MMAU_PRO_ROOT", "/path/to/MMAU-Pro"))
PARQUET_PATH = MMAU_BASE / "test.parquet"
LENGTH_TYPES = ["long", "ultra-long", "ultra_long"]

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

_ANSWER_INSTRUCTION = (
    "Answer the question based on the audio. "
    "Respond with the single correct option letter and its text, e.g. 'A. Boba tea'."
)
_LABELS = "ABCDEFGHIJ"


def load_base_df() -> pd.DataFrame:
    """Apply the long/ultra-long filters and exclude open-ended items."""
    df = pd.read_parquet(PARQUET_PATH)
    df = df[df["length_type"].isin(LENGTH_TYPES)].reset_index(drop=True)
    df = df[df["category"] != "open"].reset_index(drop=True)
    return df


def _row_audio_paths(row) -> list[str]:
    """The audio_path column (an array of relative paths) as a list, handling
    both string and array forms."""
    ap = row["audio_path"]
    if isinstance(ap, str):
        return [ap]
    if hasattr(ap, "tolist"):
        ap = ap.tolist()
    return list(ap) if ap is not None and len(ap) > 0 else []


def format_options(choices) -> str:
    """['Boba tea', ...] -> 'A. Boba tea\\nB. ...'"""
    lines = []
    for i, choice in enumerate(choices):
        if i >= len(_LABELS):
            break
        lines.append(f"{_LABELS[i]}. {choice}")
    return "\n".join(lines)


def load_items(limit: int | None = None) -> list[Item]:
    df = load_base_df()
    if limit is not None:
        df = df.head(limit).reset_index(drop=True)

    items: list[Item] = []
    for idx, (_, row) in enumerate(df.iterrows()):
        rels = _row_audio_paths(row)
        files: list[str] = []
        for rel in rels:
            full = MMAU_BASE / str(rel)
            if not full.exists():
                continue
            files.append(str(full))

        choices = list(row["choices"]) if row["choices"] is not None else []
        question = str(row["question"])
        items.append(Item(
            suite="mmau",
            qid=f"mmau-{row['id']}",
            item_id=f"mmau-{row['id']}",        # a uuid etc. -- never contains '_p'/'_a'
            audio_files=files,
            group=str(row["category"]),
            question=question,
            options=format_options(choices) if choices else "",
            correct_text=str(row["answer"]),
            correct_letters=None,               # not letter-graded -- NVEmbed matching
            meta={
                "idx": idx,
                "id": row["id"],
                "category": row["category"],
                "length_type": row["length_type"],
                "choices": choices,
                "rel_paths": rels,
                "audio_count": len(rels),
            },
        ))
    return items


def prompt_text(item: Item) -> str:
    if item.options:
        return (f"Question: {item.question}\n\nOptions:\n{item.options}\n\n"
                f"{_ANSWER_INSTRUCTION}")
    return f"Question: {item.question}\n\nAnswer the question based on the audio."


def system_text(item: Item) -> str | None:
    return SYSTEM_PROMPT


def parse_response(raw: str, item: Item) -> str | None:
    """Scoring runs NVEmbed on the full model_output, so no parsing happens here."""
    if not raw:
        return None
    return raw.strip()


def is_correct(item: Item, raw: str, parsed: str | None) -> bool:
    raise NotImplementedError(
        "MMAU is graded by the NVEmbed similarity scorer (scoring/score_mmau_pro.py), not letter comparison"
    )
