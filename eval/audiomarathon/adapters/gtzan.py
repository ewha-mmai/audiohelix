from __future__ import annotations

import json
import os
import re
from pathlib import Path

from eval.audiomarathon.adapters.common import Item, strip_qwen3_noise as _strip_qwen3_noise, item_id

_AUDIOMARATHON_ROOT = Path(os.environ.get("AUDIOMARATHON_ROOT", "/path/to/AudioMarathon"))
_DATA_ROOT = _AUDIOMARATHON_ROOT / "GTZAN/concatenated_audio"
_META_FILE = _DATA_ROOT / "music_genre_classification_meta.json"
_WAV_DIR = _DATA_ROOT / "wav"
_REQUIRED_KEYS = ["path", "question", "choice_a", "choice_b", "choice_c",
                  "choice_d", "answer_gt"]

_INSTRUCTION = "Listen to this audio segment and identify the music genre based on what you hear."
_FORMAT_TEXT = "Respond with only the letter of the correct option (A, B, C, or D)."
_QWEN_INTRO = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
               "capable of perceiving auditory and visual inputs, as well as generating "
               "text and speech.")
_TASK_PROMPT = "You are a helpful audio analysis assistant."


def _build_user_prompt(question: str, options: list[str]) -> str:
    formatted_options = ""
    for i, opt in enumerate(options):
        letter = chr(65 + i)
        formatted_options += f"{letter}. {opt}\n"
    return (f"{_INSTRUCTION}\n\nQuestion: {question}\n\n"
            f"Options:\n{formatted_options.strip()}\n\n{_FORMAT_TEXT}")


def _extract_gtzan_answer(text: str) -> str:
    """Returns an uppercase A-D letter, or "" if none found."""
    text = _strip_qwen3_noise(text)
    if not text:
        return ""
    if "assistant\n" in text:
        assistant_start = text.rfind("assistant\n") + len("assistant\n")
        text = text[assistant_start:].strip()
    text = text.strip().upper()
    if text in ['A', 'B', 'C', 'D']:
        return text
    if text == 'A' or text.startswith('A.') or text.startswith('A)') or text.endswith(' A'):
        return 'A'
    if text == 'B' or text.startswith('B.') or text.startswith('B)') or text.endswith(' B'):
        return 'B'
    if text == 'C' or text.startswith('C.') or text.startswith('C)') or text.endswith(' C'):
        return 'C'
    if text == 'D' or text.startswith('D.') or text.startswith('D)') or text.endswith(' D'):
        return 'D'
    if "option a" in text.lower() or "choice a" in text.lower() or "a)" in text.lower():
        return 'A'
    if "option b" in text.lower() or "choice b" in text.lower() or "b)" in text.lower():
        return 'B'
    if "option c" in text.lower() or "choice c" in text.lower() or "c)" in text.lower():
        return 'C'
    if "option d" in text.lower() or "choice d" in text.lower() or "d)" in text.lower():
        return 'D'
    match = re.search(r'\b([ABCD])\b', text)
    if match:
        return match.group(1)
    match = re.search(r'[(\[]?([ABCD])[)\].]?', text)
    if match:
        return match.group(1)
    return ""


def load_items(limit: int | None = None) -> list[Item]:
    metadata = json.loads(_META_FILE.read_text(encoding="utf-8"))
    valid_rows = [s for s in metadata
                  if all(k in s for k in _REQUIRED_KEYS)]
    rows = valid_rows
    if limit is not None and limit > 0 and len(rows) > limit:
        rows = rows[:limit]

    items: list[Item] = []
    for idx, s in enumerate(rows):
        audio_rel = s["path"]
        audio_full = _WAV_DIR / audio_rel
        if not audio_full.exists():
            continue
        options = [s["choice_a"], s["choice_b"], s["choice_c"], s["choice_d"]]
        question = str(s["question"])
        items.append(Item(
            suite="gtzan",
            qid=f"gtzan-{idx:04d}",
            item_id=item_id("gtzan", audio_rel),
            audio_files=[str(audio_full)],
            group="gtzan",
            question=question,
            options=_build_user_prompt(question, options),
            correct_text=str(s["answer_gt"]),
            correct_letters=[str(s["answer_gt"])],
            meta={
                "idx": idx,
                "uniq_id": s.get("uniq_id", idx),
                "genre_label": s.get("genre_label", ""),
                "task_name": s.get("task_name", ""),
                "dataset_name": s.get("dataset_name", ""),
                "audio_rel": audio_rel,
                "options_list": options,
            },
        ))
    return items


def prompt_text(item: Item) -> str:
    return item.options


def system_text(item: Item) -> str | None:
    return f"{_QWEN_INTRO} {_TASK_PROMPT}"


def parse_response(raw: str, item: Item) -> str | None:
    if not raw:
        return ""
    text = raw
    if "assistant\n" in text:
        assistant_start = text.rfind("assistant\n") + len("assistant\n")
        text = text[assistant_start:].strip()
    return _extract_gtzan_answer(text)


def is_correct(item: Item, raw: str, parsed: str | None) -> bool:
    return parsed is not None and parsed == item.correct_text
