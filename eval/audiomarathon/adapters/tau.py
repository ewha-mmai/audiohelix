from __future__ import annotations

import json
import os
import random
from pathlib import Path

from eval.audiomarathon.adapters.common import Item, strip_qwen3_noise as _strip_qwen3_noise, item_id

_AUDIOMARATHON_ROOT = Path(os.environ.get("AUDIOMARATHON_ROOT", "/path/to/AudioMarathon"))
_AUDIO_DIR = _AUDIOMARATHON_ROOT / "TAU/concatenated_resampled"
_META_FILE = _AUDIO_DIR / "acoustic_scene_task_meta.json"

_INSTR_HEAD = "Listen to this audio and identify the acoustic scene. Choose the most appropriate option.\n"
_INSTR_TAIL = "Respond with only the letter of your answer (A, B, C, or D)."
_TASK_INSTRUCTION = ("You are a helpful assistant that analyzes urban soundscape audio to "
                     "identify acoustic scenes. Please listen to the audio carefully and "
                     "classify the scene type.")
_SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
           "capable of perceiving auditory and visual inputs, as well as generating "
           "text and speech.")


def _choice_block(choices: list[str]) -> str:
    letters = "ABCD"
    return "".join(f"{letters[i]}: {choices[i]}\n" for i in range(4))


def _build_instruction(choices: list[str]) -> str:
    return _INSTR_HEAD + _choice_block(choices) + _INSTR_TAIL


def _extract_acoustic_scene_answer(text: str, choices: list[str] | None) -> str:
    """Returns an uppercase A-D letter, or "" if none found."""
    text = _strip_qwen3_noise(text)
    text_lower = text.lower().strip()
    options = ['a', 'b', 'c', 'd']
    if text_lower in options:
        return text_lower.upper()
    for opt in options:
        patterns = [f"{opt}.", f"{opt})", f"{opt}:"]
        for pattern in patterns:
            if text_lower.startswith(pattern):
                return opt.upper()
    for opt in options:
        indicators = [f"option {opt}", f"choice {opt}", f"{opt})"]
        for indicator in indicators:
            if indicator in text_lower:
                return opt.upper()
    if choices:
        best_match = None
        max_overlap = 0
        for i, choice_text in enumerate(choices):
            choice_lower = choice_text.lower()
            if choice_lower in text_lower:
                return chr(65 + i)
            keywords = choice_lower.split(' - ')[0].split()
            overlap = sum(1 for kw in keywords if kw in text_lower)
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = chr(65 + i)
        if best_match and max_overlap > 1:
            return best_match
    return ""


def _full_shuffled_order(rows: list[dict]) -> list[dict]:
    """Return the default no-limit evaluation order."""
    rng = random.Random(42)
    arr = list(rows)
    rng.shuffle(arr)
    return arr


def load_items(limit: int | None = None) -> list[Item]:
    metadata = json.loads(_META_FILE.read_text(encoding="utf-8"))
    valid_rows = []
    for item in metadata:
        wav_path = _AUDIO_DIR / item["path"]
        if not wav_path.exists():
            continue
        valid_rows.append(item)

    full_order = _full_shuffled_order(valid_rows)
    idx_of = {item["path"]: i for i, item in enumerate(full_order)}

    if limit is not None and limit > 0 and limit < len(valid_rows):
        rng = random.Random(42)
        rows = rng.sample(list(valid_rows), limit)
        rng.shuffle(rows)
    else:
        rows = full_order

    items: list[Item] = []
    for item in rows:
        idx = idx_of[item["path"]]
        rel = item["path"]
        audio_full = _AUDIO_DIR / rel
        options = [item["choice_a"], item["choice_b"],
                   item["choice_c"], item["choice_d"]]
        answer_gt = str(item["answer_gt"]).upper()
        items.append(Item(
            suite="tau",
            qid=f"tau-{idx:04d}",
            item_id=item_id("tau", rel),
            audio_files=[str(audio_full)],
            group=str(item["scene_label"]),
            question=str(item["question"]),
            options=_build_instruction(options),
            correct_text=answer_gt,
            correct_letters=[answer_gt],
            meta={
                "idx": idx,
                "uniq_id": item.get("uniq_id", idx),
                "scene_label": item["scene_label"],
                "task_name": item.get("task_name", ""),
                "dataset_name": item.get("dataset_name", ""),
                "audio_rel": rel,
                "options_list": options,
            },
        ))
    return items


def prompt_text(item: Item) -> str:
    return f"{_TASK_INSTRUCTION}\n\n{item.options}"


def system_text(item: Item) -> str | None:
    return _SYSTEM


def parse_response(raw: str, item: Item) -> str | None:
    if not raw:
        return ""
    text = raw
    if "assistant\n" in text:
        assistant_start = text.rfind("assistant\n") + len("assistant\n")
        text = text[assistant_start:].strip()
    text = text.strip()
    choices = getattr(item, "meta", {}).get("options_list")
    return _extract_acoustic_scene_answer(text, choices)


def is_correct(item: Item, raw: str, parsed: str | None) -> bool:
    return parsed is not None and parsed == item.correct_text
