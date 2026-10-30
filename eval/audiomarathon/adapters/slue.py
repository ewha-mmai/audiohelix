from __future__ import annotations

import json
import os
import re
from pathlib import Path

from eval.audiomarathon.adapters.common import Item, strip_qwen3_noise as _strip_noise, item_id

_AUDIOMARATHON_ROOT = Path(os.environ.get("AUDIOMARATHON_ROOT", "/path/to/AudioMarathon"))
_DATA_ROOT = _AUDIOMARATHON_ROOT / "SLUE"
_JSON_FILE = _DATA_ROOT / "merged_audio_data.json"

_TASK_INSTRUCTION = ("You are a helpful assistant that analyzes speech audio for named "
                     "entity recognition. Please listen carefully and extract the "
                     "requested named entities from the speech.")


def create_slue_prompt(question: str, choice_a: str, choice_b: str,
                       choice_c: str, choice_d: str) -> str:
    prompt_text = f"""{question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Please listen to the audio and select the correct answer. Reply with only the letter (A, B, C, or D)."""
    return prompt_text


def extract_answer_choice(response: str) -> str:
    """Returns an uppercase A-D letter, or "" if none found."""
    response = _strip_noise(response)
    if not response:
        return ""

    response = response.strip().upper()

    if response in ['A', 'B', 'C', 'D']:
        return response

    if response.startswith('A') and len(response) <= 3:
        return 'A'
    if response.startswith('B') and len(response) <= 3:
        return 'B'
    if response.startswith('C') and len(response) <= 3:
        return 'C'
    if response.startswith('D') and len(response) <= 3:
        return 'D'

    match = re.search(r'\b([ABCD])\b', response)
    if match:
        return match.group(1)

    match = re.search(r'[(\[]?([ABCD])[)\].]?', response)
    if match:
        return match.group(1)

    match = re.search(r'(?:option|choice)\s+([ABCD])', response)
    if match:
        return match.group(1)

    return ""


def load_items(limit: int | None = None) -> list[Item]:
    data = json.loads(_JSON_FILE.read_text(encoding="utf-8"))

    valid: list[dict] = []
    for task in data:
        relative_path = task.get("path", "")
        if relative_path:
            full_audio_path = _DATA_ROOT / relative_path
        else:
            continue
        if not full_audio_path.exists():
            continue
        valid.append(task)

    if limit is not None and limit > 0:
        valid = valid[:limit]

    items: list[Item] = []
    for idx, task in enumerate(valid):
        relative_path = task.get("path", "")
        full_audio_path = str(_DATA_ROOT / relative_path)

        question = task.get("question", "")
        choice_a = task.get("choice_a", "")
        choice_b = task.get("choice_b", "")
        choice_c = task.get("choice_c", "")
        choice_d = task.get("choice_d", "")
        answer_gt = task.get("answer_gt", "")
        uniq_id = task.get("uniq_id", idx)

        items.append(Item(
            suite="slue",
            qid=f"slue-{idx:04d}",
            item_id=item_id("slue", relative_path),
            audio_files=[full_audio_path],
            group=task.get("dataset_name", "unknown"),
            question=question,
            options=create_slue_prompt(question, choice_a, choice_b, choice_c, choice_d),
            correct_text=answer_gt,
            correct_letters=[answer_gt],
            meta={
                "idx": idx,
                "uniq_id": uniq_id,
                "id": f"slue_task_{uniq_id}",
                "task_name": task.get("task_name", "unknown"),
                "dataset_name": task.get("dataset_name", "unknown"),
                "entity_count": task.get("entity_count", 0),
                "entity_types": task.get("entity_types", []),
                "source_count": task.get("source_count", 0),
                "audio_duration_info": task.get("audio_duration_info", ""),
                "source_folder": task.get("source_folder", ""),
                "source_file": task.get("source_file", ""),
                "audio_rel": relative_path,
            },
        ))
    return items


def prompt_text(item: Item) -> str:
    return f"{_TASK_INSTRUCTION}\n\n{item.options}"


def system_text(item: Item) -> str | None:
    return None


def parse_response(raw: str, item: Item) -> str | None:
    pred = extract_answer_choice(raw if raw else "")
    return pred if pred else None


def is_correct(item: Item, raw: str, parsed: str | None) -> bool:
    pred = parsed.strip().upper() if parsed else ""
    gt = item.correct_text.strip().upper() if item.correct_text else ""
    return pred == gt
