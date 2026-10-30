from __future__ import annotations

import json
import os
import re
from pathlib import Path

from eval.audiomarathon.adapters.common import Item, strip_qwen3_noise as _strip_noise, item_id

_AUDIOMARATHON_ROOT = Path(os.environ.get("AUDIOMARATHON_ROOT", "/path/to/AudioMarathon"))
_DATA_ROOT = _AUDIOMARATHON_ROOT / "DESED/DESED_dataset/concatenated_audio"
_JSON_FILE = _DATA_ROOT / "desed_sound_event_detection_task.json"

_SYS_PROMPT = ("You are a helpful assistant that analyzes audio to detect and classify "
               "sound events. Please listen carefully and select the most appropriate "
               "answer from the given choices.")


def create_qa_prompt(question: str, choice_a: str, choice_b: str,
                     choice_c: str, choice_d: str) -> str:
    prompt_text = f"""{question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Please listen to the audio and select the correct answer. Reply with only the letter (A, B, C, or D)."""
    return prompt_text


def _cut_assistant_marker(text: str) -> str:
    if "assistant\n" in text:
        assistant_start = text.rfind("assistant\n") + len("assistant\n")
        text = text[assistant_start:].strip()
    return text


def extract_answer_choice(response: str) -> str:
    """Returns an uppercase A-D letter, or "" if none found."""
    response = _strip_noise(response)
    if not response:
        return ""

    response = response.strip().upper()

    if response in ['A', 'B', 'C', 'D']:
        return response

    match = re.search(r'\b([ABCD])\b', response)
    if match:
        return match.group(1)

    match = re.search(r'[(\[]?([ABCD])[)\].]?', response)
    if match:
        return match.group(1)

    return ""


def load_items(limit: int | None = None) -> list[Item]:
    data = json.loads(_JSON_FILE.read_text(encoding="utf-8"))

    valid: list[dict] = []
    for task in data["tasks"]:
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
        answer_gt = task.get("answer_gt", "")
        choices = task.get("choices", {})
        choice_a = choices.get("A", "")
        choice_b = choices.get("B", "")
        choice_c = choices.get("C", "")
        choice_d = choices.get("D", "")
        uniq_id = task.get("uniq_id", idx)

        items.append(Item(
            suite="desed",
            qid=f"desed-{idx:04d}",
            item_id=item_id("desed", relative_path),
            audio_files=[full_audio_path],
            group=task.get("task_type", "unknown"),
            question=question,
            options=create_qa_prompt(question, choice_a, choice_b, choice_c, choice_d),
            correct_text=answer_gt,
            correct_letters=[answer_gt],
            meta={
                "idx": idx,
                "uniq_id": uniq_id,
                "id": f"qa_task_{uniq_id}",
                "filename": relative_path.rsplit("/", 1)[-1],
                "task_type": task.get("task_type", "unknown"),
                "task_name": task.get("task_name", "unknown"),
                "dataset_name": task.get("dataset_name", "unknown"),
                "question_type": task.get("question_type", ""),
                "choices": {k: choices.get(k, "") for k in ("A", "B", "C", "D")},
                "correct_event": task.get("correct_event", ""),
                "primary_event": task.get("primary_event", ""),
                "path_extracted_event": task.get("path_extracted_event", ""),
                "all_events": task.get("all_events", []),
                "duration_seconds": task.get("duration_seconds", 0),
                "audio_rel": relative_path,
            },
        ))
    return items


def prompt_text(item: Item) -> str:
    return item.options


def system_text(item: Item) -> str | None:
    return _SYS_PROMPT


def parse_response(raw: str, item: Item) -> str | None:
    text = _cut_assistant_marker(raw if raw else "")
    text = text.strip()
    pred = extract_answer_choice(text)
    return pred if pred else None


def is_correct(item: Item, raw: str, parsed: str | None) -> bool:
    return parsed is not None and parsed == item.correct_text
