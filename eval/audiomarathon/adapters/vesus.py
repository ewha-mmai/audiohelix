from __future__ import annotations

import json
import os
from pathlib import Path

from eval.audiomarathon.adapters.common import Item, strip_qwen3_noise as _strip_noise, item_id

_AUDIOMARATHON_ROOT = Path(os.environ.get("AUDIOMARATHON_ROOT", "/path/to/AudioMarathon"))
_DATA_ROOT = _AUDIOMARATHON_ROOT / "VESUS"
_JSON_FILE = _DATA_ROOT / "audio_emotion_dataset.json"

_TASK_INSTRUCTION = ("You are a helpful assistant that analyzes speech audio to recognize "
                     "emotions. Please listen to the voice carefully and identify the "
                     "emotional state of the speaker.")


def create_emotion_prompt(question: str, choice_a: str, choice_b: str,
                          choice_c: str, choice_d: str) -> str:
    prompt = f"""{question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Please select the correct answer (A, B, C, or D)."""
    return prompt


def _cut_assistant_marker(text: str) -> str:
    if "assistant\n" in text:
        assistant_start = text.rfind("assistant\n") + len("assistant\n")
        text = text[assistant_start:].strip()
    return text


def extract_emotion_answer(text: str, choices: dict) -> str:
    """Returns an uppercase A-D letter, or "" if none found."""
    text = _strip_noise(text)
    text_lower = text.lower().strip()

    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)'):
        return "A"
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)'):
        return "B"
    if text_lower == 'c' or text_lower.startswith('c.') or text_lower.startswith('c)'):
        return "C"
    if text_lower == 'd' or text_lower.startswith('d.') or text_lower.startswith('d)'):
        return "D"

    option_patterns = {
        'A': ["option a", "choice a", "a)", "(a)"],
        'B': ["option b", "choice b", "b)", "(b)"],
        'C': ["option c", "choice c", "c)", "(c)"],
        'D': ["option d", "choice d", "d)", "(d)"],
    }

    for option, patterns in option_patterns.items():
        if any(pattern in text_lower for pattern in patterns):
            return option

    emotion_keywords = {
        'angry': ['anger', 'frustrated', 'mad', 'furious'],
        'happy': ['joy', 'cheerful', 'pleased', 'delighted'],
        'sad': ['sadness', 'melancholy', 'depressed', 'sorrow'],
        'fearful': ['fear', 'anxiety', 'scared', 'afraid'],
        'monotone': ['flat', 'emotionless', 'neutral', 'bland'],
    }

    for choice_key in ['choice_a', 'choice_b', 'choice_c', 'choice_d']:
        if choice_key in choices:
            choice_text = choices[choice_key].lower()
            for emotion, keywords in emotion_keywords.items():
                if any(keyword in text_lower for keyword in keywords) and any(keyword in choice_text for keyword in keywords):
                    option_index = ord(choice_key[-1]) - ord('a')
                    return chr(65 + option_index)

    return ""


def load_items(limit: int | None = None) -> list[Item]:
    data = json.loads(_JSON_FILE.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data["data"]

    valid: list[dict] = []
    for item in data:
        if not (isinstance(item, dict) and all(k in item for k in ['path', 'question', 'answer_gt'])):
            continue
        valid.append(item)

    if limit is not None and limit > 0:
        valid = valid[:limit]

    items: list[Item] = []
    for idx, sample in enumerate(valid):
        relative_path = sample.get("path", "")
        full_audio_path = _DATA_ROOT / relative_path
        if not full_audio_path.exists():
            continue

        question = sample.get("question", "What emotion is expressed in this audio segment?")
        choice_a = sample.get("choice_a", "")
        choice_b = sample.get("choice_b", "")
        choice_c = sample.get("choice_c", "")
        choice_d = sample.get("choice_d", "")
        answer_gt = sample.get("answer_gt", "").upper()

        choices = {
            'choice_a': choice_a,
            'choice_b': choice_b,
            'choice_c': choice_c,
            'choice_d': choice_d,
        }

        items.append(Item(
            suite="vesus",
            qid=f"vesus-{idx:04d}",
            item_id=item_id("vesus", relative_path),
            audio_files=[str(full_audio_path)],
            group="vesus",
            question=question,
            options=create_emotion_prompt(question, choice_a, choice_b, choice_c, choice_d),
            correct_text=answer_gt,
            correct_letters=[answer_gt],
            meta={
                "idx": idx,
                "uniq_id": sample.get("uniq_id", idx),
                "task_name": sample.get("task_name", "unknown"),
                "dataset_name": sample.get("dataset_name", "unknown"),
                "emotion_label": sample.get("emotion_label", "unknown"),
                "person_id": sample.get("person_id", "unknown"),
                "choices": choices,
                "audio_rel": relative_path,
            },
        ))
    return items


def prompt_text(item: Item) -> str:
    return f"{_TASK_INSTRUCTION}\n\n{item.options}"


def system_text(item: Item) -> str | None:
    return None


def parse_response(raw: str, item: Item) -> str | None:
    text = _cut_assistant_marker(raw if raw else "")
    pred = extract_emotion_answer(text, item.meta["choices"])
    return pred if pred else None


def is_correct(item: Item, raw: str, parsed: str | None) -> bool:
    return parsed is not None and parsed == item.correct_text
