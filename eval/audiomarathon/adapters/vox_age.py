from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

from eval.audiomarathon.adapters.common import Item, strip_qwen3_noise as _strip_noise, item_id

_AUDIOMARATHON_ROOT = Path(os.environ.get("AUDIOMARATHON_ROOT", "/path/to/AudioMarathon"))
_DATA_ROOT = _AUDIOMARATHON_ROOT / "VoxCeleb/concatenated_audio"
_META_FILE = _DATA_ROOT / "age_classification_task_meta.json"

_TASK_INSTRUCTION = ("You are a helpful assistant that analyzes speech audio to estimate "
                     "speaker age. Please listen to the voice carefully and classify the "
                     "speaker's age group.")
_SYSTEM_TEXT = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                "capable of perceiving auditory and visual inputs, as well as generating "
                "text and speech.")


def _create_age_prompt(question: str, choices: dict) -> str:
    prompt = f"""{question}

A) {choices['choice_a']}
B) {choices['choice_b']}
C) {choices['choice_c']}
D) {choices['choice_d']}
E) {choices['choice_e']}

Please select the correct answer (A, B, C, D, or E)."""
    return prompt


def load_items(limit: int | None = None) -> list[Item]:
    metadata = json.loads(_META_FILE.read_text(encoding="utf-8"))

    all_samples = []
    for idx, item in enumerate(metadata):
        rel_path = item["path"]
        wav_path = _DATA_ROOT / "wav" / rel_path
        if not wav_path.exists():
            continue
        age_group = item["answer_gt"].strip()
        choices = {k: item[k] for k in ("choice_a", "choice_b", "choice_c",
                                        "choice_d", "choice_e")}
        all_samples.append({
            "idx": idx,
            "uniq_id": item.get("uniq_id"),
            "speaker_id": item["speaker_id_original"],
            "speaker_age": item.get("speaker_age", 0),
            "age_group": age_group,
            "wav_path": str(wav_path),
            "rel_path": rel_path,
            "question": item["question"],
            "choices": choices,
        })

    rng = random.Random(42)
    if limit is not None and limit > 0 and len(all_samples) > limit:
        all_samples = rng.sample(all_samples, limit)
    rng.shuffle(all_samples)


    grouped: dict[str, list[dict]] = {}
    for s in all_samples:
        grouped.setdefault(s["speaker_id"], []).append(s)
    run_order = [s for samples in grouped.values() for s in samples]

    items: list[Item] = []
    seen_items: dict[str, str] = {}
    for pos, s in enumerate(run_order):
        rel = s["rel_path"]
        if rel not in seen_items:
            seen_items[rel] = item_id("vox_age", rel)
        items.append(Item(
            suite="vox_age",
            qid=f"vox_age-{s['idx']:04d}",
            item_id=seen_items[rel],
            audio_files=[s["wav_path"]],
            group=s["age_group"],
            question=s["question"],
            options=_create_age_prompt(s["question"], s["choices"]),
            correct_text=s["age_group"],
            correct_letters=None,
            meta={
                "idx": s["idx"],
                "pos": pos,
                "uniq_id": s["uniq_id"],
                "speaker_id": s["speaker_id"],
                "speaker_age": s["speaker_age"],
                "path": rel,
                "choices": s["choices"],
            },
        ))
    return items


def prompt_text(item: Item) -> str:
    return f"{_TASK_INSTRUCTION}\n\n{item.options}"


def system_text(item: Item) -> str | None:
    return _SYSTEM_TEXT


def parse_response(raw: str, item: Item) -> str | None:
    """Return an empty string when the response cannot be parsed."""
    text_lower = _strip_noise(raw).lower().strip()
    choices = item.meta["choices"]

    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)') or text_lower.endswith(' a'):
        return choices["choice_a"]
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)') or text_lower.endswith(' b'):
        return choices["choice_b"]
    if text_lower == 'c' or text_lower.startswith('c.') or text_lower.startswith('c)') or text_lower.endswith(' c'):
        return choices["choice_c"]
    if text_lower == 'd' or text_lower.startswith('d.') or text_lower.startswith('d)') or text_lower.endswith(' d'):
        return choices["choice_d"]
    if text_lower == 'e' or text_lower.startswith('e.') or text_lower.startswith('e)') or text_lower.endswith(' e'):
        return choices["choice_e"]

    if re.search(r'\ba\b', text_lower) and not any(re.search(rf'\b{letter}\b', text_lower) for letter in ['b', 'c', 'd', 'e']):
        return choices["choice_a"]
    if re.search(r'\bb\b', text_lower) and not any(re.search(rf'\b{letter}\b', text_lower) for letter in ['a', 'c', 'd', 'e']):
        return choices["choice_b"]
    if re.search(r'\bc\b', text_lower) and not any(re.search(rf'\b{letter}\b', text_lower) for letter in ['a', 'b', 'd', 'e']):
        return choices["choice_c"]
    if re.search(r'\bd\b', text_lower) and not any(re.search(rf'\b{letter}\b', text_lower) for letter in ['a', 'b', 'c', 'e']):
        return choices["choice_d"]
    if re.search(r'\be\b', text_lower) and not any(re.search(rf'\b{letter}\b', text_lower) for letter in ['a', 'b', 'c', 'd']):
        return choices["choice_e"]

    for option, choice_text in choices.items():
        option_letter = option[-1].lower()
        if f"option {option_letter}" in text_lower or f"choice {option_letter}" in text_lower or f"answer {option_letter}" in text_lower:
            return choice_text

    choice_matches = []
    for choice_text in choices.values():
        if choice_text.lower() in text_lower:
            choice_matches.append(choice_text)
    if len(choice_matches) == 1:
        return choice_matches[0]

    return ""


def is_correct(item: Item, raw: str, parsed: str | None) -> bool:
    return parsed == item.correct_text
