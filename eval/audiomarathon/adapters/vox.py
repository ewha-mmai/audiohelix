from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

from eval.audiomarathon.adapters.common import Item, strip_qwen3_noise as _strip_noise, item_id

_AUDIOMARATHON_ROOT = Path(os.environ.get("AUDIOMARATHON_ROOT", "/path/to/AudioMarathon"))
_DATA_ROOT = _AUDIOMARATHON_ROOT / "VoxCeleb/concatenated_audio"
_META_FILE = _DATA_ROOT / "gender_id_task_meta.json"

_TASK_INSTRUCTION = ("You are a helpful assistant that analyzes speech audio to classify "
                     "speaker gender. Please listen to the voice carefully and determine "
                     "the speaker's gender.")
_INSTRUCTION = ("Listen to this audio and identify the speaker's gender. Is this a male or "
                "female voice? If it is a male, answer 'a'. If it is a female, answer 'b'. "
                "Answer with only 'a' or 'b'.")
_SYSTEM_TEXT = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                "capable of perceiving auditory and visual inputs, as well as generating "
                "text and speech.")


def load_items(limit: int | None = None) -> list[Item]:
    metadata = json.loads(_META_FILE.read_text(encoding="utf-8"))

    all_samples = []
    for idx, item in enumerate(metadata):
        rel_path = item["path"]
        wav_path = _DATA_ROOT / "wav" / rel_path
        if not wav_path.exists():
            continue
        gender = item["answer_gt"].lower().strip()
        all_samples.append({
            "idx": idx,
            "uniq_id": item.get("uniq_id"),
            "speaker_id": item["speaker_id_original"],
            "gender": gender,
            "wav_path": str(wav_path),
            "rel_path": rel_path,
            "question": item["question"],
            "choice_a": item["choice_a"],
            "choice_b": item["choice_b"],
            "answer_gt": item["answer_gt"],
        })

    male_samples = [s for s in all_samples if s["gender"] == "male"]
    female_samples = [s for s in all_samples if s["gender"] == "female"]

    min_samples_per_gender = min(len(male_samples), len(female_samples))
    if limit is not None and limit > 0:
        max_per_gender = limit // 2
        min_samples_per_gender = min(min_samples_per_gender, max_per_gender)

    rng = random.Random(42)
    if len(male_samples) > min_samples_per_gender:
        male_samples = rng.sample(male_samples, min_samples_per_gender)
    if len(female_samples) > min_samples_per_gender:
        female_samples = rng.sample(female_samples, min_samples_per_gender)

    balanced_samples = male_samples + female_samples
    rng.shuffle(balanced_samples)

    items: list[Item] = []
    seen_items: dict[str, str] = {}
    for pos, s in enumerate(balanced_samples):
        rel = s["rel_path"]
        if rel not in seen_items:
            seen_items[rel] = item_id("vox", rel)
        items.append(Item(
            suite="vox",
            qid=f"vox-{s['idx']:04d}",
            item_id=seen_items[rel],
            audio_files=[s["wav_path"]],
            group=s["gender"],
            question=s["question"],
            options="",
            correct_text=s["gender"],
            correct_letters=None,
            meta={
                "idx": s["idx"],
                "pos": pos,
                "uniq_id": s["uniq_id"],
                "speaker_id": s["speaker_id"],
                "path": rel,
                "answer_gt": s["answer_gt"],
                "choice_a": s["choice_a"],
                "choice_b": s["choice_b"],
            },
        ))
    return items


def prompt_text(item: Item) -> str:
    return f"{_TASK_INSTRUCTION}\n\n{_INSTRUCTION}"


def system_text(item: Item) -> str | None:
    return _SYSTEM_TEXT


def parse_response(raw: str, item: Item) -> str | None:
    """Return an empty string when the response cannot be parsed."""
    text_lower = _strip_noise(raw).lower().strip()
    choice_a_lower = item.meta["choice_a"].lower().strip()
    choice_b_lower = item.meta["choice_b"].lower().strip()

    if text_lower == 'a' or text_lower.startswith('a.') or text_lower.startswith('a)'):
        return choice_a_lower
    if text_lower == 'b' or text_lower.startswith('b.') or text_lower.startswith('b)'):
        return choice_b_lower

    if "option a" in text_lower or "choice a" in text_lower or "a)" in text_lower:
        return choice_a_lower
    if "option b" in text_lower or "choice b" in text_lower or "b)" in text_lower:
        return choice_b_lower

    if choice_a_lower in text_lower and choice_b_lower not in text_lower:
        return choice_a_lower
    if choice_b_lower in text_lower and choice_a_lower not in text_lower:
        return choice_b_lower

    # Word-boundary matching, only valid when choice_a=="male", choice_b=="female".
    if choice_a_lower == "male" and choice_b_lower == "female":
        male_match = re.search(r'\bmale\b', text_lower) is not None
        female_match = re.search(r'\bfemale\b', text_lower) is not None
        if male_match and not female_match:
            return "male"
        if female_match and not male_match:
            return "female"

    return ""


def is_correct(item: Item, raw: str, parsed: str | None) -> bool:
    return parsed == item.correct_text
