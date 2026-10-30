from __future__ import annotations

import glob
import os
import random
import re
from pathlib import Path

from eval.audiomarathon.adapters.common import Item, strip_qwen3_noise as _strip_qwen3_noise, item_id

_AUDIOMARATHON_ROOT = Path(os.environ.get("AUDIOMARATHON_ROOT", "/path/to/AudioMarathon"))
_DATA_ROOT = _AUDIOMARATHON_ROOT / "HAD/concatenated_audio"

_QUESTION = ("Listen to this audio clip carefully. Is this audio completely authentic "
             "(real) or does it contain any artificially synthesized segments (fake)? "
             "If it is completely real, answer 'a'. If it contains any fake segments, "
             "answer 'b'. Answer with only 'a' or 'b'.")
_SYS_PROMPT = "You are a helpful assistant that analyzes audio to determine authenticity."
_TASK = "Audio_Authenticity_Detection"


def _extract_authenticity_answer(text: str) -> str:
    """Returns 'real'/'fake' or "" if neither can be determined."""
    text = _strip_qwen3_noise(text)
    text_lower = text.lower().strip()
    if (text_lower == 'a' or text_lower.startswith('a.')
            or text_lower.startswith('a)') or text_lower.endswith(' a')):
        return 'real'
    if (text_lower == 'b' or text_lower.startswith('b.')
            or text_lower.startswith('b)') or text_lower.endswith(' b')):
        return 'fake'
    if ("option a" in text_lower or "choice a" in text_lower
            or "a)" in text_lower or " a " in text_lower):
        return 'real'
    if ("option b" in text_lower or "choice b" in text_lower
            or "b)" in text_lower or " b " in text_lower):
        return 'fake'
    if "real" in text_lower and "fake" not in text_lower:
        return 'real'
    if "fake" in text_lower and "real" not in text_lower:
        return 'fake'
    real_match = re.search(
        r'\breal\b|\bauthentic\b|\bgenuine\b|\bcompletely authentic\b', text_lower) is not None
    fake_match = re.search(
        r'\bfake\b|\bartificial\b|\bsynthetic\b|\bsynthesized\b|\bcontains.*fake\b',
        text_lower) is not None
    if real_match and not fake_match:
        return "real"
    if fake_match and not real_match:
        return "fake"
    return ""


def load_items(limit: int | None = None) -> list[Item]:
    real_files = glob.glob(str(_DATA_ROOT / "real" / "*.wav"))
    fake_files = glob.glob(str(_DATA_ROOT / "fake" / "*.wav"))
    real_samples = real_files
    fake_samples = fake_files
    n_min = min(len(real_samples), len(fake_samples))

    rng = random.Random(42)
    if len(real_samples) > n_min:
        real_samples = rng.sample(real_samples, n_min)
    if len(fake_samples) > n_min:
        fake_samples = rng.sample(fake_samples, n_min)
    balanced = real_samples + fake_samples
    rng.shuffle(balanced)

    items: list[Item] = []
    for idx, path in enumerate(balanced):
        label = "real" if path in real_samples else "fake"
        rel = os.path.relpath(path, _DATA_ROOT)
        items.append(Item(
            suite="had",
            qid=f"had-{idx:04d}",
            item_id=item_id("had", rel),
            audio_files=[path],
            group=label,
            question=_QUESTION,
            options="",
            correct_text=label,
            correct_letters=None,
            meta={
                "idx": idx,
                "label": label,
                "audio_rel": rel,
                "audio_file": os.path.basename(path),
                "task": _TASK,
            },
        ))

    if limit is not None and limit > 0 and len(items) > limit:
        items = items[:limit]
    return items


def prompt_text(item: Item) -> str:
    return _QUESTION


def system_text(item: Item) -> str | None:
    return _SYS_PROMPT


def parse_response(raw: str, item: Item) -> str | None:
    if not raw:
        return ""
    text = raw
    if "assistant\n" in text:
        assistant_start = text.rfind("assistant\n") + len("assistant\n")
        text = text[assistant_start:].strip()
    text = text.strip()
    return _extract_authenticity_answer(text)


def is_correct(item: Item, raw: str, parsed: str | None) -> bool:
    return parsed is not None and parsed == item.correct_text
