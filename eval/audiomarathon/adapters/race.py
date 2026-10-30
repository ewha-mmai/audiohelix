from __future__ import annotations

import json
import os
from pathlib import Path

from eval.audiomarathon.adapters.common import Item, item_id

_AUDIOMARATHON_ROOT = Path(os.environ.get("AUDIOMARATHON_ROOT", "/path/to/AudioMarathon"))
_DATA_ROOT = _AUDIOMARATHON_ROOT / "race_audio"
_BENCH_FILE = _DATA_ROOT / "race_benchmark.json"

_TASK_INSTRUCTION = ("You are a helpful assistant that analyzes reading comprehension "
                     "passages with audio narration. Please listen to the passage and "
                     "answer the multiple-choice question based on what you heard.")
_INSTRUCTION = ("Listen to this audio of a passage being read aloud, then answer the "
                "multiple-choice question based solely on the information from the audio.")
_FORMAT_TEXT = "Respond with only the letter of the correct option (A, B, C, or D)."


def _strip_noise(text: str) -> str:
    """Remove end markers and surrounding whitespace from a response."""
    if not text:
        return ""
    return text.replace("<|im_end|>", "").strip()


def create_race_prompt(question: str, options: list[str]) -> str:
    formatted_options = ""
    for i, opt in enumerate(options):
        letter = chr(65 + i)
        formatted_options += f"{letter}. {opt}\n"
    prompt = (f"{_INSTRUCTION}\n\nQuestion: {question}\n\n"
              f"Options:\n{formatted_options.strip()}\n\n{_FORMAT_TEXT}")
    return prompt


def load_items(limit: int | None = None) -> list[Item]:
    benchmark = json.loads(_BENCH_FILE.read_text(encoding="utf-8"))
    if limit is not None:
        benchmark = benchmark[:limit]

    items: list[Item] = []
    for idx, s in enumerate(benchmark):
        audio_rel = s["audio_path"]
        audio_full = _DATA_ROOT / audio_rel
        if not audio_full.exists():
            continue
        options = list(s["options"])
        items.append(Item(
            suite="race",
            qid=f"race-{idx:04d}",
            item_id=item_id("race", audio_rel),
            audio_files=[str(audio_full)],
            group="race",
            question=str(s["question"]),
            options=create_race_prompt(str(s["question"]), options),
            correct_text=str(s["answer"]),
            correct_letters=[str(s["answer"])],
            meta={
                "idx": idx,
                "article_id": s.get("article_id", ""),
                "question_idx": s.get("question_idx", idx),
                "audio_rel": audio_rel,
                "subset": "high" if "high" in audio_rel else "middle",
                "options_list": options,
            },
        ))
    return items


def prompt_text(item: Item) -> str:
    return f"{_TASK_INSTRUCTION}\n\n{item.options}"


def system_text(item: Item) -> str | None:
    return ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
            "Group, capable of perceiving auditory and visual inputs, as well as "
            "generating text and speech.")


def parse_response(raw: str, item: Item) -> str | None:
    """First A-D letter found, or the first word if none."""
    response = _strip_noise(raw)
    if not response:
        return None
    resp = response.strip().upper()
    for ch in resp:
        if ch in ["A", "B", "C", "D"]:
            return ch
    words = resp.split()
    for word in words:
        for ch in word:
            if ch in ["A", "B", "C", "D"]:
                return ch
    return words[0] if words else None


def is_correct(item: Item, raw: str, parsed: str | None) -> bool:
    return parsed is not None and parsed == item.correct_text
