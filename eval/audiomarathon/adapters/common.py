from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path


def strip_qwen3_noise(text: str) -> str:
    """Remove Qwen3 reasoning blocks and special tokens."""
    if not text:
        return text
    s = text
    think_close = s.rfind("</think>")
    if think_close != -1:
        s = s[think_close + len("</think>"):].strip()
    s = re.sub(r"<\|[^|]*\|>", "", s)
    s = re.sub(r"(?i)^\s*assistant\s*[:\n]\s*", "", s)
    return s.strip()


def item_id(suite: str, rel_path: str) -> str:
    """Convert an audio path to a stable item id."""
    p = Path(rel_path)
    slug = re.sub(r"[^0-9A-Za-z_-]", "-", str(p.with_suffix(""))).strip("-")
    base = f"{suite}-{slug}" if slug else suite
    if re.search(r"_ch\d|_a\d", base):
        base = f"{base}-{hashlib.sha1(rel_path.encode('utf-8')).hexdigest()[:8]}"
    return base


@dataclass
class Item:
    suite: str                    # suite name (e.g. "race")
    qid: str                      # unique id within the suite
    item_id: str                  # audio item id (distinct audio) -- never contains '_p'/'_a'
    audio_files: list[str]        # absolute paths, in playback order (clip1 -> clip2 for multi-clip)
    group: str                    # group label (AM: suite name / SLUE task_name / DESED task_type,
                                  #   MMAU: category
    question: str                 # question text
    options: str                  # options string ("" if none)
    correct_text: str             # correct-answer text
    correct_letters: list[str] | None = None  # letter-graded suites only: normalized correct letters
    meta: dict = field(default_factory=dict)  # additional metadata

    def to_question_record(self, n_files: int) -> dict:
        """Serializes to the canonical record stored in manifest questions[]."""
        return {
            "qid": self.qid,
            "suite": self.suite,
            "video_id": self.item_id,       # compatibility field
            "item_id": self.item_id,
            "task_type": self.group,
            "source": f"{self.suite}:{self.qid}",
            "question": self.question,
            "options": self.options,
            "correct_options": (
                None if self.correct_letters is None
                else list(self.correct_letters)
            ),
            "answer_text": self.correct_text,
            "correct_letters": (
                None if self.correct_letters is None else list(self.correct_letters)
            ),
            "multi": n_files > 1,
            "n_files": n_files,
            "meta": self.meta,
        }
