from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Item:
    suite: str                    # suite name (e.g. "mmau")
    qid: str                      # unique id within the suite
    item_id: str                  # audio item id (distinct audio) -- never contains '_p'/'_a'
    audio_files: list[str]        # absolute paths, in playback order (clip1 -> clip2 for multi-clip)
    group: str                    # group label (category) -- the by_task_type key in stage3's summary
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
