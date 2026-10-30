from __future__ import annotations

from pathlib import Path

from audiohelix.present.content_builder import (
    DEFAULT_STYLE, UNIT_SEC, build_retrieved_content, units_from_paths,
)
from audiohelix.present.pos_emb_time import build_time_map

from ..setup import config
from ..setup.data import item_clip_offsets


def present(record: dict, unit_paths: list[list[str]], parsed_texts: list[str],
            *, model: str, concat: bool, instruction: str,
            clip_of: list[int] | None = None, cids: list[str] | None = None,
            total_counts: dict[str, int] | None = None, chunks: dict | None = None,
            item: dict | None = None, pad_waves: bool = False, args=None) -> dict:
    """Builds this question's conversation blocks.

    Returns {"content", "audio_item", "audio_sequence", "temp_files",
    "timemaps", "chunks_with_audio", "prompt_log_body", "warns"}:
        content              the items + per-chunk text blocks, in order
        audio_item           concat-only: the single merged audio input (a temp
                             wav path)
        audio_sequence       the wav paths in attachment order, concat only
                             (None otherwise -- what the old builder returned)
        temp_files           temp wavs the caller deletes in a finally block
        timemaps             [] or [TimeMap], built only when
                             args.pos_emb_time == "on" or args.pos_emb_llm == "on"
        chunks_with_audio  1-based chunk indices that contributed audio
        prompt_log_body      always None
        warns                non-fatal notes (multi-clip item without its manifest item)

    `instruction` is accepted for the shared call shape and is not placed into
    the content. `pad_waves` is traceav-only.
    """
    warns: list[str] = []
    units = units_from_paths(unit_paths, UNIT_SEC)
    rc = build_retrieved_content(
        units, parsed_texts, clip_of=clip_of, total_counts=total_counts,
        cids=cids, concat=concat, style=DEFAULT_STYLE)
    content = rc.content
    audio_item = rc.audio_item
    temp_files = list(rc.temp_files)
    audio_sequence = list(rc.audio_sequence) if concat and rc.audio_item is not None else None
    chunks_with_audio = list(rc.chunks_with_audio)

    timemaps: list = []
    pe_on = args is not None and (
        getattr(args, "pos_emb_time", config.POS_EMB_TIME_DEFAULT) == "on"
        or getattr(args, "pos_emb_llm", config.POS_EMB_LLM_DEFAULT) == "on")
    if pe_on:
        if audio_sequence is None and not concat:
            raise SystemExit(
                "--pos-emb-* is on but --concat-audio is off, so the concatenated "
                "audio order is unknown -- cannot time-align")
        if audio_sequence:
            clip_offsets: dict[str, float] = {}
            if record.get("multi"):
                if isinstance(item, dict) and item.get("files"):
                    clip_offsets = item_clip_offsets(item)
                else:
                    warns.append("multi-clip item without its manifest entry -- "
                                 "the concat time map uses clip-local times")
            timemaps = [build_time_map(audio_sequence, chunks, UNIT_SEC,
                                       clip_offsets=clip_offsets)]

    return {
        "content": content,
        "audio_item": audio_item,
        "audio_sequence": audio_sequence,
        "temp_files": temp_files,
        "timemaps": timemaps,
        "chunks_with_audio": chunks_with_audio,
        "prompt_log_body": None,
        "warns": warns,
    }
