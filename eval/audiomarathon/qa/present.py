"""AudioMarathon retrieved-content presentation."""

from __future__ import annotations

from audiohelix.present.content_builder import (
    DEFAULT_STYLE,
    build_retrieved_content,
    units_from_paths,
)
from audiohelix.present.pos_emb_time import build_time_map

from ..setup.config import UNIT_SEC


def present(record: dict, unit_paths: list[list[str]], parsed_texts: list[str], *,
            model: str, concat: bool, instruction: str, clip_of=None, cids=None,
            total_counts=None, chunks=None, item=None, pad_waves: bool = False,
            args=None) -> dict:
    """Build the retrieved content and optional time map for one question."""
    rc = build_retrieved_content(
        units_from_paths(unit_paths, UNIT_SEC), parsed_texts,
        clip_of=clip_of, total_counts=total_counts, cids=cids, concat=concat,
        style=DEFAULT_STYLE)
    content, audio_sequence = rc.content, rc.audio_sequence
    temp_files, audio_item = rc.temp_files, rc.audio_item
    chunks_with_audio = list(rc.chunks_with_audio)

    prompt_log_parts: list[str] = []
    if concat:
        if audio_item is not None:
            n_clips = sum(len(ps) for ps in unit_paths)
            prompt_log_parts.append(f"[AUDIO_CONCAT: {n_clips} clips merged]\n")
        for c in content:
            if c["type"] == "text":
                prompt_log_parts.append(f"{c['text']}\n")
    else:
        for c in content:
            if c["type"] == "audio":
                prompt_log_parts.append(f"[AUDIO_INPUT: {c['audio']}]\n")
            else:
                prompt_log_parts.append(f"{c['text']}\n")

    timemaps: list = []
    pe_on = args is not None and (
        getattr(args, "pos_emb_time", "off") == "on"
        or getattr(args, "pos_emb_llm", "off") == "on")
    if pe_on:
        if not audio_sequence and not concat:
            raise SystemExit(
                "--pos-emb-* is on but --concat-audio is off, so the concatenated "
                "audio order is unknown -- cannot time-align")
        if audio_sequence:
            timemaps = [build_time_map(audio_sequence, chunks, UNIT_SEC)]

    return {
        "content": content,
        "audio_item": audio_item,
        "audio_sequence": audio_sequence,
        "temp_files": temp_files,
        "timemaps": timemaps,
        "prompt_log_body": "".join(prompt_log_parts),
        "chunks_with_audio": chunks_with_audio,
        "warns": [],
    }
