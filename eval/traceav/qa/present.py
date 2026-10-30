from __future__ import annotations

from pathlib import Path

import numpy as np

from audiohelix.present.content_builder import (
    TRACEAV_STYLE, build_retrieved_content, prepare_wave_units,
    wave_items_to_units,
)

from ..setup.config import POS_EMB_LLM_DEFAULT, POS_EMB_TIME_DEFAULT, UNIT_SEC


def present(record: dict, unit_paths: list[list[str]], parsed_texts: list[str], *,
            model: str, concat: bool, instruction: str, clip_of=None, cids=None,
            total_counts=None, chunks=None, item=None, pad_waves: bool = False,
            args=None) -> dict:
    del chunks, item, clip_of, instruction
    sec = getattr(args, "unit_sec", None) or UNIT_SEC

    items, warns = prepare_wave_units(unit_paths, sec, model)
    if pad_waves:
        items = _pad_waves(items)

    out = _full_out(items, unit_paths, parsed_texts, model, concat, cids, total_counts)
    out["warns"] = warns
    out["timemaps"] = [it["timemap"] for it in items] if _pe_on(args) else []
    return out


def _pe_on(args) -> bool:
    """Return whether time-aligned position embeddings are enabled."""
    time_flag = getattr(args, "pos_emb_time", None) or POS_EMB_TIME_DEFAULT
    llm_flag = getattr(args, "pos_emb_llm", None) or POS_EMB_LLM_DEFAULT
    return time_flag == "on" or llm_flag == "on"


def _pad_waves(items: list[dict]) -> list[dict]:
    """Zero-pad every waveform to the longest (processor-crash retry only).
    Timemaps are untouched: each covers one segment, so PE stays identity."""
    if len(items) < 2:
        return items
    mx = max(int(np.asarray(it["wave"]).shape[-1]) for it in items)
    out = []
    for it in items:
        w = np.asarray(it["wave"])
        if w.shape[-1] < mx:
            w = np.pad(w, (0, mx - w.shape[-1]))
        out.append({**it, "wave": w})
    return out


# ---------------------------------------------------------------------------
# audiohelix's shared builder
# ---------------------------------------------------------------------------

def _full_out(items, unit_paths, parsed_texts, model, concat, cids, total_counts) -> dict:
    temp_files: list[str] = []
    if model == "audio_flamingo3":
        contrib = [i + 1 for i, ps in enumerate(unit_paths) if ps]
        merged_wave = items[0]["wave"] if items else None
        contrib_paths = [s.path for s in items[0]["timemap"].segments] if items else []
        if not items and contrib:
            # The concat failed; mark those chunks as
            # contributing (no "(no audio)" suffix) but wrote no header, while
            # audiohelix's builder emits the header whenever contrib is non-empty.
            content = [{"type": "text",
                        "text": f"audio chunk {i}'s textual information:\n{t}\n"}
                       for i, t in enumerate(parsed_texts, start=1)]
            return {"content": content, "audio_item": None, "audio_sequence": [],
                    "temp_files": [], "prompt_log_body": "".join(b["text"] for b in content),
                    "prompt_log_tail": "", "chunks_with_audio": list(contrib),
                    "unit_segments": [], "chunk_segments": []}
        units = wave_items_to_units(items, model, temp_files=temp_files)
        rc = build_retrieved_content(
            units, parsed_texts, clip_of=None, cids=cids, total_counts=total_counts,
            concat=True, style=TRACEAV_STYLE, contrib=contrib,
            merged_wave=merged_wave, contrib_paths=contrib_paths)
    else:
        units = wave_items_to_units(items, model, temp_files=temp_files)
        rc = build_retrieved_content(
            units, parsed_texts, clip_of=None, cids=cids, total_counts=total_counts,
            concat=concat, style=TRACEAV_STYLE)

    unit_segments, chunk_segments = _segments(items, unit_paths, model)
    # AF3 runs do not carry an audio-path list: their zero-pad
    # retry guard tested the *item* count (always 1 for a single concat), so a
    # merge of several clips could never arm the retry. Keep it empty here too.
    audio_sequence = [] if model == "audio_flamingo3" else list(rc.audio_sequence)
    return {
        "content": list(rc.content),
        "audio_item": rc.audio_item,
        "audio_sequence": audio_sequence,
        "temp_files": temp_files + list(rc.temp_files),
        "prompt_log_body": _prompt_body(rc.content, items, model),
        "prompt_log_tail": _concat_note(items) if model == "audio_flamingo3" else "",
        "chunks_with_audio": list(rc.chunks_with_audio),
        "unit_segments": unit_segments,
        "chunk_segments": chunk_segments,
    }


def _prompt_body(content: list[dict], items: list[dict], model: str) -> str:
    """Build the prompt log body from audio labels and chunk text."""
    parts: list[str] = []
    audio_ord = 0
    idx = 0
    n = len(content)
    for k, blk in enumerate(content):
        if blk.get("type") != "text":
            continue
        nxt = content[k + 1] if k + 1 < n else None
        if nxt is not None and nxt.get("type") == "audio" and idx < len(items):
            it = items[idx]
            idx += 1
            audio_ord += 1
            parts.append(blk["text"])
            note = f" → {nxt['audio']}" if model == "voxtral" else ""
            parts.append(f"[AUDIO {audio_ord}: chunk {it['chunk'] + 1} "
                         f"item {it['unit'] + 1} (seq {it['seq']}), "
                         f"{it['timemap'].total_sec:.2f}s{note}]\n")
        else:
            parts.append(blk["text"])
    return "".join(parts)


def _concat_note(items: list[dict]) -> str:
    """Return the AF3 concatenated-audio prompt note."""
    if not items:
        return ""
    it = items[0]
    return (f"[AUDIO_CONCAT: {it['n_units']} clips merged into a single "
            f"waveform, {it['timemap'].total_sec:.2f}s]\n")


def _segments(items: list[dict], unit_paths: list[list[str]], model: str):
    """Return per-unit and per-chunk audio layouts for the results row."""
    unit_segments: list[dict] = []
    chunk_segments: list[dict] = []

    if model == "audio_flamingo3":
        if items:
            it = items[0]
            chunk_segments.append({
                "i": None, "audio_index": 1, "audio_indices": [1],
                "n_units": it["n_units"],
                "unit_seqs": [_seq_of_path(s.path) for s in it["timemap"].segments],
                "n_samples": int(np.asarray(it["wave"]).shape[-1]),
                "local_start": it["timemap"].orig_min,
                "local_end": it["timemap"].orig_max,
                "wavs": [s.path for s in it["timemap"].segments],
            })
            bounds: dict[int, tuple[int, int]] = {}
            acc = 0
            for pi, paths in enumerate(unit_paths or []):
                bounds[pi] = (acc, acc + len(paths))
                acc += len(paths)
            for k, s in enumerate(it["timemap"].segments):
                pi = next((p for p, (a, b) in bounds.items() if a <= k < b), None)
                unit_segments.append({
                    "i": None if pi is None else pi + 1, "audio_index": 1,
                    "unit": None if pi is None else k - bounds[pi][0],
                    "seq": _seq_of_path(s.path), "wav": s.path,
                    "n_samples": s.n_samples,
                    "local_start": s.orig_start, "local_end": s.orig_end,
                })
        return unit_segments, chunk_segments

    groups: dict[int, list[dict]] = {}
    for it in items:
        groups.setdefault(it["chunk"], []).append(it)
    audio_ord = 0
    for i in sorted(groups):
        group = groups[i]
        first_k = audio_ord + 1
        for it in group:
            audio_ord += 1
            unit_segments.append({
                "i": i + 1, "audio_index": audio_ord, "unit": it["unit"],
                "seq": it["seq"], "wav": it["timemap"].segments[0].path,
                "n_samples": int(np.asarray(it["wave"]).shape[-1]),
                "local_start": it["timemap"].orig_min,
                "local_end": it["timemap"].orig_max,
            })
        chunk_segments.append({
            "i": i + 1, "audio_index": first_k,
            "audio_indices": list(range(first_k, audio_ord + 1)),
            "n_units": len(group), "unit_seqs": [it["seq"] for it in group],
            "n_samples": sum(int(np.asarray(it["wave"]).shape[-1]) for it in group),
            "local_start": min(it["timemap"].orig_min for it in group),
            "local_end": max(it["timemap"].orig_max for it in group),
            "wavs": [it["timemap"].segments[0].path for it in group],
        })
    return unit_segments, chunk_segments


def _seq_of_path(path: str) -> int | None:
    """`{chunk}_a{seq}.wav` -> seq (None when the name doesn't match)."""
    uid = Path(path).stem
    if "_a" not in uid:
        return None
    try:
        return int(uid.rsplit("_a", 1)[1])
    except ValueError:
        return None
