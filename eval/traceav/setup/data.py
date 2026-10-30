from __future__ import annotations

import json
from pathlib import Path

from audiohelix.build.manifest import probe_duration, item_chunk_count

from .config import (
    CHUNK_SEC, MANIFEST_FILE, MIN_LAST_CHUNK_SEC, source_root,
)

# Sub-tasks of TraceAV-Bench's Audio-Centric Reasoning (AR) dimension, in the
# Preserve the questions file's record order.
AR_TASKS = ["a_speech_context", "a_background_music", "a_environmental_sound"]
_KEEP_FIELDS = ("video_id", "task_type", "question", "options", "correct_options")


# ---------------------------------------------------------------------------
# questions manifest
# ---------------------------------------------------------------------------

def build_manifest(args=None) -> list[dict]:
    """Build and save the TraceAV question manifest."""
    limit = getattr(args, "limit", None)
    out_path = Path(getattr(args, "out", None) or MANIFEST_FILE)
    root = source_root()

    audio_dir = root / "audio"
    have_video = {p.stem for p in audio_dir.glob("*.wav")}
    if not have_video:
        raise SystemExit(f"no audio files found under {audio_dir} -- check TRACEAV_ROOT")

    questions: list[dict] = []
    for task in AR_TASKS:
        src_path = root / "data" / f"{task}.jsonl"
        if not src_path.exists():
            raise SystemExit(
                f"missing {src_path} -- check TRACEAV_ROOT points at a full TraceAV-Bench checkout")
        n_raw = n_kept = 0
        with src_path.open(encoding="utf-8") as f:
            for line in f:
                n_raw += 1
                rec = json.loads(line)
                if rec["video_id"] not in have_video:
                    continue
                q = {k: rec[k] for k in _KEEP_FIELDS}
                q["source"] = f"{task}.jsonl"
                questions.append(q)
                n_kept += 1
        print(f"[{task}] {n_kept}/{n_raw} questions kept (video downloaded)")
        if limit is not None and len(questions) >= limit:
            questions = questions[:limit]
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    n_videos = len({q["video_id"] for q in questions})
    print(f"wrote {out_path}: {len(questions)} questions, {n_videos} unique videos")
    return questions


def questions(manifest, args=None) -> list[dict]:
    """Return manifest questions in stored order, optionally limited."""
    limit = getattr(args, "limit", None)
    return manifest if limit is None else manifest[:limit]


# ---------------------------------------------------------------------------
# index items
# ---------------------------------------------------------------------------

def items_for_index(manifest, args=None) -> list[dict]:
    """Return one index item per TraceAV video."""
    limit = getattr(args, "limit", None)
    chunk_sec = float(getattr(args, "chunk_sec", None) or CHUNK_SEC)
    audio_dir = source_root() / "audio"

    items: list[dict] = []
    seen: set[str] = set()
    for q in manifest:
        vid = q["video_id"]
        if vid in seen:
            continue
        seen.add(vid)
        path = audio_dir / f"{vid}.wav"
        dur_s = probe_duration(str(path))
        n_chunks = item_chunk_count(dur_s, chunk_sec, MIN_LAST_CHUNK_SEC)
        items.append({
            "item_id": vid,
            "suite": "",
            "task_type": q["task_type"],
            "files": [{"path": str(path), "clip": 0, "dur_s": dur_s,
                       "start_chunk": 0, "n_chunks": n_chunks}],
            "n_chunks": n_chunks,
        })
        if limit is not None and len(items) >= limit:
            break
    return items


def caption_doc_init(item: dict, chunk_sec: float) -> dict:
    """Return a fresh TraceAV captions-document skeleton."""
    return {"video_id": item["item_id"], "task_type": item["task_type"],
            "chunk_sec": chunk_sec, "chunks": []}


# ---------------------------------------------------------------------------
# QA-side lookups
# ---------------------------------------------------------------------------

def query_text(record: dict) -> str:
    """Return the question text used to seed retrieval."""
    return record["question"]


def index_dir_for(record: dict, paths, tag: str) -> Path:
    """Return the graph directory for one TraceAV video."""
    return paths.index_dir_for("", tag) / record["video_id"]
