"""AudioMarathon manifest, item, and question loading."""

from __future__ import annotations

import json
import time
from pathlib import Path

from audiohelix.build.manifest import (
    build_manifest_dict, probe_duration, questions_from_manifest,
    item_chunk_count, items_from_manifest,
)

from . import config as cfg
from ..adapters import SUITE_NAMES, load_adapter
from ..adapters.common import Item

BENCHMARK = "audiomarathon"


def _as_suites(value) -> list[str]:
    """--suite as this bench's stages define it: None/"" -> all nine,
    "all" -> all nine, a comma-joined string or a repeated-flag list -> those."""
    if value is None:
        return list(SUITE_NAMES)
    if isinstance(value, str):
        parts = [s for s in value.split(",") if s]
    else:
        parts = [s for s in value if s]
    if not parts or "all" in parts:
        return list(SUITE_NAMES)
    return parts


def _suite_filter(value) -> str | None:
    """The index stage's --suite is a single suite (None = every suite)."""
    suites = _as_suites(value)
    return None if suites == list(SUITE_NAMES) and not value else suites[0]


def build_manifest(args, manifest_path: Path | None = None) -> dict:
    """Load suite items, compute chunk metadata, and save the manifest."""
    chunk_sec = getattr(args, "chunk_sec", None)
    chunk_sec = cfg.CHUNK_SEC if chunk_sec is None else float(chunk_sec)
    min_last_chunk_sec = getattr(args, "min_last_chunk_sec", None)
    if min_last_chunk_sec is None:
        min_last_chunk_sec = cfg.MIN_LAST_CHUNK_SEC
    min_last_chunk_sec = float(min_last_chunk_sec)
    suites = _as_suites(getattr(args, "suite", None))
    limit = getattr(args, "limit", None)
    unknown = [s for s in suites if s not in SUITE_NAMES]
    if unknown:
        raise SystemExit(f"unknown suite(s): {unknown} (valid: {SUITE_NAMES})")

    items: list[dict] = []
    questions: list[dict] = []
    item_of: dict[tuple[str, str], dict] = {}
    suite_stats: dict[str, dict] = {}
    built_at = time.strftime("%Y-%m-%d %H:%M:%S")

    for suite in suites:
        adapter = load_adapter(suite)
        raw_items = adapter.load_items(limit)
        n_q0 = len(raw_items)
        n_item = 0
        for it in raw_items:
            key = (suite, it.item_id)
            qrec = it.to_question_record(n_files=len(it.audio_files))
            u = item_of.get(key)
            if u is None:
                u = {"item_id": it.item_id, "suite": suite,
                     "files": [], "n_chunks": 0, "question_ids": []}
                item_of[key] = u
                items.append(u)
                seen_paths: set[str] = set()
                for clip, p in enumerate(it.audio_files):
                    if p in seen_paths:
                        continue
                    seen_paths.add(p)
                    dur = probe_duration(p)
                    n_chunks = item_chunk_count(dur, chunk_sec, min_last_chunk_sec)
                    if n_chunks == 0:
                        print(f"  [WARN] {it.item_id}: file {p} ({dur:.1f}s) has 0 chunks "
                              f"at chunk_sec {chunk_sec:g} -- will be skipped in QA")
                    u["files"].append({
                        "path": p, "clip": clip, "dur_s": round(dur, 3),
                        "start_chunk": u["n_chunks"], "n_chunks": n_chunks,
                    })
                    u["n_chunks"] += n_chunks
                n_item += 1
            else:
                file_paths = [f["path"] for f in u["files"]]
                if file_paths != it.audio_files:
                    raise SystemExit(
                        f"{suite}: item {it.item_id} points to a different audio file set "
                        f"across questions -- {file_paths} vs {it.audio_files}. Adapter item_id bug."
                    )
            u["question_ids"].append(qrec["qid"])
            questions.append(qrec)
        suite_stats[suite] = {"n_questions": n_q0, "n_items": n_item}
        print(f"{suite}: {n_q0} questions / {n_item} items")

    manifest = build_manifest_dict(
        BENCHMARK, chunk_sec, min_last_chunk_sec, items, questions, suite_stats, built_at,
    )
    out_path = cfg.MANIFEST_FILE if manifest_path is None else Path(manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(f"\nmanifest saved: {out_path}")
    print(f"total: {len(questions)} questions / {len(items)} items "
          f"(chunk_sec={chunk_sec:g}, min_last_chunk_sec={min_last_chunk_sec:g})")
    return manifest


def items_for_index(manifest: dict, args) -> list[dict]:
    """Return manifest items selected for index building."""
    items = items_from_manifest(manifest, _suite_filter(getattr(args, "suite", None)),
                                getattr(args, "limit", None))
    task_type_of_item: dict[str, str] = {}
    for q in manifest["questions"]:
        task_type_of_item.setdefault(q["item_id"], q.get("task_type", ""))
    return [{**u, "item_id": u["item_id"],
             "task_type": task_type_of_item.get(u["item_id"], "")} for u in items]


def caption_doc_init(item: dict, chunk_sec: float) -> dict:
    """Return a fresh captions-document skeleton."""
    item_id = item["item_id"]
    return {"video_id": item_id, "item_id": item_id,
            "task_type": item.get("task_type", ""),
            "chunk_sec": chunk_sec,
            "source_files": [f["path"] for f in item["files"]],
            "chunks": []}


def questions(manifest: dict, args) -> list[dict]:
    """Return canonical question records in suite and manifest order."""
    suites = _as_suites(getattr(args, "suite", None))
    out: list[dict] = []
    for s in suites:
        out.extend(questions_from_manifest(manifest, s, None))
    skip = getattr(args, "skip", 0) or 0
    if skip:
        out = out[skip:]
    limit = getattr(args, "limit", None)
    if limit is not None:
        out = out[:limit]
    return out


def query_text(record: dict) -> str:
    """Return the question text used as the retrieval query."""
    return record["question"]


def item_from_record(record: dict, item: dict | None) -> Item:
    """Convert a manifest question record to an adapter item."""
    return Item(
        suite=record["suite"], qid=record["qid"], item_id=record["item_id"],
        audio_files=[f["path"] for f in item["files"]] if item else [],
        group=record.get("task_type", record["suite"]),
        question=record["question"], options=record["options"],
        correct_text=record.get("answer_text", ""),
        correct_letters=record.get("correct_letters"),
        meta=record.get("meta", {}),
    )


def clip_of(record: dict, item: dict | None, cids: list[str]) -> list[int | None] | None:
    """Return the clip label for each retrieved chunk, when applicable."""
    if not record.get("multi") or item is None:
        return None
    cid_clips = item_clip_of(item)
    return [cid_clips.get(cid) for cid in cids] if cid_clips else None


def total_counts(record: dict, item: dict | None, n_total_by_cid: dict,
                 cids: list[str]) -> dict:
    """Return the per-chunk unit counts used in labels."""
    return n_total_by_cid


def item_clip_of(item: dict, cids: list[str] | None = None):
    """Return clip labels either as an aligned list or a chunk mapping."""
    if len(item["files"]) <= 1:
        cid_clips: dict[str, int | None] = {}
    else:
        item_id = item["item_id"]
        cid_clips = {}
        for f in item["files"]:
            for idx in range(f["start_chunk"], f["start_chunk"] + f["n_chunks"]):
                cid_clips[f"{item_id}_ch{idx:04d}"] = f["clip"]
    if cids is None:
        return cid_clips
    if not cid_clips:
        return None
    return [cid_clips.get(cid) for cid in cids]


def index_dir_for(record: dict, paths, tag: str,
                  chunk_sec: float | None = None) -> Path:
    """Return the per-item index folder for a question record."""
    full_tag = cfg.artifact_tag_for(cfg.CHUNK_SEC if chunk_sec is None else chunk_sec, tag)
    return paths.index_dir_for(record["suite"], full_tag) / record["item_id"]
