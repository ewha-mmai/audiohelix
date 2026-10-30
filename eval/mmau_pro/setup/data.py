from __future__ import annotations

import json
import time
from pathlib import Path

from audiohelix.present.content_builder import format_parsed_text
from audiohelix.build.manifest import (
    build_manifest_dict, load_manifest, probe_duration, questions_from_manifest,
    item_chunk_count, items_from_manifest,
)

from . import config
from ..adapters import SUITE_NAMES, load_adapter
from ..adapters.common import Item


def items_for_index(manifest: dict, args) -> list[dict]:
    """Return manifest items selected for index building."""
    suite = _opt(args, "suite", None)
    items = items_from_manifest(manifest, suite if isinstance(suite, str) else None,
                                _opt(args, "limit", None))
    if _opt(args, "only_multi", False) or _opt(args, "only_multi_clip_graphs", False):
        items = [u for u in items if len(u["files"]) > 1]
    return items


def caption_doc_init(item: dict, chunk_sec: float) -> dict:
    """The FRESH captions-doc skeleton stage1_extract_captions.load_or_init_doc()
    starts from: {"video_id", "item_id", "task_type", "chunk_sec",
    "source_files", "chunks": []}. `task_type` is the item's first question's
    task_type (stage1's task_type_of_item), "" when the item has none."""
    item_id = item["item_id"]
    return {"video_id": item_id, "item_id": item_id,
            "task_type": task_type_of_item().get(item_id, ""),
            "chunk_sec": chunk_sec,
            "source_files": [f["path"] for f in item["files"]],
            "chunks": []}


_TASK_TYPE_CACHE: dict[tuple[str, float], dict[str, str]] = {}


def task_type_of_item() -> dict[str, str]:
    """item_id -> its first question's task_type (cached; re-read when the
    manifest file changes)."""
    path = Path(config.MANIFEST_FILE)
    if not path.exists():
        return {}
    key = (str(path), path.stat().st_mtime)
    if key not in _TASK_TYPE_CACHE:
        out: dict[str, str] = {}
        for q in load_manifest(path)["questions"]:
            out.setdefault(q["item_id"], q.get("task_type", ""))
        _TASK_TYPE_CACHE.clear()
        _TASK_TYPE_CACHE[key] = out
    return _TASK_TYPE_CACHE[key]


def questions(manifest: dict, args) -> list[dict]:
    """The canonical question records, in test.parquet row order.

    Applies --limit, --only-multi, --shard, and --skip in that order."""
    qs = questions_from_manifest(manifest, config.SUITE, _opt(args, "limit", None))
    if _opt(args, "only_multi", False):
        qs = [q for q in qs if q.get("multi")]
    shard = _opt(args, "shard", None)
    if shard:
        idx_str, n_str = str(shard).split("/")
        shard_idx, shard_n = int(idx_str), int(n_str)
        qs = [q for i, q in enumerate(qs) if i % shard_n == shard_idx]
    skip = _opt(args, "skip", 0) or 0
    if skip:
        qs = qs[int(skip):]
    return qs


def _opt(args, name: str, default):
    """args.<name> if present and not None, else the bench default."""
    v = getattr(args, name, None)
    return default if v is None else v


def item_from_record(record: dict, item: dict | None) -> Item:
    """Convert a manifest question record to an adapter Item."""
    return Item(
        suite=record["suite"], qid=record["qid"], item_id=record["item_id"],
        audio_files=[f["path"] for f in item["files"]] if item else [],
        group=record.get("task_type", record["suite"]),
        question=record["question"], options=record["options"],
        correct_text=record.get("answer_text", ""),
        correct_letters=record.get("correct_letters"),
        meta=record.get("meta", {}),
    )


def build_parsed_text(chunks: dict, cids: list[str]) -> dict[str, str]:
    """Return parsed text blocks for the requested chunk IDs."""
    return {cid: format_parsed_text(chunks[cid], include_concepts=False)
            for cid in cids if cid in chunks}


def query_text(record: dict) -> str:
    """Return the question text used as the retrieval query."""
    return record["question"]


def index_dir_for(record: dict, paths, tag: str) -> Path:
    """The index folder of one item: {index}/{suite}/{tag}/{item_id}.

    `record` is a question record or a manifest item (both carry suite/item_id).
    `tag` is the artifact tag (audiohelix.build.manifest.artifact_tag(chunk_sec,
    cluster.tag(unit_sec)), e.g. "c30-k10_s6")."""
    return paths.index_dir_for(record["suite"], tag) / record["item_id"]


# ---------------------------------------------------------------------------
# multi-clip item semantics (MMAU is the only multi-file bench)
# ---------------------------------------------------------------------------

def item_chunk_clips(item: dict) -> dict[str, int | None]:
    """For a multi-clip item: chunk id (full "{item}_ch{idx:04d}") -> clip
    number (0-based). A single-file item returns {} (so its chunk labels stay
    unlabeled). Also carries a "(Clip {c})" chunk label for multi-clip items."""
    if len(item["files"]) <= 1:
        return {}
    item_id = item["item_id"]
    out: dict[str, int | None] = {}
    for f in item["files"]:
        for idx in range(f["start_chunk"], f["start_chunk"] + f["n_chunks"]):
            out[f"{item_id}_ch{idx:04d}"] = f["clip"]
    return out


def item_clip_of(index) -> list[int] | None:
    """Clip number (0-based) per chunk, in chunk order, for a multi-clip
    item's loaded index; None for a single-file item (labels stay unlabeled).

    Chunks are ordered by their global chunk idx (the `_ch{idx:04d}` suffix),
    i.e. clip playback order. The item is looked up by the index folder's name
    in the manifest, which is where the file boundaries live."""
    cids = sorted(index.chunk_id_to_text, key=_chunk_idx)
    if not cids:
        return None
    item = _items_by_id().get(Path(index.index_dir).name)
    if item is None or len(item.get("files") or []) <= 1:
        return None
    clips = item_chunk_clips(item)
    return [clips.get(cid) for cid in cids]


def clip_of(record: dict, item: dict | None, cids: list[str]) -> list[int | None] | None:
    """Return the clip number for each retrieved chunk, when applicable."""
    cid_clips = item_chunk_clips(item) if (record.get("multi") and item) else {}
    if not cid_clips:
        return None
    return [cid_clips.get(cid) for cid in cids]


def total_counts(record: dict, item: dict | None, n_total_by_cid: dict | None,
                 cids: list[str]) -> dict[str, int]:
    """Return the per-chunk audio-unit counts used in labels."""
    return dict(n_total_by_cid or {})


def item_clip_offsets(item: dict) -> dict[str, float]:
    """Return cumulative clip offsets for a multi-clip item."""
    if len(item["files"]) <= 1:
        return {}
    item_id = item["item_id"]
    out: dict[str, float] = {}
    cum = 0.0
    for f in item["files"]:
        for idx in range(f["start_chunk"], f["start_chunk"] + f["n_chunks"]):
            out[f"{item_id}_ch{idx:04d}"] = cum
        cum += float(f["dur_s"])
    return out


def _chunk_idx(cid: str) -> int:
    """Global chunk index from "{item_id}_ch{idx:04d}" (item ids never contain
    '_ch', adapters/common.py)."""
    head, sep, tail = cid.rpartition("_ch")
    return int(tail) if sep and tail.isdigit() else 0


_ITEMS_CACHE: dict[tuple[str, float], dict[str, dict]] = {}


def _items_by_id() -> dict[str, dict]:
    """Manifest items by item_id (cached; re-read when the file changes)."""
    path = Path(config.MANIFEST_FILE)
    if not path.exists():
        return {}
    key = (str(path), path.stat().st_mtime)
    if key not in _ITEMS_CACHE:
        _ITEMS_CACHE.clear()
        _ITEMS_CACHE[key] = {u["item_id"]: u for u in load_manifest(path)["items"]}
    return _ITEMS_CACHE[key]


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def build_manifest(args) -> dict:
    """Build and save the benchmark manifest."""
    chunk_sec = float(_opt(args, "chunk_sec", config.CHUNK_SEC))
    min_last_chunk_sec = float(_opt(args, "min_last_chunk_sec", config.MIN_LAST_CHUNK_SEC))
    suites = _opt(args, "suite", None) or list(SUITE_NAMES)
    if isinstance(suites, str):
        suites = [suites]
    suites = list(suites)
    limit = _opt(args, "limit", None)

    unknown = [s for s in suites if s not in SUITE_NAMES]
    if unknown:
        raise SystemExit(f"unknown suite(s): {unknown} (valid: {SUITE_NAMES})")

    items: list[dict] = []
    qs: list[dict] = []
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
            qs.append(qrec)
        suite_stats[suite] = {"n_questions": n_q0, "n_items": n_item}
        print(f"{suite}: {n_q0} questions / {n_item} items")

    manifest = build_manifest_dict(
        config.BENCHMARK, chunk_sec, min_last_chunk_sec, items, qs, suite_stats, built_at)

    manifest_path = Path(config.MANIFEST_FILE)   # read at call time (tests patch it)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(f"\nmanifest saved: {manifest_path}")
    print(f"total: {len(qs)} questions / {len(items)} items "
          f"(chunk_sec={chunk_sec:g}, min_last_chunk_sec={min_last_chunk_sec:g})")
    return manifest
