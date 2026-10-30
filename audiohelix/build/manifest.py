from __future__ import annotations

import json
import math

from pathlib import Path

from audiohelix.audio_io import ChunkRecord, TARGET_SR
from audiohelix.config import PathsConfig


def make_pkg_paths(pkg_root: Path, chunk_sec: float) -> PathsConfig:
    """PathsConfig rooted at this package's data/ dir, with every field set
    explicitly. The index/results roots come from PathsConfig.index_dir_for /
    results_dir_for (data/index/{suite}/[tag]/, data/results/{model_tag}/)."""
    data = pkg_root / "data"
    ctag = f"c{chunk_sec:g}"
    return PathsConfig(
        root=pkg_root,
        questions_file=data / "questions" / "manifest.json",
        source_audio_dir=pkg_root,          # unused -- sources come from manifest item.files paths
        chunks_dir=data / "chunks" / ctag,
        captions_dir=data / "captions" / ctag,
        units_dir=data / "units" / ctag,
        embedding_cache_dir=data / "embed_cache" / ctag,
        index_dir=data / "index",
        results_dir=data / "results",
    )


def chunk_tag(chunk_sec: float) -> str:
    return f"c{chunk_sec:g}"


def artifact_tag(chunk_sec: float, cluster_tag: str) -> str:
    """Index/results folder prefix combining chunk_sec and the full cluster/unit config."""
    return f"{chunk_tag(chunk_sec)}-{cluster_tag}"


# A manifest item is one distinct audio -- a TraceAV video, an AudioMarathon
# suite item, an MMAU-Pro recording -- and is a plain dict:
#     {"item_id", "suite", "files": [{"path", "clip", "dur_s",
#      "start_chunk", "n_chunks"}], "n_chunks", "question_ids"}
# `suite` is that item's index scope segment -- "" when the benchmark owns
# the whole index dir (TraceAV). Multi-file items (MMAU multi-clip) are one
# item whose files are played in order.

# ---------------------------------------------------------------------------
# manifest I/O
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path} (run the manifest stage first)")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def assert_manifest_chunk_cfg(manifest: dict, chunk_sec: float, min_last_chunk_sec: float) -> None:
    """Checks that this stage's --chunk-sec/--min-last-chunk-sec matches what
    the manifest was built with -- a mismatch would silently misalign chunk
    counts and path tags, so it's rejected outright."""
    if abs(manifest.get("chunk_sec", -1) - chunk_sec) > 1e-6 or abs(
        manifest.get("min_last_chunk_sec", -1) - min_last_chunk_sec
    ) > 1e-6:
        raise SystemExit(
            f"manifest config mismatch: manifest chunk_sec={manifest.get('chunk_sec')}, "
            f"min_last_chunk_sec={manifest.get('min_last_chunk_sec')} != CLI "
            f"chunk_sec={chunk_sec}, min_last_chunk_sec={min_last_chunk_sec}. "
            f"Re-run the manifest stage with the same --chunk-sec."
        )


def items_from_manifest(manifest: dict, suite: str | None = None,
                        limit: int | None = None) -> list[dict]:
    """The manifest's item list (first-appearance order), filtered by --suite/--limit."""
    items = manifest["items"]
    if suite is not None:
        items = [u for u in items if u["suite"] == suite]
    return items if limit is None else items[:limit]


def questions_from_manifest(manifest: dict, suite: str | None = None,
                            limit: int | None = None) -> list[dict]:
    """The manifest's question list, filtered by --suite/--limit."""
    questions = manifest["questions"]
    if suite is not None:
        questions = [q for q in questions if q["suite"] == suite]
    return questions if limit is None else questions[:limit]


def build_manifest_dict(benchmark: str, chunk_sec: float, min_last_chunk_sec: float,
                        items: list[dict], questions: list[dict],
                        suite_stats: dict, built_at: str) -> dict:
    """The canonical manifest document written by every benchmark's manifest stage."""
    return {
        "benchmark": benchmark,
        "chunk_sec": chunk_sec,
        "min_last_chunk_sec": min_last_chunk_sec,
        "built_at": built_at,
        "suites": suite_stats,
        "n_items": len(items),
        "n_questions": len(questions),
        "items": items,
        "questions": questions,
    }


# ---------------------------------------------------------------------------
# chunking helpers
# ---------------------------------------------------------------------------

def item_chunk_count(dur_s: float, chunk_sec: float, min_last_chunk_sec: float) -> int:
    """Same chunk-count logic as audio_io.chunk_audio_to_chunks (including
    the last-fragment rule)."""
    n = math.ceil(dur_s / chunk_sec)
    cnt = 0
    for idx in range(n):
        start_sec = idx * chunk_sec
        end_sec = min(start_sec + chunk_sec, dur_s)
        if end_sec - start_sec < min_last_chunk_sec:
            break
        cnt += 1
    return cnt


def probe_duration(audio_path: str) -> float:
    """Duration (seconds) from the wav header, falling back to a full
    librosa decode if the header can't be read."""
    import soundfile as sf
    try:
        with sf.SoundFile(audio_path) as f:
            return len(f) / f.samplerate
    except Exception:  # noqa: BLE001 -- non-wav or otherwise unreadable by soundfile
        import librosa
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        return len(y) / sr


def chunk_item_files(files: list[dict], item_id: str, out_dir: str,
                     chunk_sec: float = 45.0, min_last_chunk_sec: float = 1.0,
                     expect_n: int | None = None) -> list[ChunkRecord]:
    """Cuts a item's files, in order, into one chunk sequence. files:
    [{path, clip, dur_s, start_chunk, n_chunks}] in playback order;
    chunk idx increases globally across file boundaries, start_sec/end_sec
    are local to each file. Output is deterministic, so existing wavs
    aren't rewritten. Chunk wavs go to {out_dir}/{item_id}/{item_id}_ch{idx:04d}.wav."""
    import librosa
    import soundfile as sf

    video_out_dir = Path(out_dir) / item_id
    video_out_dir.mkdir(parents=True, exist_ok=True)

    records: list[ChunkRecord] = []
    global_idx = 0
    for f in files:
        waveform, sr = librosa.load(f["path"], sr=TARGET_SR, mono=True)
        duration = len(waveform) / TARGET_SR
        n_chunks = math.ceil(duration / chunk_sec)
        for j in range(n_chunks):
            start_sec = j * chunk_sec
            end_sec = min(start_sec + chunk_sec, duration)
            if end_sec - start_sec < min_last_chunk_sec:
                break
            i0 = int(start_sec * TARGET_SR)
            i1 = int(end_sec * TARGET_SR)
            chunk_wav = waveform[i0:i1]

            idx = global_idx
            chunk_id = f"{item_id}_ch{idx:04d}"
            audio_path = video_out_dir / f"{chunk_id}.wav"
            if not audio_path.exists():
                sf.write(str(audio_path), chunk_wav, TARGET_SR)

            records.append(ChunkRecord(
                chunk_id=chunk_id, video_id=item_id, idx=idx,
                start_sec=start_sec, end_sec=end_sec, audio_path=str(audio_path),
            ))
            global_idx += 1

    if expect_n is not None and len(records) != expect_n:
        raise SystemExit(
            f"{item_id}: actual chunk count {len(records)} != manifest n_chunks {expect_n} -- "
            f"the audio file changed or the manifest is stale. Re-run the manifest stage."
        )
    return records


def group_chunks_by_file(files: list[dict],
                           chunks: list[ChunkRecord]) -> list[tuple[str, list[ChunkRecord]]]:
    """Regroups a global chunk sequence back by source file (the graph build
    cuts audio units per file and needs file-local time). File
    boundaries come from the manifest's start_chunk/n_chunks."""
    out: list[tuple[str, list[ChunkRecord]]] = []
    for f in files:
        lo = f["start_chunk"]
        hi = lo + f["n_chunks"]
        seg = [p for p in chunks if lo <= p.idx < hi]
        if seg:
            out.append((f["path"], seg))
    return out
