from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List

import librosa
import numpy as np
import soundfile as sf

from audiohelix.config import AudioConfig
from audiohelix.utils import log

TARGET_SR = 16000


@dataclass
class ChunkRecord:
    chunk_id: str      # f"{video_id}_ch{idx:04d}"
    video_id: str
    idx: int             # 0-based order within this audio
    start_sec: float
    end_sec: float
    audio_path: str


@dataclass
class UnitRecord:
    unit_id: str     # f"{chunk_id}_a{seq:02d}"
    chunk_id: str
    seq: int             # 0-based order within the chunk
    start_sec: float     # global time in the source audio
    end_sec: float
    audio_path: str


def load_wavs_concat(audio_paths: List[str], target_sr: int = TARGET_SR) -> np.ndarray | None:
    """Concatenates wavs in order into one mono waveform, so retrieved chunks
    can be given to the model as a single audio with no chunk boundaries.
    Returns None if the list is empty or every file fails to load."""
    chunks: List[np.ndarray] = []
    for p in audio_paths:
        try:
            wav, _ = librosa.load(p, sr=target_sr, mono=True)
        except Exception as e:  # noqa: BLE001 -- one bad wav shouldn't kill the whole QA run
            log(f"  [WARN] failed to load wav, skipping in concat: {p}: {e}")
            continue
        if len(wav) > 0:
            chunks.append(wav)
    if not chunks:
        return None
    return np.concatenate(chunks)


def chunk_audio_to_records(
    source_audio_path: str,
    video_id: str,
    out_dir: str,
    chunk_sec: float = 45.0,
    min_last_chunk_sec: float = 1.0,
) -> List[ChunkRecord]:
    """Splits source_audio_path into chunk_sec-long wavs under out_dir/{video_id}/.
    Drops the final chunk if shorter than min_last_chunk_sec."""
    waveform, sr = librosa.load(source_audio_path, sr=TARGET_SR, mono=True)
    duration = len(waveform) / TARGET_SR

    video_out_dir = Path(out_dir) / video_id
    video_out_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = math.ceil(duration / chunk_sec)
    records: List[ChunkRecord] = []

    for idx in range(n_chunks):
        start_sec = idx * chunk_sec
        end_sec = min(start_sec + chunk_sec, duration)
        if end_sec - start_sec < min_last_chunk_sec:
            break

        i0 = int(start_sec * TARGET_SR)
        i1 = int(end_sec * TARGET_SR)
        chunk_wav = waveform[i0:i1]

        chunk_id = f"{video_id}_ch{idx:04d}"
        audio_path = video_out_dir / f"{chunk_id}.wav"
        # Output is deterministic for a given input, so skip if it already exists
        # (also avoids reading a half-written file from a concurrent run).
        if not audio_path.exists():
            sf.write(str(audio_path), chunk_wav, TARGET_SR)

        records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                video_id=video_id,
                idx=idx,
                start_sec=start_sec,
                end_sec=end_sec,
                audio_path=str(audio_path),
            )
        )

    return records


def make_unit_wavs(
    source_audio_path: str,
    chunk_records: List[ChunkRecord],
    out_dir: str,
    unit_sec: float = 15.0,
    min_last_unit_sec: float = 1.0,
) -> List[UnitRecord]:
    """Splits each chunk into unit_sec-long wavs under out_dir/{video_id}/,
    cut directly from the source using the chunk's global start/end so the
    result matches re-cutting the chunk wav. unit_id is {chunk_id}_a{seq:02d}."""
    video_out_dir = Path(out_dir) / chunk_records[0].video_id
    video_out_dir.mkdir(parents=True, exist_ok=True)

    waveform, sr = librosa.load(source_audio_path, sr=TARGET_SR, mono=True)
    records: List[UnitRecord] = []

    for p in chunk_records:
        n_units = math.ceil((p.end_sec - p.start_sec) / unit_sec)
        for seq in range(n_units):
            start_sec = p.start_sec + seq * unit_sec
            end_sec = min(start_sec + unit_sec, p.end_sec)
            if end_sec - start_sec < min_last_unit_sec:
                break

            i0 = int(start_sec * TARGET_SR)
            i1 = int(end_sec * TARGET_SR)
            if i0 >= i1:
                # Avoid writing an empty (header-only) wav on a boundary error --
                # PE-AV would produce NaN embeddings from it and break clustering.
                break
            unit_wav = waveform[i0:i1]

            unit_id = f"{p.chunk_id}_a{seq:02d}"
            audio_path = video_out_dir / f"{unit_id}.wav"
            if not audio_path.exists():
                sf.write(str(audio_path), unit_wav, TARGET_SR)

            records.append(
                UnitRecord(
                    unit_id=unit_id,
                    chunk_id=p.chunk_id,
                    seq=seq,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    audio_path=str(audio_path),
                )
            )

    return records
