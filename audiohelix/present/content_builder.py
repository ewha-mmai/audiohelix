from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

UNIT_SEC = 6.0
CHUNK_TEXT_FALLBACK = "(no content)"


# ---------------------------------------------------------------------------
# Per-chunk text
# ---------------------------------------------------------------------------

def format_parsed_text(caption_chunk: dict, include_concepts: bool = False) -> str:
    """The text block shown for one chunk: Transcription/Caption lines
    (+ Concepts, when the caller wants them).

    Concepts stay in the graph either way -- this only decides whether they
    also appear in the generation prompt.
    """
    parts = []
    if caption_chunk.get("transcription"):
        parts.append(f"Transcription: {caption_chunk['transcription']}")
    if caption_chunk.get("caption"):
        parts.append(f"Caption: {caption_chunk['caption']}")
    if include_concepts and caption_chunk.get("concepts"):
        parts.append(f"Entities: {', '.join(caption_chunk['concepts'])}")
    return "\n".join(parts) if parts else CHUNK_TEXT_FALLBACK


# ---------------------------------------------------------------------------
# Label formatting
# ---------------------------------------------------------------------------

def _fmt_time(sec: float) -> str:
    """Seconds -> `m:ss` for labels (25.0 -> `0:25`, 65.0 -> `1:05`). The
    value is a position inside that chunk's own audio, never an absolute
    recording timestamp."""
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def seq_and_local_time(path: str, unit_sec: float):
    """Derives (seq, start_sec, end_sec) from a `{chunk}_a{seq}.wav`
    filename. Returns (None, None, None) if the name doesn't match."""
    import soundfile as sf
    uid = Path(path).stem
    if "_a" not in uid:
        return None, None, None
    try:
        seq = int(uid.rsplit("_a", 1)[1])
    except ValueError:
        return None, None, None
    start = seq * unit_sec
    info = sf.info(path)
    dur = info.frames / info.samplerate
    return seq, start, start + dur


@dataclass(frozen=True)
class PresentStyle:
    time_sep: str = "–"              # en dash (U+2013) in every benchmark's labels
    label_newline: bool = False      # TraceAV's segment label ends with "\n"
    audio_value: str = "path"        # "path" | "waveform"
    concat_header: bool = True


DEFAULT_STYLE = PresentStyle()                       # AudioMarathon / MMAU-Pro
TRACEAV_STYLE = PresentStyle(time_sep="–", label_newline=True, audio_value="waveform")


# ---------------------------------------------------------------------------
# One retrieved audio input
# ---------------------------------------------------------------------------

@dataclass
class Unit:
    """One prepared audio input for a prompt."""
    chunk: int
    audio: object
    seq: int | None = None
    local_start: float | None = None
    local_end: float | None = None
    unit_id: str | None = None
    wav_path: str | None = None


def length_ok(wave, timemap) -> bool:
    """Whether a waveform and its time map agree on segment layout (by sample
    count; 0.1% tolerance for resampling/rounding)."""
    n = int(np.asarray(wave).shape[-1])
    return abs(n - timemap.total_samples) <= max(2, int(0.001 * timemap.total_samples))


def prepare_wave_units(unit_paths: list[list[str]], unit_sec: float,
                       model_name: str) -> tuple[list[dict], list[str]]:
    """Loads the retrieved units as waveforms + their local (chunk-relative)
    time maps -- for models fed samples rather than file paths.

    Returns (items, warns). `items` is in audio-input order (chunk rank, then
    ascending seq within a chunk); each element is
    `{"chunk", "unit", "seq", "wave", "timemap", "n_units": 1}`. A unit that
    fails on either the waveform or the time map is dropped from both and falls
    back to text-only, so audio count always matches time-map count.

    audio_flamingo3 is the exception: one concatenated waveform for the whole
    retrieved set (`chunk=None`, `n_units` = number of merged units).
    """
    from audiohelix.audio_io import load_wavs_concat
    from audiohelix.present import pos_emb_time

    items: list[dict] = []
    warns: list[str] = []

    if model_name == "audio_flamingo3":
        flat = [p for paths in unit_paths for p in paths]
        if not flat:
            return items, ["0 units selected -- text only"]
        wave = load_wavs_concat(flat)
        if wave is None or np.asarray(wave).size == 0:
            return items, ["concat failed/empty waveform -- text only"]
        try:
            timemap = pos_emb_time.build_local_time_map(unit_paths, unit_sec)
        except Exception as e:                                   # noqa: BLE001
            return items, [f"local timemap failed -- text only: {e}"]
        if len(timemap.segments) != len(flat):
            return items, [f"only {len(timemap.segments)}/{len(flat)} units mapped -- text only"]
        if not length_ok(wave, timemap):
            return items, [f"concat length {np.asarray(wave).shape[-1]} vs timemap "
                           f"{timemap.total_samples} -- text only"]
        return [{"chunk": None, "unit": None, "seq": None, "wave": wave,
                 "timemap": timemap, "n_units": len(flat)}], warns

    # one standalone audio per unit (same shape as the multiple-audio path)
    for i, paths in enumerate(unit_paths):
        if not paths:
            warns.append(f"chunk {i + 1}: 0 units selected -- text only")
            continue
        for j, path in enumerate(paths):
            what = f"chunk {i + 1} unit {j + 1} ({Path(path).name})"
            try:
                _, seq, _ = pos_emb_time._seq_of(path, unit_sec, pos_emb_time.SR)
            except Exception as e:                               # noqa: BLE001
                warns.append(f"{what}: wav id failed -- this unit text only: {e}")
                continue
            wave = load_wavs_concat([path])
            if wave is None or np.asarray(wave).size == 0:
                warns.append(f"{what}: waveform load failed/empty -- this unit text only")
                continue
            try:
                timemap = pos_emb_time.build_local_time_map([[path]], unit_sec)
            except Exception as e:                               # noqa: BLE001
                warns.append(f"{what}: local timemap failed -- this unit text only: {e}")
                continue
            if len(timemap.segments) != 1:
                warns.append(f"{what}: {len(timemap.segments)} segments -- this unit text only")
                continue
            if not length_ok(wave, timemap):
                warns.append(f"{what}: waveform length {np.asarray(wave).shape[-1]} vs timemap "
                             f"{timemap.total_samples} -- this unit text only")
                continue
            items.append({"chunk": i, "unit": j, "seq": seq, "wave": wave,
                          "timemap": timemap, "n_units": 1})
    return items, warns


def wave_items_to_units(items: list[dict], model_name: str,
                        temp_files: list[str] | None = None) -> list[Unit]:
    """`prepare_wave_units` output -> presentation Units. Voxtral's backend takes
    file paths only, so its units are written to temp wavs first (the caller
    deletes them after the call)."""
    units: list[Unit] = []
    for it in items:
        audio = it["wave"]
        if model_name == "voxtral":
            import os
            import soundfile as sf
            import tempfile
            from audiohelix.audio_io import TARGET_SR
            fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="voxtral_")
            os.close(fd)
            sf.write(tmp, it["wave"], TARGET_SR)
            if temp_files is not None:
                temp_files.append(tmp)
            audio = tmp
        units.append(Unit(
            chunk=it["chunk"] if it["chunk"] is not None else -1,
            audio=audio, seq=it["seq"],
            local_start=it["timemap"].orig_min, local_end=it["timemap"].orig_max,
            wav_path=it["timemap"].segments[0].path if it["timemap"].segments else None,
        ))
    return units


def units_from_paths(unit_paths: list[list[str]], unit_sec: float) -> list[Unit]:
    """Builds Units from on-disk unit wavs (`{chunk}_a{seq}.wav`), taking
    (seq, start, end) from the filename + the wav header."""
    units: list[Unit] = []
    for pi, paths in enumerate(unit_paths):
        for sp in paths:
            seq, start, end = seq_and_local_time(sp, unit_sec)
            units.append(Unit(chunk=pi, audio=sp, seq=seq,
                              local_start=start, local_end=end, wav_path=sp))
    return units


# ---------------------------------------------------------------------------
# Content assembly
# ---------------------------------------------------------------------------

@dataclass
class RetrievedContent:
    content: list[dict]                  # conversation content blocks
    audio_sequence: list[str]            # wav paths, in attachment order
    temp_files: list[str] = field(default_factory=list)
    audio_item: dict | None = None       # concat only: the single merged input
    chunks_with_audio: list[int] = field(default_factory=list)   # 1-based
    segments: list[dict] = field(default_factory=list)             # diagnostics


def build_retrieved_content(units: list[Unit], parsed_texts: list[str],
                            *, clip_of: list[int] | None = None,
                            total_counts: dict[str, int] | None = None,
                            cids: list[str] | None = None,
                            concat: bool = False,
                            style: PresentStyle = DEFAULT_STYLE,
                            instruction: str | None = None,
                            contrib: list[int] | None = None,
                            merged_wave=None,
                            contrib_paths: list[str] | None = None) -> RetrievedContent:
    """Assembles the retrieved content blocks.

    Non-concat: each chunk's units are listed individually, each labeled
    `audio chunk {i}'s segment {n} of {m} ({mm:ss}{sep}{mm:ss}):`, then the
    chunk's text once. `total_counts`/`cids` supply the label's `of {m}`
    (required whenever a unit's seq is known -- failing loudly beats silently
    dropping `of m` from the label).

    Concat: every unit's samples are merged into a single waveform labeled
    `audio chunks {a}-{b}'s segments (concatenated):`, and the merged input is
    returned as `audio_item` so the caller can place it after the instruction.

    `instruction`: when given, appended as the last text block (TraceAV does
    this inside its content list; the other benchmarks append it outside).
    """
    content: list[dict] = []
    temp_files: list[str] = []
    audio_item: dict | None = None
    audio_sequence: list[str] = []
    segments: list[dict] = []
    n_text_only = 0

    def chunk_label(i: int) -> str:
        """`audio chunk {i}` (1-based), with the source clip when the unit
        spans several files."""
        clip = clip_of[i - 1] if clip_of is not None else None
        base = f"audio chunk {i}"
        return f"{base} (Clip {clip})" if clip is not None else base

    if concat:
        from audiohelix.audio_io import load_wavs_concat
        # `merged_wave`/`contrib_paths`: for callers that already prepared the
        # single merged input (TraceAV's audio_flamingo3 path builds it together
        # with its one time map) instead of letting this function concatenate.
        flat = contrib_paths if contrib_paths is not None else \
            [u.wav_path for u in units if u.wav_path]
        if merged_wave is not None:
            waveform = merged_wave
        else:
            waveform = load_wavs_concat(flat) if flat else None
        # Chunks that contributed audio -- both for the header label and the
        # "(no audio)" marker in the text loop below. Callers whose units don't
        # carry chunk positions (a single merged input) pass it explicitly.
        if contrib is None:
            contrib = sorted({u.chunk + 1 for u in units})
        if contrib:
            a, b = contrib[0], contrib[-1]
            if len(contrib) == 1:
                head = f"audio chunk {a}'s segments (concatenated):\n"
            elif contrib == list(range(a, b + 1)):
                head = f"audio chunks {a}-{b}'s segments (concatenated):\n"
            else:
                head = f"audio chunks {','.join(str(c) for c in contrib)}'s segments (concatenated):\n"
            content.append({"type": "text", "text": head})
        if waveform is not None:
            audio_sequence = list(flat)
            if style.audio_value == "waveform":
                audio_item = {"type": "audio", "audio": waveform}
            else:
                import soundfile as sf
                import tempfile
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(tmp.name, waveform, 16000)
                temp_files.append(tmp.name)  # caller deletes these in a finally block
                audio_item = {"type": "audio", "audio": tmp.name}
        for i, text in enumerate(parsed_texts, start=1):
            suffix = "" if i in contrib else " (no audio)"
            content.append({"type": "text",
                            "text": f"{chunk_label(i)}'s textual information{suffix}:\n{text}\n"})
        if instruction is not None:
            content.append({"type": "text", "text": instruction})
        # The merged input itself is returned as `audio_item` (not appended):
        # each caller places it after its own instruction block.
        return RetrievedContent(content=content, audio_sequence=audio_sequence,
                                temp_files=temp_files, audio_item=audio_item,
                                chunks_with_audio=contrib)

    by_chunk: dict[int, list[Unit]] = {}
    for u in units:
        by_chunk.setdefault(u.chunk, []).append(u)

    audio_ord = 0
    for i, text in enumerate(parsed_texts):
        group = by_chunk.get(i, [])
        lbl = chunk_label(i + 1)
        if not group:
            n_text_only += 1
            part = f"{lbl}'s textual information (no audio):\n{text}\n"
        else:
            first_k = audio_ord + 1
            for n, u in enumerate(group, start=1):
                audio_ord += 1
                if u.seq is not None:
                    m_total = _chunk_total(total_counts, cids, i)
                    seg_no = u.seq + 1
                    label = (f"{lbl}'s segment {seg_no} of {m_total} "
                             f"({_fmt_time(u.local_start)}{style.time_sep}"
                             f"{_fmt_time(u.local_end)}):")
                else:
                    seg_no = n
                    label = f"--- {lbl} Audio ---"
                if style.label_newline:
                    label += "\n"
                content.append({"type": "text", "text": label})
                content.append({"type": "audio", "audio": u.audio})
                if u.wav_path:
                    audio_sequence.append(u.wav_path)
                segments.append({
                    "i": i + 1, "audio_index": audio_ord, "seq": u.seq,
                    "unit_id": u.unit_id, "wav": u.wav_path,
                    "local_start": u.local_start, "local_end": u.local_end,
                })
            last_k = audio_ord
            part = f"{lbl}'s textual information:\n{text}\n"
            segments.append({"i": i + 1, "chunk": True,
                             "audio_indices": list(range(first_k, last_k + 1)),
                             "n_units": len(group),
                             "local_start": min(u.local_start for u in group
                                                if u.local_start is not None),
                             "local_end": max(u.local_end for u in group
                                              if u.local_end is not None)})
        content.append({"type": "text", "text": part})
    if instruction is not None:
        content.append({"type": "text", "text": instruction})

    return RetrievedContent(content=content, audio_sequence=audio_sequence,
                            temp_files=temp_files, audio_item=audio_item,
                            chunks_with_audio=sorted({u.chunk + 1 for u in units}),
                            segments=segments)


def _chunk_total(total_counts: dict[str, int] | None,
                   cids: list[str] | None, i: int) -> int:
    """Total unit count `m` for chunk `i` (0-based). Raises rather than
    silently dropping `of m` from the label."""
    if not total_counts or not cids:
        raise RuntimeError(
            "Cannot build the label's `of {m}` -- total_counts/cids was not "
            "passed through to build_retrieved_content.")
    cid = cids[i]
    n_total = total_counts.get(cid)
    if not n_total:
        raise RuntimeError(
            f"Cannot build the label's `of {{m}}` -- cid {cid!r} (chunk {i + 1}) "
            f"is not in the index's audio_unit_id_to_chunk. Either this chunk has "
            f"no unit wavs at all, or the index is out of sync.")
    return n_total


def concat_waveform(units: list[Unit]) -> np.ndarray | None:
    """Merged waveform for the concat mode's single audio input (TraceAV builds
    it before the content so it can attach its one time map to it)."""
    from audiohelix.audio_io import load_wavs_concat
    flat = [u.wav_path for u in units if u.wav_path]
    return load_wavs_concat(flat) if flat else None


def system_message(model_name: str) -> dict | None:
    """The conversation's system turn. Voxtral's backend drops system messages
    and AF3's chat template injects its own default, so neither gets one."""
    if model_name in {"voxtral", "audio_flamingo3"}:
        return None
    return {"role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]}
