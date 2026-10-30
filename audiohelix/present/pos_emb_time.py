from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

MEL_FPS = 100.0   # mel frame rate shared by both model families (hop 160 @ 16kHz)
SR = 16000


# ---------------------------------------------------------------------------
# Original-time map
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Segment:
    """One segment of the concatenated waveform: its position in the
    concatenated stream (in_*) and its position in the source audio (orig_*)."""
    path: str
    in_start: float
    in_end: float
    orig_start: float
    orig_end: float
    n_samples: int


@dataclass
class TimeMap:
    """Maps concatenated-waveform frames to source-audio time."""
    segments: list[Segment]
    sr: int = SR

    @property
    def total_samples(self) -> int:
        return sum(s.n_samples for s in self.segments)

    @property
    def total_sec(self) -> float:
        return self.total_samples / self.sr

    @property
    def orig_min(self) -> float:
        return min(s.orig_start for s in self.segments)

    @property
    def orig_max(self) -> float:
        return max(s.orig_end for s in self.segments)

    def orig_times(self, t_in) -> np.ndarray:
        """Concatenated-stream time (seconds, array-like) -> source-audio
        time. Out-of-range values are clamped to the nearest segment end."""
        t = np.asarray(t_in, dtype=np.float64)
        out = np.empty(t.shape, dtype=np.float64)
        covered = np.zeros(t.shape, dtype=bool)
        for s in self.segments:
            m = (t >= s.in_start) & (t < s.in_end)
            out[m] = s.orig_start + (t[m] - s.in_start)
            covered |= m
        out[~covered] = self.segments[-1].orig_end  # exactly at total_sec, or out of range
        np.clip(out, self.orig_min, self.orig_max, out=out)
        return out

    def describe(self, max_items: int = 12) -> str:
        parts = []
        for i, s in enumerate(self.segments[:max_items]):
            parts.append(f"#{i + 1} orig {s.orig_start:7.2f}~{s.orig_end:7.2f}s"
                         f" (concat +{s.in_start:6.2f}s, {s.n_samples / self.sr:5.2f}s)")
        if len(self.segments) > max_items:
            parts.append(f"... and {len(self.segments) - max_items} more")
        span = self.orig_max - self.orig_min
        return (f"concat {self.total_sec:.2f}s / {len(self.segments)} segments, "
                f"source time {self.orig_min:.2f}~{self.orig_max:.2f}s (span {span:.2f}s)\n      "
                + "\n      ".join(parts))


def build_time_map(unit_paths: list[str], chunks: dict, unit_sec: float,
                   sr: int = SR, clip_offsets: dict[str, float] | None = None) -> TimeMap:
    """Builds a TimeMap from a concat-ordered list of unit wav paths.
    Source start time is recovered from the filename
    `{chunk_id}_a{seq:02d}.wav` + the caption chunk's start_sec +
    clip_offsets; duration comes from the wav header. `clip_offsets`
    (chunk_id -> cumulative duration of prior clips) is needed for
    MMAU-Pro multi-clip units, where a chunk's start_sec resets to 0 per
    file; None/{} leaves every offset at 0.
    """
    import soundfile as sf

    clip_offsets = clip_offsets or {}
    segments: list[Segment] = []
    cursor = 0  # samples
    for p in unit_paths:
        uid = Path(p).stem
        if "_a" not in uid:
            raise ValueError(f"audio unit id doesn't match {{chunk}}_a{{seq}}: {uid}")
        cid, seq_s = uid.rsplit("_a", 1)
        c = chunks.get(cid)
        if c is None:
            raise ValueError(f"no caption chunk for chunk {cid} -- cannot recover its time")
        seq = int(seq_s)
        orig_start = float(c["start_sec"]) + clip_offsets.get(cid, 0.0) + seq * unit_sec

        info = sf.info(p)
        n = int(info.frames)
        if info.samplerate != sr:  # concat uses librosa.load(sr=16000) -- match that axis
            n = int(round(n * sr / info.samplerate))
        if n <= 0:
            raise ValueError(f"empty wav: {p}")

        in_start = cursor / sr
        cursor += n
        segments.append(Segment(
            path=p, in_start=in_start, in_end=cursor / sr,
            orig_start=orig_start, orig_end=orig_start + n / sr, n_samples=n,
        ))

    if not segments:
        raise ValueError("unit_paths is empty")
    return TimeMap(segments=segments, sr=sr)


# ---------------------------------------------------------------------------
# TraceAV only -- a local (chunk-relative) time axis. Deliberately does not
# encode the source video's absolute time. AudioMarathon/MMAU-Pro don't use this.
# ---------------------------------------------------------------------------

def _seq_of(path: str, unit_sec: float, sr: int) -> tuple[str, int, int]:
    """wav path -> (chunk_id, seq, n_samples), length matched to the 16kHz axis."""
    import soundfile as sf

    uid = Path(path).stem
    if "_a" not in uid:
        raise ValueError(f"audio unit id doesn't match {{chunk}}_a{{seq}}: {uid}")
    cid, seq_s = uid.rsplit("_a", 1)
    seq = int(seq_s)
    if seq < 0:
        raise ValueError(f"negative seq: {uid}")

    info = sf.info(path)
    n = int(info.frames)
    if info.samplerate != sr:
        n = int(round(n * sr / info.samplerate))
    if n <= 0:
        raise ValueError(f"empty wav: {path}")
    return cid, seq, n


def build_local_time_map(per_chunk_paths: list[list[str]], unit_sec: float,
                         sr: int = SR) -> TimeMap:
    """Builds a TimeMap on a local axis where each chunk (chunk) starts at
    0 (TraceAV only): `orig_start = chunk_offset + seq * unit_sec`.
    `per_chunk_paths` is per-chunk wav path lists, seq-ascending within
    each chunk -- a single chunk gives that chunk's own local map
    (called once per chunk for qwen25/qwen3/voxtral); the full list keeps
    offsets continuous across chunks (AF3's single concat waveform). The
    offset accumulates local span, not actual length, to keep positions
    monotonic across an empty chunk.
    """
    if not per_chunk_paths:
        raise ValueError("per_chunk_paths is empty")

    segments: list[Segment] = []
    cursor = 0      # samples
    offset = 0.0    # this chunk's local-axis start (= cumulative span of prior chunks)
    for paths in per_chunk_paths:
        if not paths:
            continue    # chunk with no audio -- local axis doesn't advance either
        seqs: list[int] = []
        for p in paths:
            _, seq, n = _seq_of(p, unit_sec, sr)
            seqs.append(seq)
            orig_start = offset + seq * unit_sec
            segments.append(Segment(
                path=p, in_start=cursor / sr, in_end=(cursor + n) / sr,
                orig_start=orig_start, orig_end=orig_start + n / sr, n_samples=n,
            ))
            cursor += n
        offset += (max(seqs) + 1) * unit_sec

    if not segments:
        raise ValueError("no audio to concatenate -- every chunk was empty")
    return TimeMap(segments=segments, sr=sr)


# ---------------------------------------------------------------------------
# Active context read by the hooks (encoder layer, AudioMarathon/MMAU-Pro only)
# ---------------------------------------------------------------------------

@dataclass
class _Active:
    timemap: TimeMap
    pe_fps: float | None = None             # None = per-model default (mel/stride)
    af3_table: torch.Tensor | None = None   # (B, 1500, D) cache
    qwen_pos: torch.Tensor | None = None    # (num_chunks, T, D) cache
    stats: dict = field(default_factory=dict)
    n_applied: int = 0
    n_skipped: int = 0

    @property
    def expected_mel(self) -> float:
        return self.timemap.total_sec * MEL_FPS


# Tolerance (10ms frames) for feature-extractor rounding; still catches a dropped unit.
_MEL_TOL = 10


_ACTIVE: _Active | None = None


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Position tensor computation
# ---------------------------------------------------------------------------

def _downsample_convs(module) -> list:
    """The time-downsampling convs (k=3, s=2, p=1) in the conv frontend, in
    order (AF3: 1, Qwen2.5: 2, Qwen3: 3)."""
    convs = [m for m in module.modules()
             if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d)) and m.stride[0] > 1]
    return convs


def _conv_stride(module) -> int:
    """Total time downsampling factor = 2^(number of stride-2 convs)."""
    stride = 1
    for m in _downsample_convs(module):
        stride *= int(m.stride[0])
    return stride


def _conv_out_len(mel_len: int, convs: list) -> int:
    """Output length after the conv chain (k=3, s=2, p=1)."""
    t = mel_len
    for _ in convs:
        t = max(0, (t + 2 * 1 - 3) // 2 + 1)
    return t


def _af3_position_table(module, timemap: TimeMap, num_windows: int, pe_fps: float | None = None):
    """AF3: per-window (1500, D) position tensor -> (num_windows, 1500, D)."""
    n_frames = int(module.max_source_positions)     # 1500
    stride = _conv_stride(module)                   # 2
    mel_per_window = n_frames * stride              # 3000
    win_sec = mel_per_window / MEL_FPS              # 30s
    fps = pe_fps or (MEL_FPS / stride)              # 50 = 20ms/step

    # mel index at the center of frame t in window w = mel_per_window*w + stride*t
    mel_idx = (mel_per_window * np.arange(num_windows)[:, None]
               + stride * np.arange(n_frames)[None, :])
    t_orig = timemap.orig_times((mel_idx.ravel() / MEL_FPS)).reshape(mel_idx.shape)
    idx = np.rint(t_orig * fps).astype(np.int64)
    n_needed = int(idx.max()) + 1
    n_valid_mel = int(round(timemap.total_sec * MEL_FPS))   # actual concat length, excluding padding

    base = module.embed_positions.weight.detach()   # (1500, D), before any swap
    if n_needed > base.shape[0]:
        import torch.nn.functional as F
        # Stretch the learned table; compute in float32 (bf16 interp isn't universally supported).
        ext = F.interpolate(base.float().t().unsqueeze(0), size=n_needed,
                            mode="linear", align_corners=True)
        ext = ext.squeeze(0).t().to(base.dtype)
    else:
        ext = base
    rows = ext[torch.as_tensor(idx, device=ext.device, dtype=torch.long)]
    table = rows.view(num_windows, n_frames, -1)

    stats = {
        "n_frames": n_frames, "stride": stride, "win_sec": win_sec, "pe_fps": fps,
        "n_windows": num_windows, "n_needed": n_needed, "base_len": int(base.shape[0]),
        "stretch": n_needed / float(base.shape[0]),
        "t_orig_min": float(t_orig.min()), "t_orig_max": float(t_orig.max()),
        **_jump_stats(mel_idx.ravel(), t_orig.ravel(), idx.ravel(), n_valid_mel),
    }
    return table, stats


def _jump_stats(mel_flat: np.ndarray, t_orig_flat: np.ndarray, idx_flat: np.ndarray,
                n_valid_mel: int) -> dict:
    """Stats on segment-boundary jumps (a divergence from the expected
    mel-time rate marks a boundary). Padding frames past the valid concat
    length are excluded."""
    keep = mel_flat < n_valid_mel
    if keep.sum() < 2:
        return {"n_jumps": 0, "max_jump": 0.0, "jump_gap_sec": 0.0}
    d_orig = np.diff(t_orig_flat[keep])
    d_mel = np.diff(mel_flat[keep]) / MEL_FPS
    d_idx = np.diff(idx_flat[keep])
    is_jump = np.abs(d_orig - d_mel) > 0.01
    return {
        "n_jumps": int(is_jump.sum()),
        "max_jump": float(np.abs(d_idx[is_jump]).max()) if is_jump.any() else 0.0,
        "jump_gap_sec": float(np.abs(d_orig[is_jump]).max()) if is_jump.any() else 0.0,
    }


def _extended_sinusoid(pe_module, n: int, device, dtype) -> torch.Tensor:
    """Extends the sinusoidal position formula out to n positions (a closed
    form, so it's valid beyond the initial table's 1500)."""
    channels = int(pe_module.channels)
    log_timescale_increment = np.log(float(pe_module.max_timescale)) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
    scaled_time = torch.arange(n)[:, None] * inv_timescales[None, :]
    table = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    return table.to(device=device, dtype=dtype)


def _qwen_position_table(module, timemap: TimeMap, mel_len_total: int, pe_fps: float | None = None):
    """Qwen: per-chunk (T, D) position tensor -> (num_chunks, T, D). One
    chunk = n_window*2 mel frames (1s); Qwen2.5 adds PE after conv2
    (stride 2), Qwen3 after conv2d1-3 (stride 8), so T and the index unit
    differ automatically via _downsample_convs/_conv_out_len.
    """
    n_window = int(module.n_window)
    chunk_mel = n_window * 2
    convs = _downsample_convs(module)
    stride = _conv_stride(module)
    t_full = _conv_out_len(chunk_mel, convs)
    fps = pe_fps or (MEL_FPS / stride)              # Qwen2.5: 50 (20ms), Qwen3: 12.5 (80ms)

    n_chunks = int(np.ceil(mel_len_total / chunk_mel))
    mel_idx = (chunk_mel * np.arange(n_chunks)[:, None]
               + stride * np.arange(t_full)[None, :])
    t_orig = timemap.orig_times(mel_idx.ravel() / MEL_FPS).reshape(mel_idx.shape)
    idx = np.rint(t_orig * fps).astype(np.int64)
    n_needed = int(idx.max()) + 1
    n_valid_mel = int(round(timemap.total_sec * MEL_FPS))   # excludes tail-chunk padding

    pe = module.positional_embedding
    table = _extended_sinusoid(pe, n_needed, device=pe.positional_embedding.device,
                               dtype=torch.float32)
    rows = table[torch.as_tensor(idx, device=table.device, dtype=torch.long)]
    pos = rows.view(n_chunks, t_full, -1)

    stats = {
        "n_chunks": n_chunks, "chunk_sec": chunk_mel / MEL_FPS, "stride": stride,
        "T": t_full, "pe_fps": fps, "n_needed": n_needed, "sinusoid_len": int(pe.length),
        "t_orig_min": float(t_orig.min()), "t_orig_max": float(t_orig.max()),
        **_jump_stats(mel_idx.ravel(), t_orig.ravel(), idx.ravel(), n_valid_mel),
    }
    return pos, stats


# ---------------------------------------------------------------------------
# Proxy modules used to swap in the computed positions
# ---------------------------------------------------------------------------

class _WeightProxy(torch.nn.Module):
    """Returns a precomputed (B, N, D) tensor as `.weight` (AF3 adds
    positions via `inputs_embeds + self.embed_positions.weight`)."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self._weight = weight

    @property
    def weight(self) -> torch.Tensor:
        return self._weight


class _ChunkPE:
    """Proxy for per-chunk positional embeddings."""

    def __init__(self, tensor: torch.Tensor):
        self._t = tensor          # (num_chunks, T, D)
        self._T = int(tensor.shape[1])

    def __getitem__(self, key):
        if isinstance(key, tuple) and isinstance(key[0], slice):
            stop = key[0].stop
            if stop is not None and int(stop) != self._T:
                raise IndexError(
                    f"chunk position proxy: requested T={stop} != prepared T={self._T}")
        return self._t

    @property
    def shape(self):
        return self._t.shape


class _PEModuleProxy(torch.nn.Module):
    """Proxy swapped in for the positional_embedding submodule (Qwen)."""

    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.positional_embedding = _ChunkPE(tensor)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def _hook_inputs(args, kwargs):
    feats = kwargs.get("input_features")
    if feats is None and args:
        feats = args[0]
    second = kwargs.get("feature_lens")
    if second is None and len(args) > 1:
        second = args[1]
    mask = kwargs.get("input_features_mask")
    if mask is None and len(args) > 2:
        mask = args[2]
    return feats, second, mask


def _actual_mel_len(kind: str, feats, second, mask) -> int | None:
    """The mel frame count the model actually received, to cross-check
    against the timemap (built from the concat waveform)."""
    if kind == "af3":
        # AF3 passes (windows, mel, seq) features + (windows, seq) mask; mask sum = actual length.
        if torch.is_tensor(mask):
            return int(mask.detach().sum().item())
        return None
    if torch.is_tensor(second):
        return int(second.detach().cpu().reshape(-1)[0].item())
    if feats is not None and feats.dim() >= 2:
        return int(feats.shape[-1])
    return None


def _pre_hook(module, args, kwargs):
    ctx = _ACTIVE
    if ctx is None:
        return None
    kind = _ENCODER_KIND.get(type(module).__name__)
    feats, second, mask = _hook_inputs(args, kwargs)
    if kind is None or feats is None:
        return None

    # If a failed wav load desynced the timemap from the waveform, skip
    # rather than inject a wrong position.
    actual_mel = _actual_mel_len(kind, feats, second, mask)
    if actual_mel is not None and abs(actual_mel - ctx.expected_mel) > _MEL_TOL:
        ctx.n_skipped += 1
        if ctx.n_skipped <= 3 or ctx.n_skipped % 20 == 0:
            log(f"  [pos-emb] WARN concat length mismatch: model mel {actual_mel} vs timemap "
                f"{ctx.expected_mel:.0f} frames (diff {actual_mel - ctx.expected_mel:+.0f}) -- "
                f"skipping time alignment for this call (falls back to default PE, {ctx.n_skipped} so far)")
        return None
    ctx.n_applied += 1

    if kind == "af3":
        if feats.dim() != 3:
            log(f"  [pos-emb] AF3: unexpected input_features shape {tuple(feats.shape)} -- skipping")
            return None
        num_windows = int(feats.shape[0])
        if ctx.af3_table is None or ctx.af3_table.shape[0] != num_windows:
            ctx.af3_table, stats = _af3_position_table(module, ctx.timemap, num_windows, ctx.pe_fps)
            ctx.stats["af3"] = stats
            log(f"  [pos-emb] AF3 positions: {num_windows} windows x {stats['n_frames']} frames"
                f"({stats['win_sec']:.1f}s, {1000 / stats['pe_fps']:.0f}ms/step), learned table "
                f"{stats['base_len']} -> {stats['n_needed']} ({stats['stretch']:.2f}x interpolated), "
                f"source time {stats['t_orig_min']:.2f}~{stats['t_orig_max']:.2f}s, junction jumps "
                f"{stats['n_jumps']} (max {stats['jump_gap_sec']:.1f}s = {stats['max_jump']:.0f} steps)")
        module._pos_emb_backup = module.embed_positions
        module.embed_positions = _WeightProxy(ctx.af3_table.to(module.embed_positions.weight.device))
        return None

    # Qwen: mel length from feature_lens (or the last axis of input_features)
    mel_len_total = actual_mel if actual_mel is not None else _actual_mel_len(kind, feats, second, mask)
    if mel_len_total is None:
        mel_len_total = int(feats.shape[-1])
    if ctx.qwen_pos is None:
        ctx.qwen_pos, stats = _qwen_position_table(module, ctx.timemap, mel_len_total, ctx.pe_fps)
        ctx.stats["qwen"] = stats
        log(f"  [pos-emb] Qwen positions: {stats['n_chunks']} chunks x T={stats['T']}"
            f"({stats['chunk_sec']:.2f}s/chunk, stride {stats['stride']}, "
            f"{1000 / stats['pe_fps']:.0f}ms/step), sinusoid "
            f"{stats['sinusoid_len']} -> {stats['n_needed']}, source time "
            f"{stats['t_orig_min']:.2f}~{stats['t_orig_max']:.2f}s, junction jumps "
            f"{stats['n_jumps']} (max {stats['jump_gap_sec']:.1f}s = {stats['max_jump']:.0f} steps)")
    module._pos_emb_backup = module.positional_embedding
    module.positional_embedding = _PEModuleProxy(ctx.qwen_pos)
    return None


def _post_hook(module, args, output):
    backup = getattr(module, "_pos_emb_backup", None)
    if backup is not None:
        if type(module).__name__ == "AudioFlamingo3Encoder":
            module.embed_positions = backup
        else:
            module.positional_embedding = backup
        del module._pos_emb_backup
    return None


_ENCODER_KIND = {
    "AudioFlamingo3Encoder": "af3",
    "Qwen2_5OmniAudioEncoder": "qwen",
    "Qwen3OmniMoeAudioEncoder": "qwen",
}


# ---------------------------------------------------------------------------
# Install / context
# ---------------------------------------------------------------------------

def find_audio_encoder(llm):
    """Finds the audio encoder submodule in the backend, by class name."""
    for m in llm.model.modules():
        if type(m).__name__ in _ENCODER_KIND:
            return m
    return None


def install(llm, verbose: bool = True) -> str | None:
    """Installs the time-aligned position hooks on the audio encoder. Returns
    None if the model isn't supported."""
    enc = find_audio_encoder(llm)
    if enc is None:
        if verbose:
            log(f"  [pos-emb] no supported audio encoder found -- skipping (model: {type(llm.model).__name__})")
        return None
    enc.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    enc.register_forward_hook(_post_hook)
    if verbose:
        log(f"  [pos-emb] installed time-aligned position hooks: {type(enc).__name__}")
    return type(enc).__name__


@contextlib.contextmanager
def time_aligned_pos_emb(llm, timemap: TimeMap, pe_fps: float | None = None):
    """Applies time-aligned positional embedding to audio-encoder calls made
    inside this block. The concatenated waveform must follow timemap's
    order exactly. pe_fps overrides the index unit (default: per-model
    mel/stride, AF3/Qwen2.5 = 50, Qwen3 = 12.5).
    """
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError("nested time_aligned_pos_emb calls are not supported")
    if timemap is None:
        raise RuntimeError("time_aligned_pos_emb got timemap=None -- pass a build_time_map() result")
    ctx = _Active(timemap=timemap, pe_fps=pe_fps)
    _ACTIVE = ctx
    log(f"  [pos-emb] applying time-aligned positions: {timemap.describe()}")
    try:
        yield ctx
    finally:
        _ACTIVE = None
        st = ctx.stats.get("af3")
        if st:
            log(f"  [pos-emb] AF3 summary: table {st['base_len']}->{st['n_needed']} "
                f"({st['stretch']:.2f}x), {st['n_windows']} windows ({st['win_sec']:.1f}s)")
        st = ctx.stats.get("qwen")
        if st:
            log(f"  [pos-emb] Qwen summary: {st['n_chunks']} chunks ({st['chunk_sec']:.2f}s), "
                f"sinusoid {st['sinusoid_len']}->{st['n_needed']} ({1000 / st['pe_fps']:.0f}ms/step)")
        if ctx.n_applied or ctx.n_skipped:
            log(f"  [pos-emb] applied {ctx.n_applied}x / skipped (length mismatch) {ctx.n_skipped}x")
