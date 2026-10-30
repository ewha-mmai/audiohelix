from __future__ import annotations

import contextlib
import functools
import inspect
from dataclasses import dataclass, field

import numpy as np
import torch

from audiohelix.present.pos_emb_time import MEL_FPS, TimeMap, log

__all__ = ["install_llm", "time_aligned_llm_pos", "find_thinker"]

_MEL_TOL = 10  # tolerance (10ms frames) vs. pos_emb_time's


def _tokens_for_mel(mel_len: float) -> int:
    """mel frame count -> LLM audio token count (Qwen2.5/AF3/Qwen3's own
    conv2+pool2 formula)."""
    a = (int(round(mel_len)) - 1) // 2 + 1
    return (a - 2) // 2 + 1


# ---------------------------------------------------------------------------
# Active context
# ---------------------------------------------------------------------------

@dataclass
class _Active:
    timemaps: list[TimeMap]              # aligned 1:1 with audio input order (length 1 for AF3)
    stats: dict = field(default_factory=dict)
    n_applied: int = 0                   # number of runs repositioned (expected = number of audio inputs)
    n_skipped: int = 0
    skip_reason: str | None = None

    @property
    def total_sec(self) -> float:
        return sum(tm.total_sec for tm in self.timemaps)


_ACTIVE: _Active | None = None


# ---------------------------------------------------------------------------
# Position computation
# ---------------------------------------------------------------------------

def _run_offsets(n_tok: int, timemap: TimeMap,
                 sec_per_tok: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return token positions and their corresponding source times."""
    total = timemap.total_sec
    if sec_per_tok is None:
        sec_per_tok = total / n_tok
    t_in = (np.arange(n_tok, dtype=np.float64) + 0.5) * sec_per_tok
    t_orig = timemap.orig_times(np.minimum(t_in, total))
    # Continue past the padded tail at the last segment's rate instead of
    # freezing (orig_times clamps out-of-range inputs to the segment end).
    over = np.maximum(t_in - total, 0.0)
    if over.any():
        t_orig = t_orig + over
    rate = 1.0 / sec_per_tok                  # tokens/second (in-segment growth rate)
    rel = np.rint((t_orig - t_orig[0]) * rate).astype(np.int64)
    return rel, t_orig


def _jump_stats(t_orig: np.ndarray, step_sec: float) -> tuple[int, float]:
    """Number of segment-boundary jumps and the largest gap (seconds) --
    a deviation from the expected per-token rate marks a boundary."""
    if t_orig.size < 2:
        return 0, 0.0
    d = np.diff(t_orig)
    at_jump = np.abs(d - step_sec) > 0.01
    if not at_jump.any():
        return 0, 0.0
    return int(at_jump.sum()), float(np.abs(d[at_jump]).max())


def _audio_runs(input_ids_row: torch.Tensor, audio_token_id: int,
                mask_row: torch.Tensor | None) -> list[torch.Tensor]:
    """Contiguous spans of the audio placeholder token, in audio input
    order. Returns [] if there are none."""
    m = input_ids_row == audio_token_id
    if mask_row is not None:
        m = m & mask_row.bool()
    pos = torch.nonzero(m, as_tuple=False).flatten().tolist()
    spans: list[tuple[int, int]] = []        # (first index, last index)
    for i in pos:
        if spans and i == spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], i)
        else:
            spans.append((i, i))
    dev = input_ids_row.device
    return [torch.arange(a, b + 1, device=dev) for a, b in spans]


def _sec_per_tok_for(audio_seqlens, i: int, n_tok: int, timemap: TimeMap) -> float | None:
    """Seconds covered by one token of run i, derived from the mel length
    the model actually received (handles padding correctly). Returns None
    if audio_seqlens is missing/too short, so the caller falls back to
    `timemap.total_sec / n_tok`."""
    if audio_seqlens is None or n_tok <= 0:
        return None
    try:
        seq = torch.as_tensor(audio_seqlens).flatten()
        if seq.numel() <= i:
            return None
        mel = float(seq[i].item())
    except Exception:
        return None
    if not np.isfinite(mel) or mel <= 0:
        return None
    return mel / (MEL_FPS * n_tok)


def _short_mel(audio_seqlens, timemaps: list[TimeMap]) -> str | None:
    """Reason string if any run's received mel is shorter than that audio's
    real length (e.g. a truncated wav load), else None. A longer mel from
    padding is normal and isn't flagged."""
    if audio_seqlens is None:
        return None
    try:
        seq = torch.as_tensor(audio_seqlens).flatten().tolist()
    except Exception:
        return None
    if len(seq) != len(timemaps):
        return f"{len(seq)} audio_seqlens vs {len(timemaps)} timemaps"
    worst = None
    for i, (mel, tm) in enumerate(zip(seq, timemaps)):
        need = tm.total_sec * MEL_FPS
        if float(mel) + _MEL_TOL < need:
            d = need - float(mel)
            if worst is None or d > worst[1]:
                worst = (i, d, float(mel), need)
    if worst is None:
        return None
    i, _, mel, need = worst
    return f"audio {i + 1} too short (model mel {mel:.0f} < actual {need:.0f})"


def _mel_len_from_features(input_features_mask) -> float | None:
    """Mel frame count the model actually received. AF3 splits input into
    30s windows, so the sum across all windows (not per-row) is the real
    length."""
    if input_features_mask is None:
        return None
    try:
        return float(input_features_mask.sum().item())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Qwen2.5 / Qwen3 -- M-RoPE (get_rope_index) hook
# ---------------------------------------------------------------------------

def find_thinker(llm):
    """Finds the module with get_rope_index (Qwen Omni's thinker) by class
    name, not modules() order (Qwen3-Omni's talker has the same method)."""
    cands = [m for m in llm.model.modules()
             if callable(getattr(m, "get_rope_index", None))]
    if not cands:
        return None
    for m in cands:
        if "Thinker" in type(m).__name__:
            return m
    return cands[0]


def _audio_token_id_of(model) -> int | None:
    cfg = model.config
    for src in (getattr(cfg, "thinker_config", None), cfg):
        if src is None:
            continue
        for key in ("audio_token_index", "audio_token_id"):
            v = getattr(src, key, None)
            if isinstance(v, int):
                return v
    return None


def _rewrite_mrope(position_ids: torch.Tensor, input_ids: torch.Tensor,
                   attention_mask, audio_token_id: int, ctx: _Active,
                   audio_seqlens=None):
    """Rewrites the (3,B,L) M-RoPE positions to be time-aligned. Returns
    (number of runs applied, reason-if-none). Each run's shift accumulates
    onto everything after it, keeping the sequence monotonic."""
    B, L = input_ids.shape
    n_tm = len(ctx.timemaps)
    applied = 0
    first_reason: str | None = None
    for b in range(B):
        runs = _audio_runs(input_ids[b], audio_token_id,
                           None if attention_mask is None else attention_mask[b])
        if len(runs) != n_tm:
            # Audio/label mismatch -- skip rather than pair them up incorrectly.
            if first_reason is None:
                first_reason = f"{len(runs)} audio-token runs vs {n_tm} timemaps"
            continue
        for i, idx in enumerate(runs):
            n_tok = int(idx.numel())
            tm = ctx.timemaps[i]
            spt = _sec_per_tok_for(audio_seqlens, i, n_tok, tm)
            rel, t_orig = _run_offsets(n_tok, tm, sec_per_tok=spt)
            if spt is None:
                spt = tm.total_sec / n_tok
            base = int(position_ids[0, b, int(idx[0])])   # reflects any earlier-run shift
            new = torch.as_tensor(rel + base, dtype=position_ids.dtype,
                                  device=position_ids.device)
            old_max = int(position_ids[0, b, idx].max())
            shift = int(new.max()) - old_max
            position_ids[:, b, idx] = new
            if shift:                                        # push trailing text along
                tail = torch.arange(int(idx[-1]) + 1, L, device=input_ids.device)
                if attention_mask is not None:
                    tail = tail[attention_mask[b][tail].bool()]
                if tail.numel():
                    position_ids[:, b, tail] += shift
            applied += 1
            nj, mj = _jump_stats(t_orig, spt)
            _accum_run_stats(ctx, i, n_tok, base, shift, t_orig, nj, mj)
    if applied == 0:
        return 0, (first_reason or f"failed to reposition {n_tm} audio-token runs")
    return applied, None


def _accum_run_stats(ctx: _Active, i: int, n_tok: int, base: int, shift: int,
                     t_orig, n_jumps: int, max_jump_sec: float) -> None:
    """Accumulates per-run stats for logging."""
    st = ctx.stats.setdefault("llm", {
        "n_runs": 0, "n_tok": 0, "base": 0, "shift": 0,
        "t_orig_min": 0.0, "t_orig_max": 0.0, "n_jumps": 0, "max_jump_sec": 0.0,
        "runs": [],
    })
    st["n_runs"] += 1
    st["n_tok"] += n_tok
    st["shift"] += shift
    st["n_jumps"] += n_jumps
    st["max_jump_sec"] = max(st["max_jump_sec"], max_jump_sec)
    if st["n_runs"] == 1:
        st["base"] = base
        st["t_orig_min"], st["t_orig_max"] = float(t_orig[0]), float(t_orig[-1])
    else:
        st["t_orig_min"] = min(st["t_orig_min"], float(t_orig[0]))
        st["t_orig_max"] = max(st["t_orig_max"], float(t_orig[-1]))
    st["runs"].append({
        "i": i, "n_tok": n_tok, "base": base, "shift": shift,
        "t_min": float(t_orig[0]), "t_max": float(t_orig[-1]),
        "n_jumps": n_jumps, "max_jump_sec": max_jump_sec,
    })


def _install_mrope(thinker, verbose: bool) -> str:
    orig = thinker.get_rope_index
    sig = inspect.signature(orig)
    audio_token_id = _audio_token_id_of(thinker)
    if audio_token_id is None:
        raise RuntimeError("audio_token_index not found in config")

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        out = orig(*args, **kwargs)
        if not isinstance(out, tuple) or len(out) != 2:
            return out
        position_ids, deltas = out
        if _ACTIVE is None or not isinstance(position_ids, torch.Tensor):
            return out
        try:
            ba = sig.bind(*args, **kwargs)
            ba.apply_defaults()
            input_ids = ba.arguments.get("input_ids")
            attention_mask = ba.arguments.get("attention_mask")
            audio_seqlens = ba.arguments.get("audio_seqlens")
        except TypeError:
            return out
        if input_ids is None or not isinstance(input_ids, torch.Tensor):
            return out

        # Per-run length check: skip if the model received less mel than that
        # chunk's real length (padding to *more* is fine -- it just widens the interval).
        bad = _short_mel(audio_seqlens, _ACTIVE.timemaps)
        if bad is not None:
            _ACTIVE.n_skipped += 1
            _ACTIVE.skip_reason = bad
            return out

        n_applied, reason = _rewrite_mrope(position_ids, input_ids, attention_mask,
                                           audio_token_id, _ACTIVE, audio_seqlens=audio_seqlens)
        if n_applied == 0:
            _ACTIVE.n_skipped += 1
            _ACTIVE.skip_reason = reason
            return out
        _ACTIVE.n_applied += n_applied

        # Recompute rope_deltas from the current positions and input length.
        if isinstance(deltas, torch.Tensor) and deltas.ndim == 2:
            L = position_ids.shape[-1]
            for b in range(position_ids.shape[1]):
                if attention_mask is not None:
                    m = attention_mask[b].bool()
                    mx = position_ids[:, b, :][:, m].max() if bool(m.any()) else position_ids[:, b, :].max()
                else:
                    mx = position_ids[:, b, :].max()
                deltas[b, 0] = (mx + 1 - L).to(deltas.dtype)
        return position_ids, deltas

    thinker.get_rope_index = wrapper      # instance attribute -- called without self
    if verbose:
        log(f"  [pos-emb-llm] installed M-RoPE hook: {type(thinker).__name__}."
            f"get_rope_index (audio_token_id={audio_token_id})")
    return "qwen_mrope"


# ---------------------------------------------------------------------------
# Qwen2 (AF3) / Mistral (Voxtral) backbones -- inject position_ids directly
# ---------------------------------------------------------------------------

_AF3_DELTA_ATTR = "_pos_emb_llm_delta"

# Voxtral pads the waveform to a 30s multiple and emits 375 tokens per
# 30s chunk (12.5 tok/s, 80ms/token) -- token count alone validates length.
_VOXTRAL_SEC_PER_CHUNK = 30.0
_VOXTRAL_TOK_PER_SEC = 12.5


def _voxtral_expected_tokens(total_sec: float) -> int:
    n_chunks = max(int(np.ceil(total_sec / _VOXTRAL_SEC_PER_CHUNK - 1e-9)), 1)
    return int(round(n_chunks * _VOXTRAL_SEC_PER_CHUNK * _VOXTRAL_TOK_PER_SEC))


def _voxtral_sec_per_tok() -> float:
    return 1.0 / _VOXTRAL_TOK_PER_SEC


def _build_flat_positions(input_ids, attention_mask, audio_token_id, ctx, base=None,
                          expected_tokens: list[int] | None = None,
                          sec_per_tok: float | None = None):
    """(B,L) flat positions and the number of runs repositioned. Returns
    (None, 0) if it couldn't be applied. If `base` is given (generate()'s
    own cumsum-based positions), only the audio span is corrected on top of
    it, preserving conventions like left-padding. `expected_tokens` skips a
    run whose length doesn't match (Voxtral's truncation check)."""
    B, L = input_ids.shape
    n_tm = len(ctx.timemaps)
    if base is not None and base.shape == (B, L):
        positions = base.to(dtype=torch.long).clone()
    else:
        positions = torch.arange(L, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(B, -1).clone()
        if attention_mask is not None:
            positions = positions.masked_fill(attention_mask == 0, 0)
    out = None
    applied = 0
    for b in range(B):
        runs = _audio_runs(input_ids[b], audio_token_id,
                           None if attention_mask is None else attention_mask[b])
        if len(runs) != n_tm:
            ctx.n_skipped += 1
            ctx.skip_reason = f"{len(runs)} audio-token runs vs {n_tm} timemaps"
            return None, 0
        for i, idx in enumerate(runs):
            n_tok = int(idx.numel())
            if expected_tokens is not None and n_tok != expected_tokens[i]:
                ctx.n_skipped += 1
                ctx.skip_reason = (f"concat length mismatch (audio {i + 1} tokens {n_tok} vs "
                                   f"expected {expected_tokens[i]})")
                return None, 0
            tm = ctx.timemaps[i]
            rel, t_orig = _run_offsets(n_tok, tm, sec_per_tok=sec_per_tok)
            first = int(positions[b, int(idx[0])])       # reflects any earlier-run shift
            new = torch.as_tensor(rel + first, dtype=positions.dtype,
                                  device=positions.device)
            shift = int(new.max()) - int(positions[b, idx].max())
            positions[b, idx] = new
            if shift:
                tail = torch.arange(int(idx[-1]) + 1, L, device=input_ids.device)
                if attention_mask is not None:
                    tail = tail[attention_mask[b][tail].bool()]
                if tail.numel():
                    positions[b, tail] += shift
            nj, mj = _jump_stats(t_orig, sec_per_tok if sec_per_tok else tm.total_sec / n_tok)
            _accum_run_stats(ctx, i, n_tok, first, shift, t_orig, nj, mj)
            applied += 1
        out = positions
    return out, applied


def _install_flat_rope(inner, verbose: bool, kind: str) -> str:
    """For backbones that build position_ids directly (AF3=Qwen2,
    Voxtral=Mistral); kind picks the length safety check (mel vs. token count)."""
    audio_token_id = _audio_token_id_of(inner)
    is_vox = kind == "voxtral"

    def pre_hook(module, args, kwargs):
        ctx = _ACTIVE
        if ctx is None or audio_token_id is None:
            return None
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            return None

        provided = kwargs.get("position_ids")
        if provided is not None and provided.ndim != 2:
            return None                                  # unknown convention -- leave it alone
        if provided is not None and provided.device != input_ids.device:
            return None

        pkv = kwargs.get("past_key_values")
        try:
            past = 0 if pkv is None else int(pkv.get_seq_length())
        except Exception:
            past = 0

        if past > 0:                                     # decoding step -- just add the stored delta
            delta = getattr(module, _AF3_DELTA_ATTR, None)
            if delta is None:
                return None
            L = input_ids.shape[1]
            if provided is not None and provided.shape[-1] == L:
                pos = provided.to(dtype=torch.long) + delta
            else:
                pos = torch.arange(L, dtype=torch.long, device=input_ids.device) + past + delta
                pos = pos.unsqueeze(0).expand(input_ids.shape[0], -1)
            kwargs["position_ids"] = pos
            return args, kwargs

        expected_tokens = None
        sec_per_tok = None
        if is_vox:
            # Each chunk is a separate wav -> padded to its own 30s multiple -> per-run expected token count.
            expected_tokens = [_voxtral_expected_tokens(tm.total_sec) for tm in ctx.timemaps]
            sec_per_tok = _voxtral_sec_per_tok()
        else:
            # AF3 gets one audio input (the full concat), so the mel sum is this sample's real length.
            # (pos_emb_time.build_local_time_map given the full list also sums total_sec.)
            mel_actual = _mel_len_from_features(kwargs.get("input_features_mask"))
            need = ctx.total_sec * MEL_FPS
            if mel_actual is not None and mel_actual + _MEL_TOL < need:
                ctx.n_skipped += 1
                ctx.skip_reason = (f"concat too short (model mel {mel_actual:.0f} < "
                                   f"TimeMap {need:.0f})")
                return None

        # Takes generate()'s own positions (cumsum-1) as the base and rewrites only the audio span.
        positions, n_runs = _build_flat_positions(input_ids, kwargs.get("attention_mask"),
                                                  audio_token_id, ctx, base=provided,
                                                  expected_tokens=expected_tokens,
                                                  sec_per_tok=sec_per_tok)
        if positions is None:
            if ctx.skip_reason is None:
                ctx.n_skipped += 1
                ctx.skip_reason = "failed to reposition audio-token runs"
            return None
        ctx.n_applied += n_runs
        setattr(module, _AF3_DELTA_ATTR, int(positions.max()) + 1 - input_ids.shape[1])
        kwargs["position_ids"] = positions
        return args, kwargs

    inner.register_forward_pre_hook(pre_hook, with_kwargs=True)
    if verbose:
        log(f"  [pos-emb-llm] installed position_ids hook on {type(inner).__name__} "
            f"(audio_token_id={audio_token_id})")
    return "qwen2_rope" if not is_vox else "voxtral_rope"


# ---------------------------------------------------------------------------
# Install / context
# ---------------------------------------------------------------------------

def install_llm(llm, verbose: bool = True) -> str | None:
    """Installs LLM-side time-aligned position hooks. Returns None if the
    model isn't supported.

    Returns: 'qwen_mrope' (Qwen2.5/Qwen3-Omni) | 'qwen2_rope' (AF3) |
          'voxtral_rope' (Voxtral) | None
    """
    thinker = find_thinker(llm)
    if thinker is not None:
        try:
            return _install_mrope(thinker, verbose)
        except Exception as e:                            # noqa: BLE001
            if verbose:
                log(f"  [pos-emb-llm] failed to install M-RoPE hook, skipping: {e}")
            return None

    # Backbones that take position_ids -- distinguished by class name, since
    # AF3 and Voxtral both have a `language_model` attribute.
    inner = getattr(llm.model, "model", None)
    kind = None
    if inner is not None:
        cname = type(inner).__name__
        if "Voxtral" in cname:
            kind = "voxtral"
        elif "AudioFlamingo3" in cname or hasattr(inner, "language_model"):
            kind = "af3"
    if kind is not None:
        try:
            return _install_flat_rope(inner, verbose, kind)
        except Exception as e:                            # noqa: BLE001
            if verbose:
                log(f"  [pos-emb-llm] failed to install position_ids hook, skipping: {e}")
            return None

    if verbose:
        log(f"  [pos-emb-llm] unsupported model, skipping (model: {type(llm.model).__name__})")
    return None


@contextlib.contextmanager
def time_aligned_llm_pos(llm, timemaps):
    """Aligns audio-token positions to (local) time for LLM calls inside
    this block. timemaps: list of TimeMap aligned 1:1 with audio input
    order (a single TimeMap is also accepted, as for AF3)."""
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError("nested time_aligned_llm_pos calls are not supported")
    if timemaps is None:
        raise RuntimeError("time_aligned_llm_pos got timemaps=None")
    if isinstance(timemaps, TimeMap):
        timemaps = [timemaps]
    timemaps = list(timemaps)
    if not timemaps:
        raise RuntimeError("time_aligned_llm_pos got an empty timemaps list")
    ctx = _Active(timemaps=timemaps)
    _ACTIVE = ctx
    try:
        yield ctx
    finally:
        _ACTIVE = None
        st = ctx.stats.get("llm")
        if st:
            log(f"  [pos-emb-llm] repositioned {st['n_runs']} audio runs / {st['n_tok']} tokens "
                f"(first run start position {st['base']}, cumulative trailing-text shift +{st['shift']}), "
                f"local {st['t_orig_min']:.2f}~{st['t_orig_max']:.2f}s, "
                f"junction jumps {st['n_jumps']} (max {st['max_jump_sec']:.1f}s)")
        if ctx.n_applied or ctx.n_skipped:
            tail = f" -- {ctx.skip_reason}" if ctx.skip_reason else ""
            log(f"  [pos-emb-llm] applied {ctx.n_applied}x / skipped {ctx.n_skipped}x{tail}")
