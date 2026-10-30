from __future__ import annotations

import os
import re
from pathlib import Path

from audiohelix.config import ClusterConfig, PathsConfig

NAME = "traceav"
BENCH_ROOT = Path(__file__).resolve().parents[1]      # eval/traceav
DATA_DIR = BENCH_ROOT / "data"

CHUNK_SEC = 45.0                                      # chunk length
MIN_LAST_CHUNK_SEC = 1.0                              # drop a shorter trailing chunk
UNIT_SEC = 5.0                                        # audio unit length

CLUSTER = ClusterConfig(
    spectral_n_clusters=10,
)

ENV_ROOT_VAR = "TRACEAV_ROOT"
AUDIO_SOURCE = "single"                               # exactly one source file per item
ID_FIELD = "video_id"                                 # results/resume key field

# Position embedding: each audio input gets its own local time map (chunk
# start = 0), applied on the LLM layer only. (AM/MMAU:
# "encoder_and_llm_absolute".)
PE_MODE = "llm_local_timemaps"

DIAGNOSTICS = True

MANIFEST_FILE = DATA_DIR / "questions" / "traceav_full.json"

# TraceAV has no suite/scope segment: its index lives directly under
# data/index/{cluster_tag}/ (AM/MMAU put a suite name there, and one index dir
# holds every suite). index_location() returns the empty scope for that.
CLUSTER_TAG = None                                    # AM/MMAU: a frozen artifact tag constant

# ---------------------------------------------------------------------------
# run contract (read by audiohelix/qa.py through getattr)
# ---------------------------------------------------------------------------

POS_EMB_LLM_DEFAULT = "on"
POS_EMB_TIME_DEFAULT = "off"                          # no encoder-layer time PE in traceav

RETRIEVAL_TOP_K_DEFAULT = 15                          # chunks; audio = m per chunk
AUDIO_TOP_M_DEFAULT = 2                               # audio units per chunk
NEAREST_CENTROIDS = 3
CENTROID_SIM_THRESHOLD = 0.85

SEED_TOP_K_TEXT = 1
SEED_TOP_K_AUDIO = 1

INDEX_CAP = "raw"                                     # top-k is never capped by n_chunks

MISSING_INDEX_POLICY = "skip"
NO_CHUNKS_POLICY = "skip"
RETRIEVAL_ERROR_POLICY = "raise"

# One zero-pad retry when the qwen processor rejects N waveforms of differing
# lengths (rare, kept as insurance). The prefix is prepended to qa_prompt only
# on that retry.
PAD_RETRY = True
PAD_RETRY_PREFIX = "[ZERO-PAD RETRY: all unit waveforms padded to max length]\n"

ONLY_MULTI_FLAG = False                               # MMAU-Pro-only CLI flag
SHARD_FLAG = False                                    # MMAU-Pro-only --shard i/n

# Presentation ids recorded in every row's `audio_mode` (the resume guard
# compares against the run's own constant).
AUDIO_MODE = "unit_separate_seg_of_total"             # qwen/voxtral: one audio per item + label
AUDIO_MODE_AF3 = "concat_single_with_labels"          # AF3: single concatenated audio

_DEFAULT_SOURCE_ROOT = "/path/to/TraceAV-Bench"


def source_root() -> Path:
    """$TRACEAV_ROOT (expects `data/*.jsonl` + `audio/`), resolved per call."""
    return Path(os.environ.get(ENV_ROOT_VAR, _DEFAULT_SOURCE_ROOT))


SOURCE_ROOT = source_root()


# ---------------------------------------------------------------------------
# Presentation (SPEC §4) -- TraceAV's frozen content-builder style.
#
# audiohelix/present/content_builder.py owns the style switch (PresentStyle + the two style
# instances); this only names TraceAV's. The values are the verified
# differences from AM/MMAU: a trailing "\\n" after the segment label, waveform
# arrays (not paths) as the audio value, and one local TimeMap per audio input.
# ---------------------------------------------------------------------------

def _traceav_style():
    try:
        from audiohelix.present.content_builder import TRACEAV_STYLE
        return TRACEAV_STYLE
    except ImportError:
        pass
    try:
        from audiohelix.present.content_builder import PresentStyle
    except ImportError:
        # audiohelix/present/content_builder.py has not landed the style switch yet (the
        # shared builder for AM/MMAU is still TraceAV's-only-free). Placeholder
        # type carrying the same four frozen values, so this package imports
        # and runs; replaced by audiohelix's TRACEAV_STYLE as soon as it exists.
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class PresentStyle:  # noqa: D101 -- placeholder for audiohelix's dataclass
            time_sep: str = "–"
            label_newline: bool = False
            audio_value: str = "path"
            per_unit_timemap: bool = False

    return PresentStyle(time_sep="–", label_newline=True,
                        audio_value="waveform", per_unit_timemap=True)


PRESENT = _traceav_style()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def paths(chunk_sec: float | None = None) -> PathsConfig:
    """Return TraceAV artifact paths rooted at the benchmark directory."""
    del chunk_sec  # applied by PathsConfig._chunk_key at every *_dir_for call
    return PathsConfig(
        root=BENCH_ROOT,
        questions_file=MANIFEST_FILE,
        source_audio_dir=source_root() / "audio",
        chunks_dir=DATA_DIR / "chunks",
        captions_dir=DATA_DIR / "captions",
        units_dir=DATA_DIR / "units",
        embedding_cache_dir=DATA_DIR / "embed_cache",
        index_dir=DATA_DIR / "index",
        results_dir=DATA_DIR / "results",
    )


def index_paths(paths: PathsConfig, chunk_sec: float):
    """Return index paths for the selected chunk size."""
    from audiohelix.build.index_builder import IndexPaths
    return IndexPaths(paths=paths, chunk_sec=chunk_sec)


def index_location(item: dict, chunk_sec: float, cluster_tag: str) -> tuple[str, str]:
    """(scope, tag) for `IndexPaths.graph_out_dir(scope, tag, item_id)`. TraceAV
    has no suite segment -- the scope is always "" -- and the tag is the one
    built *with* chunk_sec: `ClusterConfig.tag(UNIT_SEC, chunk_sec)`,
    e.g. "k10_s5" at 45s, "k10_c15_s5" at 15s. A `cluster_tag`
    built without the chunk key (ClusterConfig.tag(UNIT_SEC)) is re-tagged here
    so the index path does not depend on how the caller built it."""
    return "", _with_chunk_key(cluster_tag, chunk_sec)


_TAG_UNIT_SUFFIX_RE = re.compile(r"_s[\d.]+$")


def _with_chunk_key(cluster_tag: str, chunk_sec: float) -> str:
    """`{base}_s{sec}` -> `{base}_c{chunk}_s{sec}` (idempotent; the 45s key is
    "" so the configured tag is returned unchanged)."""
    if chunk_sec == CHUNK_SEC:
        return cluster_tag
    key = f"_c{chunk_sec:g}"
    if f"{key}_s" in cluster_tag:
        return cluster_tag
    m = _TAG_UNIT_SUFFIX_RE.search(cluster_tag)       # insert before the final "_s{sec}"
    if m:
        return f"{cluster_tag[:m.start()]}{key}{cluster_tag[m.start():]}"
    return f"{cluster_tag}{key}"


# ---------------------------------------------------------------------------
# captions / units / chunks dirs (audiohelix/qa.py's three path hooks)
# ---------------------------------------------------------------------------

def captions_dir(paths: PathsConfig, chunk_sec: float) -> Path:
    """Return the caption directory for the selected chunk size."""
    return paths.captions_dir_for(chunk_sec)


def units_dir(paths: PathsConfig, unit_sec: float, chunk_sec: float) -> Path:
    """Return the audio-unit directory for the selected sizes."""
    return paths.units_dir_for(unit_sec, chunk_sec)


def chunks_dir(paths: PathsConfig, chunk_sec: float) -> Path:
    """Return the chunk directory for the selected chunk size."""
    return paths.chunks_dir_for(chunk_sec)


# ---------------------------------------------------------------------------
# cluster / run tags
# ---------------------------------------------------------------------------

def cluster_config(args=None) -> ClusterConfig:
    """The ClusterConfig this run's index was built with. Clustering is
    spectral, fixed; `--cluster-k` is the one knob audiohelix/run.py exposes
    (unset -> the bench default above)."""
    k = getattr(args, "cluster_k", None)
    return ClusterConfig(
        spectral_n_clusters=k or CLUSTER.spectral_n_clusters,
    )


def model_tag(qa_model: str, args=None) -> str:
    """Return the QA model name used as the results directory key."""
    return qa_model


def out_dir(paths: PathsConfig, args, model_tag: str, cluster_tag: str = "") -> Path:
    """The results folder (no timestamp -- audiohelix/qa.py appends one, and
    `--out` overrides the whole thing):

        data/results/{model_tag}/

    The same shape as AudioMarathon's and MMAU-Pro's: the run's
    knobs -- the retrieval settings and the LLM time PE switch -- are recorded
    in this run's summary.json (`run_flags`, `pos_emb_*`), and the cluster tag
    is part of the *index* path (the run's k is in the meta's `cluster_config`),
    so none of them belongs in this path. Two runs of one QA model therefore
    differ only by their timestamp.
    """
    _fill_run_args(args)
    out = getattr(args, "out", None)
    if out:
        return Path(out)

    return paths.results_dir_for(model_tag)


def _fill_run_args(args=None) -> bool:
    """Fill the run knobs audiohelix/qa.py reads directly off `args` but audiohelix/run.py
    does not (yet) define from this config's *_DEFAULT constants:
    `retrieval_top_k`, `audio_top_m`, `nearest_centroids`,
    `centroid_sim_threshold`.

    Stopgap: every value set here is exactly the *_DEFAULT above, so once
    audiohelix/run.py defaults them from the bench config this becomes a no-op. It
    only ever fills a missing/None attribute -- a real CLI value always wins.
    """
    if args is None:
        return False
    filled = False
    for name, value in (("retrieval_top_k", RETRIEVAL_TOP_K_DEFAULT),
                        ("audio_top_m", AUDIO_TOP_M_DEFAULT),
                        ("nearest_centroids", NEAREST_CENTROIDS),
                        ("centroid_sim_threshold", CENTROID_SIM_THRESHOLD)):
        if getattr(args, name, None) is None:
            setattr(args, name, value)
            filled = True
    return filled
