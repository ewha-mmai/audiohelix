from __future__ import annotations

import os
import time
from pathlib import Path

from audiohelix.config import ClusterConfig, PathsConfig
from audiohelix.config import INDEX_MODEL  # noqa: F401 -- re-exported as config.INDEX_MODEL
from audiohelix.present.content_builder import DEFAULT_STYLE
from audiohelix.build.index_builder import IndexPaths
from audiohelix.build.manifest import artifact_tag, make_pkg_paths

NAME = "mmau_pro"
BENCH_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = BENCH_ROOT / "data"

BENCHMARK = "mmau-pro"

SUITE = "mmau"

ID_FIELD = "item_id"

ENV_ROOT_VAR = "MMAU_PRO_ROOT"
AUDIO_SOURCE = "multi"

MANIFEST_FILE = DATA_DIR / "questions" / "manifest.json"

CHUNK_SEC = 30.0
MIN_LAST_CHUNK_SEC = 1.0
UNIT_SEC = 6.0

CLUSTER = ClusterConfig(spectral_n_clusters=10)


def cluster_config(args=None) -> ClusterConfig:
    """Return the spectral clustering configuration used by index building."""
    k = getattr(args, "cluster_k", None)
    return ClusterConfig(
        spectral_n_clusters=k or CLUSTER.spectral_n_clusters,
    )

PRESENT = DEFAULT_STYLE


def paths(chunk_sec: float | None = None) -> PathsConfig:
    """Return paths rooted at this benchmark's data directory."""
    return make_pkg_paths(BENCH_ROOT, CHUNK_SEC if chunk_sec is None else chunk_sec)


def index_paths(paths: PathsConfig, chunk_sec: float | None = None) -> IndexPaths:
    """Return index paths; the chunk size is already encoded in ``paths``."""
    return IndexPaths(paths=paths, chunk_sec=None)


def index_location(item: dict, chunk_sec: float, cluster_tag: str) -> tuple[str, str]:
    """Return the index scope and artifact tag for an item."""
    return item["suite"], artifact_tag(chunk_sec, cluster_tag)


def captions_dir(paths: PathsConfig, chunk_sec: float | None = None) -> Path:
    """Return the captions directory."""
    return paths.captions_dir_for()


def units_dir(paths: PathsConfig, unit_sec: float,
                  chunk_sec: float | None = None) -> Path:
    """Return the audio-unit directory."""
    return paths.units_dir_for(unit_sec)


def chunks_dir(paths: PathsConfig, chunk_sec: float | None = None) -> Path:
    """The per-chunk caption-doc folder: data/chunks/c30 (already baked)."""
    return paths.chunks_dir


def model_tag(qa_model: str, args=None) -> str:
    """Return the QA model name used as the results folder key."""
    return qa_model


def out_dir(paths: PathsConfig, args, model_tag: str, cluster_tag: str = "") -> Path:
    """Return the output folder, using ``--out`` when provided."""
    if getattr(args, "out", None):
        return Path(args.out)
    return paths.results_dir / model_tag


def run_timestamp() -> str:
    """Return a timestamp suitable for a run folder name."""
    return time.strftime("%Y%m%d_%H%M%S")


RETRIEVAL_TOP_K_DEFAULT = int(os.environ.get("RETRIEVAL_TOP_K", "8"))
RETRIEVAL_TOP_K = RETRIEVAL_TOP_K_DEFAULT
AUDIO_TOP_M_DEFAULT = 2              # m audio units per chunk
AUDIO_TOP_M = AUDIO_TOP_M_DEFAULT    # alias
NEAREST_CENTROIDS = 3                # soft centroid links, top-N
CENTROID_SIM_THRESHOLD = 0.85
SEED_TOP_K_TEXT = 1
SEED_TOP_K_AUDIO = 1

CLUSTER_TAG = "c30-k10_s6"

POS_EMB_LLM_DEFAULT = "off"
POS_EMB_TIME_DEFAULT = "off"
INDEX_CAP = "min_n_chunks"
MISSING_INDEX_POLICY = "row"
NO_CHUNKS_POLICY = "row"
RETRIEVAL_ERROR_POLICY = "row"
PAD_RETRY = False
DIAGNOSTICS = False
ONLY_MULTI_FLAG = True
SHARD_FLAG = True

PE_MODE = "encoder_and_llm_absolute"

# Multi-clip retrieval can use per-clip graphs instead of pooled retrieval.
# The graphs must already exist when this option is enabled.
MULTI_CLIP_GRAPHS = os.environ.get("MULTI_CLIP_GRAPHS", "0") == "1"
