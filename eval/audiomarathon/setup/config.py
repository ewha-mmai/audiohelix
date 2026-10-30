from __future__ import annotations

import os
from pathlib import Path

from audiohelix.config import ClusterConfig, PathsConfig
from audiohelix.present.content_builder import DEFAULT_STYLE, PresentStyle
from audiohelix.build.index_builder import IndexPaths
from audiohelix.build.manifest import artifact_tag, chunk_tag, make_pkg_paths

NAME = "audiomarathon"
BENCH_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = BENCH_ROOT / "data"

CHUNK_SEC = 30.0
MIN_LAST_CHUNK_SEC = 1.0
UNIT_SEC = 6.0
CLUSTER = ClusterConfig(
    spectral_n_clusters=10,
)


def cluster_config(args=None) -> ClusterConfig:
    """Return the spectral clustering configuration used by index building."""
    k = getattr(args, "cluster_k", None)
    return ClusterConfig(
        spectral_n_clusters=k or CLUSTER.spectral_n_clusters,
    )

ENV_ROOT_VAR = "AUDIOMARATHON_ROOT"

AUDIO_SOURCE = "multi"

ID_FIELD = "item_id"

MANIFEST_FILE = DATA_DIR / "questions" / "manifest.json"

PRESENT: PresentStyle = DEFAULT_STYLE

ALL_SUITES = ["race", "had", "gtzan", "tau", "slue", "vesus", "desed", "vox", "vox_age"]

CLUSTER_TAG = "c30-k10_s6"
NEAREST_CENTROIDS = 3
CENTROID_SIM_THRESHOLD = 0.85
SEED_TOP_K_TEXT = 1
SEED_TOP_K_AUDIO = 1
RETRIEVAL_TOP_K_DEFAULT = int(os.environ.get("RETRIEVAL_TOP_K", "8"))
AUDIO_TOP_M_DEFAULT = 2
INDEX_CAP = "min_n_chunks"
MISSING_INDEX_POLICY = "skip"
NO_CHUNKS_POLICY = "skip"
RETRIEVAL_ERROR_POLICY = "skip"
PAD_RETRY = False
DIAGNOSTICS = False
ONLY_MULTI_FLAG = False
SHARD_FLAG = False
PE_MODE = "encoder_and_llm_absolute"
POS_EMB_LLM_DEFAULT = "off"
POS_EMB_TIME_DEFAULT = "off"

def paths(chunk_sec: float | None = None) -> PathsConfig:
    """Return paths rooted at the AudioMarathon data directory."""
    return make_pkg_paths(BENCH_ROOT, CHUNK_SEC if chunk_sec is None else chunk_sec)


def index_paths(paths: PathsConfig, chunk_sec: float | None = None) -> IndexPaths:
    """Return index paths for the benchmark."""
    return IndexPaths(paths=paths, chunk_sec=None)


def chunks_dir(paths: PathsConfig, chunk_sec: float | None = None) -> Path:
    """Return the chunk directory."""
    return paths.chunks_dir


def captions_dir(paths: PathsConfig, chunk_sec: float | None = None) -> Path:
    """Return the caption directory."""
    return paths.captions_dir_for()


def units_dir(paths: PathsConfig, unit_sec: float,
                  chunk_sec: float | None = None) -> Path:
    """Unit wavs: data/units/c30/s{unit_sec:g}."""
    return paths.units_dir_for(unit_sec)


def index_location(item: dict, chunk_sec: float, cluster_tag: str) -> tuple[str, str]:
    """Return the index scope and artifact tag for an item."""
    return item["suite"], artifact_tag_for(chunk_sec, cluster_tag)


def artifact_tag_for(chunk_sec: float, cluster_tag: str) -> str:
    """Return a chunk-prefixed artifact tag without double-prefixing."""
    ctag = chunk_tag(chunk_sec)
    if cluster_tag == ctag or cluster_tag.startswith(ctag + "-"):
        return cluster_tag
    return artifact_tag(chunk_sec, cluster_tag)


def results_dir() -> Path:
    """Return the benchmark results root."""
    return DATA_DIR / "results"


def model_tag(qa_model: str, args=None) -> str:
    """Return the QA model name used as the results folder key."""
    return qa_model


def out_dir(paths: PathsConfig, args, model_tag: str, cluster_tag: str = "") -> Path:
    """Return the output folder, using ``--out`` when provided."""
    if getattr(args, "out", None):
        return Path(args.out)
    return results_dir() / model_tag
