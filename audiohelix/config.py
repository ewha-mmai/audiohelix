from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent  # repo root


def default_gpu() -> str:
    return os.environ.get("CUDA_VISIBLE_DEVICES", "4,6")


# ---------------------------------------------------------------------------
# The caption / index model
# ---------------------------------------------------------------------------

# Fixed, not a setting: every chunk's textual information (transcription +
# caption + concepts) comes from this model, and it owns every index. It is a
# separate role from the QA backbone, which the run does select (`qa --model`).
INDEX_MODEL = "qwen25_omni"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@dataclass
class PathsConfig:
    """Where one benchmark's artifacts live. Benchmarks construct this
    explicitly (see `eval/<name>/setup/config.py:paths()`), because each one owns
    a different root -- these defaults are only a fallback shape.

    `index_dir_for`/`results_dir_for` build the artifact roots: the index is
    keyed by an optional scope (the suite name for the multi-suite benchmarks,
    empty when the benchmark owns the whole index dir) and by the cluster tag;
    the results dir holds one folder per QA model.
    """
    root: Path = _HERE
    questions_file: Path = _HERE / "data" / "questions" / "questions.json"
    # Raw source audio, for single-file benchmarks (TraceAV:
    # $TRACEAV_ROOT/audio/{video_id}.wav). Multi-file benchmarks (AudioMarathon/
    # MMAU-Pro) take every path from the manifest instead.
    source_audio_dir: Path = Path(os.environ.get("TRACEAV_ROOT", "/path/to/TraceAV-Bench")) / "audio"

    chunks_dir: Path = _HERE / "data" / "chunks"
    captions_dir: Path = _HERE / "data" / "captions"
    units_dir: Path = _HERE / "data" / "units"
    embedding_cache_dir: Path = _HERE / "data" / "embed_cache"
    index_dir: Path = _HERE / "data" / "index"
    results_dir: Path = _HERE / "data" / "results"

    def _chunk_key(self, chunk_sec: float) -> str:
        """Return the path key for a chunk duration."""
        return "" if chunk_sec == 45.0 else f"c{chunk_sec:g}"

    def chunks_dir_for(self, chunk_sec: float = 45.0) -> Path:
        k = self._chunk_key(chunk_sec)
        return self.chunks_dir if not k else self.chunks_dir / k

    def captions_dir_for(self, chunk_sec: float = 45.0) -> Path:
        """The caption doc of every chunk. One captioner (INDEX_MODEL), so the
        path carries no per-model segment."""
        k = self._chunk_key(chunk_sec)
        return self.captions_dir if not k else self.captions_dir / k

    def units_dir_for(self, unit_sec: float, chunk_sec: float = 45.0) -> Path:
        """Split per unit_sec (s15, s30, ...) since wav filenames don't encode
        it, and per chunk_sec (c30_s5) since unit wavs are cut from the
        chunk's audio."""
        k = self._chunk_key(chunk_sec)
        name = f"s{unit_sec:g}" if not k else f"{k}_s{unit_sec:g}"
        return self.units_dir / name

    def embedding_cache_dir_for(self, unit_sec: float,
                                chunk_sec: float = 45.0) -> Path:
        """Keyed by (unit_sec, chunk_sec) -- the settings that change the
        embedding itself. Cached per video so changing the cluster config
        doesn't force re-running the expensive encoding pass."""
        k = self._chunk_key(chunk_sec)
        tag = f"s{unit_sec:g}" if not k else f"{k}_s{unit_sec:g}"
        return self.embedding_cache_dir / tag

    def index_dir_for(self, scope: str, cluster_tag: str = "") -> Path:
        """`index[/{scope}]/{cluster_tag}` -- the scope separates benchmarks
        that share one index dir (AudioMarathon's suites, MMAU-Pro); the cluster
        tag separates one graph build from another (`--cluster-k`)."""
        base = self.index_dir / scope if scope else self.index_dir
        if cluster_tag:
            base = base / cluster_tag
        return base

    def results_dir_for(self, model_tag: str) -> Path:
        """`results/{model_tag}` -- model_tag is the QA model's, the only model
        a run selects."""
        return self.results_dir / model_tag


# ---------------------------------------------------------------------------
# Audio chunking / unit splitting / clustering
# ---------------------------------------------------------------------------

@dataclass
class AudioConfig:
    chunk_sec: float = 45.0
    min_last_chunk_sec: float = 1.0
    unit_sec: float = 15.0
    target_sr: int = 16000


@dataclass
class ClusterConfig:
    """Configuration for spectral clustering of audio units."""
    spectral_n_clusters: int = 10

    def tag(self, unit_sec: float, chunk_sec: float = 45.0) -> str:
        """Path tag identifying this cluster config, e.g. "k10_s5"."""
        base = f"k{self.spectral_n_clusters}"
        if chunk_sec != 45.0:
            return f"{base}_c{chunk_sec:g}_s{unit_sec:g}"
        return f"{base}_s{unit_sec:g}"


# ---------------------------------------------------------------------------
# Retrieval / PPR
# ---------------------------------------------------------------------------

@dataclass
class RetrieveConfig:
    seed_top_k_text: int = 1
    seed_top_k_audio: int = 1
    max_iterations: int = 3
    iteration_threshold: float = 0.5
    damping: float = 0.5
    top_k_unit: int = 1
    retrieval_top_k: int = 5
    chunk_ratio: float = 1.5
    chunk_node_weight: float = 0.05


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    model: str = "qwen3_omni"
    qwen3_model_id: str = os.environ.get(
        "QWEN3_OMNI_MODEL_PATH", "/path/to/Qwen3-Omni-30B-A3B-Instruct")
    qwen25_model_id: str = "Qwen/Qwen2.5-Omni-7B"
    voxtral_model_id: str = "mistralai/Voxtral-Mini-3B-2507"
    audio_flamingo3_model_id: str = "nvidia/audio-flamingo-3-hf"

    gpu: str = field(default_factory=default_gpu)
    caption_min_new_tokens: int = 32
    caption_max_new_tokens: int = 1024
    qa_max_new_tokens: int = 1024
    qa_min_new_tokens: int = 0


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

@dataclass
class EmbedConfig:
    # PE-AV audio encoding uses GB-scale activation memory per clip, so batch
    # sizes are kept small (measured against a 47.5GB GPU).
    chunk_batch_size: int = 4
    unit_batch_size: int = 8
    concept_batch_size: int = 64
    pe_av_model_id: str = "facebook/pe-av-large"
