#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))          # runnable from any cwd

from audiohelix.build.manifest import probe_duration, item_chunk_count  # noqa: E402
from audiohelix.config import ClusterConfig, INDEX_MODEL, PathsConfig  # noqa: E402
from audiohelix.utils import log  # noqa: E402
from eval.traceav import present as tv_present  # noqa: E402
from eval.traceav import prompts as tv_prompts  # noqa: E402

SCRIPT = Path(__file__).resolve()

DEFAULT_WORKDIR = "audiohelix_run"
DEFAULT_QA_MODEL = "qwen3_omni"            # default QA backbone
PAPER_CLUSTER_K = 10                       # default spectral k
STAGES = ("all", "manifest", "index", "captions", "graph", "qa")
_INDEX_SUBSTAGE = {"index": "all", "captions": "captions", "graph": "graph"}


_INSTRUCTION = """\
You are given audio segments retrieved from a long audio as relevant to the question, along with textual information associated with the retrieved segments. Carefully analyze the provided audio and text, integrating information across the retrieved segments to answer the question. Answer in a short phrase and do not explain your reasoning.

Question:
{question}"""


def local_instruction(record: dict, model: str, *, concat: bool = False,
                      evidence_framing: bool = False) -> str:
    """`prompts.instruction`. The same text for every backbone: TraceAV's
    template switches wording by model only because its runs needed a
    "concatenated" phrasing for AF3, which this instruction does not use."""
    del model, concat, evidence_framing
    return _INSTRUCTION.replace("{question}", record["question"])


class _LocalConfig:
    """Configuration for inference on one local audio file."""

    NAME = "local"
    AUDIO_SOURCE = "single"
    ID_FIELD = "audio_id"
    DIAGNOSTICS = False
    PE_MODE = "llm_local_timemaps"

    CHUNK_SEC = 45.0
    MIN_LAST_CHUNK_SEC = 1.0
    UNIT_SEC = 5.0
    CLUSTER = ClusterConfig(
        spectral_n_clusters=PAPER_CLUSTER_K,
    )

    POS_EMB_LLM_DEFAULT = "on"
    POS_EMB_TIME_DEFAULT = "off"
    RETRIEVAL_TOP_K_DEFAULT = 15
    AUDIO_TOP_M_DEFAULT = 2
    NEAREST_CENTROIDS = 3
    CENTROID_SIM_THRESHOLD = 0.85
    INDEX_CAP = "raw"
    MISSING_INDEX_POLICY = "skip"
    NO_CHUNKS_POLICY = "skip"
    RETRIEVAL_ERROR_POLICY = "raise"
    PAD_RETRY = True
    PAD_RETRY_PREFIX = "[ZERO-PAD RETRY: all unit waveforms padded to max length]\n"
    ONLY_MULTI_FLAG = False
    SHARD_FLAG = False

    # Row provenance (audiohelix/qa.py `_audio_mode`); a local run has no
    # per-model presentation variants, so one id serves every backbone.
    AUDIO_MODE = "local_freeform"

    def __init__(self, workdir: Path, args=None):
        self.WORKDIR = workdir
        self.DATA_DIR = workdir                          # logs/ (build.py)
        self.MANIFEST_FILE = workdir / "questions" / "local.json"
        self._args = args                                # this run's flags (CLUSTER_TAG)

    @property
    def CLUSTER_TAG(self) -> str:
        """Return the artifact tag for the effective run configuration."""
        args = self._args
        cfg = self.cluster_config(args) if args is not None else self.CLUSTER
        unit_sec = getattr(args, "unit_sec", None) or self.UNIT_SEC
        chunk_sec = getattr(args, "chunk_sec", None) or self.CHUNK_SEC
        return cfg.tag(unit_sec, chunk_sec)          # with the chunk key, as qa.py builds it

    # --- paths ------------------------------------------------------------
    def paths(self, chunk_sec: float | None = None) -> PathsConfig:
        """`{workdir}/{questions,chunks,captions,units,embed_cache,index,
        results}` -- TraceAV's layout with the benchmark root swapped for the
        workdir. `chunk_sec` is applied by the PathsConfig `*_dir_for` helpers
        (the 45s key is ""), exactly as traceav's own `paths()` does."""
        del chunk_sec
        return PathsConfig(
            root=self.WORKDIR,
            questions_file=self.MANIFEST_FILE,
            source_audio_dir=self.WORKDIR,       # unused: paths come from the manifest
            chunks_dir=self.WORKDIR / "chunks",
            captions_dir=self.WORKDIR / "captions",
            units_dir=self.WORKDIR / "units",
            embedding_cache_dir=self.WORKDIR / "embed_cache",
            index_dir=self.WORKDIR / "index",
            results_dir=self.WORKDIR / "results",
        )

    def index_paths(self, paths: PathsConfig, chunk_sec: float):
        from audiohelix.build.index_builder import IndexPaths
        return IndexPaths(paths=paths, chunk_sec=chunk_sec)

    def index_location(self, item: dict, chunk_sec: float,
                       cluster_tag: str) -> tuple[str, str]:
        """(scope, tag) -> `IndexPaths.graph_out_dir(...)`, i.e.
        `index/local/{tag}/{audio_id}/`. The build stage tags with
        `ClusterConfig.tag(unit_sec)` (no chunk key), the QA stage with
        `tag(unit_sec, chunk_sec)`; the chunk key is added here so both name
        the same directory (the same helper traceav's config uses)."""
        del item
        return "local", _with_chunk_key(cluster_tag, chunk_sec)

    def captions_dir(self, paths: PathsConfig, chunk_sec: float) -> Path:
        return paths.captions_dir_for(chunk_sec)

    def units_dir(self, paths: PathsConfig, unit_sec: float, chunk_sec: float) -> Path:
        return paths.units_dir_for(unit_sec, chunk_sec)

    def chunks_dir(self, paths: PathsConfig, chunk_sec: float) -> Path:
        return paths.chunks_dir_for(chunk_sec)

    # --- run tags ---------------------------------------------------------
    def cluster_config(self, args) -> ClusterConfig:
        """Return the spectral clustering configuration for this run."""
        return ClusterConfig(
            spectral_n_clusters=getattr(args, "cluster_k", None) or self.CLUSTER.spectral_n_clusters,
        )

    def model_tag(self, qa_model: str, args=None) -> str:
        suffix = "__pos-emb-llm" if getattr(args, "pos_emb_llm", None) == "on" else ""
        return f"{qa_model}{suffix}"

    def out_dir(self, paths: PathsConfig, args, model_tag: str, cluster_tag: str = "") -> Path:
        """The results folder. Deterministic on purpose (no timestamp): a rerun
        resumes in place, so the same knobs must land in the same folder. The
        folder name spells out every *QA* knob that changes the answers, so
        changing one can never silently mix rows; the cluster config is not one
        of them -- it lives in the index path (`index/local/{tag}/`) and in the
        run fingerprint this bench's evaluate hook guards on. `--out` overrides
        the whole path."""
        out = getattr(args, "out", None)
        if out:
            return Path(out)
        sim = getattr(args, "centroid_sim_threshold", None)
        sim = self.CENTROID_SIM_THRESHOLD if sim is None else sim
        seed_t = getattr(args, "seed_top_k_text", None)
        seed_a = getattr(args, "seed_top_k_audio", None)
        if seed_t is None or seed_a is None:
            seed_t, seed_a = 1, 1         # audiohelix/qa.py's per-modality default
        nearest = getattr(args, "nearest_centroids", None) or self.NEAREST_CENTROIDS
        top_k = getattr(args, "retrieval_top_k", None) or self.RETRIEVAL_TOP_K_DEFAULT
        m = getattr(args, "audio_top_m", None) or self.AUDIO_TOP_M_DEFAULT
        return (paths.results_dir
                / model_tag
                / f"nearest-centroids-{nearest}"
                / f"centroid-sim-{sim:g}"
                / f"seed-top-k-text-{seed_t}"
                / f"seed-top-k-audio-{seed_a}"
                / f"retrieval-top-{top_k}"
                / f"audio-top-m-{m}")


_TAG_UNIT_SUFFIX_RE = re.compile(r"_s[\d.]+$")


def _with_chunk_key(cluster_tag: str, chunk_sec: float) -> str:
    """`{base}_s{sec}` -> `{base}_c{chunk}_s{sec}` (idempotent; the 45s key is
    "" so the configured tag is returned unchanged)."""
    if chunk_sec == _LocalConfig.CHUNK_SEC:
        return cluster_tag
    key = f"_c{chunk_sec:g}"
    if f"{key}_s" in cluster_tag:
        return cluster_tag
    m = _TAG_UNIT_SUFFIX_RE.search(cluster_tag)       # insert before the final "_s{sec}"
    if m:
        return f"{cluster_tag[:m.start()]}{key}{cluster_tag[m.start():]}"
    return f"{cluster_tag}{key}"


# ---------------------------------------------------------------------------
# data -- the manifest and the index item
# ---------------------------------------------------------------------------

class _LocalData:
    """`data`: one item (your audio file) and one record per question. The
    manifest is the run's source of truth -- the later stages read the audio
    path from it, so `--stage qa` needs nothing but the manifest file."""

    def __init__(self, audio: Path, questions: list[str], manifest: Path):
        self.audio = audio
        self.audio_id = audio.stem
        self.questions_text = questions
        self.manifest = manifest

    def build_manifest(self, args=None) -> list[dict]:
        """Writes the manifest and returns it -- one record per question, in the
        same flat-list form TraceAV's own manifest uses."""
        records = [{"audio_id": self.audio_id, "audio_path": str(self.audio),
                    "task_type": "local", "question": q} for q in self.questions_text]
        limit = getattr(args, "limit", None)
        if limit is not None:
            records = records[:limit]
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        self.manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        dur = probe_duration(str(self.audio))
        chunk_sec = float(getattr(args, "chunk_sec", None) or _LocalConfig.CHUNK_SEC)
        unit_sec = float(getattr(args, "unit_sec", None) or _LocalConfig.UNIT_SEC)
        n_chunks = item_chunk_count(dur, chunk_sec, _LocalConfig.MIN_LAST_CHUNK_SEC)
        log(f"wrote {self.manifest}: {len(records)} question(s) about {self.audio.name} "
            f"({dur:.1f}s -> {n_chunks} chunk(s) of {chunk_sec:g}s, "
            f"~{int(dur // unit_sec)} unit(s) of {unit_sec:g}s)")
        return records

    def questions(self, manifest, args=None) -> list[dict]:
        limit = getattr(args, "limit", None)
        return manifest if limit is None else manifest[:limit]

    def items_for_index(self, manifest, args=None) -> list[dict]:
        """Return the single local audio file as an index item."""
        if not manifest:
            return []
        rec = manifest[0]
        path = Path(rec["audio_path"])
        chunk_sec = float(getattr(args, "chunk_sec", None) or _LocalConfig.CHUNK_SEC)
        dur = probe_duration(str(path))
        n_chunks = item_chunk_count(dur, chunk_sec, _LocalConfig.MIN_LAST_CHUNK_SEC)
        return [{
            "item_id": rec["audio_id"],
            "suite": "local",
            "task_type": "local",
            "files": [{"path": str(path), "clip": 0, "dur_s": dur,
                       "start_chunk": 0, "n_chunks": n_chunks}],
            "n_chunks": n_chunks,
        }]

    def caption_doc_init(self, item: dict, chunk_sec: float) -> dict:
        """Return a fresh captions-document skeleton."""
        return {"video_id": item["item_id"], "task_type": item["task_type"],
                "chunk_sec": chunk_sec, "chunks": []}

    def index_dir_for(self, record: dict, paths: PathsConfig, tag: str) -> Path:
        """Return the graph directory for the local audio item."""
        return paths.index_dir_for("local", tag) / record["audio_id"]


# ---------------------------------------------------------------------------
# evaluate -- the sink (no scoring: a local run has no gold answer)
# ---------------------------------------------------------------------------

class _LocalEvaluate:
    """`evaluate`: the QA loop's sink and resume hooks. `parse` keeps the raw
    answer, `is_correct` returns None (there is nothing to compare against),
    and a row carries the answer next to what retrieval actually selected."""

    _RESULTS = "results.json"

    def __init__(self, mode: str):
        self._mode = mode           # what a resume must match (see run_mode)
        self._rows: dict[str, list[dict]] = {}

    def startup(self, out_dir, args, questions) -> None:
        del args
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        log(f"local: {len(questions)} question(s) -> {out_dir}")

    def row_key(self, record: dict) -> tuple:
        return (record["audio_id"], record["question"])

    def load_done(self, out_dir, args) -> set:
        rows = self._read(out_dir, force=getattr(args, "force", False))
        return {self.row_key(r) for r in rows}

    def resume_guard(self, out_dir, args) -> None:
        """Refuse to append to a folder whose rows came from another run (a
        different QA/index model or cluster config) -- mixing them would produce
        one results.json describing the two runs."""
        rows = self._read(out_dir, force=getattr(args, "force", False))
        if any(r.get("run") != self._mode for r in rows):
            raise SystemExit(
                f"the existing results.json in {out_dir} came from a different run "
                f"({self._mode!r} now) -- use a new --workdir/--out folder, or --force.")

    def error_row(self, out_dir, record, args, kind) -> None:
        del out_dir, args
        log(f"  {record['audio_id']}: {kind}")

    def parse(self, raw: str, record: dict):
        del record
        return raw.strip()

    def is_correct(self, record: dict, raw: str, parsed) -> None:
        del record, raw, parsed
        return None                  # no gold answer for a local recording

    def result_row(self, record, raw, parsed, correct: bool | None, extra: dict) -> dict:
        del parsed, correct
        return {
            "audio_id": record["audio_id"],
            "question": record["question"],
            "answer": raw,
            "retrieved_chunks": extra.get("retrieved_chunks"),
            "parsed_texts": extra.get("parsed_texts"),
            "audio_units": extra.get("audio_units"),
            "kept_cids": extra.get("kept_cids"),
            "n_audio_inputs": extra.get("n_audio_inputs"),
            "n_total_by_cid": extra.get("n_total_by_cid"),
            "unit_segments": extra.get("unit_segments"),
            "chunk_segments": extra.get("chunk_segments"),
            "audio_mode": extra.get("audio_mode"),
            "pos_emb_stats": extra.get("pos_emb_stats") or None,
            "elapsed_sec": extra.get("elapsed_sec"),
            "qa_prompt": extra.get("qa_prompt"),
            "run": self._mode,
        }

    def emit(self, out_dir, record, raw, parsed, row) -> None:
        del record, raw, parsed
        rows = self._rows.setdefault(str(out_dir), [])
        rows.append(row)
        (Path(out_dir) / self._RESULTS).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    def finalize(self, out_dir, run: dict, args) -> dict:
        out_dir = Path(out_dir)
        rows = json.loads((out_dir / self._RESULTS).read_text(encoding="utf-8"))
        summary = {
            "run": self._mode,
            "n_questions": len(rows),
            "index_model": getattr(args, "index_model", None),
            "qa_model": getattr(args, "model", None),
            "cluster_tag": run.get("cluster_tag"),
            "chunk_sec": getattr(args, "chunk_sec", None),
            "unit_sec": getattr(args, "unit_sec", None),
            "retrieval_top_k": getattr(args, "retrieval_top_k", None),
            "audio_top_m": getattr(args, "audio_top_m", None),
            "results": str(out_dir / self._RESULTS),
            "answers": [{"question": r["question"], "answer": r["answer"]} for r in rows],
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"saved: {out_dir / self._RESULTS} ({len(rows)} question(s))")
        return summary

    def _read(self, out_dir, *, force: bool = False) -> list[dict]:
        """The rows already on disk ([] when there are none), cached per folder
        for the run -- `emit` appends to the same list it hands out here."""
        key = str(out_dir)
        if force:
            self._rows[key] = []
            return []
        if key not in self._rows:
            path = Path(out_dir) / self._RESULTS
            self._rows[key] = json.loads(path.read_text(encoding="utf-8")) \
                if path.exists() else []
        return self._rows[key]


# ---------------------------------------------------------------------------
# the bench and the run arguments
# ---------------------------------------------------------------------------

def make_bench(audio: Path, workdir: Path, questions: list[str],
               mode: str = "local", args=None) -> SimpleNamespace:
    """The inline benchmark: the five hooks `eval/<bench>/` provides. `args` is
    this run's flag namespace (the config's `CLUSTER_TAG` is derived from it)."""
    config = _LocalConfig(workdir, args)
    return SimpleNamespace(
        config=config,
        data=_LocalData(audio, questions, config.MANIFEST_FILE),
        prompts=SimpleNamespace(system=tv_prompts.system, instruction=local_instruction),
        present=tv_present,                      # TraceAV's frozen presentation
        evaluate=_LocalEvaluate(mode),
    )


def run_mode(args) -> str:
    """What a resume must match: the caption model (fixed, but recorded), the QA
    backbone, and the index's cluster config -- the results folder names the QA
    model only, so a `--cluster-k`/`--unit-sec` change has to be caught here
    instead. Called before `resolve_defaults`, hence the getattr/fallbacks."""
    k = getattr(args, "cluster_k", None) or _LocalConfig.CLUSTER.spectral_n_clusters
    unit_sec = getattr(args, "unit_sec", None) or _LocalConfig.UNIT_SEC
    chunk_sec = getattr(args, "chunk_sec", None) or _LocalConfig.CHUNK_SEC
    index_model = getattr(args, "index_model", None) or INDEX_MODEL
    tag = ClusterConfig(spectral_n_clusters=k).tag(unit_sec, chunk_sec)
    return f"local|index={index_model}|qa={args.model}|cluster={tag}"


def run_args(rest: list[str]) -> argparse.Namespace:
    """Parse the run flags with the CLI parser (so every
    `audiohelix.run qa` flag works here, with its own defaults and validation).
    `--bench traceav` is injected only to satisfy the required argument -- this
    run uses the inline bench above, never a benchmark package."""
    from audiohelix.run import build_parser
    return build_parser().parse_args(["qa", "--bench", "traceav", *rest])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="inference.py",
        description="Run AudioHelix on one local audio file.",
        epilog="Every `audiohelix.run qa` flag also works: --chunk-sec, --unit-sec, "
               "--cluster-k, --retrieval-top-k, --audio-top-m, "
               "--gpu, --limit, --force, ...",
    )
    ap.add_argument("--audio", required=True, metavar="FILE",
                    help="the recording to ask about (anything soundfile or librosa "
                         "can decode: wav/flac/mp3/ogg/...)")
    ap.add_argument("-q", "--question", action="append", default=[], metavar="TEXT",
                    help="a question about the recording; repeat for several")
    ap.add_argument("--questions-file", metavar="FILE",
                    help="read questions from a text file, one per line "
                         "(blank lines and '#' comments ignored)")
    ap.add_argument("--workdir", metavar="DIR",
                    help=f"where the index and results live "
                         f"(default: ./{DEFAULT_WORKDIR}/<audio name>)")
    ap.add_argument("--stage", default="all", choices=list(STAGES),
                    help="run the whole pipeline (default) or one stage only")
    ap.add_argument("--plan", action="store_true",
                    help="print what would run (paths, derived cluster k, child "
                         "commands) and exit without loading anything")
    return ap


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

def _probe_or_die(path: Path) -> float:
    if not path.exists():
        raise SystemExit(f"audio file not found: {path}")
    try:
        return probe_duration(str(path))
    except Exception as e:                                        # noqa: BLE001
        raise SystemExit(f"cannot read {path} as audio ({type(e).__name__}: {e})") from e


def _collect_questions(mine) -> list[str]:
    qs = [q.strip() for q in mine.question if q.strip()]
    if mine.questions_file:
        path = Path(mine.questions_file).expanduser()
        if not path.exists():
            raise SystemExit(f"questions file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                qs.append(line)
    if not qs:
        raise SystemExit("no questions given -- pass --question/-q (or --questions-file)")
    dupes = {q for q in qs if qs.count(q) > 1}
    if dupes:
        print(f"warning: {len(dupes)} duplicated question(s); identical text shares one "
              f"resume key, so only the first is run")
    return qs


def _has_flag(rest: list[str], name: str) -> bool:
    return any(a == name or a.startswith(f"{name}=") for a in rest)


def _cluster_k_for(dur: float, unit_sec: float) -> tuple[int, bool]:
    """Return a valid cluster count and whether it was capped."""
    n_units = int(dur // unit_sec) if unit_sec > 0 else 0
    k = max(2, min(PAPER_CLUSTER_K, n_units))
    return k, k < PAPER_CLUSTER_K


def plan(mine, audio: Path, workdir: Path, rest: list[str],
         questions: list[str]) -> dict:
    """Everything the run derives before any model loads: the QA backbone, the
    cluster k, the results folder, and the flags each stage is invoked with."""
    if _has_flag(rest, "--bench"):
        raise SystemExit("--bench is not used here -- this file runs the local "
                         "benchmark (one audio file); pass --audio instead.")
    dur = _probe_or_die(audio)
    # The captioning model is not a flag (the build reads audiohelix.config
    # .INDEX_MODEL), so the build side is replayed with the user's flags as-is.
    # `qa --model`, on the other hand, is required and has no default, so this
    # file supplies one -- for the QA child only, and for the in-process stages
    # to read off `args`.
    qa_rest = rest if _has_flag(rest, "--model") else [*rest, "--model", DEFAULT_QA_MODEL]

    args = run_args(qa_rest)
    bench = make_bench(audio, workdir, questions, mode=run_mode(args), args=args)
    from audiohelix.run import resolve_defaults
    resolve_defaults(bench, args)

    capped = False
    if not _has_flag(rest, "--cluster-k"):
        k, capped = _cluster_k_for(dur, args.unit_sec)
        rest = [*rest, "--cluster-k", str(k)]
        qa_rest = [*qa_rest, "--cluster-k", str(k)]
        args = run_args(qa_rest)                 # re-parse, so args carries the k
        bench = make_bench(audio, workdir, questions, mode=run_mode(args), args=args)
        resolve_defaults(bench, args)

    paths = bench.config.paths(args.chunk_sec)
    # The tag must name the folder the build writes -- i.e. carry the k this run
    # ends up with (see _LocalConfig.CLUSTER_TAG), not the configured k.
    cluster_tag = bench.config.CLUSTER_TAG
    model_tag = bench.config.model_tag(args.model, args)
    return {
        # argv: what the build child is replayed with (no model flag). argv_qa
        # adds the QA backbone the run resolved to.
        "argv": [*mine_argv(mine), *rest],
        "argv_qa": [*mine_argv(mine), *qa_rest],
        "audio": audio, "dur": dur,
        "workdir": workdir, "rest": rest, "args": args, "bench": bench, "paths": paths,
        "manifest": bench.config.MANIFEST_FILE, "cluster_tag": cluster_tag,
        "model_tag": model_tag,
        "out_dir": bench.config.out_dir(paths, args, model_tag, cluster_tag),
        "capped": capped,
        "n_units": int(dur // args.unit_sec) if args.unit_sec else 0,
        "stage": mine.stage,
    }


def mine_argv(mine) -> list[str]:
    """The user's own flags, replayed to the child processes (`--stage` is left
    out: each child is told the one stage it must run)."""
    argv = ["--audio", str(mine.audio)]
    for q in mine.question:
        argv += ["-q", q]
    if mine.questions_file:
        argv += ["--questions-file", str(mine.questions_file)]
    if mine.workdir:
        argv += ["--workdir", str(mine.workdir)]
    return argv


def _child_argv(argv: list[str], *extra: str) -> list[str]:
    """This file, re-invoked with the run's flags plus the stage to run and the
    pins for that stage. The appends win (argparse keeps the last occurrence),
    so the injected --stage/--out cannot be shadowed by a user flag."""
    return [sys.executable, str(SCRIPT), *argv, *extra]


def _index_child(p: dict) -> list[str]:
    """The build child: `--stage index` (captions + graph, one process). The
    captioning model is not a flag -- the build reads it from
    audiohelix.config.INDEX_MODEL -- so no model is pinned here."""
    return _child_argv(p["argv"], "--stage", "index")


def _qa_child(p: dict) -> list[str]:
    """The QA child: `--out` pinned to the derived results folder, so a resumed
    run lands in the same place instead of a timestamped one, plus the QA
    backbone (`--model`, injected by plan() into argv_qa)."""
    return _child_argv(p["argv_qa"], "--stage", "qa", "--out", str(p["out_dir"]))


def _steps(p: dict) -> list[tuple[str, list[str]]]:
    """The child processes a `--stage all` run spawns, in order."""
    return [("index", _index_child(p)), ("qa", _qa_child(p))]


def _require_manifest(p: dict) -> None:
    if not p["manifest"].exists():
        raise SystemExit(f"no manifest at {p['manifest']} -- run "
                         f"`python {SCRIPT.name} --audio {p['audio']}` "
                         f"(or --stage manifest) first")


def _run_child(label: str, argv: list[str]) -> None:
    print(f"\n== stage {label}\n$ {' '.join(shlex.quote(a) for a in argv)}", flush=True)
    rc = subprocess.call(argv)
    if rc != 0:
        raise SystemExit(f"stage {label} failed (exit {rc}) -- logs are under the workdir")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    mine, rest = build_parser().parse_known_args(argv)

    audio = Path(mine.audio).expanduser().resolve()
    workdir = Path(mine.workdir).expanduser().resolve() if mine.workdir \
        else Path.cwd() / DEFAULT_WORKDIR / audio.stem
    questions = _collect_questions(mine)

    p = plan(mine, audio, workdir, rest, questions)
    args = p["args"]

    if mine.plan:
        _print_plan(p, questions)
        return 0

    if mine.stage in _INDEX_SUBSTAGE or mine.stage == "qa":
        _require_manifest(p)

    k_note = (f", cluster k={args.cluster_k}"
              + (f" (capped from {PAPER_CLUSTER_K})" if p["capped"] else ""))
    print(f"== {audio.name}: {p['dur']:.1f}s, {p['n_units']} unit(s) of "
          f"{args.unit_sec:g}s{k_note}")
    print(f"   workdir   {workdir}")
    print(f"   manifest  {p['manifest']}")
    print(f"   results   {p['out_dir']}")

    # --- manifest (cheap, no model, always in this process) ----------------
    if mine.stage in ("all", "manifest"):
        from audiohelix.build import run_manifest
        summary = run_manifest(p["bench"], args)
        if mine.stage == "manifest":
            _print_summary(summary)
            return 0

    # --- the stage chain ---------------------------------------------------
    if mine.stage == "all":
        # One process per model-holding stage, so the captioning model and the
        # QA backbone are never resident at the same time.
        for label, child in _steps(p):
            _run_child(label, child)
        _report(p["out_dir"])
        return 0

    if mine.stage in _INDEX_SUBSTAGE:
        from audiohelix.build import run_index
        args.stage = _INDEX_SUBSTAGE[mine.stage]   # what run_index dispatches on
        _print_summary(run_index(p["bench"], args))
        return 0

    from audiohelix.qa import run_qa
    summary = run_qa(p["bench"], args)
    _print_summary(summary)
    _report(p["out_dir"], summary.get("results") if isinstance(summary, dict) else None)
    return 0


def _print_summary(summary) -> None:
    """The stage's return value, as `python -m audiohelix.run` prints it."""
    print(json.dumps(summary if isinstance(summary, (dict, list)) else {"result": str(summary)},
                     ensure_ascii=False, indent=2, default=str))


def _print_plan(p: dict, questions: list[str]) -> None:
    args = p["args"]
    print(f"audio      {p['audio']} ({p['dur']:.1f}s, {p['n_units']} unit(s) of "
          f"{args.unit_sec:g}s)")
    print(f"questions  {len(questions)}")
    print(f"workdir    {p['workdir']}")
    print(f"manifest   {p['manifest']}")
    print(f"index      model={args.index_model} chunk={args.chunk_sec:g}s "
          f"cluster={p['cluster_tag']}"
          + (" (k capped to the recording's unit count)" if p["capped"] else ""))
    print(f"qa         model={args.model} top-k={args.retrieval_top_k} "
          f"audio-top-m={args.audio_top_m} pe-llm={args.pos_emb_llm}")
    print(f"results    {p['out_dir']}")
    if p["stage"] in ("all", "manifest"):
        print(f"\n[this process] manifest -> {p['manifest']}")
    if p["stage"] == "all":
        for label, child in _steps(p):
            print(f"\n$ {' '.join(shlex.quote(a) for a in child)}")
    elif p["stage"] in _INDEX_SUBSTAGE:
        print(f"\n[this process] index ({_INDEX_SUBSTAGE[p['stage']]})")
    elif p["stage"] == "qa":
        print("\n[this process] qa")


def _report(out_dir: Path, results_path: str | None = None) -> None:
    """The answers, after the QA stage -- what the run was for."""
    # The QA stage saves under a per-run timestamp (`audiohelix/qa.py` appends
    # one unless `--out` was given), so the flat path below is the `--out`
    # case; otherwise take the path the stage reported, or the newest one it
    # wrote (`--stage all`, where the QA stage ran in a child process).
    out_path = Path(results_path) if results_path else Path(out_dir) / "results.json"
    if not out_path.exists():
        found = sorted(Path(out_dir).glob("*/results.json"))
        if found:
            out_path = found[-1]
    if not out_path.exists():
        print(f"\nno results at {out_path}")
        return
    rows = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"\n== {len(rows)} answer(s) -- {out_dir}")
    for r in rows:
        print(f"\nQ: {r['question']}\nA: {r['answer']}")
    print(f"\nfull rows (retrieved chunks, prompt, timings): {out_path}")


if __name__ == "__main__":
    sys.exit(main())
