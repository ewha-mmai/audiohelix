from __future__ import annotations

import argparse
import json
import sys

from audiohelix.config import INDEX_MODEL


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bench", required=True,
                   choices=["traceav", "audiomarathon", "mmau_pro"])
    p.add_argument("--chunk-sec", type=float, default=None,
                   help="chunk length in seconds (default: the bench's own)")
    p.add_argument("--min-last-chunk-sec", type=float, default=None,
                   help="drop a trailing fragment shorter than this")
    p.add_argument("--unit-sec", type=float, default=None,
                   help="audio-unit length in seconds (default: the bench's own)")
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N items/questions")
    p.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES for this run")


def _add_cluster(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cluster-k", type=int, default=None,
                   help="n_clusters for the audio-unit clustering "
                        "(default: the bench's own)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="audiohelix.run")
    sub = ap.add_subparsers(dest="command", required=True)

    m = sub.add_parser("manifest", help="build the benchmark's question/item file")
    _add_common(m)

    i = sub.add_parser("index", help="cut chunks, caption them, build the graph")
    _add_common(i)
    _add_cluster(i)
    i.add_argument("--stage", default="all", choices=["all", "captions", "graph"],
                   help="run only one part of the build")
    i.add_argument("--suite", default=None,
                   help="only this suite (the manifest stage's item suites)")
    i.add_argument("--force", action="store_true",
                   help="redo chunks that already have captions")

    q = sub.add_parser("qa", help="retrieve, ask, and score")
    _add_common(q)
    _add_cluster(q)
    q.add_argument("--model", required=True,
                   choices=["qwen3_omni", "voxtral", "audio_flamingo3"],
                   help="the QA backbone")
    q.add_argument("--suite", default=None,
                   help="only this suite (default: every suite)")
    q.add_argument("--only-multi", action="store_true",
                   help="only questions whose audio spans several clips")
    q.add_argument("--retrieval-top-k", type=int, default=None,
                   help="chunks to retrieve (default setting)")
    q.add_argument("--audio-top-m", type=int, default=None,
                   help="audio units picked per retrieved chunk")
    q.add_argument("--seed-top-k-text", type=int, default=None,
                   help="text concepts seeding the retrieval walk (default: 1)")
    q.add_argument("--seed-top-k-audio", type=int, default=None,
                   help="audio concepts seeding the retrieval walk (default: 1)")
    q.add_argument("--nearest-centroids", type=int, default=None,
                   help="nearest centroids linked per audio unit")
    q.add_argument("--centroid-sim-threshold", type=float, default=None,
                   help="cosine-similarity floor for a centroid link")
    q.add_argument("--concat-audio", action="store_true",
                   help="feed one concatenated audio input instead of one per segment")
    q.add_argument("--pos-emb-time", default=None, choices=["on", "off"],
                   help="time-aligned positional embedding in the encoder")
    q.add_argument("--pos-emb-llm", default=None, choices=["on", "off"],
                   help="time-aligned positional embedding in the LLM")
    q.add_argument("--pos-emb-fps", type=float, default=None,
                   help="encoder positional-embedding frame rate")
    q.add_argument("--evidence-framing", action="store_true",
                   help="allow the model to answer 'insufficient evidence'")
    q.add_argument("--out", default=None,
                   help="write results here instead of the timestamped default dir")
    q.add_argument("--force", action="store_true",
                   help="redo questions that already have results")
    q.add_argument("--skip", type=int, default=0, help="skip the first N questions")
    q.add_argument("--shard", default=None, metavar="I/N",
                   help="process only shard I of N questions (parallel runners; "
                        "MMAU-Pro only, for sharded evaluation)")
    q.add_argument("--dry-run", action="store_true",
                   help="assemble and dump the conversations without loading the QA model")
    return ap


def resolve_defaults(bench, args) -> None:
    """Fills unset flags from the benchmark's own config.

    Only flags the subcommand actually defines are filled -- `manifest` has no
    --model, so it stays unset instead of being invented."""
    cfg = bench.config

    def fill(name: str, value) -> None:
        if hasattr(args, name) and getattr(args, name, None) is None:
            setattr(args, name, value)

    fill("chunk_sec", cfg.CHUNK_SEC)
    fill("min_last_chunk_sec", cfg.MIN_LAST_CHUNK_SEC)
    fill("unit_sec", cfg.UNIT_SEC)
    fill("cluster_k", cfg.CLUSTER.spectral_n_clusters)
    # Which index to read is not a choice either: the same fixed caption model
    # owns every graph, so this is a constant, not a flag.
    args.index_model = INDEX_MODEL
    fill("retrieval_top_k", getattr(cfg, "RETRIEVAL_TOP_K_DEFAULT", 15))
    fill("audio_top_m", getattr(cfg, "AUDIO_TOP_M_DEFAULT", 2))
    fill("seed_top_k_text", getattr(cfg, "SEED_TOP_K_TEXT", 1))
    fill("seed_top_k_audio", getattr(cfg, "SEED_TOP_K_AUDIO", 1))
    fill("nearest_centroids", getattr(cfg, "NEAREST_CENTROIDS", 3))
    fill("centroid_sim_threshold", getattr(cfg, "CENTROID_SIM_THRESHOLD", 0.85))
    pe_mode = getattr(cfg, "PE_MODE", "llm_local_timemaps")
    fill("pos_emb_llm", getattr(cfg, "POS_EMB_LLM_DEFAULT",
                                "on" if pe_mode == "llm_local_timemaps" else "off"))
    fill("pos_emb_time", getattr(cfg, "POS_EMB_TIME_DEFAULT", "off"))


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    from eval import get_bench
    bench = get_bench(args.bench)
    resolve_defaults(bench, args)

    if args.command == "manifest":
        from audiohelix.build import run_manifest
        out = run_manifest(bench, args)
    elif args.command == "index":
        from audiohelix.build import run_index
        out = run_index(bench, args)
    else:
        from audiohelix.qa import run_qa
        out = run_qa(bench, args)

    summary = out if isinstance(out, (dict, list)) else {"result": str(out)}
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
