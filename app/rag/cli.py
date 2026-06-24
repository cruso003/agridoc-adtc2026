"""Command-line interface.

    python -m rag index
    python -m rag query "why are my chickens coughing?" --server http://127.0.0.1:8080
    python -m rag bench --prompts eval_prompts.jsonl --models smollm2-360m-q4,llama-3.2-1b-q4,qwen2.5-1.5b-q4
"""
from __future__ import annotations

import argparse
import sys

from . import config

# Windows consoles default to cp1252; corpus + model output is UTF-8. Force it so
# accented sources ("Wikipedia — Maize") and any unicode in answers never crash.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def _cmd_index(args) -> int:
    from .index import build
    build(chunks_path=args.chunks, index_dir=args.index_dir, embed_model=args.embed_model)
    return 0


def _cmd_query(args) -> int:
    from . import llm
    from .index import LoadedIndex
    from .embedder import Embedder
    from .retriever import Retriever
    from .pipeline import Pipeline

    if not llm.health(args.server):
        print(f"ERROR: no healthy llama-server at {args.server} "
              f"(start one with a candidate GGUF first)", file=sys.stderr)
        return 2

    index = LoadedIndex(args.index_dir)
    pipeline = Pipeline(Retriever(index, Embedder(index.embed_model)))
    result = pipeline.answer(args.question, server=args.server, top_k=args.top_k)

    if args.json:
        import json
        print(json.dumps({
            "query": result.query,
            "answer": result.answer,
            "citations": result.citations,
        }, indent=2, ensure_ascii=False))
    else:
        print(result.format_cli())
    return 0


def _cmd_serve(args) -> int:
    import os
    os.environ["AGRIDOC_PORT"] = str(args.port)
    os.environ["RAG_LLM_SERVER"] = args.server
    os.environ["RAG_INDEX_DIR"] = args.index_dir
    from .app import main as serve_main
    serve_main()
    return 0


def _cmd_bench(args) -> int:
    from .bench import run_bench
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("ERROR: --models is empty", file=sys.stderr)
        return 2
    run_bench(
        prompts_path=args.prompts,
        models=models,
        index_dir=args.index_dir,
        submissions=args.submissions,
        out_md=args.out,
        launcher=args.launcher,
        port=args.port,
        image=args.image,
        server_bin=args.server_bin,
        top_k=args.top_k,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag", description="SaharaSprout offline RAG pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    # index
    pi = sub.add_parser("index", help="(re)build the hybrid index from chunks.jsonl")
    pi.add_argument("--chunks", default=str(config.DEFAULT_CHUNKS))
    pi.add_argument("--index-dir", default=str(config.DEFAULT_INDEX_DIR))
    pi.add_argument("--embed-model", default=config.EMBED_MODEL)
    pi.set_defaults(func=_cmd_index)

    # query
    pq = sub.add_parser("query", help="grounded answer + citations for one question")
    pq.add_argument("question")
    pq.add_argument("--server", default=config.DEFAULT_SERVER)
    pq.add_argument("--index-dir", default=str(config.DEFAULT_INDEX_DIR))
    pq.add_argument("--top-k", type=int, default=config.TOP_K)
    pq.add_argument("--json", action="store_true", help="emit JSON instead of text")
    pq.set_defaults(func=_cmd_query)

    # serve (AgriDoc app API)
    ps = sub.add_parser("serve", help="run the AgriDoc /ask API for the UI")
    ps.add_argument("--port", type=int, default=8000)
    ps.add_argument("--server", default=config.DEFAULT_SERVER, help="llama-server URL (the GGUF)")
    ps.add_argument("--index-dir", default=str(config.DEFAULT_INDEX_DIR))
    ps.set_defaults(func=_cmd_serve)

    # bench
    pb = sub.add_parser("bench", help="base-model tiebreaker across candidate GGUFs")
    pb.add_argument("--prompts", default=str(config.REPO_ROOT / "eval_prompts.jsonl"))
    pb.add_argument("--models", required=True,
                    help="comma-separated model keys (submissions/<key>/model/*.gguf)")
    pb.add_argument("--index-dir", default=str(config.DEFAULT_INDEX_DIR))
    pb.add_argument("--submissions", default=str(config.DEFAULT_SUBMISSIONS))
    pb.add_argument("--out", default=str(config.REPO_ROOT / "base_tiebreaker.md"))
    pb.add_argument("--launcher", choices=["docker", "exec"], default="docker")
    pb.add_argument("--port", type=int, default=8080)
    pb.add_argument("--image", default="adtc-profiler:latest",
                    help="docker image whose entrypoint is llama-server")
    pb.add_argument("--server-bin", default="llama-server",
                    help="native llama-server binary (launcher=exec)")
    pb.add_argument("--top-k", type=int, default=config.TOP_K)
    pb.set_defaults(func=_cmd_bench)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
