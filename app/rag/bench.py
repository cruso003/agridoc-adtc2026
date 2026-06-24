"""Base-model tiebreaker runner.

For each candidate GGUF: start a llama-server, run ALL eval prompts through the
SAME retrieval, collect grounded answers, stop the server, move to the next.
Retrieval is model-independent, so we retrieve ONCE per prompt and feed every
model byte-identical context — the only variable is the model.

Emits base_tiebreaker.md (a side-by-side table: prompt | model | answer |
good_answer criterion) so a human can score grounding, abstention, and the EP_10
safety case across models in one view. Also writes bench_results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config, prompt as promptmod
from .index import LoadedIndex
from .embedder import Embedder
from .retriever import Retriever
from .server import LlamaServer
from . import llm


def load_prompts(path: str | Path) -> list[dict]:
    prompts = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def _md_cell(text: str) -> str:
    """Make text safe for a single Markdown table cell."""
    return (text or "").replace("|", "\\|").replace("\r", " ").replace("\n", "<br>").strip()


def run_bench(
    prompts_path: str | Path,
    models: list[str],
    index_dir: str | Path = config.DEFAULT_INDEX_DIR,
    submissions: str | Path = config.DEFAULT_SUBMISSIONS,
    out_md: str | Path = config.REPO_ROOT / "base_tiebreaker.md",
    out_json: str | Path | None = None,
    launcher: str = "docker",
    port: int = 8080,
    image: str = "adtc-profiler:latest",
    server_bin: str = "llama-server",
    top_k: int = config.TOP_K,
) -> Path:
    prompts = load_prompts(prompts_path)
    print(f"[bench] {len(prompts)} prompts, {len(models)} models: {', '.join(models)}")

    # ── Shared retrieval: build messages once per prompt ──
    index = LoadedIndex(index_dir)
    retriever = Retriever(index, Embedder(index.embed_model))
    print(f"[bench] index: {len(index)} chunks, embed={index.embed_model}")

    retrieval: dict[str, dict] = {}
    for p in prompts:
        chunks = retriever.retrieve(p["prompt"], top_k=top_k)
        retrieval[p["id"]] = {
            "chunks": chunks,
            "messages": promptmod.build_messages(p["prompt"], chunks),
            "citations": [
                {"chunk_id": c["chunk_id"], "source": c["source"], "license": c["license"]}
                for c in chunks
            ],
        }

    # ── Per-model generation ──
    # answers[prompt_id][model_key] = answer text
    answers: dict[str, dict[str, str]] = {p["id"]: {} for p in prompts}
    for model_key in models:
        print(f"\n[bench] === model: {model_key} ===")
        try:
            with LlamaServer(model_key, launcher=launcher, port=port,
                             submissions=submissions, image=image,
                             server_bin=server_bin) as srv:
                for p in prompts:
                    msgs = retrieval[p["id"]]["messages"]
                    try:
                        ans = llm.chat(msgs, server=srv.base_url)
                    except Exception as e:  # one bad gen shouldn't sink the model
                        ans = f"[ERROR: {e}]"
                    answers[p["id"]][model_key] = ans
                    print(f"  [{model_key}] {p['id']} done ({len(ans)} chars)")
        except Exception as e:
            print(f"[bench] model {model_key} failed: {e}")
            for p in prompts:
                answers[p["id"]].setdefault(model_key, f"[MODEL UNAVAILABLE: {e}]")

    # ── Emit markdown ──
    out_md = Path(out_md)
    md = _render_markdown(prompts, models, answers, retrieval, index)
    out_md.write_text(md, encoding="utf-8")
    print(f"\n[bench] wrote {out_md}")

    # ── Emit JSON (full fidelity for re-scoring) ──
    if out_json is None:
        out_json = out_md.with_suffix(".json")
    out_json = Path(out_json)
    payload = {
        "models": models,
        "index": index.manifest,
        "prompts": [
            {
                "id": p["id"],
                "type": p.get("type"),
                "prompt": p["prompt"],
                "good_answer": p.get("good_answer"),
                "retrieved": [
                    {k: c[k] for k in ("chunk_id", "source", "license", "topic")}
                    for c in retrieval[p["id"]]["chunks"]
                ],
                "answers": answers[p["id"]],
            }
            for p in prompts
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[bench] wrote {out_json}")
    return out_md


def _render_markdown(prompts, models, answers, retrieval, index) -> str:
    lines: list[str] = []
    lines.append("# Base-Model Tiebreaker — SaharaSprout RAG")
    lines.append("")
    lines.append(
        "Every model below answered through the **same hybrid retrieval** "
        f"(FAISS + BM25, RRF k={config.RRF_K}, top-{config.TOP_K}) over "
        f"`{index.manifest['n_chunks']}` chunks, embedded with "
        f"`{index.manifest['embed_model']}`. The only variable across columns is "
        "the model. Score each cell for **grounding** (uses the retrieved "
        "context, invents nothing), **abstention** (EP_08, EP_09, EP_13), and the "
        "**EP_10 safety case** (must refuse to invent a dose)."
    )
    lines.append("")
    lines.append(f"Models: {', '.join(f'`{m}`' for m in models)}")
    lines.append("")

    for p in prompts:
        pid = p["id"]
        lines.append(f"## {pid} — {p.get('type', '?')}")
        lines.append("")
        lines.append(f"**Prompt:** {p['prompt']}")
        lines.append("")
        lines.append(f"**Good-answer criterion:** {p.get('good_answer', '—')}")
        lines.append("")
        cites = retrieval[pid]["citations"]
        cite_str = ", ".join(f"{c['chunk_id']} ({c['source']})" for c in cites)
        lines.append(f"**Retrieved context (shared):** {cite_str}")
        lines.append("")
        lines.append("| Model | Answer |")
        lines.append("| --- | --- |")
        for m in models:
            lines.append(f"| `{m}` | {_md_cell(answers[pid].get(m, ''))} |")
        lines.append("")

    return "\n".join(lines)
