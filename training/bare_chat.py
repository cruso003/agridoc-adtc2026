"""Bare-model chat harness: POST identical domain prompts to a llama-server,
NO system prompt, NO RAG — the way ADTC judges chat the .gguf in LM Studio/Ollama.

Usage:  python bare_chat.py http://127.0.0.1:8080 llama1b [prompts.jsonl]
  3rd arg optional: a JSONL of {"tag","prompt"} rows (e.g. held_out_eval.jsonl).
  Default: the built-in 5 domain prompts below.
"""
import json
import sys
import urllib.request

# Spread across the Agriculture domain: livestock, crop disease, post-harvest,
# market advisory, and a second livestock case the prompt was NOT hand-walked for.
PROMPTS = [
    ("livestock-poultry", "At night my broilers are dying two at a time, I don't know why."),
    ("crop-disease", "My maize leaves have long pale yellow streaks running along them and the "
                     "plants are stunted. What is wrong and what should I do?"),
    ("post-harvest", "How should I store my maize harvest so it doesn't get mouldy or attacked "
                     "by weevils?"),
    ("market-advisory", "Tomato prices at my local market are very low right now. Should I sell "
                        "today or wait, and why?"),
    ("livestock-goat", "My goats are losing weight even though they are eating well. What could "
                       "be causing this?"),
]


def chat(server: str, prompt: str) -> str:
    payload = {
        "model": "local", "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4, "max_tokens": 340, "seed": 7, "stream": False,
        # match what judges' tools (LM Studio / Ollama) apply by default, so behaviour
        # is representative and we don't measure looping the runtime would have suppressed.
        "repeat_penalty": 1.1,
    }
    req = urllib.request.Request(
        server.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=420) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def main() -> None:
    server, label = sys.argv[1], sys.argv[2]
    prompts = PROMPTS
    if len(sys.argv) > 3:
        rows = [json.loads(l) for l in open(sys.argv[3], encoding="utf-8") if l.strip()]
        prompts = [(r["tag"], r["prompt"]) for r in rows]
    out = []
    for tag, p in prompts:
        ans = chat(server, p)
        print(f"\n{'='*74}\n[{label} · {tag}] {p}\n{'-'*74}\n{ans}", flush=True)
        out.append({"tag": tag, "prompt": p, "answer": ans})
    with open(f"bare_chat_{label}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] bare_chat_{label}.json")


if __name__ == "__main__":
    main()
