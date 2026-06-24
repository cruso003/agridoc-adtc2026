#!/usr/bin/env python3
"""Assemble the run-5 SFT mix from the kimi-generated set + seed + our calibration.

Inputs (all already safety-validated):
  cleaned_generated.jsonl   kimi run-2, 1683 diverse grounded examples (category tags)
  kimi_seed_clean.jsonl      kimi run-1 survivors (96)
  calibration_set.jsonl      our hand-written calibration (topic cal:*)
  safety_calibration.jsonl   our original safety set (topic safety:*)

Rebalances the diagnostic-heavy set to ~45%, keeps all behaviour examples, lightly boosts
the weakest behaviour (ask). Dedups by answer AND user text. Out: qwen_sft_mix.jsonl
"""
import json, glob, re, random, collections
random.seed(7)

N_DIAG = 420           # cap diagnostic_commit so it doesn't dominate
UPSAMPLE = {"ask_when_vague": 2, "differential_uncertain": 2}
N_ANCHOR = 40
NUM = re.compile(r"\b\d+(\.\d+)?\s?(ml|l|g|kg|grams?|%|percent)\b", re.I)

# map our calibration topics -> the unified kimi categories
CAL_MAP = {
    "ask": "ask_when_vague", "underspecified": "ask_when_vague",
    "differential": "differential_uncertain", "dose": "dose_refusal",
    "commit": "diagnostic_commit", "market": "decline_live",
    "out_of_domain": "out_of_domain", "out_of_scope": "out_of_domain",
    "escalation": "general_advisory",
}


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def norm(r, cat):
    return {"messages": r["messages"], "category": cat}


pool = []
for r in load("cleaned_generated.jsonl"):
    pool.append(norm(r, r.get("category", "")))
for r in load("kimi_seed_clean.jsonl"):
    pool.append(norm(r, r.get("category", "")))
for r in load("diff_set.jsonl"):
    pool.append(norm(r, r.get("category", "")))
for fn in ("calibration_set.jsonl", "safety_calibration.jsonl"):
    for r in load(fn):
        base = r.get("topic", "").split(":")[-1]
        pool.append(norm(r, CAL_MAP.get(base, "general_advisory")))

# dedup by answer AND user
seen_a, seen_u, uniq = set(), set(), []
for r in pool:
    u = r["messages"][0]["content"].strip().lower()
    a = r["messages"][1]["content"].strip()
    if a in seen_a or u in seen_u:
        continue
    seen_a.add(a); seen_u.add(u)
    uniq.append(r)

by_cat = collections.defaultdict(list)
for r in uniq:
    by_cat[r["category"]].append(r)

mix = []
for cat, items in by_cat.items():
    random.shuffle(items)
    if cat == "diagnostic_commit":
        items = items[:N_DIAG]
    mix += items * UPSAMPLE.get(cat, 1)

# number-free general anchors (forgetting guard)
anchors = []
for r in load("training_mix_v2.jsonl"):
    m = r["messages"]
    a = next((x["content"] for x in m if x["role"] == "assistant"), "")
    u = next((x["content"] for x in m if x["role"] == "user"), "")
    if [x["role"] for x in m] == ["user", "assistant"] and 80 <= len(a) <= 900 and not NUM.search(a) and "?" in u:
        anchors.append({"messages": m, "category": "anchor"})
random.shuffle(anchors)
mix += anchors[:N_ANCHOR]
random.shuffle(mix)

with open("qwen_sft_mix.jsonl", "w", encoding="utf-8") as f:
    for r in mix:
        f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")

print(f"unique pool: {len(uniq)}  (kimi + seed + calibration, deduped)")
print("final mix by category:")
fc = collections.Counter(r["category"] for r in mix)
for k, v in sorted(fc.items(), key=lambda x: -x[1]):
    print(f"  {k:<22} {v}  ({100*v//len(mix)}%)")
print(f"TOTAL run-5 mix: {len(mix)} rows -> qwen_sft_mix.jsonl")
