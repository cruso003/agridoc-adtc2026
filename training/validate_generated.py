#!/usr/bin/env python3
"""Curate kimi-generated SFT data: auto-reject unsafe/broken examples before training.

The whole point of the new data is to teach refusal/calibration — so any generated example
that itself emits a dose/rate/price, loops, or misses its category's required behaviour is
REJECTED here (never reaches the model). Survivors go to cleaned_generated.jsonl; rejects go
to rejected_generated.jsonl with a reason for human review.

Usage:  python validate_generated.py kimi_out.jsonl [more.jsonl ...]
"""
import json
import re
import sys
import collections

DOSE_NUM = re.compile(
    r"\b\d+(\.\d+)?\s?(ml|l|litres?|liters?|g|kg|grams?|kilograms?|tablets?|sachets?|cups?|"
    r"teaspoons?|tablespoons?|cc|ppm)\b"
    r"|\b\d+(\.\d+)?\s?(kg|g|l|ml)\s?(/|per)\s?(ha|acre|bag|plant|kg|litre|liter|bird)\b"
    r"|\b\d+(\.\d+)?\s?(g|kg|ml|l)\b.{0,15}\bper\s+(plant|bird|bag|acre|hectare|ha)\b"
    r"|\b\d+\s?[-–]\s?\d+\s?(kg|g|ml|l|%)\b", re.I)
PRICE_NUM = re.compile(r"\b(ksh|tsh|naira|ngn|usd|\$|€|£|cedis?|birr|rand)\s?\d|\b\d+\s?(ksh|tsh|naira|usd|dollars?|shillings?)\b", re.I)
REFUSE = re.compile(
    r"\b(can'?t|cannot|won'?t|will not|unable to)\b.{0,25}\b(give|provide|tell|recommend|put|state|name|specify|quote|advise)"
    r"|follow the (product )?label|label rate|read the label|label('s| ?s)? (rate|instruction)"
    r"|ask (a|an|your) (local )?(vet|extension|animal[- ]health|agro|specialist)"
    r"|(see|consult) (a|an|your) (vet|veterinarian|extension|specialist|officer|expert)"
    r"|not something i can|outside (what i|my scope|my area)|i work offline", re.I)
VET = re.compile(r"\bvet(erinarian)?\b|animal[- ]health worker", re.I)
# any competent redirect for out-of-domain (not just a vet — e.g. fisheries/aquaculture)
REDIRECT = re.compile(r"\bvet(erinarian)?\b|animal[- ]health|(consult|see|ask) (a|an|your).{0,30}(specialist|officer|expert|extension)|specialist|fisheries|aquaculture|outside (my|what)", re.I)
DECLINE = re.compile(r"don'?t have (live|current|today)|no (live|current) (data|prices?)|run offline|offline|can'?t see (today|current|live)|check (with )?(your )?(local )?(market|trader|weather|met|extension)", re.I)
# a request for more info — a "?" OR an imperative clarification ("please tell me…")
ASK = re.compile(r"\?|tell me|let me know|i need (more|to know|a clearer|some)|more (details|information|specifics)|can you (tell|describe|share|let)|could you (tell|describe|share)|which (crop|variety|animal|part|bird)|what (crop|exactly|symptoms?|type|part|colour|color)", re.I)
CATS = {"diagnostic_commit", "differential_uncertain", "ask_when_vague", "dose_refusal",
        "out_of_domain", "decline_live", "general_advisory"}


def looping(text):
    w = re.findall(r"\w+", text.lower())
    if len(w) < 8:
        return False
    run = 1
    for a, b in zip(w, w[1:]):
        run = run + 1 if a == b else 1
        if run >= 4:
            return True
    for n in range(3, 15):   # sentence-level immediate repeats too
        for i in range(len(w) - n * 2 + 1):
            if w[i:i+n] == w[i+n:i+2*n]:
                return True
    return len(w) > 25 and len(set(w)) / len(w) < 0.35   # collapsed vocabulary (loops ~0.2)


def check(ex):
    """Return reason string if the example must be REJECTED, else None."""
    m = ex.get("messages", [])
    if [x.get("role") for x in m] != ["user", "assistant"] or not all(x.get("content", "").strip() for x in m):
        return "bad format"
    u, a = m[0]["content"].strip(), m[1]["content"].strip()
    cat = ex.get("category", "")
    if cat not in CATS:
        return f"unknown category {cat!r}"
    if DOSE_NUM.search(a):
        return "emits a dose/rate number"          # hard reject — re-poisons refusal
    if PRICE_NUM.search(a):
        return "emits a price"
    if looping(a):
        return "looping/repetition"
    if not (120 <= len(a) <= 1500):
        return f"length {len(a)}"
    # category-specific behaviour the example MUST demonstrate
    if cat == "dose_refusal" and not REFUSE.search(a):
        return "dose_refusal without a refusal"
    if cat == "ask_when_vague" and not ASK.search(a):
        return "ask_when_vague without a clarification request"
    if cat == "out_of_domain" and not (REDIRECT.search(a) or REFUSE.search(a)):
        return "out_of_domain without redirect"
    if cat == "decline_live" and not DECLINE.search(a):
        return "decline_live without declining"
    return None


def main():
    rows = []
    for f in sys.argv[1:]:
        for l in open(f, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    kept, rejected, seen = [], [], set()
    cat_kept = collections.Counter()
    rej_reason = collections.Counter()
    for ex in rows:
        r = check(ex)
        if r:
            rejected.append({**ex, "_reject": r}); rej_reason[r.split(" ")[0] if "format" not in r else r] += 1
            continue
        key = ex["messages"][0]["content"].strip().lower()
        if key in seen:
            rejected.append({**ex, "_reject": "duplicate"}); rej_reason["duplicate"] += 1
            continue
        seen.add(key)
        kept.append({"messages": ex["messages"], "category": ex.get("category", ""), "topic": ex.get("topic", "")})
        cat_kept[ex.get("category", "")] += 1
    with open("cleaned_generated.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open("rejected_generated.jsonl", "w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(rows)
    print(f"in: {n} | KEPT: {len(kept)} ({100*len(kept)//max(n,1)}%) | REJECTED: {len(rejected)}")
    print("kept by category:", dict(cat_kept))
    print("reject reasons:", dict(rej_reason))
    print("→ cleaned_generated.jsonl (train) | rejected_generated.jsonl (review)")


if __name__ == "__main__":
    main()
