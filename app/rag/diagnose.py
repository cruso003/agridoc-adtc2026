"""Reasoning-first diagnosis for the AgriDoc workbench.

Design (rebuilt after DR-0018): the model REASONS like a field extension officer
— it weighs the specific pattern the farmer describes (timing, count, age, where,
recent change), forms a differential, and proposes the single most useful question
to narrow it down. Retrieval is REFERENCE, not a cage: only genuinely-relevant
library notes are offered, and the model is told to ignore them when they don't fit.

What stays deterministic / honest:
  - low-relevance chunks never reach the prompt (REF_FLOOR) — no more mulch noise;
  - a cause is labelled "from your library" ONLY when a retrieved note actually
    supports that named cause (term overlap), not merely because the query matched
    something — so the label can't overstate grounding;
  - the model may not invent a dose, rate, or price (enforced by the prompt).

The whole thing streams, so the UI can show the reasoning as it is written.
"""
from __future__ import annotations

import re

from . import config, llm
from .retriever import Retriever

# Only chunks at/above this cosine count as a usable reference (drops 0.0 noise).
REF_FLOOR = 0.35
# A named cause is "library-backed" if a reference note shares this fraction of the
# cause's significant words.
SUPPORT_OVERLAP = 0.5

DX_SYSTEM = (
    "You are an experienced agricultural and veterinary extension officer in East and "
    "West Africa, helping a farmer offline. Diagnose like a good field officer: reason "
    "from the SPECIFIC PATTERN the farmer describes — the timing (day or night), how "
    "many are affected and how fast, the age or growth stage, where on the plant or "
    "body, and any recent change in weather, feed, or management. Those patterns are "
    "your strongest clues.\n\n"
    "Weigh the FULL range of causes, not just disease: management and environment (cold, "
    "heat, drafts, crowding or birds piling at night, poor ventilation), predators, feed "
    "or water problems, poisoning, and injury — as well as pests, infections, and "
    "nutrient problems. Match the cause to the pattern; do NOT default to 'a virus' or "
    "'an infection' unless the pattern really points there. Prefer practical management "
    "and observation steps; for any medicine, advise seeing a vet rather than naming a "
    "drug.\n\n"
    "Think it through first, then give a short structured assessment of up to three "
    "causes, most likely first. Reason from the farmer's specific pattern using your own "
    "knowledge. Do NOT assume a common disease just because the words sound similar — a "
    "bird dying is not automatically a respiratory infection; match the cause to the "
    "actual pattern (timing, count, age, what is seen).\n\n"
    "Rules:\n"
    "- Never invent a pesticide or drug dose, rate, or price. For those, say to follow "
    "the product label and ask a local vet or extension officer.\n"
    "- It is good to ask the farmer for one key observation rather than guessing.\n\n"
    "Reply in EXACTLY this format and nothing else:\n"
    "THINKING: <3-5 sentences: what the pattern suggests, the causes it could be, "
    "which fit best and which you can rule out, and the one observation that would tell "
    "them apart>\n"
    "CAUSE: <the single most likely cause, named>\n"
    "WHY: <why it fits THIS farmer's pattern, 1-2 sentences>\n"
    "DO: <action or check 1>; <action or check 2>; <action or check 3>\n"
    "ALSO: <second cause> — <why>; <third cause> — <why>   (write 'none' if nothing "
    "else fits)\n"
    "ASK: <the single observation or question that would best narrow it down>"
)

# A *completed prior case*, supplied as its own conversation turn (NOT inline in the
# system prompt) — a 1B copies an inline example verbatim, but treats a prior turn as
# done and answers the new question fresh. The example also models reasoning-from-pattern.
DX_EXAMPLE = [
    {"role": "user", "content":
        "FARMER'S PROBLEM:\nMy maize is going yellow — the oldest bottom leaves first, "
        "now creeping up the plant, and I see no insects on it."},
    {"role": "assistant", "content":
        "THINKING: Yellowing that starts on the oldest lower leaves and creeps upward, "
        "with no pests in sight, usually means the plant is moving a mobile nutrient up "
        "to the new growth — classic nitrogen shortage, common on sandy soil or after "
        "heavy rain leaches it out. If the youngest top leaves were worst, or there were "
        "spots, I would think differently. The quickest check is which leaves lead.\n"
        "CAUSE: Nitrogen deficiency\n"
        "WHY: Yellowing that begins on the older lower leaves and moves upward fits "
        "nitrogen being remobilised to the younger leaves.\n"
        "DO: Confirm the oldest leaves are worst affected; top-dress with a nitrogen "
        "source at the label rate; avoid waterlogging that leaches nitrogen\n"
        "ALSO: Sulphur deficiency — similar yellowing but worst on the young top leaves; "
        "Waterlogging stress — roots starved in saturated soil\n"
        "ASK: Is the yellowing worst on the oldest bottom leaves, or on the new top leaves?"},
]

_STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "to", "by", "with", "for",
         "is", "are", "from", "at", "as", "that", "this", "disease", "virus", "stress",
         "deficiency", "infection", "problem", "cause", "issue", "syndrome"}


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", (text or "").lower()) if w not in _STOP}


def _origin_title(source: str, topic: str) -> tuple[str, str]:
    s = source or ""
    if "synthesized advisory" in s.lower():
        return "AgriDoc knowledge base", "Advisory note"
    if " — " in s:
        o, t = (p.strip() for p in s.split(" — ", 1))
        return o, t
    if s.startswith("http"):
        return "Web source", topic or s
    return s, topic or s


def _source_view(c: dict) -> dict:
    origin, title = _origin_title(c.get("source", ""), c.get("topic", ""))
    text = re.sub(r"\s+", " ", c.get("text", "")).strip()
    return {
        "id": c["chunk_id"], "origin": origin, "title": title,
        "snippet": text[:220] + ("…" if len(text) > 220 else ""),
        "match": c["_dense_score"], "topic": c.get("topic", ""),
        "_terms": _terms(c.get("text", "")),
    }


def _supporting(cause: str, sources: list[dict]) -> list[dict]:
    """Sources whose text actually supports the *named cause* (not just the query)."""
    ct = _terms(cause)
    if not ct:
        return []
    hits = []
    for s in sources:
        overlap = len(ct & s["_terms"]) / len(ct)
        if overlap >= SUPPORT_OVERLAP:
            hits.append(s)
    return hits


# reconcile: only surface a library candidate when retrieval is CONFIDENT (not just
# topically related) and it names a diagnosis the model missed.
STRONG_REF = 0.75
DIAG_KW = re.compile(
    r"\b(rust|blight|wilt|rot|mildew|mosaic|virus|disease|deficiency|mould|mold|bacterial|"
    r"fungal|smut|anthracnose|streak|spot|canker|scald|mite|weevil|borer|aphid|nematode|"
    r"worm|thrip|whitefl|leafhopper|caterpillar|armyworm|coccidio|newcastle|bronchitis|"
    r"cholera|blast|damping)\b", re.I)


def _subject(snippet: str) -> str:
    """Leading subject of a reference note, e.g. 'Southern rust', 'Cassava mosaic disease'."""
    m = re.match(r"\s*([A-Z][A-Za-z'\-]+(?: [A-Za-z'\-]+){0,4}?)(?=,| is | are | \(|caused by| affects)", snippet)
    return (m.group(1).strip() if m else snippet.split(",")[0][:48].strip())


def _parse(text: str) -> dict:
    out = {"thinking": "", "cause": "", "why": "", "do": [], "also": [], "ask": ""}
    for key, pat in (("thinking", r"THINKING:\s*(.+?)(?=\n[A-Z]{2,}:|\Z)"),
                     ("cause", r"CAUSE:\s*(.+)"), ("why", r"WHY:\s*(.+)"),
                     ("ask", r"ASK:\s*(.+)")):
        m = re.search(pat, text, re.I | re.S if key == "thinking" else re.I)
        if m:
            out[key] = m.group(1).strip()
    m = re.search(r"DO:\s*(.+?)(?:\n[A-Z]{2,}:|\Z)", text, re.I | re.S)
    if m:
        items = re.split(r";|\n\s*\d+[.)]\s*|\n[-*]\s*", m.group(1))
        out["do"] = [re.sub(r"^\d+[.)]\s*", "", a).strip() for a in items if a.strip()][:4]
    m = re.search(r"ALSO:\s*(.+?)(?:\n[A-Z]{2,}:|\Z)", text, re.I | re.S)
    if m and m.group(1).strip().lower() not in ("none", "n/a", "-", ""):
        for part in re.split(r";|\n", m.group(1)):
            part = part.strip(" .-")
            if not part:
                continue
            if " — " in part:
                name, why = part.split(" — ", 1)
            elif " - " in part:
                name, why = part.split(" - ", 1)
            else:
                name, why = part, ""
            out["also"].append({"name": name.strip(), "why": why.strip()})
        out["also"] = out["also"][:2]
    return out


def _build(query: str, raw: str, sources: list[dict]) -> dict:
    p = _parse(raw)
    public = [{k: v for k, v in s.items() if k != "_terms"} for s in sources]

    causes = []
    primary_support = _supporting(p["cause"], sources) if p["cause"] else []
    causes.append({
        "name": p["cause"] or "Not clear yet — needs a closer look",
        "rank": "most_likely",
        "grounding": "library" if primary_support else "reasoning",
        "explanation": p["why"],
        "actions": p["do"],
        "sources": [{k: v for k, v in s.items() if k != "_terms"} for s in primary_support[:3]],
    })
    for alt in p["also"]:
        sup = _supporting(alt["name"], sources)
        causes.append({
            "name": alt["name"], "rank": "also_consider",
            "grounding": "library" if sup else "reasoning",
            "explanation": alt["why"] or "Worth ruling out — verify before acting.",
            "actions": [],
            "sources": [{k: v for k, v in s.items() if k != "_terms"} for s in sup[:2]],
        })

    # RECONCILE: if a CONFIDENT diagnostic note names a DIFFERENT diagnosis than the model's
    # causes, surface it — even when the model's (wrong) cause is itself in the library — so the
    # references can correct a confident miss (model "anthracnose"/"sooty mould" vs notes "rust").
    # Conservative gate: match >= STRONG_REF + a diagnosis keyword + a subject not already named.
    strong = [s for s in sources if s.get("match", 0) >= STRONG_REF and DIAG_KW.search(s.get("snippet", ""))]
    if strong:
        subj = _subject(strong[0]["snippet"])
        named = any(_terms(subj) & _terms(c["name"]) for c in causes)
        if subj and not named:
            causes.append({
                "name": subj,
                "rank": "also_consider",
                "grounding": "library",
                "explanation": "My offline reference notes' closest match for these signs — it "
                               "differs from my answer above, so check which one actually fits.",
                "actions": [],
                "sources": [{k: v for k, v in strong[0].items() if k != "_terms"}],
            })

    return {
        "query": query,
        "thinking": p["thinking"],
        "causes": causes,
        "ask": p["ask"],
        "related": public,        # relevant library notes the officer can open to verify
        "grounded": any(c["grounding"] == "library" for c in causes),
        "raw": raw,
    }


def _ctx(sources: list[dict], chunks: list[dict]) -> str:
    if not sources:
        return "(No closely matching notes in the library — rely on your own knowledge.)"
    keep = {s["id"] for s in sources}
    return "\n\n".join(
        f"[{c['chunk_id']} | {c['source']}]\n{c['text'].strip()}"
        for c in chunks if c["chunk_id"] in keep
    )


def diagnose_stream(retriever: Retriever, query: str, *,
                    server: str = config.DEFAULT_SERVER, top_k: int = config.TOP_K):
    """Generator of ('meta'|'token'|'result', payload). Streams the reasoning live."""
    chunks = retriever.retrieve(query, top_k=top_k)
    refs = [c for c in chunks if c.get("_dense_score", 0.0) >= REF_FLOOR]
    sources = [_source_view(c) for c in refs]
    public_sources = [{k: v for k, v in s.items() if k != "_terms"} for s in sources]
    yield ("meta", {"query": query, "sources": public_sources})

    messages = [
        {"role": "system", "content": DX_SYSTEM},
        *DX_EXAMPLE,
        {"role": "user", "content":
            f"FARMER'S PROBLEM:\n{query}"},
    ]
    buf = []
    for delta in llm.chat_stream(messages, server=server, temperature=0.2, max_tokens=460):
        buf.append(delta)
        yield ("token", delta)
    yield ("result", _build(query, "".join(buf), sources))


def diagnose(retriever: Retriever, query: str, *, server: str = config.DEFAULT_SERVER,
             top_k: int = config.TOP_K) -> dict:
    """Non-streaming convenience wrapper (used by tests / non-SSE callers)."""
    result = None
    for kind, payload in diagnose_stream(retriever, query, server=server, top_k=top_k):
        if kind == "result":
            result = payload
    return result or {"query": query, "causes": [], "raw": ""}
