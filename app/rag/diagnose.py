"""Natural multi-turn diagnosis for the AgriDoc workbench.

The model is trained to be a conversational field officer in NATURAL PROSE: it asks a
discriminating question when the farmer's description is thin, and commits when the
pattern is clear. So the app no longer imposes a rigid schema (the old
`READY:`/`CAUSE:` prototype is retired) — it threads the conversation, streams the
model's own prose, and RECONCILES against the offline reference corpus.

Honest / offline:
  - the model reasons from its OWN knowledge — retrieved notes are NOT fed into the
    prompt (that anchored the 1.5B on topical-but-wrong notes);
  - RAG reconciles AFTER generation, over the accumulated farmer text: a *confident*
    diagnostic note (cosine >= 0.75 + a disease keyword) that names a diagnosis the
    model did NOT mention is surfaced as a cited "reference note" (where this comes
    from, bundled mid-2026, confirm locally) — not as gospel, and silent when the
    corpus genuinely can't help;
  - the model never invents a dose, rate, or price (trained + persona-reinforced).

Everything streams, so the UI shows the reply as it is written.
"""
from __future__ import annotations

import re

from . import config, llm
from .retriever import Retriever

# Only chunks at/above this cosine count as a usable reference (drops 0.0 noise).
REF_FLOOR = 0.35
# Surface a reference note only when retrieval is CONFIDENT (not merely topical).
STRONG_REF = 0.75

# The SAME light persona the model was fine-tuned with — this reliably elicits
# the trained ask-or-assess behaviour. No few-shot examples: the behaviour is in the
# weights now, and inline examples made the 1.5B copy/contaminate.
DX_SYSTEM = (
    "You are a careful, experienced agricultural and poultry extension officer helping "
    "smallholder farmers in East and West Africa, offline. Reason from the specific pattern "
    "the farmer describes and give safe, practical first-opinion advice. When the description "
    "is too thin to be sure, say what you are leaning toward and ask the one or two questions "
    "that would settle it, rather than guessing. Never invent a pesticide or drug dose — point "
    "to the product label and a local vet or extension officer. Animals other than poultry are "
    "outside your focus: give the most useful check and refer to an animal-health worker."
)

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


DIAG_KW = re.compile(
    r"\b(rust|blight|wilt|rot|mildew|mosaic|virus|disease|deficiency|mould|mold|bacterial|"
    r"fungal|smut|anthracnose|streak|spot|canker|scald|mite|weevil|borer|aphid|nematode|"
    r"worm|thrip|whitefl|leafhopper|caterpillar|armyworm|coccidio|newcastle|bronchitis|"
    r"cholera|blast|damping)\b", re.I)


def _subject(snippet: str) -> str:
    """Leading subject of a reference note, e.g. 'Southern rust', 'Cassava mosaic disease'."""
    m = re.match(r"\s*([A-Z][A-Za-z'\-]+(?: [A-Za-z'\-]+){0,4}?)(?=,| is | are | \(|caused by| affects)", snippet)
    return (m.group(1).strip() if m else snippet.split(",")[0][:48].strip())


# The reply is "gathering" (still asking) if it poses a discriminating question — a '?'
# alongside question-seeking language. Otherwise it has committed/advised. Drives the UI
# (a gathering turn invites the farmer's answer; a committed turn offers save/print).
_Q = re.compile(r"\?")
_QSEEK = re.compile(
    r"are the|is it|is the|do you|does it|what (colour|color|do|exactly|age|kind|type)|"
    r"where\b|how (old|many|fast|long|wet|is)|when you|are they|which|could you|"
    r"can you (tell|check|describe|look|see)|tell me|let me know|have you", re.I)


def _asked(text: str) -> bool:
    return bool(_Q.search(text) and _QSEEK.search(text))


def _reconcile(text: str, sources: list[dict]) -> dict | None:
    """A confident diagnostic note that names a diagnosis the model did NOT mention."""
    strong = [s for s in sources if s.get("match", 0) >= STRONG_REF and DIAG_KW.search(s.get("snippet", ""))]
    if not strong:
        return None
    subj = _subject(strong[0]["snippet"])
    if subj and not (_terms(subj) & _terms(text)):
        return {"subject": subj, "source": {k: v for k, v in strong[0].items() if k != "_terms"}}
    return None


def _normalize(conversation) -> tuple[list[dict], str]:
    """Accept a plain string (fresh case) or a list of {role, content} turns (continuing).

    No 'FARMER'S PROBLEM:' prefix — the model was trained on plain farmer text. Returns
    (history, retrieval_query) where the query is all farmer turns joined (so retrieval
    sees the fuller picture after a clarifying answer).
    """
    if isinstance(conversation, str):
        history = [{"role": "user", "content": conversation.strip()}]
    else:
        history = [{"role": "assistant" if t.get("role") == "assistant" else "user",
                    "content": (t.get("content") or "").strip()}
                   for t in conversation if (t.get("content") or "").strip()]
    farmer = " ".join(h["content"] for h in history if h["role"] == "user")
    return history, farmer


def _profile_note(profile: dict | None) -> str:
    """Standing farm context, formatted for the system prompt.

    DISABLED (2026-07-26): injecting this note MEASURABLY REGRESSED SAFETY. With a
    "…do not re-ask these: …grows maize…" note appended, a fertiliser-dose question that
    the bare model correctly refuses ("read the package, ask an extension officer") flipped
    to leaking a rate ("10–25 kg/ha"), and looping got worse. The "do not re-ask" steer
    fights the ask-or-assess + dose-refusal behaviour the model was TRAINED and GATED with —
    and profile-conditioning was never part of that gate. So we do NOT inject it: the app
    ships exactly the behaviour that was gated. Re-enable only when profile context is part
    of the training mix and passes the same greedy safety gate. The frontend still sends
    `profile`; the backend accepts and ignores it.
    """
    return ""


def _strip(s: dict) -> dict:
    return {k: v for k, v in s.items() if k != "_terms"}


# App-layer DOSE GUARD — defence-in-depth safety net. The model is trained to refuse doses
# and prices; the app additionally redacts any prescriptive rate/ratio from the final advisory
# as a backstop and points to the product label + a local extension officer or vet.
# Belt-and-suspenders by design.
_DOSE_RATE = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?\s*"
    r"(?:kg|g|grams?|mg|ml|mls|l|litres?|liters?|cc|tonnes?|tons?|tablespoons?|teaspoons?|caps?|sachets?)"
    r"(?:\s+[\w+/-]+){0,3}?\s*(?:/|per)\s*"
    r"(?:ha|hectare|acre|bird|plant|animal|hole|stand|knapsack|sprayer|application|bag|litre|liter|l\b|\d+\s*l)"
    r"|\b1\s*:\s*\d{1,4}\b"
    r"|\b\d+(?:\.\d+)?\s*(?:litres?|liters?|ml)\b(?=[^.]{0,30}(?:sprayer|knapsack|concentrate|dilut))"
    r"|\b\d+(?:\.\d+)?\s*%\s*(?:solution|concentration|dilution|ai|active)?",
    re.I)
_DOSE_NOTE = ("[a specific rate isn't given here — read the product label and confirm the amount "
              "with your local extension officer or vet]")


def dose_guard(text: str) -> str:
    """Replace any prescriptive dose/rate/ratio in a reply with a refer-to-label note."""
    return _DOSE_RATE.sub(_DOSE_NOTE, text or "")


def diagnose_stream(retriever: Retriever, conversation, *, profile: dict | None = None,
                    server: str = config.DEFAULT_SERVER, top_k: int = config.TOP_K):
    """Generator of ('meta'|'token'|'result', payload). Streams the reply live.

    `conversation` is the farmer's problem (str) for a fresh case, or the full list of
    {role, content} turns for a continuing case. The model decides each turn whether to
    ask for more or to advise.
    """
    history, retrieval_query = _normalize(conversation)
    chunks = retriever.retrieve(retrieval_query, top_k=top_k)
    refs = [c for c in chunks if c.get("_dense_score", 0.0) >= REF_FLOOR]
    sources = [_source_view(c) for c in refs]
    public_sources = [_strip(s) for s in sources]
    yield ("meta", {"query": retrieval_query, "sources": public_sources})

    messages = [{"role": "system", "content": DX_SYSTEM + _profile_note(profile)}, *history]
    buf = []
    # Greedy + full-context repeat penalty (config defaults) — matches how the model was
    # gated (a diagnostic tool should be deterministic, and sampling made it loop).
    for delta in llm.chat_stream(messages, server=server):
        buf.append(delta)
        yield ("token", delta)
    text = dose_guard("".join(buf).strip())   # redact any rare dose leak from the final advisory

    yield ("result", {
        "mode": "gather" if _asked(text) else "commit",
        "asked": _asked(text),
        "text": text,
        "reference_note": _reconcile(text, sources),   # honest RAG surface, or None
        "related": public_sources,                     # notes the officer can open to verify
        "raw": text,
    })


def diagnose(retriever: Retriever, conversation, *, profile: dict | None = None,
             server: str = config.DEFAULT_SERVER, top_k: int = config.TOP_K) -> dict:
    """Non-streaming convenience wrapper (tests / non-SSE callers)."""
    result = None
    for kind, payload in diagnose_stream(retriever, conversation, profile=profile,
                                         server=server, top_k=top_k):
        if kind == "result":
            result = payload
    return result or {"mode": "commit", "text": "", "asked": False, "related": []}
