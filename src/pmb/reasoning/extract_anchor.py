"""C2 — keyed-fact extraction by HYPOTHESIS MARGIN (no per-language regex).

For each candidate value span (C1, `spans.value_spans`) and each personal
attribute, build a short ENGLISH hypothesis with the RAW span injected, plus an
anti-hypothesis chosen to catch that attribute's most likely false positive.
Embed the sentence + all hypotheses in ONE batch and decide by margin:

    margin(attr, v) = cos(sentence, "the user lives in {v}")
                    − cos(sentence, "the user no longer lives in {v}")

The multilingual embedder transfers, so RU "Ya zhivu v Kieve" extracts city=Kieve and
"Ich arbeite bei Stripe" extracts employer=Stripe with NO Russian/German data.
The anti-hypothesis also handles negation/tense for free: "I no longer live in
Tampa" makes the anti side win (negative margin) → the caller routes that to the
existing close-on-negation flow instead of asserting (F2).

Cost: ≤ 4 attrs × 4 spans × 2 + 1 ≈ 33 short embeds in one batch (~20–50 ms),
WRITE path only, WARM-ONLY (cold engine → []). Inflected values are stored raw
("Kieve"); keyed canonicalization clusters them by cosine (C2 dedup).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pmb.reasoning.spans import value_spans

# attr → (positive hypothesis, anti-hypothesis). `{v}` = raw value span.
# The anti-hypothesis is the NEGATION/tense flip, so a "no longer" sentence makes
# the anti side win and routes to close-on-negation (F2). job_title is omitted on
# purpose: its values are common nouns (not capitalised → not proper-noun spans),
# and as a hypothesis it only ever collided with `employer` on org names.
HYPOTHESES: dict[str, tuple[str, str]] = {
    "city":      ("the user lives in {v}",        "the user no longer lives in {v}"),
    "employer":  ("the user works at {v}",        "the user no longer works at {v}"),
    "name":      ("the user's own name is {v}",   "{v} is a different person, not the user"),
}

PREGATE_ANCHOR = "statement.about_self"


@dataclass(frozen=True)
class KeyedCandidate:
    attr: str
    value: str        # RAW span (case + inflection preserved)
    margin: float     # cos(pos) − cos(anti)
    pos: float        # cos(sentence, positive hypothesis) — F1 confidence input


def extract_keyed_anchor(sentence: str, engine, *, tau: float = 0.05,
                         pos_floor: float = 0.45, attr_gap: float = 0.035,
                         max_spans: int = 4) -> list[KeyedCandidate]:
    """Keyed (attr, value) candidates from one sentence. For each value span the
    WINNING attribute is the one whose positive hypothesis best fits the sentence
    (cross-attribute argmax = "which relationship does the sentence assert about
    this value?"). It fires only when that winner (a) clears `pos_floor`,
    (b) beats its own anti-hypothesis by `tau` (the negation/tense guard), and
    (c) beats the runner-up attribute by `attr_gap` (so an ambiguous span yields
    NO keyed fact rather than a wrong one — precision over recall on writes).

    WARM-ONLY and best-effort: returns [] on a cold engine, when the self
    pre-gate doesn't fire, or on any failure. Never loads the model."""
    sentence = (sentence or "").strip()
    if not sentence:
        return []
    try:
        idx = engine.anchor_index()
    except Exception:
        idx = None
    if idx is None:                                   # cold → regex tier only
        return []
    try:
        if not idx.fires(sentence, PREGATE_ANCHOR):   # cheap one-anchor pre-gate
            return []
    except Exception:
        return []

    spans = value_spans(sentence, max_spans=max_spans)
    if not spans:
        return []

    attrs = list(HYPOTHESES.keys())
    texts: list[str] = [sentence]
    # texts layout per span: [pos_a0, anti_a0, pos_a1, anti_a1, ...]
    for sp in spans:
        for attr in attrs:
            pos_t, neg_t = HYPOTHESES[attr]
            texts.append(pos_t.format(v=sp.text))
            texts.append(neg_t.format(v=sp.text))

    try:
        vecs = np.asarray(engine.search.embed_batch(texts), dtype=np.float32)
    except Exception:
        return []
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    s = vecs[0]

    out: list[KeyedCandidate] = []
    per_span = 2 * len(attrs)
    for si, sp in enumerate(spans):
        base = 1 + si * per_span
        pos = {attr: float(vecs[base + 2 * j] @ s) for j, attr in enumerate(attrs)}
        anti = {attr: float(vecs[base + 2 * j + 1] @ s) for j, attr in enumerate(attrs)}
        win = max(attrs, key=lambda a: pos[a])
        second = max((pos[a] for a in attrs if a != win), default=0.0)
        if (pos[win] > pos_floor and pos[win] - anti[win] > tau
                and pos[win] - second > attr_gap):
            out.append(KeyedCandidate(win, sp.text,
                                      round(pos[win] - anti[win], 4),
                                      round(pos[win], 4)))
    return _dedupe_best_per_attr(out)


def values_are_alias(engine, a: str, b: str, thresh: float = 0.92) -> bool:
    """C2 dedup: are two keyed VALUES the same thing inflected differently
    ("Kieve"/"Kiev", a German dative form)? Warm-only embedding cosine; byte-
    equal (case-insensitive) short-circuits to True; returns False on a cold
    engine or any failure (so the caller keeps both, never crashes a write)."""
    a, b = (a or "").strip(), (b or "").strip()
    if not a or not b:
        return False
    if a.lower() == b.lower():
        return True
    try:
        idx = engine.anchor_index()
        if idx is None:
            return False
        vecs = np.asarray(engine.search.embed_batch([a, b]), dtype=np.float32)
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        return float(vecs[0] @ vecs[1]) >= thresh
    except Exception:
        return False


def _dedupe_best_per_attr(cands: list[KeyedCandidate]) -> list[KeyedCandidate]:
    """Keep the highest-margin value per attribute, best margin first."""
    best: dict[str, KeyedCandidate] = {}
    for c in cands:
        if c.attr not in best or c.margin > best[c.attr].margin:
            best[c.attr] = c
    return sorted(best.values(), key=lambda c: -c.margin)
