"""Semantic intent fallback for the auto-recall hook.

The lexical intent detector (`detect_intents`) covers EN/RU/UK question
patterns. For a query in another language ("Wo wohne ich?", "¿Dónde vivo?") it
finds nothing and the hook stays silent. When the engine is warm (i.e. served by
the daemon - hooks must stay <100ms, so we never load the model on the cold
per-process path), this tier classifies the message semantically.

TWO implementations, in precedence order:

  * B1 - the Semantic Anchor Engine (`engine.anchor_index()`): margin-based,
    one-vs-rest, with per-set thresholds CALIBRATED at FPR ≤ 1% (Phase A2).
    This is the default tier (`lang.anchors`, on) and the one that actually
    earns multilingual coverage - a German/Japanese/Spanish query lands on the
    English anchors with no per-language data.
  * C5 (legacy) - per-intent exemplar CENTROIDS by raw cosine. Kept only for
    `hooks.semantic_intents` with anchors explicitly disabled. The measured
    finding was that raw-cosine centroids do NOT beat lexical; margins + FPR
    calibration are what changed that, which is why anchors supersede this.
"""
from __future__ import annotations

from pmb.hooks.auto_recall import Intent

# B1: anchor set name → coarse recall intent. `intent.self_intent` ("where do I
# live", "what's my name") is recall-worthy → PAST_QUERY. `intent.trivial_ack`
# is intentionally ABSENT: when it is the strongest hit the message is a
# foreign-language ack and the hook should stay silent (return None).
_ANCHOR_TO_INTENT: dict[str, str] = {
    "intent.goals_query": Intent.GOALS_QUERY,
    "intent.past_query": Intent.PAST_QUERY,
    "intent.recent_query": Intent.RECENT_QUERY,
    "intent.lessons_query": Intent.LESSONS_QUERY,
    "intent.work_request": Intent.WORK_REQUEST,
    "intent.self_intent": Intent.PAST_QUERY,
}


def classify_anchor_intent(engine, message: str) -> str | None:
    """B1 tier: map the strongest anchor hit to a recall intent, or None.

    Returns None when the engine is cold / anchors are off / nothing fires, and
    explicitly when the top hit is a trivial ack (so we don't surface memory on
    "了解" / "merci"). Best-effort - never raises into the hook."""
    msg = (message or "").strip()
    if not msg:
        return None
    try:
        idx = engine.anchor_index()
    except Exception:
        idx = None
    if idx is None:
        return None
    try:
        hits = idx.classify(msg)          # best margin first
    except Exception:
        return None
    for h in hits:
        # D1: log the winning fire (any anchor, incl. trivial_ack) so Phase D can
        # distil predictive n-grams from real traffic into auto.yaml.
        _log_fire(engine, h.name, msg)
        if h.name == "intent.trivial_ack":
            return None                   # strongest signal is "ack" → stay silent
        mapped = _ANCHOR_TO_INTENT.get(h.name)
        if mapped:
            return mapped
    return None


def _log_fire(engine, anchor: str, msg: str) -> None:
    """D1 best-effort fire log, gated on `lang.anchor_log`. Never raises."""
    try:
        if engine.config.get("lang.anchor_log"):
            from pmb.maintenance.distill import log_anchor_fire
            log_anchor_fire(engine, anchor, msg)
    except Exception:
        pass

# A handful of canonical exemplars per recall-worthy intent. Embedded once and
# averaged into a centroid; the multilingual model places a same-meaning query
# in another language near the matching English centroid.
_EXEMPLARS: dict[str, list[str]] = {
    Intent.PAST_QUERY: [
        "when did I do that", "what did I decide", "why did we choose this",
        "what is my configuration", "where do I live", "what was my setup",
    ],
    Intent.RECENT_QUERY: [
        "what did we just do", "what were we just discussing",
        "what happened a moment ago",
    ],
    Intent.GOALS_QUERY: [
        "what are my goals", "what is in progress", "what am I working towards",
        "list my open goals",
    ],
    Intent.LESSONS_QUERY: [
        "what rules apply here", "what conventions should I follow",
        "what lessons did we learn", "how do we work in this project",
    ],
}


def _centroids(engine):
    """Per-intent centroid vectors, computed once and cached on the engine."""
    cached = getattr(engine, "_semantic_intent_centroids", None)
    if cached is not None:
        return cached
    import numpy as np
    out = {}
    embed = engine.search.embed
    for intent, phrases in _EXEMPLARS.items():
        vecs = [np.asarray(embed(p), dtype="float32") for p in phrases]
        c = np.mean(vecs, axis=0)
        n = float(np.linalg.norm(c)) or 1.0
        out[intent] = c / n
    engine._semantic_intent_centroids = out
    return out


def classify_semantic_intent(engine, message: str,
                             threshold: float = 0.45) -> str | None:
    """Best semantic intent for `message`, or None. Prefers the calibrated
    anchor tier (B1); only falls back to the legacy centroid cosine when
    `lang.anchors` is off. Best-effort: returns None on any failure (so the
    caller just keeps the lexical/SKIP result)."""
    msg = (message or "").strip()
    if not msg:
        return None
    # B1: anchors are the authoritative semantic tier when enabled. If they fire
    # nothing we do NOT fall through to the weaker centroid path - anchors are
    # FPR-calibrated, so "anchors found nothing" is a real negative, not a gap.
    try:
        if engine.config.get("lang.anchors"):
            return classify_anchor_intent(engine, msg)
    except Exception:
        pass
    try:
        import numpy as np
        centroids = _centroids(engine)
        v = np.asarray(engine.search.embed(msg), dtype="float32")
        n = float(np.linalg.norm(v)) or 1.0
        v = v / n
        best, best_sim = None, 0.0
        for intent, c in centroids.items():
            sim = float(np.dot(v, c))
            if sim > best_sim:
                best, best_sim = intent, sim
        return best if best_sim >= float(threshold) else None
    except Exception:
        return None
