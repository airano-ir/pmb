"""C5: semantic intent fallback for the auto-recall hook (opt-in, default OFF).

The lexical intent detector (`detect_intents`) covers EN/RU/UK question
patterns. For a query in another language ("Wo wohne ich?", "?Dónde vivo?") it
finds nothing and the hook stays silent. When enabled AND the engine is warm
(i.e. served by the daemon — hooks must stay <100ms, so we never load the model
on the cold per-process path), this classifies the message by embedding it and
comparing to per-intent exemplar centroids.

DEFAULT OFF and eval-gated by design: the measured project finding is that the
semantic tier does NOT beat the lexical one with the default embedder, so this
exists for languages with NO lexical coverage, not to replace the lexical path.
Flipping the default on should be backed by an intent-classification eval that
shows >= lexical accuracy on EN/RU.
"""
from __future__ import annotations

from typing import Optional

from pmb.hooks.auto_recall import Intent

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
                             threshold: float = 0.45) -> Optional[str]:
    """Best-matching intent for `message` by embedding cosine, or None if no
    centroid clears `threshold`. Best-effort: returns None on any failure (so
    the caller just falls back to the lexical/SKIP result)."""
    msg = (message or "").strip()
    if not msg:
        return None
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
