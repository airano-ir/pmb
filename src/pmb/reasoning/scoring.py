"""X1/X2/X3 - the canonical scoring + confidence math in ONE explainable place.

Recall's final score is a base channel-combination followed by a long chain of
config-gated boosts (graph / causation / arc / ppr / temporal / keyed) and the
PAMVR reranker. Unifying the WHOLE chain would rewire the recall hot path - the
core the project guards most - so this module owns the two CLEAN, well-defined
pieces and leaves the boost chain where it is:

  * ``combine_base_score`` (X1) - the base channel combination (hybrid hit score
    × importance factor × recency reward) as one named function, with an
    optional per-CHANNEL weight vector (X2). Weights default to identity, so
    default recall is byte-identical to the old inline formula.
  * ``calibrated_confidence`` (X3) - RecallPack.confidence as a named, tested,
    monotonic, bounded function.

X2 adaptive weights: ``recall.channel_weights`` (a JSON object string in config)
is a per-workspace vector the system COULD learn from feedback;
``propose_channel_weights`` suggests an update but is NEVER auto-applied -
default identity means zero behaviour change until a user opts in.
"""
from __future__ import annotations

import json

# The three base channels the weight vector scales. Identity = no change.
CHANNELS = ("hit", "importance", "recency")
_IDENTITY: dict[str, float] = {c: 1.0 for c in CHANNELS}


def importance_factor(importance: float, weights: dict[str, float] | None = None) -> float:
    """0.5..1.0 - an importance-0 event keeps half its hit score, importance-1
    keeps all of it (recall's ``0.5 + 0.5*importance``), scaled by the importance
    CHANNEL weight (X2). Weight 1.0 → the original.

    Recall computes this ONCE per candidate and feeds it to BOTH
    ``combine_base_score`` and the downstream boost terms, so the importance
    weight is applied consistently across the whole score - not just the base."""
    w = (weights or _IDENTITY).get("importance", 1.0)
    return 0.5 + 0.5 * max(0.0, min(1.0, importance)) * float(w)


def combine_base_score(
    hit_score: float,
    imp_factor: float,
    recency: float,
    *,
    rec_mul: float = 1.0,
    weights: dict[str, float] | None = None,
) -> float:
    """Base recall score BEFORE the boost chain + PAMVR. Takes the ALREADY
    -computed importance factor (from ``importance_factor``) so the importance
    weight is applied once and SHARED with the boosts. With identity weights:

        hit_score * imp_factor * (1 + 0.2*recency*rec_mul)

    The hit / recency weights scale those two channels; 1.0 = no change."""
    w = weights or _IDENTITY
    recf = 1.0 + 0.2 * recency * rec_mul * float(w.get("recency", 1.0))
    return hit_score * float(w.get("hit", 1.0)) * imp_factor * recf


def calibrated_confidence(top1: float, top2: float | None = None) -> float:
    """RecallPack confidence in [0,1] (X3). Monotonic non-decreasing in top1 and
    in the top1→top2 gap. With identity inputs reproduces the old inline calc."""
    t1 = max(0.0, min(1.0, top1))
    if top2 is None:
        return min(1.0, t1 * 0.7 + 0.1)
    t2 = max(0.0, min(1.0, top2))
    gap = t1 - t2
    return min(1.0, t1 * 0.7 + gap * 0.3 + 0.1)


def channel_weights(config) -> dict[str, float]:
    """Per-workspace channel weights (X2) parsed from the ``recall.channel_weights``
    JSON-object string. Empty / malformed → identity (no behaviour change).
    Unknown keys are ignored; missing channels default to 1.0."""
    out = dict(_IDENTITY)
    try:
        raw = config.get("recall.channel_weights")
    except Exception:
        return out
    if not raw or not isinstance(raw, str):
        return out
    try:
        parsed = json.loads(raw)
    except Exception:
        return out
    if isinstance(parsed, dict):
        for c in CHANNELS:
            if c in parsed:
                try:
                    out[c] = float(parsed[c])
                except (TypeError, ValueError):
                    pass
    return out


def is_identity(weights: dict[str, float]) -> bool:
    return all(abs(float(weights.get(c, 1.0)) - 1.0) < 1e-9 for c in CHANNELS)


def propose_channel_weights(
    samples: list[tuple[dict[str, float], bool]],
    *,
    lr: float = 0.1,
) -> dict[str, float]:
    """X2 learning hook (NOT auto-applied). Given ``(channel_scores, was_useful)``
    feedback samples, nudge each channel's weight up when high values of that
    channel coincided with useful recalls and down otherwise - a simple
    gradient-free heuristic, clamped to [0.5, 1.5] around identity. The result is
    a SUGGESTION a user can inspect and set via ``recall.channel_weights``; recall
    never adopts it on its own (default identity)."""
    if not samples:
        return dict(_IDENTITY)
    out = dict(_IDENTITY)
    for c in CHANNELS:
        # mean channel value among useful vs not-useful recalls
        useful = [s.get(c, 0.0) for s, ok in samples if ok]
        notok = [s.get(c, 0.0) for s, ok in samples if not ok]
        mu = sum(useful) / len(useful) if useful else 0.0
        mn = sum(notok) / len(notok) if notok else 0.0
        out[c] = max(0.5, min(1.5, 1.0 + lr * (mu - mn)))
    return out
