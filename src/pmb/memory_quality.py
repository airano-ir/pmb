"""
Memory-quality signals: freshness (staleness) and confidence.

These are DISPLAY/CAPTURE signals - they do NOT change the hot-path ranking
(BM25 + vector + PAMVR), so they can't regress the LoCoMo benchmark or slow
recall. They make recall output more trustworthy:

  - freshness: "is this fact probably stale?" (old + rarely confirmed)
  - confidence: "how reliable is the source?" (user-said > imported > inferred)

Pure functions, no I/O - trivially testable.
"""

from __future__ import annotations

from pmb import lang as _lang

# A fact older than this (days) and not recently re-confirmed is flagged stale.
STALE_DAYS = 180.0

# How reliable is the source the fact came from. Independent of importance.
_SOURCE_CONFIDENCE = {
    "lesson": 0.9,       # explicit correction/technique
    "cli-note": 0.85,    # the user typed it directly
    "note": 0.85,
    "user": 0.9,         # user statement captured by the agent
    "git": 0.8,          # commit metadata - factual
    "markdown": 0.7,     # user's own notes
    "chatgpt": 0.65,     # imported conversation (could be assistant text)
    "claude": 0.65,
    "mem0": 0.7,         # already-distilled facts
    "watch": 0.7,        # user's journal/notes
    "agent": 0.5,        # agent-inferred / decision
    "failure": 0.9,      # a recorded failure is a reliable "don't do this"
}


def days_old(timestamp: float, now: float) -> float:
    return max(0.0, (now - timestamp) / 86400.0)


def is_stale(timestamp: float, now: float, *, access_count: int = 0,
             threshold_days: float = STALE_DAYS) -> bool:
    """Old AND rarely accessed → likely stale. Frequent access keeps a fact
    'warm' even if old (it's clearly still relevant)."""
    if days_old(timestamp, now) < threshold_days:
        return False
    return access_count < 3


def freshness_label(timestamp: float, now: float, *, access_count: int = 0) -> str | None:
    """Human staleness note, or None if the fact is fresh enough to not warn."""
    if not is_stale(timestamp, now, access_count=access_count):
        return None
    d = days_old(timestamp, now)
    if d >= 365:
        age = f"{d / 365:.1f}y"
    elif d >= 30:
        age = f"{int(d // 30)}mo"
    else:
        age = f"{int(d)}d"
    return f"may be stale, {age} old"


def confidence_from(metadata: dict | None) -> float:
    """Source-derived reliability in [0,1]. Defaults to 0.6 (unknown source)."""
    meta = metadata or {}
    # explicit override wins
    if isinstance(meta.get("confidence"), (int, float)):
        return float(meta["confidence"])
    kind = (meta.get("kind") or "").lower()
    if kind in _SOURCE_CONFIDENCE:
        return _SOURCE_CONFIDENCE[kind]
    src = (meta.get("source") or "").lower()
    if src in _SOURCE_CONFIDENCE:
        return _SOURCE_CONFIDENCE[src]
    if meta.get("agent_id") or meta.get("actor") == "agent":
        return _SOURCE_CONFIDENCE["agent"]
    return 0.6


def confidence_label(c: float) -> str:
    if c >= 0.8:
        return "high"
    if c >= 0.6:
        return "med"
    return "low"


# Lesson-intent: does this query look like "how should I work here / what's
# the convention / what to avoid"? Used to gently boost lesson/failure
# memories on coding-style queries. Pure + cheap.
# EN floor inline; the RU/UK markers ("how to configure" / "what rules" / …)
# live in the lang packs under `lesson_intent_markers` (L1).
_LESSON_INTENT_MARKERS = (
    "how do", "how to", "how should", "should i", "should we",
    "what's the right", "what is the right", "best way", "right way",
    "convention", "conventions", "do we use", "which tool", "setup",
    "configure", "config", "avoid", "gotcha", "lesson", "lessons",
    "remember to", "rule", "rules", "workflow", "policy",
    *(_lang.merged_list("lesson_intent_markers")),
)


def is_lesson_intent(query: str) -> bool:
    q = (query or "").lower()
    return any(m in q for m in _LESSON_INTENT_MARKERS)
