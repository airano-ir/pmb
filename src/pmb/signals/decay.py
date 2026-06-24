"""
Importance scoring and forgetting curve.

This is an **anti-degradation** mechanism:
- Frequently accessed memories gain importance (reinforcement)
- Those unused for a long time decline (forgetting)
- When importance < 0.05 and age > 90 days → archive (not deleted permanently)

Formula:
- On each recall hit: importance += boost * (1 - importance)  // saturating boost
- Daily decay: importance *= 0.985  (~half-life ~46 days if unused)
- Pin: importance = 1.0 + pinned flag (does not decay)

Entry points:
- recompute_importance(engine) - recompute after a batch
- apply_decay(engine) - daily ritual

In Phase 2 these are manual functions; in Phase 3 there will be a cron-like background scheduler.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmb.core.engine import Engine

DAILY_DECAY_FACTOR = 0.985  # ~46 day half-life
RECALL_BOOST_RATE = 0.10
ARCHIVE_THRESHOLD_IMPORTANCE = 0.05
ARCHIVE_MIN_AGE_DAYS = 90

# Floor for lesson / decision events. Without it, a critical-but-rarely-
# recalled rule drifts below `find_lessons` ranking and stops surfacing
# on auto-recall - the agent then repeats the same mistake because the
# rule went invisible, not because it was wrong. The floor sits well above
# the archive threshold (0.05) but below "pinned" (0.99) so explicitly
# pinned events still rank higher. To intentionally retire a rule use
# `archive()` / `forget()`; auto-decay alone will not drop it.
LESSON_DECISION_IMPORTANCE_FLOOR = 0.6


def _is_lesson_or_decision(ev) -> bool:
    """A lesson is `event_type='lesson'` OR `metadata.kind='lesson'`. Decisions
    use `metadata.kind='decision'` (or, historically, `activity_kind`). Both
    are durable knowledge - the rules and the why behind them - and are the
    types that hurt most when they quietly decay out of recall."""
    if getattr(ev, "event_type", None) == "lesson":
        return True
    meta = getattr(ev, "metadata", None) or {}
    kind = meta.get("kind") or meta.get("activity_kind")
    return kind in ("lesson", "decision")


def boost_on_recall(current_importance: float, hit_score: float) -> float:
    """
    Apply a boost to importance after a recall hit.

    Saturating: the closer to 1.0, the smaller the increment.
    hit_score (0..1) - how relevant the hit was (we only boost on strong ones).
    """
    if hit_score < 0.3:
        return current_importance
    boost = RECALL_BOOST_RATE * hit_score
    new_imp = current_importance + boost * (1.0 - current_importance)
    return min(1.0, new_imp)


def apply_decay(engine: Engine, days_since_last_decay: float = 1.0) -> dict:
    """
    Tier-aware daily decay - COMPOUNDED over `days_since_last_decay`.

    Each tier has its own per-day decay factor (see core/events.py):
      - working   factor=0.70  -> half-life ~1.94 days
      - episodic  factor=0.985 -> half-life ~46 days
      - semantic  factor=0.998 -> half-life ~346 days

    Compounding: factor^N. So after 30 days of inactivity:
      - working event: 0.70^30 = 2.25e-5  (effectively zero, archive-ready)
      - episodic     : 0.985^30 = 0.636
      - semantic     : 0.998^30 = 0.942

    This is by design - working memory IS supposed to be ephemeral.
    Pinned events (importance >= 0.99) are still fully protected.
    Working-tier events that fall below threshold are archived MUCH sooner
    than episodic/semantic ones - keeps the hot working memory clean.

    `days_since_last_decay` is the time since this function was last
    called for this workspace; the scheduler tracks this so calling
    `apply_decay` daily vs weekly produces the same long-term curve.
    """
    from pmb.core.events import TIER_DECAY_FACTORS, TIER_WORKING

    workspace_id = engine.workspace.id
    active = engine.events.list_active(workspace_id, limit=100000)

    n_decayed = 0
    n_archived = 0
    by_tier: dict[str, int] = {}
    now = time.time()
    archive_age_seconds = ARCHIVE_MIN_AGE_DAYS * 86400

    for ev in active:
        # Pinned (importance >= 0.99) does not decay
        if ev.importance >= 0.99:
            continue
        # Tier-specific factor; unknown tier falls back to the old constant
        tier_factor = TIER_DECAY_FACTORS.get(ev.tier, DAILY_DECAY_FACTOR)
        factor = tier_factor ** days_since_last_decay
        new_imp = ev.importance * factor

        # Bonus for recently accessed: does not decline as steeply
        recent_boost = 0.0
        days_since_access = (now - ev.last_accessed) / 86400.0
        if days_since_access < 7.0:
            recent_boost = 0.05 * (1.0 - days_since_access / 7.0)
        new_imp = min(1.0, new_imp + recent_boost)

        # Floor for lessons / decisions - the "rules" and "why" behind the
        # work. A rarely-recalled lesson can decay below find_lessons ranking
        # and silently stop surfacing, which directly causes the agent to
        # repeat the same mistake. Cap the downward drift here, not via
        # `pin` (pinning is for the user's explicit "this is important").
        if _is_lesson_or_decision(ev):
            new_imp = max(new_imp, LESSON_DECISION_IMPORTANCE_FLOOR)

        engine.events.update_importance(ev.ulid, new_imp)
        n_decayed += 1
        by_tier[ev.tier] = by_tier.get(ev.tier, 0) + 1

        # Tier-aware archive policy:
        # - working tier: archive when low importance AND >3 days old (fast forget)
        # - other tiers: keep current 90-day floor
        if ev.tier == TIER_WORKING:
            min_age_for_archive = 3 * 86400
        else:
            min_age_for_archive = archive_age_seconds
        age_seconds = now - ev.timestamp
        if new_imp < ARCHIVE_THRESHOLD_IMPORTANCE and age_seconds > min_age_for_archive:
            engine.events.archive(ev.ulid)
            n_archived += 1

    return {
        "n_active_processed": len(active),
        "n_decayed": n_decayed,
        "n_archived": n_archived,
        "decayed_by_tier": by_tier,
    }


def recompute_importance(engine: Engine) -> dict:
    """
    Recompute importance based on access patterns.

    Used when you want to reset state - usually apply_decay is sufficient.
    """
    active = engine.events.list_active(engine.workspace.id, limit=100000)
    now = time.time()

    n_updated = 0
    for ev in active:
        if ev.importance >= 0.99:
            continue

        # Base importance by event_type
        base = {
            "qa": 0.5,
            "fact": 0.7,
            "git": 0.5,
            "file": 0.4,
            "test": 0.5,
            "pin": 1.0,
        }.get(ev.event_type, 0.5)

        # Access bonus
        access_factor = min(0.3, ev.access_count * 0.05)

        # Recency factor
        days_old = (now - ev.timestamp) / 86400.0
        if days_old < 7:
            recency = 0.1
        elif days_old < 30:
            recency = 0.05
        elif days_old < 90:
            recency = 0.0
        else:
            recency = -0.1

        new_imp = max(0.0, min(1.0, base + access_factor + recency))
        engine.events.update_importance(ev.ulid, new_imp)
        n_updated += 1

    return {"n_updated": n_updated}
