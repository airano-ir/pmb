"""
Adaptive Importance - learns from failed self-test queries.

When a self-test fails on query X (the expected event E wasn't in the top-K),
that's a signal that E doesn't have enough importance to compete with others.

Adaptation strategies:
1. Boost the importance of failed events by 10% (saturating)
2. Additionally log the failure pattern in adaptive_log.jsonl
3. If the same event fails > 3 times - apply a bigger blanket importance boost

This is slow learning: each weekly self-test → a small adjustment.

After 2-3 months the system should reach stable accuracy on its own for the
queries the user actually asks.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmb.core.engine import Engine
    from pmb.health.self_test import SelfTestResult


BOOST_PER_FAILURE = 0.10
SUPERBOOST_THRESHOLD = 3  # after 3 failures - bigger boost
SUPERBOOST_VALUE = 0.85


def _adaptive_log_path(engine: Engine) -> Path:
    return engine.workspace.storage_dir / "adaptive_log.jsonl"


def _load_failure_counts(engine: Engine) -> dict[str, int]:
    """Load failure counts keyed by ulid."""
    log = _adaptive_log_path(engine)
    if not log.exists():
        return {}
    counts: dict[str, int] = {}
    with open(log, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                ulid = data.get("ulid")
                if ulid:
                    counts[ulid] = counts.get(ulid, 0) + 1
            except Exception:
                continue
    return counts


def apply_adaptive_boost(
    engine: Engine,
    self_test_result: SelfTestResult,
) -> dict:
    """
    After a self-test, apply an adaptive boost to the failed events.

    Pinned events (importance >= 0.99) are skipped.

    Returns:
        {
            "n_failed": int,
            "n_boosted": int,
            "n_superboosted": int,
        }
    """
    failed = self_test_result.failed_queries
    if not failed:
        return {"n_failed": 0, "n_boosted": 0, "n_superboosted": 0}

    log_path = _adaptive_log_path(engine)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Append all failures to log
    with open(log_path, "a", encoding="utf-8") as f:
        for failure in failed:
            f.write(json.dumps({
                "timestamp": time.time(),
                "ulid": failure["ulid"],
                "query": failure.get("query"),
                "expected_preview": failure.get("expected_content_preview"),
            }, ensure_ascii=False) + "\n")

    # Recalculate failure counts (incl. just-added)
    counts = _load_failure_counts(engine)

    n_boosted = 0
    n_super = 0
    for failure in failed:
        ulid = failure["ulid"]
        ev = engine.events.get_by_ulid(ulid)
        if not ev:
            continue
        if ev.importance >= 0.99:
            continue  # pinned

        n_failures = counts.get(ulid, 1)
        if n_failures >= SUPERBOOST_THRESHOLD:
            new_imp = max(ev.importance, SUPERBOOST_VALUE)
            n_super += 1
        else:
            new_imp = min(1.0, ev.importance + BOOST_PER_FAILURE)
        engine.events.update_importance(ulid, new_imp)
        n_boosted += 1

    return {
        "n_failed": len(failed),
        "n_boosted": n_boosted,
        "n_superboosted": n_super,
    }


def adaptive_history(engine: Engine, limit: int = 100) -> list[dict]:
    """Read historic failures."""
    log = _adaptive_log_path(engine)
    if not log.exists():
        return []
    items = []
    with open(log, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
    return items[-limit:]


# ---------------------------------------------------------------------------
# Feedback-driven adaptive - uses REAL user signal, not synthetic self-test
# ---------------------------------------------------------------------------

FEEDBACK_USEFUL_PROMOTE_AT = 3   # n useful → strong promote
FEEDBACK_EXPECTED_PROMOTE_AT = 2  # n times flagged as expected-but-missed → promote
FEEDBACK_WRONG_DEMOTE_AT = 3     # n wrong → demote
FEEDBACK_PROMOTE_TARGET = 0.85
FEEDBACK_EXPECTED_PROMOTE_TARGET = 0.90
FEEDBACK_DEMOTE_FACTOR = 0.5
FEEDBACK_DEMOTE_FLOOR = 0.05


def apply_feedback_adaptive(engine: Engine) -> dict:
    """
    Aggregate feedback counts and promote / demote importance.

    Run periodically (e.g. weekly with self-test). Operates on totals,
    so repeated calls don't compound - promoting to a target is idempotent.

    Returns counts of events touched.
    """
    from pmb.health.feedback import history

    entries = history(engine)
    if not entries:
        return {
            "n_feedback_entries": 0,
            "n_promoted_useful": 0,
            "n_promoted_expected": 0,
            "n_demoted_wrong": 0,
        }

    useful_counts: dict[str, int] = {}
    wrong_counts: dict[str, int] = {}
    expected_counts: dict[str, int] = {}

    for e in entries:
        if e.verdict == "useful":
            useful_counts[e.ulid] = useful_counts.get(e.ulid, 0) + 1
        elif e.verdict in ("wrong", "irrelevant"):
            wrong_counts[e.ulid] = wrong_counts.get(e.ulid, 0) + 1
        if e.expected_ulid and e.verdict == "wrong":
            expected_counts[e.expected_ulid] = expected_counts.get(e.expected_ulid, 0) + 1

    n_promo_useful = 0
    n_promo_expected = 0
    n_demoted = 0

    for ulid, cnt in useful_counts.items():
        if cnt < FEEDBACK_USEFUL_PROMOTE_AT:
            continue
        ev = engine.events.get_by_ulid(ulid)
        if ev and ev.importance < 0.99 and ev.importance < FEEDBACK_PROMOTE_TARGET:
            engine.events.update_importance(ulid, FEEDBACK_PROMOTE_TARGET)
            n_promo_useful += 1

    for ulid, cnt in expected_counts.items():
        if cnt < FEEDBACK_EXPECTED_PROMOTE_AT:
            continue
        ev = engine.events.get_by_ulid(ulid)
        if ev and ev.importance < 0.99 and ev.importance < FEEDBACK_EXPECTED_PROMOTE_TARGET:
            engine.events.update_importance(ulid, FEEDBACK_EXPECTED_PROMOTE_TARGET)
            n_promo_expected += 1

    for ulid, cnt in wrong_counts.items():
        if cnt < FEEDBACK_WRONG_DEMOTE_AT:
            continue
        ev = engine.events.get_by_ulid(ulid)
        if ev and ev.importance < 0.99:
            new_imp = max(FEEDBACK_DEMOTE_FLOOR, ev.importance * FEEDBACK_DEMOTE_FACTOR)
            engine.events.update_importance(ulid, new_imp)
            n_demoted += 1

    return {
        "n_feedback_entries": len(entries),
        "n_promoted_useful": n_promo_useful,
        "n_promoted_expected": n_promo_expected,
        "n_demoted_wrong": n_demoted,
    }
