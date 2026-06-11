"""Tests for the memory tier system (working/episodic/semantic)."""
from __future__ import annotations

import time

from pmb.core.engine import Engine
from pmb.core.events import (
    PROMOTE_WORKING_TO_EPISODIC_ACCESS,
    TIER_EPISODIC,
    TIER_SEMANTIC,
    TIER_WORKING,
    default_tier_for_event_type,
)
from pmb.signals.decay import apply_decay


def test_default_tier_for_fact_is_semantic():
    assert default_tier_for_event_type("fact") == TIER_SEMANTIC


def test_default_tier_for_qa_is_working():
    assert default_tier_for_event_type("qa") == TIER_WORKING


def test_remember_lands_in_working_tier(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ulid = eng.remember("Q", "A test event")
    ev = eng.events.get_by_ulid(ulid)
    assert ev.tier == TIER_WORKING


def test_record_fact_lands_in_semantic_tier(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ulid = eng.record_fact("user prefers dark mode")
    ev = eng.events.get_by_ulid(ulid)
    # Currently record_fact uses Event default tier (working). Fact-events get
    # promoted via default_tier_for_event_type only when caller honours it,
    # so we expect the appender to keep whatever tier the dataclass had — but
    # the API contract is that a 'fact' is born semantic. Engine should set it.
    # (This test documents the EXPECTED behaviour; if it fails, fix the engine.)
    assert ev.tier == TIER_SEMANTIC, (
        "facts should be born semantic — see Engine.record_fact"
    )


def test_promotion_working_to_episodic_after_n_recalls(
    tmp_pmb_home, tmp_workspace_dir,
):
    # Disable recall cache: cached hits skip apply_recall_updates so the
    # access_count would stay frozen and tier promotion would never fire.
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={"recall.cache_size": 0},
    )
    ulid = eng.remember("Postgres setup", "Use Postgres 17 on port 5432")
    assert eng.events.get_by_ulid(ulid).tier == TIER_WORKING

    # Same content as query → strong recall hit. Repeat 3 times.
    for _ in range(PROMOTE_WORKING_TO_EPISODIC_ACCESS):
        eng.recall("postgres setup port 5432", top_k=3)

    after = eng.events.get_by_ulid(ulid)
    assert after.tier == TIER_EPISODIC, (
        f"expected promotion to episodic after {PROMOTE_WORKING_TO_EPISODIC_ACCESS} "
        f"recalls, got tier={after.tier}, access_count={after.access_count}"
    )


def test_working_tier_decays_much_faster_than_semantic(
    tmp_pmb_home, tmp_workspace_dir,
):
    """A working-tier event should decay far more than a semantic one over
    the same time interval. This is the core human-memory analogy."""
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    u_work = eng.remember("ephemeral", "this is a passing thought")
    u_sem = eng.record_fact("this is a stable project decision")

    # Set both to the same starting importance so we measure decay alone
    eng.events.update_importance(u_work, 0.6)
    eng.events.update_importance(u_sem, 0.6)

    # Simulate 7 days of no access
    apply_decay(eng, days_since_last_decay=7.0)

    after_work = eng.events.get_by_ulid(u_work).importance
    after_sem = eng.events.get_by_ulid(u_sem).importance
    assert after_work < after_sem, (
        f"working ({after_work:.3f}) must decay below semantic ({after_sem:.3f})"
    )
    # Semantic barely moves
    assert after_sem > 0.55
    # Working moves a lot (0.7^7 ≈ 0.082, plus recent_boost saves it some)
    assert after_work < 0.4


def test_pinned_protected_across_tiers(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ulid = eng.remember("pin me", "important even if working tier")
    eng.pin(ulid)
    apply_decay(eng, days_since_last_decay=30.0)
    after = eng.events.get_by_ulid(ulid)
    assert after.importance >= 0.99


def test_working_archived_sooner_than_episodic(tmp_pmb_home, tmp_workspace_dir):
    """Working-tier event with low importance >3 days old → archived.
    Semantic-tier event of similar age stays alive (90-day floor)."""
    import sqlite3
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    u_w = eng.remember("w1", "noise content")
    u_e = eng.record_fact("stable project fact about something")
    # Drive both to very low importance and backdate them so:
    #   - last_accessed is >7 days ago → recent_boost contributes 0
    #   - timestamp is >5 days ago → above working's 3-day archive floor
    long_ago = time.time() - 30 * 86400
    with sqlite3.connect(eng.workspace.db_path) as conn:
        conn.execute(
            "UPDATE events SET timestamp = ?, last_accessed = ?, importance = ? "
            "WHERE ulid IN (?, ?)",
            (time.time() - 5 * 86400, long_ago, 0.04, u_w, u_e),
        )
    # One day of decay: working factor 0.7 → 0.028 (< 0.05), semantic 0.998 → 0.0399
    apply_decay(eng, days_since_last_decay=1.0)
    assert eng.events.get_by_ulid(u_w).archived_at is not None, (
        "working tier with importance≈0.028 and age=5d should archive"
    )
    assert eng.events.get_by_ulid(u_e).archived_at is None, (
        "semantic tier of same age should NOT archive — 90-day floor"
    )
