"""Tests for the auto-consolidation trigger (sleep on a schedule)."""
from __future__ import annotations

import time

import pytest

from pmb.core.engine import Engine
from pmb.health.auto_consolidate import (
    TriggerState,
    load_state,
    mark_consolidation_done,
    save_state,
    should_trigger,
)


def test_disabled_by_default(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    out = should_trigger(eng)
    assert out["should_run"] is False
    assert out["reason"] == "auto_trigger_disabled"


# platform_sensitive: the event-count trigger races with async write visibility
# under macOS-CI load; gates on the Linux reference runner.
@pytest.mark.platform_sensitive
def test_event_count_threshold(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={
            "consolidate.auto_trigger": True,
            "consolidate.auto_min_new_events": 5,
            "consolidate.auto_min_days": 365.0,  # high enough to not fire on time alone
        },
    )
    # Pretend we just consolidated with 0 events
    mark_consolidation_done(eng)
    # Add 4 events — below threshold
    for i in range(4):
        eng.record_fact(f"fact {i}")
    assert should_trigger(eng)["should_run"] is False
    # 5th event crosses threshold
    eng.record_fact("fact 5")
    decision = should_trigger(eng)
    assert decision["should_run"] is True
    assert "5 new events" in decision["reason"]


def test_time_threshold(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={
            "consolidate.auto_trigger": True,
            "consolidate.auto_min_new_events": 100000,  # never via count
            "consolidate.auto_min_days": 1.0,
        },
    )
    # Mark a consolidation 2 days ago
    state = TriggerState(
        last_consolidation_at=time.time() - 2 * 86400,
        n_events_at_last=0, n_runs=1,
    )
    save_state(eng, state)
    decision = should_trigger(eng)
    assert decision["should_run"] is True
    assert "since last" in decision["reason"]


def test_never_consolidated_always_due(tmp_pmb_home, tmp_workspace_dir):
    """A fresh workspace with auto_trigger=true is always due."""
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={"consolidate.auto_trigger": True},
    )
    decision = should_trigger(eng)
    assert decision["should_run"] is True
    assert "never" in decision["reason"]


def test_mark_resets_counters(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={
            "consolidate.auto_trigger": True,
            "consolidate.auto_min_new_events": 3,
            "consolidate.auto_min_days": 365.0,
        },
    )
    eng.record_fact("a")
    eng.record_fact("b")
    eng.record_fact("c")
    eng.record_fact("d")
    assert should_trigger(eng)["should_run"] is True
    mark_consolidation_done(eng)
    # Now no new events since last consolidation → not due
    assert should_trigger(eng)["should_run"] is False
    state = load_state(eng)
    assert state.n_runs == 1
    assert state.n_events_at_last == 4


def test_maybe_auto_consolidate_skips_when_not_due(
    tmp_pmb_home, tmp_workspace_dir,
):
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={"consolidate.auto_trigger": False},
    )
    assert eng.maybe_auto_consolidate() is None


def test_maybe_auto_consolidate_uses_stub_llm(
    tmp_pmb_home, tmp_workspace_dir,
):
    """End-to-end: trigger fires, consolidation runs with our stub LLM."""

    class _StubLLM:
        def consolidate(self, texts):
            return {
                "consolidate": True, "summary": "Stub generalisation",
                "confidence": 0.9, "reasoning": "",
            }

    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={
            "consolidate.auto_trigger": True,
            "consolidate.auto_min_new_events": 3,
            "consolidate.auto_min_days": 365.0,
            # Keep the 4 near-identical fixtures as 4 distinct events — L2
            # semantic dedup would otherwise merge them (timing-dependent) to
            # below min_cluster_size, so consolidation found nothing (flake).
            "dedup.enable_semantic": False,
        },
    )
    # Cluster of similar facts
    for i in range(4):
        eng.record_fact(f"User prefers no comments in code part {i}")

    result = eng.maybe_auto_consolidate(
        llm=_StubLLM(), similarity_threshold=0.3, min_cluster_size=3,
    )
    assert result is not None
    assert result["n_consolidated"] >= 1
    # State updated
    state = load_state(eng)
    assert state.n_runs == 1
