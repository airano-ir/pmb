"""Tests for goals + milestone chains (Improvement R)."""
from __future__ import annotations

from pmb.core.engine import Engine

# ----------------------------------------------------------------------
# Goals
# ----------------------------------------------------------------------

def test_record_goal_creates_event(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ulid = eng.record_goal("Ship v1.0 by Q3", status="in_progress")
    ev = eng.events.get_by_ulid(ulid)
    assert ev.event_type == "goal"
    assert ev.metadata.get("goal_status") == "in_progress"
    assert ev.metadata.get("goal_progress") == 0


def test_update_goal_creates_update_event(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    g = eng.record_goal("Learn Rust", status="pending")
    result = eng.update_goal(g, status="in_progress", progress=30)
    assert result["status"] == "in_progress"
    assert result["progress"] == 30
    # An update event exists pointing at the goal
    update_ulid = result["update_ulid"]
    upd = eng.events.get_by_ulid(update_ulid)
    assert upd.event_type == "goal_update"
    assert upd.metadata.get("goal_ulid") == g


def test_list_goals_filters_by_status(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_goal("A", status="pending")
    eng.record_goal("B", status="done")
    eng.record_goal("C", status="pending")
    pending = eng.list_goals(status="pending")
    assert len(pending) == 2
    done = eng.list_goals(status="done")
    assert len(done) == 1


def test_goal_hierarchy(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    parent = eng.record_goal("Big project")
    child = eng.record_goal("Sub task", parent_goal_ulid=parent)
    ev = eng.events.get_by_ulid(child)
    assert ev.metadata.get("parent_goal_ulid") == parent


# ----------------------------------------------------------------------
# Milestone chains
# ----------------------------------------------------------------------

def test_record_milestone_starts_chain(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    m1 = eng.record_milestone(
        "architecture_layers",
        "6 layers (initial)",
        state={"count": 6},
    )
    ev = eng.events.get_by_ulid(m1)
    assert ev.event_type == "milestone"
    assert ev.metadata.get("chain_name") == "architecture_layers"
    # First milestone has no previous
    assert ev.metadata.get("previous_milestone_ulid") is None


def test_milestone_chain_links_prev(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    m1 = eng.record_milestone("layers", "6", state={"count": 6})
    m2 = eng.record_milestone("layers", "7", state={"count": 7})
    m3 = eng.record_milestone("layers", "11", state={"count": 11})
    ev2 = eng.events.get_by_ulid(m2)
    ev3 = eng.events.get_by_ulid(m3)
    assert ev2.metadata.get("previous_milestone_ulid") == m1
    assert ev3.metadata.get("previous_milestone_ulid") == m2


def test_chain_history_chronological(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_milestone("growth", "6 layers", state={"count": 6})
    eng.record_milestone("growth", "7 layers", state={"count": 7})
    eng.record_milestone("growth", "11 layers", state={"count": 11})
    hist = eng.chain_history("growth")
    assert len(hist) == 3
    assert hist[0]["state"]["count"] == 6
    assert hist[1]["state"]["count"] == 7
    assert hist[2]["state"]["count"] == 11


def test_chain_current_returns_latest(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_milestone("x", "a", state={"v": 1})
    eng.record_milestone("x", "b", state={"v": 2})
    eng.record_milestone("x", "c", state={"v": 3})
    cur = eng.chain_current("x")
    assert cur is not None
    assert cur["state"]["v"] == 3


def test_chain_isolated_per_name(tmp_pmb_home, tmp_workspace_dir):
    """Two different chains don't mix."""
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_milestone("layers", "6", state={"count": 6})
    eng.record_milestone("tests", "100", state={"count": 100})
    eng.record_milestone("layers", "7", state={"count": 7})
    layers_hist = eng.chain_history("layers")
    tests_hist = eng.chain_history("tests")
    assert len(layers_hist) == 2
    assert len(tests_hist) == 1


def test_triggered_by_creates_causation_edge(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    # Some implementation event that "caused" the milestone
    trigger = eng.record_fact("Implemented activity log layer")
    milestone = eng.record_milestone(
        "layers", "11 layers",
        state={"count": 11},
        triggered_by_ulid=trigger,
    )
    # Edge from trigger → milestone exists
    import sqlite3
    with sqlite3.connect(eng.workspace.db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM event_edges WHERE source_ulid = ? AND target_ulid = ?",
            (trigger, milestone),
        ).fetchall()
    assert len(rows) == 1


def test_chain_history_includes_triggered_by(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    trigger = eng.record_fact("Built typo correction")
    eng.record_milestone("features", "added typo fix",
                         state={"feature": "typo"},
                         triggered_by_ulid=trigger)
    hist = eng.chain_history("features")
    assert hist[0]["triggered_by_ulid"] == trigger
