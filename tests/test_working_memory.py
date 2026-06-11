"""Tests for working memory / activity log (Improvement Q)."""
from __future__ import annotations

import time

from pmb.core.engine import Engine

# ----------------------------------------------------------------------
# record_activity
# ----------------------------------------------------------------------

def test_record_activity_creates_activity_event(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ulid = eng.record_activity(
        "Refactored auth module to use JWT",
        actor="agent", kind="edit",
    )
    ev = eng.events.get_by_ulid(ulid)
    assert ev.event_type == "activity"
    assert ev.tier == "working"  # default tier for activities
    assert ev.metadata.get("actor") == "agent"
    assert ev.metadata.get("activity_kind") == "edit"


def test_record_activity_defaults(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ulid = eng.record_activity("Did something")
    ev = eng.events.get_by_ulid(ulid)
    assert ev.metadata.get("actor") == "agent"
    assert ev.metadata.get("activity_kind") == "action"


# ----------------------------------------------------------------------
# recent_activity
# ----------------------------------------------------------------------

def test_recent_activity_returns_chronological(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_activity("first action")
    eng.record_activity("second action")
    eng.record_activity("third action")
    out = eng.recent_activity(minutes=60, limit=10)
    assert len(out) == 3
    # Newest first
    assert "third" in out[0]["content"]
    assert "first" in out[-1]["content"]


def test_recent_activity_filters_by_actor(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_activity("agent action", actor="agent")
    eng.record_activity("user action", actor="user")
    eng.record_activity("system action", actor="system")
    agent_only = eng.recent_activity(actor="agent")
    assert len(agent_only) == 1
    assert "agent" in agent_only[0]["content"]


def test_recent_activity_filters_by_kind(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_activity("a", kind="edit")
    eng.record_activity("b", kind="tool_call")
    eng.record_activity("c", kind="edit")
    edits = eng.recent_activity(kind="edit")
    assert len(edits) == 2


def test_recent_activity_window(tmp_pmb_home, tmp_workspace_dir):
    """Old activities outside the time window are excluded."""
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ulid = eng.record_activity("very old")
    # Manually backdate
    import sqlite3
    with sqlite3.connect(eng.workspace.db_path) as conn:
        conn.execute("UPDATE events SET timestamp = ? WHERE ulid = ?",
                     (time.time() - 3 * 3600, ulid))  # 3 hours ago
    eng.record_activity("fresh")
    out = eng.recent_activity(minutes=60)  # last hour only
    contents = [o["content"] for o in out]
    assert "fresh" in contents[0]
    assert all("old" not in c for c in contents)


# ----------------------------------------------------------------------
# what_just_happened
# ----------------------------------------------------------------------

def test_what_just_happened_returns_n_events(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    for i in range(7):
        eng.record_fact(f"fact {i}")
    out = eng.what_just_happened(n=3)
    assert len(out) == 3


def test_what_just_happened_includes_activities(tmp_pmb_home, tmp_workspace_dir):
    """Should mix activity + fact events in chronological order."""
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_fact("user fact 1")
    eng.record_activity("agent did X")
    eng.record_fact("user fact 2")
    eng.record_activity("agent did Y")
    out = eng.what_just_happened(n=4)
    types = [o["event_type"] for o in out]
    assert "activity" in types
    assert "fact" in types


# ----------------------------------------------------------------------
# session_timeline
# ----------------------------------------------------------------------

def test_session_timeline_chronological(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    sess_id = eng.session_tracker.touch().id
    eng.record_fact("first", session_id=sess_id)
    eng.record_activity("second", session_id=sess_id)
    eng.record_fact("third", session_id=sess_id)
    tl = eng.session_timeline(session_id=sess_id)
    # Oldest first
    assert len(tl) >= 3
    assert "first" in tl[0]["content"]


def test_session_timeline_excludes_other_sessions(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_fact("session A item", session_id="session-a")
    eng.record_fact("session B item", session_id="session-b")
    tl_a = eng.session_timeline(session_id="session-a")
    contents = [t["content"] for t in tl_a]
    assert any("session A" in c for c in contents)
    assert not any("session B" in c for c in contents)


# ----------------------------------------------------------------------
# integration: activity also searchable via recall (long-term too)
# ----------------------------------------------------------------------

def test_activity_searchable_via_recall(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={"recall.cache_size": 0},
    )
    eng.record_activity(
        "Implemented multi-algorithm fuzzy typo correction with Levenshtein",
        kind="edit",
    )
    pack = eng.recall("fuzzy typo correction", top_k=3)
    contents = [r.content for r in pack.results]
    assert any("fuzzy" in c.lower() for c in contents)
