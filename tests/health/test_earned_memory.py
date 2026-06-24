"""Tests for Earned Memory - the surface -> outcome lesson-impact join."""
from __future__ import annotations

import sqlite3
import time

from pmb.core.ambient_log import ensure_agent_actions_table
from pmb.health.earned_memory import lesson_impact


def _seed(eng):
    db, ws, now = eng.workspace.db_path, eng.workspace.id, time.time()
    with sqlite3.connect(db) as c:
        ensure_agent_actions_table(c)
        c.execute(
            "CREATE TABLE IF NOT EXISTS lesson_surfaces (id INTEGER PRIMARY KEY "
            "AUTOINCREMENT, workspace_id TEXT, lesson_ulid TEXT, query TEXT, "
            "source TEXT, surfaced_at REAL, session_id TEXT, followed INTEGER)"
        )
        c.execute("INSERT INTO lesson_surfaces (id,workspace_id,lesson_ulid,query,"
                  "source,surfaced_at) VALUES (1,?,'L_good','','test',?)", (ws, now))
        c.execute("INSERT INTO lesson_surfaces (id,workspace_id,lesson_ulid,query,"
                  "source,surfaced_at) VALUES (2,?,'L_bad','','test',?)", (ws, now))

        def act(sess, status, sids):
            c.execute(
                "INSERT INTO agent_actions (workspace_id,timestamp,tool,target,"
                "status,significant,session_id,surface_ids) VALUES (?,?,?,?,?,?,?,?)",
                (ws, now, "Bash", "pytest", status, 1, sess, sids),
            )
        # Turn A: L_good active, tests pass -> success
        act("sA", "ok", "1")
        # Turn B: L_bad active, tests fail -> failure
        act("sB", "error", "2")
        # Baseline turn: no lesson active, tests pass
        act("sC", "ok", "")
        c.commit()


def test_lesson_impact_lift(isolated_engine):
    _seed(isolated_engine)
    r = lesson_impact(isolated_engine, window_days=30)
    assert r["n_outcome_turns"] == 3
    assert r["n_baseline_turns"] == 1
    assert r["baseline_success_rate"] == 1.0  # the one no-lesson turn passed

    by = {L["lesson_ulid"]: L for L in r["lessons"]}
    assert by["L_good"]["success_rate"] == 1.0
    assert by["L_bad"]["success_rate"] == 0.0
    # L_bad precedes a failure vs a passing baseline -> negative lift (harmful)
    assert by["L_bad"]["lift"] < 0
    # worst lift first
    assert r["lessons"][0]["lesson_ulid"] == "L_bad"


def test_lesson_impact_empty(isolated_engine):
    # No agent_actions at all -> no outcome turns, no lessons, no crash.
    r = lesson_impact(isolated_engine, window_days=30)
    assert r["n_outcome_turns"] == 0
    assert r["lessons"] == []
