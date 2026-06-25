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


def test_lesson_impact_refuses_to_trust_a_fluke(isolated_engine):
    # The original seed has n=1 per lesson and only 3 outcome turns. Every
    # verdict must be "insufficient", the whole report flagged untrustworthy,
    # and the CIs genuinely wide - a 1-sample result must NEVER read as a real
    # useful/harmful effect (that was the trust gap).
    _seed(isolated_engine)
    r = lesson_impact(isolated_engine, window_days=30)
    assert r["signal_sufficiency"] == "insufficient"
    assert r["trustworthy"] is False
    assert r["n_confident"] == 0
    assert r["caveat"]                                   # confounding caveat present
    for L in r["lessons"]:
        assert L["verdict"] == "insufficient"            # n=1 < min_n
        assert 0.0 <= L["ci_low"] <= L["ci_high"] <= 1.0
        assert L["ci_high"] - L["ci_low"] > 0.3          # no false precision at n=1
        # follow flag is NULL in this seed -> no causal arm at all
        assert L["causal_verdict"] == "insufficient"
        assert L["n_followed"] == 0 and L["n_ignored"] == 0


def test_lesson_impact_can_fire_a_confident_verdict(isolated_engine):
    # Counter to the test above: when a lesson HAS enough turns and its success
    # CI clears the baseline, the verdict actually fires "useful". The change
    # suppresses noise without muting real signal.
    db, ws, now = (isolated_engine.workspace.db_path,
                   isolated_engine.workspace.id, time.time())
    with sqlite3.connect(db) as c:
        ensure_agent_actions_table(c)
        c.execute(
            "CREATE TABLE IF NOT EXISTS lesson_surfaces (id INTEGER PRIMARY KEY "
            "AUTOINCREMENT, workspace_id TEXT, lesson_ulid TEXT, query TEXT, "
            "source TEXT, surfaced_at REAL, session_id TEXT, followed INTEGER)"
        )
        c.execute("INSERT INTO lesson_surfaces (id,workspace_id,lesson_ulid,query,"
                  "source,surfaced_at) VALUES (1,?,'L_useful','','test',?)", (ws, now))

        def act(sess, status, sids):
            c.execute(
                "INSERT INTO agent_actions (workspace_id,timestamp,tool,target,"
                "status,significant,session_id,surface_ids) VALUES (?,?,?,?,?,?,?,?)",
                (ws, now, "Bash", "pytest", status, 1, sess, sids),
            )
        for i in range(6):                       # baseline ~50%
            act(f"b{i}", "ok" if i % 2 == 0 else "error", "")
        for i in range(6):                       # lesson: 6/6 pass
            act(f"u{i}", "ok", "1")
        c.commit()
    r = lesson_impact(isolated_engine, window_days=30, min_n=5)
    by = {L["lesson_ulid"]: L for L in r["lessons"]}
    assert by["L_useful"]["verdict"] == "useful"
    assert by["L_useful"]["ci_low"] > r["baseline_success_rate"]
    assert r["n_confident"] >= 1


def test_lesson_impact_causal_verdict_separates_followed_from_ignored(isolated_engine):
    # Same lesson: FOLLOWED -> passes, IGNORED -> failures. The within-lesson
    # causal read must fire "helps". Unlike baseline lift, this holds the
    # surfacing trigger fixed (both arms are "this lesson was relevant").
    db, ws, now = (isolated_engine.workspace.db_path,
                   isolated_engine.workspace.id, time.time())
    with sqlite3.connect(db) as c:
        ensure_agent_actions_table(c)
        c.execute(
            "CREATE TABLE IF NOT EXISTS lesson_surfaces (id INTEGER PRIMARY KEY "
            "AUTOINCREMENT, workspace_id TEXT, lesson_ulid TEXT, query TEXT, "
            "source TEXT, surfaced_at REAL, session_id TEXT, followed INTEGER)"
        )
        state = {"sid": 0}

        def surface(followed):
            state["sid"] += 1
            c.execute(
                "INSERT INTO lesson_surfaces (id,workspace_id,lesson_ulid,query,"
                "source,surfaced_at,followed) VALUES (?,?,?,'','test',?,?)",
                (state["sid"], ws, "L_c", now, followed),
            )
            return state["sid"]

        def act(sess, status, sids):
            c.execute(
                "INSERT INTO agent_actions (workspace_id,timestamp,tool,target,"
                "status,significant,session_id,surface_ids) VALUES (?,?,?,?,?,?,?,?)",
                (ws, now, "Bash", "pytest", status, 1, sess, sids),
            )
        for i in range(6):                       # followed -> pass
            act(f"f{i}", "ok", str(surface(1)))
        for i in range(6):                       # ignored -> fail
            act(f"g{i}", "error", str(surface(0)))
        c.commit()
    r = lesson_impact(isolated_engine, window_days=30, min_n=5)
    Lc = {L["lesson_ulid"]: L for L in r["lessons"]}["L_c"]
    assert Lc["n_followed"] == 6 and Lc["n_ignored"] == 6
    assert Lc["followed_success_rate"] == 1.0
    assert Lc["ignored_success_rate"] == 0.0
    assert Lc["causal_verdict"] == "helps"
    assert r["n_causal"] >= 1
