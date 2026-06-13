"""End-to-end tests for ambient memory (auto-write) on a real engine.

The full loop the two new hooks form:

    PostToolUse  → record_agent_action  (observe what the agent did)
    Stop         → run_autowrite:
                     • agent journaled itself this turn  → STAY SILENT
                     • else, enough significant actions  → synthesize +
                       record_activity(source=autowrite)

Plus the significance filter and the user-control escape hatch
(forget_auto_written). Pure SQL + template — no model, fast.
"""

from __future__ import annotations

import sqlite3
import time

import pytest


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("PMB_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PMB_WORKSPACE", "ambient_e2e")
    from pmb.core.engine import Engine
    return Engine()


def _seed_turn(engine):
    """Agent edits code, runs tests, commits — and a noisy read."""
    engine.record_agent_action("Edit", "src/auth.py", "ok")
    engine.record_agent_action("Write", "tests/test_auth.py", "ok")
    engine.record_agent_action("Bash", "pytest tests/", "ok", command="pytest tests/")
    engine.record_agent_action("Bash", "git commit -m fix", "ok",
                               command="git commit -m fix")
    engine.record_agent_action("Read", "README.md", "ok")  # noise


# ─── significance filter ────────────────────────────────────────────────


def test_significance_filter(engine):
    assert engine.is_significant_action("Edit", "a.py")
    assert engine.is_significant_action("Write", "b.py")
    assert not engine.is_significant_action("Read", "a.py")
    assert not engine.is_significant_action("Grep", "x")
    assert engine.is_significant_action("Bash", command="git commit -m x")
    assert engine.is_significant_action("Bash", command="pytest")
    assert not engine.is_significant_action("Bash", command="ls -la")
    assert not engine.is_significant_action("Bash", command="git status")


# ─── observe → read ─────────────────────────────────────────────────────


def test_actions_logged_and_filtered(engine):
    _seed_turn(engine)
    allr = engine.recent_agent_actions(minutes=60)
    sig = engine.recent_agent_actions(minutes=60, significant_only=True)
    assert len(allr) == 5
    assert len(sig) == 4  # the Read is dropped


def test_action_links_active_lesson_surface(engine):
    lesson_ulid = engine.record_fact(
        "Use pnpm build when changing src/auth.py",
        metadata={"kind": "lesson", "source": "lesson"},
    )
    surfaced = [{"ulid": lesson_ulid, "content": "Use pnpm build"}]
    engine._log_lesson_surfaces(surfaced, query="change auth", source="hook.prepare")
    sid = surfaced[0]["surface_id"]

    engine.record_agent_action("Edit", "src/auth.py", "ok")
    actions = engine.recent_agent_actions(minutes=60, significant_only=True)

    assert actions
    assert str(sid) in actions[0]["surface_ids"].split(",")


# ─── auto-write happens when the agent stays silent ─────────────────────


def test_autowrite_journals_when_agent_silent(engine):
    from pmb.hooks import run_autowrite
    _seed_turn(engine)
    res = run_autowrite(engine, window_minutes=60, min_actions=2,
                        synthesizer="template", apply=True)
    assert res.wrote is True
    assert res.synthesizer == "template"
    assert "auth.py" in res.summary
    assert "committed" in res.summary
    # It landed as an activity tagged source=autowrite.
    acts = engine.recent_activity(minutes=60, limit=10)
    assert any("auth.py" in (a.get("content") or "") for a in acts)


# ─── coordination: silent if the agent journaled itself ─────────────────


def test_autowrite_silent_when_agent_recorded(engine):
    from pmb.hooks import run_autowrite
    _seed_turn(engine)
    # Simulate the agent having called record_batch this turn (mcp_calls row).
    with sqlite3.connect(engine.workspace.db_path) as c:
        c.execute(
            "CREATE TABLE IF NOT EXISTS mcp_calls (id INTEGER PRIMARY KEY, "
            "workspace_id TEXT, tool_name TEXT, timestamp REAL, duration_ms REAL,"
            " success INTEGER, error TEXT, args_size INTEGER)"
        )
        c.execute("INSERT INTO mcp_calls (workspace_id, tool_name, timestamp) "
                  "VALUES (?,?,?)", (engine.workspace.id, "record_batch", time.time()))
        c.commit()
    res = run_autowrite(engine, window_minutes=60, min_actions=2, apply=True)
    assert res.wrote is False
    assert "recorded its own work" in (res.skipped_reason or "")


def test_agent_wrote_recently_detects_record(engine):
    assert engine.agent_wrote_recently(60) is False
    with sqlite3.connect(engine.workspace.db_path) as c:
        c.execute(
            "CREATE TABLE IF NOT EXISTS mcp_calls (id INTEGER PRIMARY KEY, "
            "workspace_id TEXT, tool_name TEXT, timestamp REAL, duration_ms REAL,"
            " success INTEGER, error TEXT, args_size INTEGER)"
        )
        c.execute("INSERT INTO mcp_calls (workspace_id, tool_name, timestamp) "
                  "VALUES (?,?,?)", (engine.workspace.id, "record_fact", time.time()))
        c.commit()
    assert engine.agent_wrote_recently(60) is True


# ─── threshold + dry-run ────────────────────────────────────────────────


def test_autowrite_below_threshold_stays_silent(engine):
    from pmb.hooks import run_autowrite
    engine.record_agent_action("Edit", "only.py", "ok")  # 1 significant
    res = run_autowrite(engine, window_minutes=60, min_actions=2, apply=True)
    assert res.wrote is False
    assert "significant action" in (res.skipped_reason or "")


def test_autowrite_dry_run_writes_nothing(engine):
    from pmb.hooks import run_autowrite
    _seed_turn(engine)
    res = run_autowrite(engine, window_minutes=60, min_actions=2, apply=False)
    assert res.wrote is True  # would-write
    assert res.summary
    # but nothing actually landed
    acts = engine.recent_activity(minutes=60, limit=10)
    assert not any("auth.py" in (a.get("content") or "") for a in acts)


# ─── user control: forget auto-written ──────────────────────────────────


def test_autowrite_gate(engine):
    """The cheap gate the Stop hook uses to decide whether to spawn a
    background LLM worker — no synthesis, just yes/no."""
    from pmb.hooks import autowrite_gate
    # nothing yet
    assert autowrite_gate(engine, window_minutes=60, min_actions=2) is not None
    _seed_turn(engine)
    # now qualifies
    assert autowrite_gate(engine, window_minutes=60, min_actions=2) is None
    # higher threshold → doesn't qualify
    reason = autowrite_gate(engine, window_minutes=60, min_actions=10)
    assert reason and "significant action" in reason


def test_llm_synthesizer_falls_back_to_template(engine):
    """When the LLM backend is unavailable, the worker path still records —
    via the template fallback — so the journal entry is never lost."""
    from pmb.hooks import run_autowrite
    _seed_turn(engine)
    # llm:ollama with ollama almost certainly absent in CI → fallback.
    res = run_autowrite(engine, window_minutes=60, min_actions=2,
                        synthesizer="llm:ollama", llm_timeout=3.0, apply=True)
    assert res.wrote is True
    assert res.synthesizer == "template"   # fell back
    assert res.summary


def test_forget_auto_written_removes_only_ambient(engine):
    from pmb.hooks import run_autowrite
    # An explicit user/agent activity that must SURVIVE.
    engine.record_activity("Manually noted: shipped the release", kind="completed")
    # An ambient auto-write.
    _seed_turn(engine)
    run_autowrite(engine, window_minutes=60, min_actions=2, apply=True)

    before = engine.recent_activity(minutes=60, limit=20)
    assert any("auth.py" in (a.get("content") or "") for a in before)

    n = engine.forget_auto_written()
    assert n >= 1
    after = engine.recent_activity(minutes=60, limit=20)
    # ambient gone...
    assert not any("auth.py" in (a.get("content") or "") for a in after)
    # ...explicit note survives
    assert any("shipped the release" in (a.get("content") or "") for a in after)
