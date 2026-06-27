"""The three automatic write paths, and the division of labor between them.

The contract the user relies on (and a refactor must not break):

  - auto-recall  (UserPromptSubmit / prepare-context)  writes ALMOST NOTHING:
    on a normal message it appends NO durable events; its only write is a
    lesson-surface MARKER (the surface_id the agent later confirms).
  - track-action (PostToolUse)  journals every tool call into `agent_actions`,
    a side table - never `events`.
  - autowrite    (Stop)  reads that journal and, when the agent did not record
    its own work, synthesizes exactly ONE durable activity event tagged
    source=autowrite.

These tests pin that split so auto-recall can't silently become a writer and
the journal -> summary handoff can't regress.
"""
from __future__ import annotations

import sqlite3

from pmb.core.ambient_log import insert_agent_action
from pmb.core.engine import Engine
from pmb.hooks.autowrite import run_autowrite


def _engine(ws, home):
    return Engine(cwd=ws, pmb_home=home, config_overrides={"recall.cache_size": 0})


def _count(eng, table, where="", params=()):
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        try:
            return c.execute(
                f"SELECT COUNT(*) FROM {table} {where}", params).fetchone()[0]
        except sqlite3.OperationalError:
            return 0  # table not created yet → nothing written there


# ── auto-recall: read-mostly; its only write is a surface marker ───────────

def test_auto_recall_does_not_append_durable_events(tmp_pmb_home, tmp_workspace_dir):
    from pmb.hooks.auto_recall import run_auto_context
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.record_fact("Always use pnpm, never npm in this repo.",
                    metadata={"kind": "lesson", "source": "lesson"})
    before = _count(eng, "events")
    res = run_auto_context(eng, "which package manager should I use in this repo?")
    assert res.correction is None          # not a correction → no draft written
    assert _count(eng, "events") == before, "auto-recall must not write durable memory"


def test_auto_recall_write_is_a_surface_marker_not_an_event(
    tmp_pmb_home, tmp_workspace_dir,
):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    ul = eng.record_fact("Use ruff, not flake8, for linting here.",
                         metadata={"kind": "lesson", "source": "lesson"})
    ev_before = _count(eng, "events")
    sf_before = _count(eng, "lesson_surfaces")
    surfaced = eng._log_lesson_surfaces(
        [{"ulid": ul, "content": "Use ruff, not flake8"}],
        query="what linter do we use", source="hook.auto-recall",
    )
    assert surfaced and surfaced[0].get("surface_id")
    assert _count(eng, "lesson_surfaces") == sf_before + 1   # a marker was written
    assert _count(eng, "events") == ev_before               # but NOT a durable event


# ── track-action: journals to agent_actions, never events ──────────────────

def test_track_action_journals_to_side_table_not_events(
    tmp_pmb_home, tmp_workspace_dir,
):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    ev_before = _count(eng, "events")
    insert_agent_action(eng.workspace.db_path, eng.workspace.id,
                        tool="Edit", target="auth.py", status="ok")
    assert _count(eng, "agent_actions") == 1
    assert _count(eng, "events") == ev_before, "the journal must not touch durable memory"


# ── autowrite: reads the journal → writes ONE durable event ────────────────

def test_autowrite_synthesizes_one_event_from_the_journal(
    tmp_pmb_home, tmp_workspace_dir,
):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    # track-action has journaled a real turn (edits + a passing test run)
    insert_agent_action(eng.workspace.db_path, eng.workspace.id,
                        tool="Edit", target="src/auth.py", status="ok")
    insert_agent_action(eng.workspace.db_path, eng.workspace.id,
                        tool="Write", target="src/api.py", status="ok")
    insert_agent_action(eng.workspace.db_path, eng.workspace.id,
                        tool="Bash", target="pytest -q", status="ok",
                        command="pytest -q")
    ev_before = _count(eng, "events")
    res = run_autowrite(eng, window_minutes=60.0, min_actions=2)
    assert res.wrote, res.skipped_reason
    assert _count(eng, "events") == ev_before + 1            # exactly one summary
    auto = _count(eng, "events",
                  "WHERE json_extract(metadata_json,'$.source')=?", ("autowrite",))
    assert auto == 1


def test_autowrite_stays_silent_when_agent_recorded_its_own_work(
    tmp_pmb_home, tmp_workspace_dir, monkeypatch,
):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    for t, f in (("Edit", "a.py"), ("Write", "b.py")):
        insert_agent_action(eng.workspace.db_path, eng.workspace.id,
                            tool=t, target=f, status="ok")
    # the agent journaled its own work this turn (record_* via MCP) → ambient
    # must defer to the agent's richer summary and stay silent.
    monkeypatch.setattr(eng, "agent_wrote_recently", lambda minutes=30.0: True)
    ev_before = _count(eng, "events")
    res = run_autowrite(eng, window_minutes=60.0, min_actions=2)
    assert not res.wrote
    assert "recorded its own work" in (res.skipped_reason or "")
    assert _count(eng, "events") == ev_before                # nothing synthesized
