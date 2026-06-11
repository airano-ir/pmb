"""R1: lesson-surface logging must not double-count.

A single prepare()/recall() logs the same lesson on two paths
(project_overview + find_lessons), and the same rule re-surfaces minutes later.
Before R1 each minted a NEW surface_id, inflating the adherence denominator so
the dashboard measured logging artifacts, not what the agent saw. Now a surface
is deduped per (lesson, session, hour): the existing id is reused.
"""
from __future__ import annotations

import sqlite3

from pmb.core.engine import Engine


def _engine(ws, home):
    return Engine(cwd=ws, pmb_home=home, config_overrides={"recall.cache_size": 0})


def _n_surfaces(eng) -> int:
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        return c.execute("SELECT COUNT(*) FROM lesson_surfaces").fetchone()[0]


def test_same_lesson_same_session_logged_once(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    u = eng.record_fact("this repo uses pnpm, never npm",
                        metadata={"kind": "lesson", "source": "lesson"})
    L = {"ulid": u, "content": "this repo uses pnpm, never npm"}
    # two paths in ONE turn (project_overview + find_lessons) + a re-surface
    eng._log_lesson_surfaces([dict(L)], query="q1", source="recall", session_id="s1")
    r2 = eng._log_lesson_surfaces([dict(L)], query="q1",
                                  source="recall.project_context", session_id="s1")
    r3 = eng._log_lesson_surfaces([dict(L)], query="q2", source="recall", session_id="s1")
    assert _n_surfaces(eng) == 1
    # all calls report the SAME surface_id (one rule, one id to mark followed)
    assert r2[0]["surface_id"] == r3[0]["surface_id"]


def test_different_session_logs_separately(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    u = eng.record_fact("always run the linter before pushing",
                        metadata={"kind": "lesson", "source": "lesson"})
    L = {"ulid": u, "content": "always run the linter before pushing"}
    eng._log_lesson_surfaces([dict(L)], query="q", source="recall", session_id="s1")
    eng._log_lesson_surfaces([dict(L)], query="q", source="recall", session_id="s2")
    assert _n_surfaces(eng) == 2  # distinct sessions are distinct surfaces


def test_two_lessons_one_call_two_rows(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    a = eng.record_fact("use ruff not flake8", metadata={"kind": "lesson"})
    b = eng.record_fact("commit messages in imperative", metadata={"kind": "lesson"})
    eng._log_lesson_surfaces(
        [{"ulid": a, "content": "x"}, {"ulid": b, "content": "y"}],
        query="q", source="recall", session_id="s1")
    assert _n_surfaces(eng) == 2  # two distinct lessons, two surfaces
