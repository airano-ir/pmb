"""S5: project_overview / _known_projects memoized by write-generation, and
session_brief / active_arcs_for_project keep working after the SQL-scope and
N+1 fixes.

The cache identity check (`a is b`) proves the second call was served from the
memo without recomputing; a write bumps the recall-cache generation counter,
which must invalidate the memo (`c is not a`).
"""
from __future__ import annotations

from pmb.core.engine import Engine
from pmb.hooks.auto_recall import _known_projects


def _engine(ws, home):
    return Engine(cwd=ws, pmb_home=home, config_overrides={"recall.cache_size": 0})


def test_project_overview_cached_then_invalidated(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    a = eng.project_overview("acme")
    b = eng.project_overview("acme")
    assert a is b, "second project_overview must be served from the memo"
    # a write bumps the generation -> memo is stale -> recompute
    eng.record_fact("acme runs on postgres", metadata={"kind": "fact"})
    c = eng.project_overview("acme")
    assert c is not a, "write must invalidate the project_overview memo"
    # different args key separately
    d = eng.project_overview("acme", max_per_section=3)
    assert d is not c


def test_known_projects_cached_then_invalidated(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    a = _known_projects(eng)
    b = _known_projects(eng)
    assert a is b, "second _known_projects must be served from the memo"
    eng.record_fact("some new fact", metadata={"kind": "fact"})
    c = _known_projects(eng)
    assert c is not a, "write must invalidate the _known_projects memo"


def test_active_arcs_batched_query_ok(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    # No arcs yet -> empty, but exercises the single batched arc_events fetch
    # path (S5.3) without the per-arc N+1.
    assert eng.active_arcs_for_project("acme") == []


def test_session_brief_sql_scope_ok(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.record_fact("did the thing", metadata={"kind": "completed"})
    brief = eng.session_brief()
    assert isinstance(brief, dict)
