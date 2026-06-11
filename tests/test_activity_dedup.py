"""0.2 (former E6): exact-duplicate suppression for `activity` events.

`record_activity` had NO dedup — agents re-log the same action seconds apart
and live workspaces accumulated identical activities ~60s apart. Within
`write.dedup_window_h` (default 24h, 0 = off) an exact-normalized duplicate of
an ACTIVE activity bumps the existing row instead of inserting a twin, and
returns the existing ULID (with `{deduped: true}` surfaced by record_batch).

Facts/goals keep their own L1+L2 dedup path; these tests cover the new activity
window only.
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from pmb.core.engine import Engine


def _engine(ws, home, **over):
    cfg = {"recall.cache_size": 0}
    cfg.update(over)
    return Engine(cwd=ws, pmb_home=home, config_overrides=cfg)


def _count_activities(eng) -> int:
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        return c.execute(
            "SELECT COUNT(*) FROM events "
            "WHERE event_type='activity' AND archived_at IS NULL"
        ).fetchone()[0]


def _access_count(eng, ulid) -> int:
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        r = c.execute("SELECT access_count FROM events WHERE ulid=?", (ulid,)).fetchone()
    return int(r[0] or 0)


# ── core plan tests ─────────────────────────────────────────────────────────

def test_same_activity_twice_within_window_one_row(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)  # default window 24h
    u1 = eng.record_activity("benchmarked Neo4j AuraDB latency", kind="completed")
    u2 = eng.record_activity("benchmarked Neo4j AuraDB latency", kind="completed")
    assert u1 == u2, "second identical activity should return the canonical ULID"
    assert _count_activities(eng) == 1


def test_window_zero_disables_dedup_two_rows(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home, **{"write.dedup_window_h": 0})
    u1 = eng.record_activity("explored Memgraph", kind="completed")
    u2 = eng.record_activity("explored Memgraph", kind="completed")
    assert u1 != u2
    assert _count_activities(eng) == 2


def test_dedup_is_workspace_scoped(tmp_pmb_home):
    # two independent workspaces, each gets its own row for the same text
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        eng_a = _engine(Path(a), tmp_pmb_home)
        eng_b = _engine(Path(b), tmp_pmb_home)
        assert eng_a.workspace.id != eng_b.workspace.id
        ua = eng_a.record_activity("added pytest-benchmark", kind="completed")
        ub = eng_b.record_activity("added pytest-benchmark", kind="completed")
        assert ua != ub
        assert _count_activities(eng_a) == 1
        assert _count_activities(eng_b) == 1


# ── behavior details ────────────────────────────────────────────────────────

def test_different_content_not_merged(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.record_activity("explored Memgraph", kind="completed")
    eng.record_activity("explored Neo4j", kind="completed")
    assert _count_activities(eng) == 2


def test_dup_bumps_access_count(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    u1 = eng.record_activity("ran the full suite", kind="completed")
    before = _access_count(eng, u1)
    eng.record_activity("ran the full suite", kind="completed")
    after = _access_count(eng, u1)
    assert after == before + 1


def test_normalization_collapses_whitespace_and_case(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    u1 = eng.record_activity("Deployed the Daemon", kind="completed")
    u2 = eng.record_activity("deployed   the daemon", kind="completed")
    assert u1 == u2
    assert _count_activities(eng) == 1


# ── record_batch surfaces the deduped flag ──────────────────────────────────

def test_record_batch_surfaces_deduped_flag(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.record_activity("merged the lang packs PR", kind="completed")
    res = eng.record_batch(items=[
        {"type": "activity", "content": "merged the lang packs PR", "kind": "completed"},
    ])
    act = [r for r in res["results"] if r.get("type") == "activity"][0]
    assert act.get("deduped") is True
    assert _count_activities(eng) == 1


def test_record_batch_first_activity_not_flagged(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    res = eng.record_batch(items=[
        {"type": "activity", "content": "first time only", "kind": "completed"},
    ])
    act = [r for r in res["results"] if r.get("type") == "activity"][0]
    assert "deduped" not in act
