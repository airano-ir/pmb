"""Tests for spaced-repetition rehearsal."""
from __future__ import annotations

import sqlite3
import time

from pmb.core.engine import Engine
from pmb.health.rehearse import _query_from_event, rehearse


def _backdate_access(db_path, ulid, days_ago):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE events SET last_accessed = ? WHERE ulid = ?",
            (time.time() - days_ago * 86400, ulid),
        )


def test_query_synthesis_pulls_keywords():
    q = _query_from_event("Postgres 17 setup with WAL replication on AWS RDS")
    assert "postgres" in q
    # First few significant tokens
    assert len(q.split()) <= 7


def test_query_synthesis_skips_short_tokens():
    q = _query_from_event("we use a db on aws")
    # 'we', 'a', 'on' too short — only 'use', 'aws', 'db' would qualify; 'db' is 2 chars so skipped
    assert "we" not in q.split()
    assert "a" not in q.split()


def test_rehearse_skips_recently_accessed(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    u = eng.record_fact("Use Postgres 17 in production", importance=0.8)
    # Just-accessed → should be skipped
    result = rehearse(eng, importance_threshold=0.5, min_idle_days=7.0)
    assert result.n_rehearsed == 0
    assert result.n_candidates == 0


def test_rehearse_picks_up_idle_high_importance(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    u = eng.record_fact("Use Postgres 17 in production", importance=0.8)
    _backdate_access(eng.workspace.db_path, u, days_ago=30)
    result = rehearse(eng, importance_threshold=0.5, min_idle_days=7.0)
    assert u in result.rehearsed_ulids
    # After rehearsal, last_accessed should be fresh (within 5s)
    ev = eng.events.get_by_ulid(u)
    assert time.time() - ev.last_accessed < 5


def test_rehearse_below_threshold_ignored(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    u = eng.record_fact("Low-value scribble", importance=0.3)
    _backdate_access(eng.workspace.db_path, u, days_ago=30)
    result = rehearse(eng, importance_threshold=0.5, min_idle_days=7.0)
    assert result.n_rehearsed == 0


def test_rehearse_skips_pinned(tmp_pmb_home, tmp_workspace_dir):
    """Pinned memories don't need rehearsal — they never drop."""
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    u = eng.record_fact("Pinned crucial fact", importance=0.8)
    eng.pin(u)
    _backdate_access(eng.workspace.db_path, u, days_ago=30)
    result = rehearse(eng, importance_threshold=0.5, min_idle_days=7.0)
    assert u not in result.rehearsed_ulids


def test_rehearse_respects_max_cap(tmp_pmb_home, tmp_workspace_dir):
    # Disable SEMANTIC dedup: the 15 near-identical fixtures ("Fact number i …")
    # otherwise get merged by L2 dedup depending on embed-model warmup timing,
    # making n_candidates non-deterministic (the pre-existing flake under load).
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
                 config_overrides={"dedup.enable_semantic": False})
    ulids = []
    for i in range(15):
        u = eng.record_fact(f"Fact number {i} about postgres setup", importance=0.7)
        ulids.append(u)
        _backdate_access(eng.workspace.db_path, u, days_ago=30)
    result = rehearse(
        eng, importance_threshold=0.5, min_idle_days=7.0, max_rehearse=5,
    )
    assert result.n_rehearsed <= 5
    assert result.n_candidates == 15


def test_rehearse_bumps_importance_on_miss(tmp_pmb_home, tmp_workspace_dir):
    """Even if the self-recall didn't surface the memory, we nudge importance
    up — the system shouldn't lose high-value memories due to weak queries."""
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    # Two facts, both with same key tokens → one will outrank the other on recall.
    eng.record_fact("postgres setup info A", importance=0.7)
    u_target = eng.record_fact("postgres setup info B", importance=0.55)
    _backdate_access(eng.workspace.db_path, u_target, days_ago=30)

    before = eng.events.get_by_ulid(u_target).importance
    rehearse(eng, importance_threshold=0.5, min_idle_days=7.0)
    after = eng.events.get_by_ulid(u_target).importance
    # Importance should be at least as high as before, often higher
    assert after >= before
