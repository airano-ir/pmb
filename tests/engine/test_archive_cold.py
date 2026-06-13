"""T6: `decay --archive-cold` — time-based forgetting of cold low-value facts.

Decay only down-weights; this archives facts/activities that are old AND never
recalled AND low-value, while protecting accessed / pinned / keyed / lessons /
goals. Archive-only + dry-run by default.
"""
from __future__ import annotations

import json
import sqlite3
import time

import pytest

from pmb.core.engine import Engine
from pmb.core.events import Event

NOW = time.time()
OLD = NOW - 200 * 86400      # 200 days
RECENT = NOW - 10 * 86400    # 10 days






@pytest.fixture
def eng(tmp_workspace_dir, tmp_pmb_home):
    return Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
                  config_overrides={"recall.cache_size": 0})


def _seed(eng, content, *, etype="fact", imp=0.1, ac=0, ts=OLD, meta=None):
    ev = Event(workspace_id=eng.workspace.id, event_type=etype, content=content,
               metadata=meta or {}, importance=imp, access_count=ac, timestamp=ts)
    return eng.events.append(ev).ulid


def _is_active(eng, ulid):
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        row = c.execute("SELECT archived_at FROM events WHERE ulid=?",
                        (ulid,)).fetchone()
    return row is not None and row[0] is None


def _meta(eng, ulid):
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        row = c.execute("SELECT metadata_json FROM events WHERE ulid=?",
                        (ulid,)).fetchone()
    return json.loads(row[0] or "{}") if row else {}


def test_cold_fact_is_archived(eng):
    u = _seed(eng, "an old trivial scratch note nobody ever recalled")
    res = eng.archive_cold(dry_run=False)
    assert u in {c["ulid"] for c in res["candidates"]}
    assert not _is_active(eng, u)
    assert _meta(eng, u).get("archived_reason") == "decay_cold"


def test_protected_events_survive(eng):
    accessed = _seed(eng, "recalled before", ac=3)
    pinned = _seed(eng, "pinned note", imp=1.0)
    recent = _seed(eng, "fresh note", ts=RECENT)
    important = _seed(eng, "important note", imp=0.8)
    keyed = _seed(eng, "user city: Tampa", meta={"keyed_fact_key": "user::city"})
    lesson = _seed(eng, "always use pnpm not npm", meta={"kind": "lesson"})
    goal = _seed(eng, "ship v1", etype="goal", meta={"goal_status": "pending"})

    eng.archive_cold(dry_run=False)
    for u in (accessed, pinned, recent, important, keyed, lesson, goal):
        assert _is_active(eng, u), u


def test_dry_run_changes_nothing(eng):
    u = _seed(eng, "old trivial note")
    res = eng.archive_cold(dry_run=True)
    assert res["dry_run"] is True
    assert u in {c["ulid"] for c in res["candidates"]}
    assert _is_active(eng, u)


def test_defaults_come_from_config(eng):
    res = eng.archive_cold(dry_run=True)
    assert res["days"] == 90
    assert res["max_importance"] == 0.25


def test_explicit_thresholds_override(eng):
    # a 30-day-old, 0.4-importance fact is NOT cold by default, but IS under
    # relaxed thresholds.
    u = _seed(eng, "semi-recent mid-value note", imp=0.4, ts=NOW - 30 * 86400)
    assert eng.archive_cold(dry_run=True)["n"] == 0  # default 90d / 0.25
    res = eng.archive_cold(days=20, max_importance=0.5, dry_run=True)
    assert u in {c["ulid"] for c in res["candidates"]}
