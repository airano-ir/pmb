"""Lessons / decisions must not silently decay out of recall ranking.

A critical-but-rarely-touched lesson sliding below `find_lessons`' top-K
is the mechanic by which the same mistake repeats: the rule still EXISTS,
but auto-recall stops surfacing it. The decay floor in
`pmb.signals.decay.LESSON_DECISION_IMPORTANCE_FLOOR` caps the slide.

Pinned events (importance >= 0.99) keep their own behaviour (no decay
applied at all) and ordinary working/episodic events are not affected.
"""
from __future__ import annotations

import sqlite3
import time

from pmb.core.engine import Engine
from pmb.signals.decay import (
    LESSON_DECISION_IMPORTANCE_FLOOR,
    apply_decay,
)


def _engine(ws, home):
    return Engine(cwd=ws, pmb_home=home, config_overrides={"recall.cache_size": 0})


def _importance(eng, ulid) -> float:
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        r = c.execute("SELECT importance FROM events WHERE ulid=?", (ulid,)).fetchone()
    return float(r[0]) if r else 0.0


def _backdate(eng, ulid, days):
    ts = time.time() - days * 86400.0
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        c.execute("UPDATE events SET timestamp=?, last_accessed=? WHERE ulid=?",
                  (ts, ts, ulid))
        c.commit()


def _force_importance(eng, ulid, imp):
    """Skip the public API to seed a low starting importance directly."""
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        c.execute("UPDATE events SET importance=? WHERE ulid=?", (imp, ulid))
        c.commit()


def test_lesson_does_not_decay_below_floor(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    u = eng.record_fact(
        "always use ruff, never flake8",
        metadata={"kind": "lesson", "source": "lesson"},
    )
    # Seed at the floor and age 365 days untouched. Without the floor, the
    # episodic-tier factor (0.985 ** 365 ≈ 0.0038) would crush it to zero.
    _force_importance(eng, u, LESSON_DECISION_IMPORTANCE_FLOOR)
    _backdate(eng, u, 365)
    apply_decay(eng, days_since_last_decay=365)
    assert _importance(eng, u) >= LESSON_DECISION_IMPORTANCE_FLOOR


def test_decision_does_not_decay_below_floor(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    u = eng.record_activity("Chose Postgres 17 over Mongo for JSONB",
                            kind="decision")
    _force_importance(eng, u, LESSON_DECISION_IMPORTANCE_FLOOR)
    _backdate(eng, u, 365)
    apply_decay(eng, days_since_last_decay=365)
    assert _importance(eng, u) >= LESSON_DECISION_IMPORTANCE_FLOOR


def test_ordinary_qa_event_still_decays(tmp_pmb_home, tmp_workspace_dir):
    """Floor must be type-targeted - generic QA / activity must STILL decay
    so the working-tier sweep can archive stale chatter."""
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    u = eng.record_activity("opened the config file", kind="completed")
    _force_importance(eng, u, 0.5)
    _backdate(eng, u, 14)
    apply_decay(eng, days_since_last_decay=14)
    # 0.5 * 0.70**14 ≈ 0.0035 (well below the floor) — confirms the floor
    # is NOT applied to working-tier kind=completed events.
    assert _importance(eng, u) < LESSON_DECISION_IMPORTANCE_FLOOR


def test_pinned_lesson_unchanged_by_decay(tmp_pmb_home, tmp_workspace_dir):
    """Pinning (importance>=0.99) is the explicit "never touch this" path
    and predates the floor; it must keep that contract."""
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    u = eng.record_fact(
        "never amend a pushed commit",
        metadata={"kind": "lesson", "source": "lesson"},
    )
    _force_importance(eng, u, 1.0)  # pinned
    _backdate(eng, u, 365)
    apply_decay(eng, days_since_last_decay=365)
    assert _importance(eng, u) == 1.0
