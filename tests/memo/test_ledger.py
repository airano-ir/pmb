"""Tests for the Memory Delta ledger + delta-render.

Two layers:
  * ledger primitives (register / lookup / rehydrate / expired_since)
  * delta-render: collapses unchanged lessons to compact handle lines, adds
    "Still active" / "Expired" summary lines.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field

from pmb.memo.ledger import (
    _KIND_LESSON,
    active_handles,
    ensure_table,
    expired_since,
    lookup,
    register,
    rehydrate,
)


@dataclass
class _Res:
    """Stand-in for AutoContextResult - only fields the delta renderer reads."""
    lessons: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    recall_hits: list[dict] = field(default_factory=list)
    project: dict | None = None


def test_register_new_then_unchanged(tmp_path):
    db = tmp_path / "x.sqlite"
    with sqlite3.connect(db) as c:
        ensure_table(c)
        r1 = register(c, "ws", "S1", _KIND_LESSON, "42", "use pnpm not npm")
        assert r1["status"] == "new"
        assert r1["handle"] == "M01"
        r2 = register(c, "ws", "S1", _KIND_LESSON, "42", "use pnpm not npm")
        assert r2["status"] == "unchanged"
        assert r2["handle"] == "M01"


def test_register_updated_keeps_handle(tmp_path):
    db = tmp_path / "x.sqlite"
    with sqlite3.connect(db) as c:
        ensure_table(c)
        register(c, "ws", "S1", _KIND_LESSON, "42", "old text")
        r = register(c, "ws", "S1", _KIND_LESSON, "42", "new text")
        assert r["status"] == "updated"
        assert r["handle"] == "M01"


def test_rehydrate_clears_session(tmp_path):
    db = tmp_path / "x.sqlite"
    with sqlite3.connect(db) as c:
        ensure_table(c)
        register(c, "ws", "S1", _KIND_LESSON, "42", "a")
        register(c, "ws", "S2", _KIND_LESSON, "43", "b")
        n = rehydrate(c, "ws", "S1")
        assert n == 1
        assert lookup(c, "ws", "S1", _KIND_LESSON, "42") is None
        assert lookup(c, "ws", "S2", _KIND_LESSON, "43") is not None


def test_expired_since_only_old_and_absent(tmp_path):
    db = tmp_path / "x.sqlite"
    with sqlite3.connect(db) as c:
        ensure_table(c)
        register(c, "ws", "S1", _KIND_LESSON, "1", "x")  # old, absent now
        register(c, "ws", "S1", _KIND_LESSON, "2", "y")  # old, present now
        # Force both rows to look old:
        c.execute("UPDATE memory_ledger SET last_seen_at=? WHERE session_id='S1'",
                  (time.time() - 9999,))
        register(c, "ws", "S1", _KIND_LESSON, "3", "z")  # new, present
        seen = {(_KIND_LESSON, "2"), (_KIND_LESSON, "3")}
        exp = expired_since(c, "ws", "S1", seen, stale_after_s=5.0)
        assert {e["ref_id"] for e in exp} == {"1"}


def test_active_handles_newest_first(tmp_path):
    db = tmp_path / "x.sqlite"
    with sqlite3.connect(db) as c:
        ensure_table(c)
        register(c, "ws", "S1", _KIND_LESSON, "1", "a")
        register(c, "ws", "S1", _KIND_LESSON, "2", "b")
        rows = active_handles(c, "ws", "S1")
        assert [r["ref_id"] for r in rows] == ["2", "1"]


# ── delta-render ────────────────────────────────────────────────────────


def test_apply_delta_collapses_unchanged_lesson(isolated_engine):
    """Second injection of the SAME lesson collapses to a handle line."""
    from pmb.memo.delta_render import apply_delta
    res = _Res(lessons=[{"surface_id": 42, "content": "use pnpm, never npm"}])
    full = ("== PMB auto-context ==\n"
            "(matched on message: 'q')\n"
            "Lessons matching this message:\n"
            "  ! use pnpm, never npm [surface_id=42]")
    first = apply_delta(isolated_engine, "S1", res, full)
    # First time it's NEW: tagged with a fresh handle, full text kept.
    assert "M01" in first
    assert "use pnpm" in first

    second = apply_delta(isolated_engine, "S1", res, full)
    # Second time: compact handle line replaces the full lesson.
    assert "[M01] still active" in second
    # The lesson body must NOT be restated on a 'still active' line.
    assert "use pnpm, never npm" not in second.split("[M01]")[1].splitlines()[0]


def test_apply_delta_off_when_no_session(isolated_engine):
    from pmb.memo.delta_render import apply_delta
    res = _Res(lessons=[{"surface_id": 1, "content": "x"}])
    text = "  ! x [surface_id=1]"
    assert apply_delta(isolated_engine, "", res, text) == text
