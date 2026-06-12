"""End-to-end tests for the LESSON lifecycle on a real engine.

The full self-improvement loop, with real SQLite and real surface logging:

    record lesson
      → surfaces (auto-recall logs a surface_id)
      → confirmed three ways:
          • explicit mark_lesson_followed(True)
          • explicit mark_lesson_followed(False)  (ignored)
          • inferred by the Stop-hook followcheck
      → lesson_follow_stats aggregates surfaces / followed / ignored / unknown
        — the exact data the dashboard Lessons tab renders (follow-rate,
          DEAD vs UNVERIFIED badges)

These pin the backend numbers behind the dashboard fix (unconfirmed surfaces
must count as 'unknown', never as 'ignored').
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("PMB_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PMB_WORKSPACE", "lessons_e2e")
    from pmb.core.engine import Engine
    return Engine()


def _seed_lesson(engine, text: str) -> str:
    return engine.record_fact(text, metadata={"kind": "lesson", "source": "lesson"})


# ─── surfacing logs a surface_id ─────────────────────────────────────────


def test_lesson_surfaces_get_logged_with_id(engine):
    from pmb.hooks import run_auto_context

    _seed_lesson(engine, "This repo uses pnpm, never npm — lockfile is pnpm-lock")
    res = run_auto_context(engine, "do we have a rule about pnpm and npm")
    assert res.lessons, "lesson should surface"
    sid = res.lessons[0]["surface_id"]
    assert isinstance(sid, int)

    # The surface is now in the unconfirmed pool.
    pending = {s["surface_id"] for s in engine.recent_unconfirmed_surfaces(minutes=60)}
    assert sid in pending


# ─── explicit follow / ignore feed lesson_follow_stats ───────────────────


def test_explicit_follow_updates_stats(engine):
    from pmb.hooks import run_auto_context

    _seed_lesson(engine, "Always pin numpy below 2.x for lancedb compatibility")
    res = run_auto_context(engine, "do we have a rule about numpy and lancedb")
    sid = res.lessons[0]["surface_id"]

    engine.mark_lesson_followed(sid, followed=True, note="pinned numpy to 1.26")

    stats = engine.lesson_follow_stats(days=1.0)
    assert stats["total_surfaces"] >= 1
    assert stats["followed"] >= 1
    assert stats["follow_rate"] > 0.0
    # per-lesson row reflects the follow
    row = next((r for r in stats["per_lesson"]
                if "numpy" in (r["content"] or "")), None)
    assert row is not None
    assert row["followed"] >= 1
    assert row["ignored"] == 0


def test_explicit_ignore_counts_as_ignored_not_followed(engine):
    from pmb.hooks import run_auto_context

    _seed_lesson(engine, "Prefer ruff over flake8 for linting in this repo")
    res = run_auto_context(engine, "do we have a rule about ruff and the linter")
    sid = res.lessons[0]["surface_id"]

    engine.mark_lesson_followed(sid, followed=False, note="used flake8, legacy CI")

    stats = engine.lesson_follow_stats(days=1.0)
    assert stats["ignored"] >= 1
    row = next((r for r in stats["per_lesson"]
                if "ruff" in (r["content"] or "")), None)
    assert row is not None
    assert row["ignored"] >= 1
    assert row["followed"] == 0


def test_unconfirmed_surface_counts_as_unknown_not_ignored(engine):
    """The dashboard-fix invariant: a surfaced-but-unmarked lesson is
    'unknown', never 'ignored'. A frequently-surfaced unconfirmed lesson
    must NOT look dead."""
    from pmb.hooks import run_auto_context

    _seed_lesson(engine, "The websocket reconnect uses exponential backoff capped at 30s")
    # Surface it 3 times in one session, never mark it. R1 dedups same-session
    # surfaces within the hour, so this is ONE surface (counting shows, not rows).
    for _ in range(3):
        run_auto_context(engine, "do we have a rule about websocket reconnect backoff")

    stats = engine.lesson_follow_stats(days=1.0)
    assert stats["ignored"] == 0, "nothing was explicitly ignored"
    assert stats["unknown"] >= 1, "unconfirmed surfaces are 'unknown'"
    row = next((r for r in stats["per_lesson"]
                if "websocket" in (r["content"] or "")), None)
    assert row is not None
    assert row["surfaces"] >= 1
    assert row["followed"] == 0
    assert row["ignored"] == 0
    # The frontend DEAD rule is (ignored >= 2 AND ignored > followed) — this
    # row has ignored=0, so it is NOT dead despite surfacing 3×.
    is_dead = row["ignored"] >= 2 and row["ignored"] > row["followed"]
    assert not is_dead


# ─── auto followcheck closes the loop without explicit marking ───────────


def test_followcheck_inferred_follow_shows_in_stats(engine):
    from pmb.hooks import run_auto_context, run_followcheck

    _seed_lesson(engine,
                 "Use record_batch for multi-fact writes, never many record_fact calls")
    res = run_auto_context(engine, "do we have a rule about record_batch writes")
    sid = res.lessons[0]["surface_id"]

    # Agent records what it did, naming the lesson's distinctive tokens.
    engine.record_activity(
        "Migrated the importer to record_batch for multi-fact writes",
        kind="completed",
    )
    fc = run_followcheck(engine, window_minutes=60, min_overlap=2, min_strong=1,
                         apply=True)
    assert fc.marked_followed >= 1

    stats = engine.lesson_follow_stats(days=1.0)
    assert stats["followed"] >= 1
    # surface left the unconfirmed pool
    pending = {s["surface_id"] for s in engine.recent_unconfirmed_surfaces(minutes=60)}
    assert sid not in pending

    # The follow_note records that this was inferred, not self-reported.
    import sqlite3
    with sqlite3.connect(engine.workspace.db_path) as conn:
        note = conn.execute(
            "SELECT follow_note FROM lesson_surfaces WHERE id=?", (sid,)
        ).fetchone()[0]
    assert "auto-detected" in (note or "")


# ─── per-lesson aggregation across multiple surfaces ─────────────────────


def test_per_lesson_aggregates_mixed_verdicts(engine):
    # A lesson surfacing across THREE DISTINCT sessions (R1 dedups within a
    # session-hour, so distinct surfaces require distinct sessions — which is
    # exactly how a recurring rule accrues a follow-history in real use).
    u = _seed_lesson(engine, "Tailscale ACLs must allow the memo node on port 8765")
    L = {"ulid": u, "content": "Tailscale ACLs must allow the memo node"}
    sids = []
    for s in ("s1", "s2", "s3"):
        r = engine._log_lesson_surfaces([dict(L)], query="tailscale acl port",
                                        source="recall", session_id=s)
        sids.append(r[0]["surface_id"])
    assert len(set(sids)) == 3, "distinct sessions → distinct surfaces"
    engine.mark_lesson_followed(sids[0], followed=True, note="a")
    engine.mark_lesson_followed(sids[1], followed=True, note="b")
    # sids[2] left unmarked

    stats = engine.lesson_follow_stats(days=1.0)
    row = next((r for r in stats["per_lesson"]
                if "tailscale" in (r["content"] or "").lower()), None)
    assert row is not None
    assert row["surfaces"] == 3
    assert row["followed"] == 2
    assert row["ignored"] == 0
    # USEFUL badge rule on the frontend is followed >= 2 → this qualifies
    assert row["followed"] >= 2


# ─── empty / fresh workspace must not crash ──────────────────────────────


def test_lesson_stats_empty_workspace(engine):
    stats = engine.lesson_follow_stats(days=7.0)
    assert stats["total_surfaces"] == 0
    assert stats["followed"] == 0
    assert stats["ignored"] == 0
    assert stats["unknown"] == 0
    assert stats["follow_rate"] == 0.0
    assert stats["per_lesson"] == []


def test_adherence_stats_works_without_mcp_calls_table(engine):
    """Regression for the e2e-found bug: adherence_stats must still report
    lesson metrics on a workspace that never ran the MCP server (no
    mcp_calls table)."""
    from pmb.hooks import run_auto_context

    _seed_lesson(engine, "Dashboard SVG overlay uses requestAnimationFrame, not setTimeout")
    res = run_auto_context(engine, "do we have a rule about dashboard svg overlay")
    engine.mark_lesson_followed(res.lessons[0]["surface_id"], followed=True, note="x")

    stats = engine.adherence_stats(days=1.0)
    # These would all be 0 if the mcp_calls query crashed the whole function.
    assert stats["lesson_surfaces"] >= 1
    assert stats["lesson_followed"] >= 1
    assert stats["lesson_followthrough"] > 0.0
