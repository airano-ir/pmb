"""End-to-end tests for the PMB hook stack on a REAL engine.

Unlike test_auto_recall / test_followcheck (which use a FakeEngine), these
drive a genuine Engine on a throwaway workspace: real SQLite, real
record/read paths, real surface logging. They exercise the full loop the
three hooks form in production:

    record memory
        → auto-recall surfaces it (UserPromptSubmit)
        → agent "does the work" (records activity)
        → followcheck infers follow-through (Stop)
        → the surface is confirmed in the DB
    and separately
        → session-restore rebuilds context (SessionStart)

No embeddings are needed — the hook layer reads via SQL — so these run in
well under a second and don't load sentence-transformers.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("PMB_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PMB_WORKSPACE", "e2e")
    # Keep records synchronous + skip background embedding noise.
    monkeypatch.setenv("PMB_MCP_RECORD_BATCH_ASYNC", "0")
    from pmb.core.engine import Engine
    return Engine()


# ─── full loop: surface → act → followcheck → confirmed ──────────────────


def test_e2e_lesson_surface_then_followcheck_confirms(engine):
    from pmb.hooks import format_context, run_auto_context, run_followcheck

    # 1. Seed a lesson with distinctive tokens.
    engine.record_fact(
        "Always run pmb warmup before recall to avoid cold-start latency on lancedb",
        metadata={"kind": "lesson", "source": "lesson"},
    )

    # 2. Auto-recall on a matching message → lesson surfaces and is logged.
    res = run_auto_context(engine, "какие правила про warmup и recall",
                           log_surfaces=True)
    assert res.lessons, "lesson should surface for a lessons query"
    sid = res.lessons[0].get("surface_id")
    assert isinstance(sid, int), "surface must be logged with an integer id"
    block = format_context(res)
    assert "warmup" in block.lower()

    # 3. The agent 'does the work' and records what it did, naming the
    #    lesson's distinctive tokens.
    engine.record_activity(
        "Ran pmb warmup to fix cold-start latency before recall on lancedb",
        kind="completed",
    )

    # 4. Stop-hook followcheck infers the lesson was followed.
    fc = run_followcheck(engine, window_minutes=60, activity_minutes=60,
                         min_overlap=2, min_strong=1, apply=True)
    assert fc.marked_followed >= 1, "should infer at least one follow"

    # 5. That surface is now confirmed — no longer unconfirmed in the DB.
    remaining = {s["surface_id"] for s in
                 engine.recent_unconfirmed_surfaces(minutes=60)}
    assert sid not in remaining, "surface should be marked followed"


def test_e2e_followcheck_no_false_positive_without_activity(engine):
    """A lesson surfaces, the agent does UNRELATED work → no follow inferred
    (and the lesson is classified not_applicable, since it had nothing to do
    with the turn)."""
    from pmb.hooks import run_auto_context, run_followcheck

    engine.record_fact(
        "Never mix two embedding models in one lancedb table — dimensions clash",
        metadata={"kind": "lesson", "source": "lesson"},
    )
    res = run_auto_context(engine, "какие правила про embedding model lancedb")
    assert res.lessons
    sid = res.lessons[0]["surface_id"]

    # Agent did something totally unrelated.
    engine.record_activity("Refactored the calculator UI button styles",
                           kind="completed")

    fc = run_followcheck(engine, window_minutes=60, min_overlap=3, min_strong=2,
                         apply=True)
    # No false positive: unrelated work must NOT be inferred as a follow.
    assert fc.marked_followed == 0
    # But the lesson clearly didn't pertain to this turn (zero token overlap),
    # so it's classified not_applicable (followed=-1) — excluded from the
    # adherence denominator instead of dangling as a phantom 'not followed'.
    assert fc.not_applicable == 1
    remaining = {s["surface_id"] for s in
                 engine.recent_unconfirmed_surfaces(minutes=60)}
    assert sid not in remaining


# ─── decisions surface in auto-recall ────────────────────────────────────


def test_e2e_decisions_surface(engine):
    from pmb.hooks import format_context, run_auto_context

    engine.record_fact(
        "Chose Postgres over Mongo for JSONB query support and partial indexes",
        metadata={"kind": "decision"},
    )
    # find_decisions is token-overlap (not cross-lingual) — the query must
    # share a distinctive token with the decision. "Postgres" anchors it.
    res = run_auto_context(engine, "почему мы выбрали Postgres для проекта")

    # The decision must reach the agent — but it can arrive via either
    # channel: a standalone `decisions` list, OR folded into the project
    # overview if "Postgres" got promoted to a graph entity (the dedup then
    # correctly drops it from the standalone list to avoid repetition).
    standalone = any("Postgres" in d["content"] for d in res.decisions)
    via_project = any(
        "Postgres" in d.get("content", "")
        for d in ((res.project or {}).get("decisions") or [])
    )
    assert standalone or via_project, "the past decision should surface somehow"

    # What ultimately matters: the rendered block the agent sees names it.
    text = format_context(res)
    assert "Postgres" in text


# ─── cold-start guard: recall skipped, SQL intents still work ────────────


def test_e2e_cold_engine_skips_recall_but_lessons_work(engine):
    from pmb.hooks import Intent, run_auto_context

    # Fresh engine is not warm (no warmup called).
    assert not engine.is_warm()

    engine.record_fact("This repo pins numpy below 2.x for lancedb compat",
                       metadata={"kind": "lesson", "source": "lesson"})

    res = run_auto_context(engine, "когда я последний раз трогал numpy и какие правила")
    # PAST_QUERY fires but recall is cold-skipped...
    assert Intent.PAST_QUERY in res.intents
    assert "RECALL_COLD_SKIP" in res.intents
    assert res.recall_hits == []
    # ...yet the SQL-only lesson path still surfaces.
    assert res.lessons


# ─── session-restore rebuilds context from what the session recorded ─────


def test_e2e_session_restore_from_recorded_work(engine):
    from pmb.hooks import build_session_restore

    engine.record_fact("Chose SQLite-only, dropped the LanceDB dependency",
                       metadata={"kind": "decision"})
    engine.record_fact("Fixed the UTF-8 stdin bug in prepare-context",
                       metadata={"kind": "completed"})
    engine.record_fact("On Windows stdin defaults to cp1251 — force utf-8 decode",
                       metadata={"kind": "lesson", "source": "lesson"})

    out = build_session_restore(engine, minutes=180, include_project=False)
    assert out, "should produce a restore block"
    assert "SQLite-only" in out
    assert "UTF-8 stdin" in out
    assert "cp1251" in out
    assert "pick the thread back up" in out.lower()


def test_e2e_session_restore_empty_when_nothing_recorded(engine):
    from pmb.hooks import build_session_restore
    out = build_session_restore(engine, minutes=5, include_project=False)
    assert out == ""


# ─── adherence stats reflect the inferred follow ─────────────────────────


def test_e2e_followcheck_feeds_adherence_stats(engine):
    from pmb.hooks import run_auto_context, run_followcheck

    engine.record_fact(
        "Use record_batch for multi-fact writes, never many record_fact calls",
        metadata={"kind": "lesson", "source": "lesson"},
    )
    res = run_auto_context(engine, "какие правила про record_batch и запись")
    assert res.lessons

    engine.record_activity(
        "Switched the importer to record_batch for multi-fact writes",
        kind="completed",
    )
    fc = run_followcheck(engine, window_minutes=60, min_overlap=2, min_strong=1,
                         apply=True)
    assert fc.marked_followed >= 1

    stats = engine.adherence_stats(days=1.0)
    # At least one lesson surfaced and at least one was followed now.
    assert stats.get("lesson_surfaces", 0) >= 1 or \
        stats.get("surfaces", 0) >= 1
