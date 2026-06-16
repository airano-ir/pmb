"""Soft (archive) vs hard (purge) deletion at the engine + store layer.

Soft is reversible (restore via `unforget`); hard removes the SQLite row, the
vector, and the event's graph links so recall can never surface it again.
"""
from __future__ import annotations

import sqlite3

import pytest


def _recall_ulids(eng, query, k=10):
    pack = eng.recall(query, top_k=k)
    return {getattr(r, "ulid", None) for r in getattr(pack, "results", [])}


def test_soft_delete_archives_and_is_reversible(isolated_engine):
    eng = isolated_engine
    u = eng.record_fact("The deploy key lives in the vault under prod/deploy")
    assert eng.events.get_by_ulid(u) is not None

    res = eng.delete_event(u, hard=False)
    assert res["mode"] == "soft" and res["ok"] is True

    ev = eng.events.get_by_ulid(u)
    assert ev is not None and ev.archived_at is not None, "soft = archived, not gone"
    active = {e.ulid for e in eng.events.list_active(eng.workspace.id, limit=100)}
    assert u not in active, "archived events drop out of the active set"

    eng.unforget(u)
    active = {e.ulid for e in eng.events.list_active(eng.workspace.id, limit=100)}
    assert u in active, "restore brings it back"


def test_hard_delete_purges_row_vector_and_recall(isolated_engine):
    eng = isolated_engine
    u = eng.record_fact("Postgres connection pool size is capped at 20 in prod")
    assert u in _recall_ulids(eng, "postgres connection pool size"), "recallable first"

    res = eng.delete_event(u, hard=True)
    assert res["mode"] == "hard" and res["ok"] is True

    assert eng.events.get_by_ulid(u) is None, "row is gone for good"
    assert u not in _recall_ulids(eng, "postgres connection pool size"), \
        "purged memory can never be recalled again"
    assert eng.purge(u) is False, "purging an already-gone event reports nothing deleted"


def test_hard_delete_clears_graph_links(isolated_engine):
    eng = isolated_engine
    u = eng.record_fact("PMB uses LanceDB and SQLite in the Acme project")
    try:
        eng.regraph()
    except Exception:
        pass
    db = str(eng.workspace.db_path)
    with sqlite3.connect(db) as c:
        before = c.execute(
            "SELECT COUNT(*) FROM graph_event_entities WHERE event_ulid=?", (u,)
        ).fetchone()[0]
    if before == 0:
        pytest.skip("entity extractor produced no graph links for this text")

    eng.purge(u)
    with sqlite3.connect(db) as c:
        after = c.execute(
            "SELECT COUNT(*) FROM graph_event_entities WHERE event_ulid=?", (u,)
        ).fetchone()[0]
    assert after == 0, "purge must not leave graph links dangling at a gone event"
