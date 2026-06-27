"""Cold writes (no model loaded) must still land in the BM25 index.

Bug 3 root cause: a one-shot CLI `pmb fact` runs with no embedding model, so
the write deferred its whole index update to a background thread that dies on
process exit - leaving the event in SQLite but unsearchable until a reindex.
BM25 needs no model, so we index it synchronously on the cold path; the vector
embed stays deferred to the durable queue.
"""
from __future__ import annotations

from pmb.core.engine import Engine


def test_cold_write_lands_in_bm25_without_model(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    # Simulate a cold writer: the embedding model is NOT loaded.
    monkeypatch.setattr(eng.search, "is_ready", lambda: False)
    ulid = eng.record_fact("zog distinctive cold marker xyz never embedded")
    # The fix: the event is in the BM25 index synchronously (no model needed),
    # so it is lexically searchable immediately instead of waiting for a reindex.
    assert ulid in eng.search._bm25_ulids


def test_add_bm25_only_is_idempotent(tmp_pmb_home, tmp_workspace_dir):
    """add_bm25_only must not double-list a ulid, so the later full add() for
    the deferred vector embed adds only the vector."""
    s = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home).search
    s.add_bm25_only("uX", "alpha bravo charlie unique")
    s.add_bm25_only("uX", "alpha bravo charlie unique")
    assert s._bm25_ulids.count("uX") == 1
