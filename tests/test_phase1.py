"""
Phase 1 smoke tests.

Покрывают:
- Workspace detection
- EventStore CRUD
- HybridSearch index
- Engine end-to-end
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Path setup для запуска без install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmb.core.events import Event, EventStore
from pmb.core.engine import Engine
from pmb.core.workspace import detect_workspace, list_workspaces


# tmp_pmb_home / tmp_workspace_dir come from conftest.py (tmp_path-based).
# The old tempfile.TemporaryDirectory fixtures here hard-failed teardown on
# Windows when the engine wasn't closed (open SQLite/LanceDB handle → the dir
# can't be deleted → PermissionError). pytest's tmp_path cleanup is
# best-effort, so the shared fixtures don't error on a lingering handle; they
# also carry the deterministic recall config.


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

def test_workspace_detect_cwd(tmp_pmb_home, tmp_workspace_dir):
    ws = detect_workspace(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    assert ws.id
    assert ws.root == tmp_workspace_dir
    assert ws.source == "cwd"


def test_workspace_persistent(tmp_pmb_home, tmp_workspace_dir):
    ws1 = detect_workspace(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ws1.ensure_dirs()
    ws1.save_meta()

    ws2 = detect_workspace(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    assert ws1.id == ws2.id
    assert ws1.created_at == ws2.created_at


def test_list_workspaces(tmp_pmb_home, tmp_workspace_dir):
    ws = detect_workspace(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ws.ensure_dirs()
    ws.save_meta()

    items = list_workspaces(tmp_pmb_home)
    assert len(items) == 1
    assert items[0].id == ws.id


# ---------------------------------------------------------------------------
# EventStore
# ---------------------------------------------------------------------------

def test_eventstore_append_and_read(tmp_pmb_home, tmp_workspace_dir):
    ws = detect_workspace(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ws.ensure_dirs()
    store = EventStore(ws.db_path)

    ev = Event(
        workspace_id=ws.id,
        event_type="qa",
        content="Use Postgres 17",
        metadata={"query": "What database?"},
    )
    ev = store.append(ev)
    assert ev.id is not None

    roundtrip = store.get_by_ulid(ev.ulid)
    assert roundtrip is not None
    assert roundtrip.content == "Use Postgres 17"
    assert roundtrip.metadata["query"] == "What database?"


def test_eventstore_archive(tmp_pmb_home, tmp_workspace_dir):
    ws = detect_workspace(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ws.ensure_dirs()
    store = EventStore(ws.db_path)

    ev = store.append(Event(workspace_id=ws.id, content="X"))
    assert store.count(ws.id) == 1

    store.archive(ev.ulid)
    assert store.count(ws.id, include_archived=False) == 0
    assert store.count(ws.id, include_archived=True) == 1


def test_eventstore_pin(tmp_pmb_home, tmp_workspace_dir):
    ws = detect_workspace(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ws.ensure_dirs()
    store = EventStore(ws.db_path)

    ev = store.append(Event(workspace_id=ws.id, content="X", importance=0.3))
    store.pin(ev.ulid)
    after = store.get_by_ulid(ev.ulid)
    assert after.importance == 1.0


def test_eventstore_stats(tmp_pmb_home, tmp_workspace_dir):
    ws = detect_workspace(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ws.ensure_dirs()
    store = EventStore(ws.db_path)

    store.append(Event(workspace_id=ws.id, event_type="qa", content="Q1"))
    store.append(Event(workspace_id=ws.id, event_type="fact", content="F1"))
    store.append(Event(workspace_id=ws.id, event_type="fact", content="F2"))

    s = store.stats(ws.id)
    assert s["total"] == 3
    assert s["active"] == 3
    assert s["by_type"]["qa"] == 1
    assert s["by_type"]["fact"] == 2


# ---------------------------------------------------------------------------
# Engine end-to-end
# ---------------------------------------------------------------------------

def test_engine_remember_and_recall(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)

    eng.remember("What database do we use?", "Postgres 17, port 5432")
    eng.remember("What auth method?", "JWT with refresh tokens, 15m access / 7d refresh")
    eng.remember("Which CSS framework?", "Tailwind CSS with shadcn/ui components")

    pack = eng.recall("authentication", top_k=3)
    assert len(pack.results) > 0
    # Top hit should be the auth one
    top = pack.results[0]
    assert "JWT" in top.content or "refresh" in top.content.lower()


def test_engine_pin_forget_cycle(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ulid = eng.remember("Test query", "Test response")

    eng.pin(ulid)
    s = eng.stats()
    assert s["events"]["active"] == 1

    eng.forget(ulid)
    s = eng.stats()
    assert s["events"]["active"] == 0
    assert s["events"]["archived"] == 1

    # Recall не должен возвращать archived
    pack = eng.recall("Test query")
    assert all(r.ulid != ulid for r in pack.results)


def test_engine_stats_format(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    s = eng.stats()
    assert "workspace" in s
    assert "events" in s
    assert "search_index_size" in s


def test_engine_isolated_workspaces(tmp_pmb_home):
    """Два разных workspace не видят друг друга."""
    with tempfile.TemporaryDirectory() as tmp_a:
        with tempfile.TemporaryDirectory() as tmp_b:
            ws_a = Path(tmp_a)
            ws_b = Path(tmp_b)

            eng_a = Engine(cwd=ws_a, pmb_home=tmp_pmb_home)
            eng_b = Engine(cwd=ws_b, pmb_home=tmp_pmb_home)

            assert eng_a.workspace.id != eng_b.workspace.id

            eng_a.remember("A query", "A response — Postgres in workspace A")
            eng_b.remember("B query", "B response — MongoDB in workspace B")

            pack_a = eng_a.recall("database")
            pack_b = eng_b.recall("database")

            # A видит свой Postgres, не MongoDB
            a_contents = " ".join(r.content for r in pack_a.results)
            assert "Postgres" in a_contents
            assert "MongoDB" not in a_contents

            # B видит свой MongoDB, не Postgres
            b_contents = " ".join(r.content for r in pack_b.results)
            assert "MongoDB" in b_contents
            assert "Postgres" not in b_contents


def test_engine_persistence_across_instances(tmp_pmb_home, tmp_workspace_dir):
    """Перезапуск engine — данные сохраняются."""
    eng1 = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng1.remember("Q", "Postgres on port 5432")
    eng1.remember("Q2", "Redis 7 for caching")
    del eng1

    eng2 = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    s = eng2.stats()
    assert s["events"]["active"] == 2

    pack = eng2.recall("postgres")
    assert len(pack.results) > 0
    assert "Postgres" in pack.results[0].content


def test_engine_recall_signals(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.remember("Q1", "Postgres 17 setup with docker-compose")
    pack = eng.recall("docker")

    assert len(pack.results) > 0
    r = pack.results[0]
    assert 0.0 <= r.bm25_score <= 1.0
    assert 0.0 <= r.vec_score <= 1.0
    assert 0.0 <= r.importance <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
