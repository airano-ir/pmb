"""Phase 2 / issue #7 (migration): `migrate_workspace_into` merges a
per-project workspace into a unified memory, tagged project=<name>, leaving
the SOURCE fully intact (reversible) and being idempotent on re-run."""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmb.core.engine import Engine


def _engine(cwd, home):
    return Engine(cwd=cwd, pmb_home=home, config_overrides={"recall.cache_size": 0})


def _active_count(db_path):
    with sqlite3.connect(str(db_path)) as c:
        return c.execute(
            "SELECT COUNT(*) FROM events WHERE archived_at IS NULL"
        ).fetchone()[0]


def test_migrate_copies_tagged_and_is_idempotent(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("PMB_HOME", str(home))
    src_dir = tmp_path / "src_proj"; src_dir.mkdir()
    tgt_dir = tmp_path / "tgt_proj"; tgt_dir.mkdir()

    src = _engine(src_dir, home)
    src_id = src.workspace.id
    src.record_fact("source decided to use Postgres for storage")
    src.record_fact("source note about the cache layer")
    src_db = src.workspace.db_path
    try:
        src.close()
    except Exception:
        pass

    tgt = _engine(tgt_dir, home)
    assert tgt.workspace.id != src_id

    # dry-run reports, writes nothing
    dry = tgt.migrate_workspace_into(src_id, project="legacy", dry_run=True)
    assert dry["dry_run"] is True
    assert dry["n_to_migrate"] == 2
    assert _active_count(tgt.workspace.db_path) == 0

    # apply copies, tags project=legacy + migrated_from
    res = tgt.migrate_workspace_into(src_id, project="legacy", dry_run=False)
    assert res["n_migrated"] == 2

    with sqlite3.connect(str(tgt.workspace.db_path)) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT content, metadata_json FROM events "
            "WHERE workspace_id = ? AND archived_at IS NULL",
            (tgt.workspace.id,),
        ).fetchall()
    metas = [json.loads(r["metadata_json"] or "{}") for r in rows]
    migrated = [m for m in metas if m.get("migrated_from") == src_id]
    assert len(migrated) == 2
    assert all(m.get("project") == "legacy" for m in migrated)
    assert all(m.get("migrated_ulid") for m in migrated)

    # idempotent: a second apply migrates nothing
    res2 = tgt.migrate_workspace_into(src_id, project="legacy", dry_run=False)
    assert res2["n_migrated"] == 0
    assert res2["n_already"] == 2

    # SOURCE left fully intact
    assert _active_count(src_db) == 2
    try:
        tgt.close()
    except Exception:
        pass


def test_migrate_rejects_unknown_and_self(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("PMB_HOME", str(home))
    d = tmp_path / "p"; d.mkdir()
    eng = _engine(d, home)
    assert "error" in eng.migrate_workspace_into("does_not_exist", dry_run=True)
    assert "error" in eng.migrate_workspace_into(eng.workspace.id, dry_run=True)
    try:
        eng.close()
    except Exception:
        pass
