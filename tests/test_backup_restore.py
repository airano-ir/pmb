"""X6 — backup / restore + schema-migration guarantees.

Two reliability guarantees a memory tool must keep:

  1. A plain file-copy of `$PMB_HOME` is a valid backup: reopening an Engine on
     the copy returns every event and recall still works (no hidden state
     outside the workspace dir).
  2. Schema migrations are forward-safe and self-healing: an older DB missing a
     newer index (e.g. S7's idx_meta_kind) gets it back automatically on the
     next Engine open — opening an old backup with a new build never errors and
     never silently runs un-indexed.
"""
from __future__ import annotations

import shutil
import sqlite3

from pmb.core.engine import Engine


def _checkpoint(ws) -> None:
    con = sqlite3.connect(str(ws.db_path), timeout=2.0)
    try:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        con.close()


def test_file_copy_backup_round_trips(tmp_pmb_home, tmp_workspace_dir, tmp_path):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
                 config_overrides={"recall.cache_size": 0})
    ulids = [
        eng.record_fact("we use PostgreSQL in production", metadata={"kind": "fact"}),
        eng.record_fact("always run ruff before pushing", metadata={"kind": "lesson"}),
        eng.record_fact("Я живу в Киеве", metadata={"kind": "fact"}),
    ]
    _checkpoint(eng.workspace)

    # back up the WHOLE pmb_home with a plain recursive copy
    backup_home = tmp_path / "backup_home"
    shutil.copytree(tmp_pmb_home, backup_home)

    # restore = open a fresh Engine on the copy (same cwd → same workspace id)
    eng2 = Engine(cwd=tmp_workspace_dir, pmb_home=backup_home,
                  config_overrides={"recall.cache_size": 0})
    rows = eng2.events.get_many(ulids, workspace_id=eng2.workspace.id,
                                only_active=True)
    assert set(rows) == set(ulids), "every event must survive a file-copy backup"
    # lessons indexed query still works on the restored DB
    assert any("ruff" in (x["content"] or "")
               for x in eng2.find_lessons(limit=10))


def test_schema_migration_self_heals_missing_index(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
                 config_overrides={"recall.cache_size": 0})
    eng.record_fact("seed", metadata={"kind": "fact"})
    db = str(eng.workspace.db_path)

    def _has_index(name: str) -> bool:
        with sqlite3.connect(db) as c:
            return bool(c.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?",
                (name,)).fetchone())

    assert _has_index("idx_meta_kind")
    # simulate an OLDER DB created before S7 by dropping the index
    with sqlite3.connect(db) as c:
        c.execute("DROP INDEX idx_meta_kind")
    assert not _has_index("idx_meta_kind")

    # reopening with the current build must recreate it (idempotent DDL replay)
    Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
           config_overrides={"recall.cache_size": 0})
    assert _has_index("idx_meta_kind"), "Engine init must re-apply the missing index"
