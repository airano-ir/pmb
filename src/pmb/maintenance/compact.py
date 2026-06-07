"""
Storage Compaction.

Цели:
1. Active SQLite остаётся компактным (быстрые queries)
2. Archived events перемещаются в холодное хранилище (cold.sqlite)
3. SQLite VACUUM для возврата места
4. LanceDB compact (через native compaction)

Запускается:
- pmb compact
- Автоматически через scheduler (раз в неделю)

Что делает:
1. Archived events старше 30 дней → переносятся в cold.sqlite
2. Vector embeddings таких events удаляются из LanceDB
3. VACUUM на основной events.sqlite
4. LanceDB compaction (если поддерживает)

Cold storage схема такая же как main, но в отдельном файле. При желании
можно его не открывать на старте — экономия памяти.
"""

from __future__ import annotations

import shutil
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmb.core.engine import Engine


COLD_AGE_THRESHOLD_DAYS = 30


class StorageCompactor:
    """Перемещает старые archived events в cold storage и компактит main DB."""

    def __init__(self, engine: "Engine"):
        self.engine = engine

    def cold_db_path(self) -> Path:
        return self.engine.workspace.storage_dir / "cold.sqlite"

    def _ensure_cold_schema(self):
        cold = self.cold_db_path()
        with sqlite3.connect(cold) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ulid TEXT UNIQUE NOT NULL,
                    workspace_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT,
                    timestamp REAL NOT NULL,
                    importance REAL,
                    access_count INTEGER,
                    last_accessed REAL,
                    archived_at REAL,
                    source_session_id TEXT,
                    cold_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cold_workspace ON events(workspace_id);
                CREATE INDEX IF NOT EXISTS idx_cold_archived ON events(archived_at DESC);
            """)

    def compact(self, dry_run: bool = False, age_days: int = COLD_AGE_THRESHOLD_DAYS) -> dict:
        """
        Перенести archived events старше age_days в cold storage.

        Returns:
            {"moved_to_cold": N, "main_size_before": ..., "main_size_after": ..., "dry_run": bool}
        """
        workspace_id = self.engine.workspace.id
        cutoff = time.time() - age_days * 86400.0

        main_db = self.engine.workspace.db_path
        size_before = main_db.stat().st_size if main_db.exists() else 0

        # Find candidates
        with sqlite3.connect(main_db) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                """
                SELECT * FROM events
                WHERE workspace_id = ? AND archived_at IS NOT NULL AND archived_at < ?
                """,
                (workspace_id, cutoff),
            )
            candidates = [dict(r) for r in cur.fetchall()]

        if not candidates:
            return {
                "moved_to_cold": 0,
                "main_size_before": size_before,
                "main_size_after": size_before,
                "dry_run": dry_run,
            }

        if dry_run:
            return {
                "moved_to_cold": len(candidates),
                "main_size_before": size_before,
                "main_size_after": size_before,
                "dry_run": True,
                "would_move_ulids": [c["ulid"] for c in candidates[:20]],
            }

        # Перенести в cold
        self._ensure_cold_schema()
        cold_db = self.cold_db_path()

        with sqlite3.connect(cold_db) as cold_conn:
            for c in candidates:
                cold_conn.execute(
                    """
                    INSERT OR IGNORE INTO events (
                        ulid, workspace_id, event_type, content, metadata_json,
                        timestamp, importance, access_count, last_accessed,
                        archived_at, source_session_id, cold_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        c["ulid"], c["workspace_id"], c["event_type"], c["content"],
                        c["metadata_json"], c["timestamp"], c["importance"],
                        c["access_count"], c["last_accessed"], c["archived_at"],
                        c["source_session_id"], time.time(),
                    ),
                )

        # Удалить из main + удалить из vector index
        ulids = [c["ulid"] for c in candidates]
        with sqlite3.connect(main_db) as conn:
            placeholders = ",".join("?" * len(ulids))
            conn.execute(
                f"DELETE FROM events WHERE ulid IN ({placeholders})", ulids,
            )

        for u in ulids:
            try:
                self.engine.search.remove(u)
            except Exception:
                pass

        # VACUUM
        with sqlite3.connect(main_db) as conn:
            conn.execute("VACUUM")

        # LanceDB compaction (best-effort, API name varies by version)
        try:
            tbl = self.engine.search._table
            if hasattr(tbl, "optimize"):
                tbl.optimize()
            elif hasattr(tbl, "compact_files"):
                tbl.compact_files()
        except Exception:
            pass

        # Cheap graph cleanup as part of weekly compact. We prune one-off
        # co-mention edges older than 30d AND the orphan entities they leave
        # behind. This is safe (drops noise, never touches events) and keeps
        # the entity graph from drifting upward forever. A full `pmb regraph`
        # (rerunning the extractor over all active events) is intentionally
        # NOT automatic — it's expensive with LLM backends and the user
        # should run it manually after upgrading the extractor.
        graph_prune = None
        try:
            graph_prune = self.engine.prune_graph(
                max_weight=1,
                older_than_days=30.0,
                also_drop_orphan_entities=True,
            )
        except Exception:
            pass

        size_after = main_db.stat().st_size if main_db.exists() else 0

        return {
            "moved_to_cold": len(candidates),
            "main_size_before": size_before,
            "main_size_after": size_after,
            "size_saved": max(0, size_before - size_after),
            "cold_db_path": str(cold_db),
            "graph_prune": graph_prune,
            "dry_run": False,
        }

    def cold_stats(self) -> dict:
        cold = self.cold_db_path()
        if not cold.exists():
            return {"exists": False, "n_events": 0, "size_bytes": 0}
        with sqlite3.connect(cold) as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM events WHERE workspace_id = ?",
                (self.engine.workspace.id,),
            )
            n = cur.fetchone()[0]
        return {
            "exists": True,
            "n_events": n,
            "size_bytes": cold.stat().st_size,
            "path": str(cold),
        }
