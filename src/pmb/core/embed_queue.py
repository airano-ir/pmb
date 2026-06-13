"""
Durable embed queue (Hardening H3).

Background:
  Writes to PMB are fire-and-forget: MCP returns in ~2ms, the embedding +
  vector-index insertion happen asynchronously. The original in-memory
  queue (`Engine._embed_queue`) had two durability gaps:

  1. Process crash between SQLite insert and embed → the event lives in
     BM25 but never gets a vector. No way to recover after restart.

  2. Any exception inside `search.add()` (LanceDB temporarily down,
     model load error, OOM during encode) silently dropped the work.
     `except Exception: pass` - no retry, no log.

Solution:
  A small SQLite table `embed_queue_pending` colocated with the events
  DB. Inserts persist across process restarts. A worker thread drains
  it with exponential backoff. After `max_attempts` failures a row is
  moved to dead-letter status (still queryable, marked `failed`).

  Cost: ~0.5ms per enqueue (one INSERT). Worth it for the durability.

Public API:
  PersistentEmbedQueue(db_path)
    .enqueue(ulid, text)            -> persist + wake worker
    .pending_count()                -> for diagnostics
    .dead_letter_count()
    .recover_on_start(adder, ready) -> drain pending rows
    .drain_once(adder)              -> single best-effort pass

`adder` is callable (ulid, text) -> None that performs the actual
search.add. `ready` returns True when the embedder is loaded.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embed_queue_pending (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ulid TEXT NOT NULL,
    text TEXT NOT NULL,
    enqueued_at REAL NOT NULL,
    last_attempt_at REAL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    status TEXT NOT NULL DEFAULT 'pending'  -- pending | failed
);
CREATE INDEX IF NOT EXISTS embed_queue_status_idx
    ON embed_queue_pending(status, attempts);
"""


DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BACKOFF_BASE = 1.5  # seconds


class PersistentEmbedQueue:
    def __init__(
        self,
        db_path: Path,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
    ):
        self.db_path = db_path
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self._migrate()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _migrate(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def enqueue(self, ulid: str, text: str) -> None:
        """Persist a pending embed. Survives crash + restart."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO embed_queue_pending(ulid, text, enqueued_at) "
                "VALUES (?, ?, ?)",
                (ulid, text, time.time()),
            )

    def pending_count(self) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM embed_queue_pending "
                "WHERE status = 'pending'"
            ).fetchone()
            return int(row[0] or 0)

    def dead_letter_count(self) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM embed_queue_pending "
                "WHERE status = 'failed'"
            ).fetchone()
            return int(row[0] or 0)

    def drain_once(
        self,
        adder: Callable[[str, str], None],
        max_items: int = 100,
    ) -> dict:
        """Single sweep over pending rows. Returns counts.

        `adder(ulid, text)` performs the actual search.add. Raises on
        failure → we record attempt + error, eventually dead-letter.
        """
        ok = 0
        failed = 0
        skipped = 0
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, ulid, text, attempts, last_attempt_at "
                "FROM embed_queue_pending "
                "WHERE status = 'pending' "
                "ORDER BY id ASC LIMIT ?",
                (max_items,),
            ).fetchall()
        now = time.time()
        for row in rows:
            # Backoff: don't retry too aggressively
            if row["last_attempt_at"]:
                wait = self.backoff_base ** max(row["attempts"], 0)
                if now - row["last_attempt_at"] < wait:
                    skipped += 1
                    continue
            try:
                adder(row["ulid"], row["text"])
            except Exception as e:
                failed += 1
                new_attempts = row["attempts"] + 1
                new_status = (
                    "failed" if new_attempts >= self.max_attempts else "pending"
                )
                with self._conn() as conn:
                    conn.execute(
                        "UPDATE embed_queue_pending "
                        "SET attempts = ?, last_attempt_at = ?, "
                        "    last_error = ?, status = ? "
                        "WHERE id = ?",
                        (new_attempts, time.time(),
                         f"{type(e).__name__}: {e}"[:500],
                         new_status, row["id"]),
                    )
                continue
            ok += 1
            with self._conn() as conn:
                conn.execute(
                    "DELETE FROM embed_queue_pending WHERE id = ?",
                    (row["id"],),
                )
        return {"ok": ok, "failed": failed, "skipped": skipped}

    def recover_on_start(
        self,
        adder: Callable[[str, str], None],
        ready: Callable[[], bool],
        timeout_seconds: float = 600.0,
        poll_interval: float = 0.5,
    ) -> dict:
        """Wait for the embedder to become ready, then drain everything
        that's pending. Use this from a background thread on Engine init.
        """
        deadline = time.time() + timeout_seconds
        while not ready() and time.time() < deadline:
            time.sleep(poll_interval)
        if not ready():
            return {"ok": 0, "failed": 0, "skipped": 0, "timeout": True}
        # Drain in passes until either nothing pending or only dead-letters
        total_ok = total_failed = total_skipped = 0
        while True:
            result = self.drain_once(adder)
            total_ok += result["ok"]
            total_failed += result["failed"]
            total_skipped += result["skipped"]
            if result["ok"] == 0 and result["failed"] == 0:
                break
        return {
            "ok": total_ok, "failed": total_failed,
            "skipped": total_skipped, "timeout": False,
        }

    def list_dead_letters(self, limit: int = 100) -> list[dict]:
        """For diagnostics / `pmb doctor`."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT ulid, attempts, last_error, last_attempt_at "
                "FROM embed_queue_pending WHERE status = 'failed' "
                "ORDER BY last_attempt_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def retry_dead_letter(self, ulid: str | None = None) -> int:
        """Move dead-letter rows back to pending (optionally just one)."""
        with self._conn() as conn:
            if ulid:
                cur = conn.execute(
                    "UPDATE embed_queue_pending SET status='pending', "
                    "attempts = 0, last_error = NULL "
                    "WHERE status = 'failed' AND ulid = ?",
                    (ulid,),
                )
            else:
                cur = conn.execute(
                    "UPDATE embed_queue_pending SET status='pending', "
                    "attempts = 0, last_error = NULL "
                    "WHERE status = 'failed'"
                )
            return cur.rowcount
