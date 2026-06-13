"""Persistent error breadcrumbs.

PMB's write/maintenance paths are full of best-effort `try/except` blocks that
must NOT crash a user turn - but a bare `except Exception: pass` makes a real
failure invisible. This module is the middle ground: swallow the exception so
the turn survives, but drop ONE durable row describing it, so `pmb doctor` and
the status panel can show "N errors in the last 24h" instead of silence.

The logger itself must never raise (it is called from inside except blocks), so
every operation is wrapped and failures are dropped. The `error_log` table is
created by the events DDL (`pmb.core.events._DDL`); this module only writes.
"""
from __future__ import annotations

import sqlite3
import time
import traceback
from pathlib import Path

# Prune counter - every Nth insert we drop rows older than the retention window
# so the table can't grow unbounded on a chronically-failing path.
_PRUNE_EVERY = 200
_RETENTION_S = 14 * 24 * 3600.0
_insert_count = 0


def log_error(
    db_path: str | Path,
    component: str,
    exc: BaseException,
    note: str = "",
) -> None:
    """Record one swallowed exception. NEVER raises.

    `component` is a short stable tag ("negation_tombstone", "write_outbox",
    "daemon_hook") used to group rows in `pmb doctor`. `note` is optional
    free-text context (e.g. the ulid being processed)."""
    global _insert_count
    try:
        message = f"{type(exc).__name__}: {exc}"[:500]
        trace_head = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )[-2000:]
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "INSERT INTO error_log (ts, component, message, trace_head, note) "
                "VALUES (?, ?, ?, ?, ?)",
                (time.time(), str(component)[:80], message, trace_head,
                 str(note)[:300]),
            )
            _insert_count += 1
            if _insert_count % _PRUNE_EVERY == 0:
                conn.execute("DELETE FROM error_log WHERE ts < ?",
                             (time.time() - _RETENTION_S,))
    except Exception:
        pass  # the logger must never break the path that called it


def recent_errors(
    db_path: str | Path,
    since_s: float = 24 * 3600.0,
    limit: int = 200,
) -> list[dict]:
    """Return error rows from the last `since_s` seconds (newest first).
    Best-effort - returns [] if the table is missing or unreadable."""
    try:
        cutoff = time.time() - float(since_s)
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ts, component, message, trace_head, note FROM error_log "
                "WHERE ts >= ? ORDER BY ts DESC LIMIT ?",
                (cutoff, int(limit)),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def error_counts(
    db_path: str | Path,
    since_s: float = 24 * 3600.0,
) -> dict[str, int]:
    """{component: count} over the window, for the status panel / doctor."""
    out: dict[str, int] = {}
    try:
        cutoff = time.time() - float(since_s)
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                "SELECT component, COUNT(*) FROM error_log WHERE ts >= ? "
                "GROUP BY component ORDER BY 2 DESC",
                (cutoff,),
            ).fetchall()
        for comp, n in rows:
            out[str(comp)] = int(n)
    except Exception:
        pass
    return out
