"""Centralised SQLite connection setup.

Why: PMB opens dozens of short-lived sqlite3 connections from
record_batch, recall, dashboard, MCP tools. Without per-connection
`busy_timeout`, concurrent writes (MCP server + dashboard + seed
script all touching the same file) fail with `sqlite3.OperationalError:
database is locked` after the default 0-ms wait.

WAL mode allows ONE writer + many readers concurrently; busy_timeout
keeps callers patient when contention happens. Cache size and synchronous
mode are also tuned here so every caller sees the same trade-offs.

Use `connect(path)` instead of `sqlite3.connect(path)` for any code that
might race with another writer.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

# Tune via env vars for power users / benchmarks.
_BUSY_TIMEOUT_MS  = 10_000   # 10s - survives a normal record_batch burst
_CACHE_SIZE_KB    = -2_000   # negative = KB of cache; 2 MB is the sweet spot
_TEMP_STORE       = "MEMORY" # in-memory temp tables (we have RAM)


def connect(
    path: str | Path,
    *,
    isolation_level: str | None = "DEFERRED",
    row_factory: type | None = None,
) -> sqlite3.Connection:
    """Open a SQLite connection with PMB's standard pragmas applied.

    Args:
        path: SQLite file path.
        isolation_level: Python's sqlite3 isolation_level. Default DEFERRED
            keeps standard transaction semantics; pass None for autocommit.
        row_factory: optional row factory (e.g. sqlite3.Row).

    Returns:
        A configured sqlite3.Connection ready for safe concurrent use.
    """
    conn = sqlite3.connect(str(path), isolation_level=isolation_level)
    if row_factory is not None:
        conn.row_factory = row_factory
    apply_pragmas(conn)
    return conn


def apply_pragmas(conn: sqlite3.Connection, *, wal: bool = True) -> None:
    """Apply PMB's standard pragmas to an existing connection.

    Safe to call multiple times. Each pragma is idempotent.

    `wal` gates ONLY `journal_mode=WAL` - the one pragma that PERSISTS in the
    database file header. The global monkey-patch passes wal=False for
    connections to non-PMB databases (see `_is_pmb_db`) so a third-party
    caller's file is never silently switched to WAL on disk. The remaining
    pragmas are per-connection (ephemeral) and harmless to set anywhere.
    """
    try:
        if wal:
            # WAL persists in the DB file header - PMB-owned DBs only.
            conn.execute("PRAGMA journal_mode=WAL")
        # Normal = durable on commit, no fsync on every write. Tradeoff:
        # ~0.1% of writes can be lost on power loss; not OS crash.
        conn.execute("PRAGMA synchronous=NORMAL")
        # Patience for concurrent writes - without this, second writer fails
        # IMMEDIATELY with 'database is locked' instead of waiting.
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        # 2 MB cache per connection. Cheap, faster repeated reads.
        conn.execute(f"PRAGMA cache_size={_CACHE_SIZE_KB}")
        # Temp tables live in RAM, not on disk.
        conn.execute(f"PRAGMA temp_store={_TEMP_STORE}")
    except Exception:
        # Pragmas are advisory - never block opening on a quirky DB
        # (e.g. :memory: in tests that doesn't support all pragmas).
        pass


_PATCH_APPLIED = False


def _is_pmb_db(database) -> bool:
    """True if `database` is a PMB-owned SQLite file, an in-memory DB, or we
    can't tell. Used by the global patch to decide whether to switch a
    connection to WAL - a PERSISTENT change to the file header - which we only
    do for our own databases.

    PMB stores every workspace under PMB_HOME (default ~/.pmb) as
    `.../workspaces/<id>/events.sqlite`, so a path under PMB_HOME or named
    `events.sqlite` is ours. Defaults to True on any uncertainty: missing WAL
    on a real PMB DB risks 'database is locked' under concurrent writes, so we
    return False ONLY for a real on-disk file we can confidently place OUTSIDE
    PMB_HOME.
    """
    try:
        if database is None:
            return True
        s = str(database)
        if not s or ":memory:" in s:
            return True
        import os as _os
        from pathlib import Path as _Path
        if _Path(s).name == "events.sqlite":
            return True  # PMB's canonical DB name (incl. the durable queue)
        home = _Path(_os.environ.get("PMB_HOME") or (_Path.home() / ".pmb")).resolve()
        p = _Path(s).resolve()
        return p == home or home in p.parents
    except Exception:
        return True


def patch_global_sqlite3() -> None:
    """Monkey-patch `sqlite3.connect` so PMB's connections get PMB's pragmas
    automatically.

    Rationale: there are ~50 direct `sqlite3.connect()` calls across the
    codebase (graph, dedup, dashboard, MCP, tests). Touching every one is
    error-prone; patching once at engine import means the pragmas are
    guaranteed for every PMB connection.

    Third-party safety: the patch is process-wide, so it also intercepts
    connections opened by code that merely imported PMB. To avoid silently
    rewriting a third-party database's on-disk header to WAL, the PERSISTENT
    pragma (journal_mode=WAL) is applied ONLY to PMB-owned databases (see
    `_is_pmb_db`). Non-PMB connections still get the harmless per-connection
    pragmas (busy_timeout etc.), which vanish when the connection closes.

    Idempotent: safe to call multiple times. The patch is applied once per
    process lifetime.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    _orig_connect = sqlite3.connect

    def _patched_connect(*args, **kwargs):
        conn = _orig_connect(*args, **kwargs)
        db = kwargs.get("database")
        if db is None and args:
            db = args[0]
        apply_pragmas(conn, wal=_is_pmb_db(db))
        return conn

    sqlite3.connect = _patched_connect
    _PATCH_APPLIED = True
