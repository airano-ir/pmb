"""MCP call performance tracking.

Records every MCP tool invocation into a tiny SQLite table so the dashboard
can show p50/p95 latencies, error rates, and trends.

Overhead: ~1-2ms per call (one SQLite INSERT). Negligible compared to LLM
thinking time. Auto-cleans rows older than 7 days on each insert.
"""
from __future__ import annotations

import functools
import json
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

# S9: buffer perf rows and flush them in batches instead of one
# connect+INSERT+commit per MCP tool call (which contended with the events DB
# write lock on the hot path). Telemetry is best-effort, so at most
# `_PERF_FLUSH_EVERY-1` rows can be lost on a hard crash - acceptable for perf
# stats. Same DB target as before, so readers (dashboard / `pmb stats`) are
# unchanged; only the write cadence moves off the per-call path.
_PERF_BUF: dict[str, list[tuple]] = {}
_PERF_BUF_LOCK = threading.Lock()
_PERF_FLUSH_EVERY = 25


def _flush_perf_rows(db_path_str: str, rows: list[tuple]) -> None:
    if not rows:
        return
    try:
        _ensure_schema(Path(db_path_str))
        with sqlite3.connect(db_path_str, timeout=2.0) as conn:
            conn.executemany(
                "INSERT INTO mcp_calls "
                "(workspace_id, tool_name, timestamp, duration_ms, success, "
                "error, args_size, stages_json, backend, cache_hit, "
                "client_timeout, outcome) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            cutoff = time.time() - 7 * 86400
            conn.execute("DELETE FROM mcp_calls WHERE timestamp < ?", (cutoff,))
            conn.commit()
    except Exception:
        pass  # perf tracking must never break MCP


def flush_perf() -> None:
    """Drain all buffered perf rows now (call on shutdown / in tests)."""
    with _PERF_BUF_LOCK:
        pending = {k: v for k, v in _PERF_BUF.items() if v}
        _PERF_BUF.clear()
    for db_path_str, rows in pending.items():
        _flush_perf_rows(db_path_str, rows)

_DDL = """
CREATE TABLE IF NOT EXISTS mcp_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT,
    tool_name TEXT NOT NULL,
    timestamp REAL NOT NULL,
    duration_ms REAL NOT NULL,
    success INTEGER NOT NULL DEFAULT 1,
    error TEXT,
    args_size INTEGER DEFAULT 0
);
"""

_INDEX_DDLS = [
    "CREATE INDEX IF NOT EXISTS idx_mcp_calls_ts ON mcp_calls(timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_mcp_calls_tool ON mcp_calls(tool_name, timestamp DESC)",
]


_schema_ready_paths: set = set()

# Columns added after the original schema (issue #11 - recall_smart stage
# visibility). Added via idempotent ALTER so existing perf DBs upgrade in place.
_MIGRATION_COLUMNS = [
    ("stages_json", "TEXT"),       # recall_smart/recall_deep escalation diag
    ("backend", "TEXT"),           # which LLM/embedding backend ran (if known)
    ("cache_hit", "INTEGER"),      # 1 = served from a cache, 0 = computed
    ("client_timeout", "INTEGER"), # 1 = server ran past the client's timeout
    ("outcome", "TEXT"),           # ok | error | client_timeout
]


def _ensure_schema(db_path: Path) -> None:
    if str(db_path) in _schema_ready_paths:
        return
    with sqlite3.connect(db_path) as conn:
        conn.execute(_DDL)
        for d in _INDEX_DDLS:
            conn.execute(d)
        for col, typ in _MIGRATION_COLUMNS:
            try:
                conn.execute(f"ALTER TABLE mcp_calls ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:
                pass  # column already exists
        conn.commit()
    _schema_ready_paths.add(str(db_path))


def record_call(
    db_path: Path,
    workspace_id: str | None,
    tool_name: str,
    duration_ms: float,
    success: bool = True,
    error: str | None = None,
    args_size: int = 0,
    stages_json: str | None = None,
    backend: str | None = None,
    cache_hit: bool | None = None,
    client_timeout: bool | None = None,
    outcome: str | None = None,
) -> None:
    """Buffer one MCP-call row; flush in batches (S9). Silent on failure (perf
    tracking must never break MCP itself)."""
    try:
        row = (workspace_id, tool_name, time.time(), duration_ms,
               1 if success else 0, error, args_size, stages_json, backend,
               (None if cache_hit is None else (1 if cache_hit else 0)),
               (None if client_timeout is None else (1 if client_timeout else 0)),
               outcome)
        flush: tuple[str, list[tuple]] | None = None
        with _PERF_BUF_LOCK:
            buf = _PERF_BUF.setdefault(str(db_path), [])
            buf.append(row)
            if len(buf) >= _PERF_FLUSH_EVERY:
                flush = (str(db_path), buf[:])
                buf.clear()
        if flush is not None:
            _flush_perf_rows(*flush)
    except Exception:
        pass  # never break MCP


def make_timing_wrapper(db_path: Path, workspace_id: str | None):
    """Return a decorator that times the wrapped tool function and logs the
    call. Use to wrap each MCP tool body:

        @_maybe_tool(mcp, "recall")
        @timed
        def recall(query: str, top_k: int = 5) -> dict:
            ...
    """
    # The server's client-side timeout (ms): a call finishing AFTER this had
    # already elapsed means the client gave up - flag it, don't count it as a
    # clean success (#11).
    try:
        _client_to_ms = float(os.environ.get("PMB_MCP_CLIENT_TIMEOUT_MS") or 120000)
    except Exception:
        _client_to_ms = 120000.0

    def decorator(fn: Callable) -> Callable:
        tool_name = fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            success = True
            err_msg: str | None = None
            result = None
            try:
                # S9: cheap size estimate - `len(str(kwargs))` instead of a full
                # json.dumps re-serialization of (possibly large) tool args.
                args_size = len(str(kwargs)) if kwargs else 0
            except Exception:
                args_size = 0
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                err_msg = f"{type(e).__name__}: {str(e)[:200]}"
                raise
            finally:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                # Pull recall_smart/recall_deep stage diagnostics + cache flag
                # straight out of the returned pack dict (issue #11).
                stages_json = None
                cache_hit = None
                try:
                    if isinstance(result, dict):
                        esc = result.get("escalation")
                        if esc:
                            stages_json = json.dumps(esc, default=str)
                        if "cache_hit" in result:
                            cache_hit = bool(result.get("cache_hit"))
                except Exception:
                    pass
                client_timeout = dt_ms >= _client_to_ms
                outcome = (
                    "error" if not success
                    else ("client_timeout" if client_timeout else "ok")
                )
                record_call(
                    db_path=db_path,
                    workspace_id=workspace_id,
                    tool_name=tool_name,
                    duration_ms=dt_ms,
                    success=success,
                    error=err_msg,
                    args_size=args_size,
                    stages_json=stages_json,
                    cache_hit=cache_hit,
                    client_timeout=client_timeout,
                    outcome=outcome,
                )
        return wrapper
    return decorator


def get_perf_stats(db_path: Path, workspace_id: str | None = None,
                   hours: float = 24.0) -> dict:
    """Aggregate MCP call stats for the dashboard.

    Returns:
      {
        "window_hours": 24,
        "total_calls": 123,
        "error_count": 2,
        "by_tool": [{name, count, p50, p95, p99, max, errors}, ...],
        "recent": [{tool, ts, ms, success, error}, ...],   (last 50)
        "timeline_buckets": [{bucket_start, count, avg_ms}, ...]  (last 24 1h buckets)
      }
    """
    flush_perf()  # S9: drain buffered rows so the reader sees current data
    _ensure_schema(db_path)
    cutoff = time.time() - hours * 3600
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        ws_clause = ""
        params: list = [cutoff]
        if workspace_id:
            ws_clause = " AND (workspace_id = ? OR workspace_id IS NULL)"
            params.append(workspace_id)

        # Per-tool aggregates (compute p50/p95 in Python since SQLite has no native percentile)
        rows = conn.execute(
            f"SELECT tool_name, duration_ms, success, error, timestamp, "
            f"stages_json, backend, cache_hit, client_timeout, outcome "
            f"FROM mcp_calls WHERE timestamp >= ?{ws_clause} "
            f"ORDER BY timestamp DESC",
            params,
        ).fetchall()

    by_tool: dict[str, list[float]] = {}
    errors_by_tool: dict[str, int] = {}
    for r in rows:
        by_tool.setdefault(r["tool_name"], []).append(r["duration_ms"])
        if not r["success"]:
            errors_by_tool[r["tool_name"]] = errors_by_tool.get(r["tool_name"], 0) + 1

    def pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        k = int(len(s) * p)
        return float(s[min(k, len(s) - 1)])

    tool_stats = []
    for name, durs in by_tool.items():
        tool_stats.append({
            "name": name,
            "count": len(durs),
            "p50": round(pct(durs, 0.50), 2),
            "p95": round(pct(durs, 0.95), 2),
            "p99": round(pct(durs, 0.99), 2),
            "max": round(max(durs), 2),
            "avg": round(sum(durs) / len(durs), 2),
            "errors": errors_by_tool.get(name, 0),
        })
    tool_stats.sort(key=lambda t: -t["count"])

    def _parse_stages(s):
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None

    recent = [
        {
            "tool": r["tool_name"],
            "ts": r["timestamp"],
            "ms": round(r["duration_ms"], 2),
            "success": bool(r["success"]),
            "error": r["error"],
            "outcome": r["outcome"],
            "client_timeout": (
                None if r["client_timeout"] is None else bool(r["client_timeout"])
            ),
            "cache_hit": (
                None if r["cache_hit"] is None else bool(r["cache_hit"])
            ),
            "backend": r["backend"],
            "stages": _parse_stages(r["stages_json"]),
        }
        for r in rows[:50]
    ]

    # 1-hour buckets for timeline
    bucket_size = 3600.0
    buckets: dict[int, list[float]] = {}
    now = time.time()
    for r in rows:
        bucket = int((now - r["timestamp"]) // bucket_size)
        if bucket < 24:
            buckets.setdefault(bucket, []).append(r["duration_ms"])
    timeline = []
    for b in range(24):
        durs = buckets.get(b, [])
        timeline.append({
            "hours_ago": b,
            "count": len(durs),
            "avg_ms": round(sum(durs) / len(durs), 2) if durs else 0.0,
        })

    # Recent turns - cluster MCP calls by time-gap. A new turn starts when
    # the gap to the previous call exceeds 30 seconds. For each turn we show:
    # - time-range (first → last call)
    # - tools used (sequence)
    # - context (matched from events table - research-activity / user-fact)
    turns = _cluster_into_turns(rows, gap_seconds=30.0)
    enriched_turns = _enrich_turns_with_context(db_path, workspace_id, turns[:20])

    return {
        "window_hours": hours,
        "total_calls": len(rows),
        "error_count": sum(errors_by_tool.values()),
        "client_timeout_count": sum(1 for r in rows if r["client_timeout"]),
        "by_tool": tool_stats,
        "recent": recent,
        "timeline_buckets": timeline,
        "turns": enriched_turns,
    }


def _cluster_into_turns(rows: list, gap_seconds: float = 30.0) -> list[dict]:
    """Group MCP calls into turns (gap > N seconds = new turn).
    Rows are ordered DESC by timestamp; we reverse for clustering.
    Returns turns newest-first.
    """
    if not rows:
        return []
    asc = list(reversed(rows))
    turns: list[dict] = []
    current: dict = {
        "first_ts": asc[0]["timestamp"],
        "last_ts": asc[0]["timestamp"],
        "tools": [asc[0]["tool_name"]],
        "total_ms": float(asc[0]["duration_ms"]),
        "n_errors": 0 if asc[0]["success"] else 1,
    }
    for r in asc[1:]:
        gap = r["timestamp"] - current["last_ts"]
        if gap > gap_seconds:
            turns.append(current)
            current = {
                "first_ts": r["timestamp"],
                "last_ts": r["timestamp"],
                "tools": [r["tool_name"]],
                "total_ms": float(r["duration_ms"]),
                "n_errors": 0 if r["success"] else 1,
            }
        else:
            current["last_ts"] = r["timestamp"]
            current["tools"].append(r["tool_name"])
            current["total_ms"] += float(r["duration_ms"])
            if not r["success"]:
                current["n_errors"] += 1
    turns.append(current)
    return list(reversed(turns))  # newest first


def _enrich_turns_with_context(
    db_path: Path, workspace_id: str | None, turns: list[dict],
) -> list[dict]:
    """For each turn, look up events in the same time-window from the
    events table - gives us the actual question/content the user wrote.

    Priority:
      1. activity(kind="research")  - what AI saved as 'user asked about X'
      2. Any fact / activity created in the same window
    """
    if not turns:
        return []
    out = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for turn in turns:
            ctx = None
            try:
                # Search events created during this turn's MCP activity window.
                # Loose bounds: ±10s either side to catch the user's save.
                lo = turn["first_ts"] - 10
                hi = turn["last_ts"] + 30
                ws_cl = ""
                params: list = [lo, hi]
                if workspace_id:
                    ws_cl = " AND workspace_id = ?"
                    params.append(workspace_id)
                row = conn.execute(
                    f"""SELECT event_type, content, metadata_json
                        FROM events
                        WHERE timestamp BETWEEN ? AND ?{ws_cl}
                        ORDER BY
                          CASE
                            WHEN json_extract(metadata_json, '$.activity_kind') = 'research' THEN 0
                            WHEN event_type = 'goal' THEN 1
                            WHEN event_type = 'fact' THEN 2
                            ELSE 3
                          END,
                          timestamp ASC
                        LIMIT 1""",
                    params,
                ).fetchone()
                if row:
                    ctx = {
                        "event_type": row["event_type"],
                        "content": (row["content"] or "")[:200],
                    }
            except Exception:
                pass
            out.append({**turn, "context": ctx})
    return out
