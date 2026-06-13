"""Lightweight agent-action log - the hot path for the PostToolUse hook.

`pmb track-action` runs on EVERY tool the agent uses, so it must not pull
the heavy engine stack (numpy / lancedb / sentence-transformers). This
module is dependency-light (re + sqlite3 only) so the hook stays ~fast: a
small SQL transaction, no model, no vector index.

The full AmbientMixin reuses the significance filter and writer here, so
there's one source of truth.
"""

from __future__ import annotations

import re
import sqlite3
import time

# ── significance filter ──────────────────────────────────────────────────

_SIGNIFICANT_TOOLS = frozenset({
    "Edit", "Write", "MultiEdit", "NotebookEdit", "Update", "Create",
    "str_replace_editor", "create_file", "edit_file", "apply_patch",
})
_TRIVIAL_TOOLS = frozenset({
    "Read", "Glob", "Grep", "LS", "TodoWrite", "Task", "WebFetch",
    "WebSearch", "NotebookRead", "list_files", "read_file", "search_files",
})
_SIGNIFICANT_BASH = re.compile(
    r"\b(git\s+(commit|push|merge|tag|rebase|cherry-pick|revert)|"
    r"pytest|tox|nox|unittest|"
    r"npm\s+(run|test|build|publish|ci)|pnpm\s+(run|test|build)|"
    r"yarn\s+(run|test|build)|"
    r"cargo\s+(build|test|run|publish)|go\s+(build|test|run)|"
    r"make\b|cmake\b|docker\s+(build|push|compose)|"
    r"pip\s+install|poetry\s+(install|build|publish)|uv\s+(pip|run)|"
    r"deploy|migrate|alembic|terraform\s+(apply|plan)|kubectl\s+apply)\b",
    re.IGNORECASE,
)
_TRIVIAL_BASH = re.compile(
    r"^\s*(cat|ls|cd|echo|pwd|grep|rg|head|tail|find|which|where|env|"
    r"export|printf|wc|sort|uniq|cut|sed\s+-n|awk|stat|file|tree|"
    r"git\s+(status|log|diff|show|branch|remote)|"
    # PowerShell read cmdlets (Codex on Windows uses these heavily)
    r"Get-(ChildItem|Content|Command|Item|Location|Process|ItemProperty)|"
    r"Select-String|Test-Path|Measure-Object|Where-Object|ForEach-Object|"
    r"Sort-Object|Select-Object|Format-(Table|List)|Out-(String|Host)|"
    r"Resolve-Path|Split-Path|Join-Path|Set-Location)\b",
    re.IGNORECASE,
)


def is_significant_action(tool: str, target: str = "", command: str = "") -> bool:
    """True if this action is worth journaling. Edits/writes/tests/commits
    count; reads/searches/`ls` don't."""
    t = (tool or "").strip()
    if t in _TRIVIAL_TOOLS:
        return False
    if t in _SIGNIFICANT_TOOLS:
        return True
    if t in ("Bash", "shell", "run_command", "execute"):
        cmd = command or target or ""
        if _TRIVIAL_BASH.search(cmd):
            return False
        if _SIGNIFICANT_BASH.search(cmd):
            return True
        return False
    return False


# ── schema + writer ──────────────────────────────────────────────────────

def ensure_agent_actions_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_actions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT,
            timestamp    REAL NOT NULL,
            tool         TEXT NOT NULL,
            target       TEXT,
            status       TEXT,
            significant  INTEGER DEFAULT 0,
            session_id   TEXT,
            surface_ids  TEXT
        )
        """
    )
    cols = {r[1] for r in conn.execute("PRAGMA table_info(agent_actions)").fetchall()}
    if "surface_ids" not in cols:
        conn.execute("ALTER TABLE agent_actions ADD COLUMN surface_ids TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_actions_ts "
        "ON agent_actions(workspace_id, timestamp DESC)"
    )


def _active_surface_ids(
    conn: sqlite3.Connection,
    workspace_id: str,
    now: float,
    *,
    session_id: str | None = None,
    window_seconds: float = 1800.0,
    limit: int = 25,
) -> str:
    """Comma-separated recent unconfirmed lesson surface ids.

    This lets the Stop-hook later say "this action happened while these
    lessons were active" instead of relying only on a broad time window. Best
    effort: fresh schemas or non-PMB workspaces simply return "".
    """
    try:
        sql = (
            "SELECT id FROM lesson_surfaces "
            "WHERE workspace_id = ? AND surfaced_at >= ? AND surfaced_at <= ? "
            "AND followed IS NULL"
        )
        params: list = [workspace_id, now - window_seconds, now]
        if session_id:
            sql += " AND (session_id IS NULL OR session_id = ?)"
            params.append(session_id)
        sql += " ORDER BY surfaced_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return ",".join(str(r[0]) for r in rows if r and r[0] is not None)
    except Exception:
        return ""


def insert_agent_action(
    db_path,
    workspace_id: str,
    tool: str,
    target: str = "",
    status: str = "",
    command: str = "",
    session_id: str | None = None,
) -> int | None:
    """Append one observed action. Returns row id or None on failure.

    The hot path: one small SQLite transaction, no engine, no model.
    """
    sig = 1 if is_significant_action(tool, target, command) else 0
    now = time.time()
    try:
        # The workspace dir may not exist yet on a brand-new workspace (the
        # engine usually creates it; the hot path doesn't load the engine).
        from pathlib import Path as _P
        _P(str(db_path)).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db_path)) as conn:
            ensure_agent_actions_table(conn)
            surface_ids = _active_surface_ids(
                conn, workspace_id, now, session_id=session_id,
            )
            cur = conn.execute(
                "INSERT INTO agent_actions "
                "(workspace_id, timestamp, tool, target, status, "
                " significant, session_id, surface_ids) VALUES (?,?,?,?,?,?,?,?)",
                (workspace_id, now, (tool or "")[:80],
                 (target or "")[:400], (status or "")[:40], sig, session_id,
                 surface_ids),
            )
            conn.commit()
            return cur.lastrowid
    except Exception:
        return None


# ── observability: "how much did the AUTO layer do?" ─────────────────────


def auto_layer_stats(db_path, workspace_id: str, days: float = 7.0) -> dict:
    """Tally what PMB's automatic layer did on its own - no model cooperation.

    Light (sqlite only). Counts, for the last `days` and all-time:
      • ambient_writes      - memory entries the auto-writer saved
                              (events with metadata.source == 'autowrite')
      • auto_surfaces       - lessons the auto-recall HOOK injected
                              (lesson_surfaces.source LIKE 'hook.%')
      • auto_followed       - of those, ones followcheck auto-confirmed followed
      • observed_significant- significant actions the observer logged
      • agent_records       - times the agent self-recorded via MCP (record_*),
                              i.e. turns where ambient deliberately stayed silent

    Every value degrades to 0 if a table is missing (fresh / non-MCP ws).
    """
    cutoff = time.time() - days * 86400.0
    ws = workspace_id
    out = {
        "days": days,
        "ambient_writes": 0, "ambient_writes_window": 0,
        "ambient_importance_avg": 0.0,
        "auto_surfaces": 0, "auto_surfaces_window": 0,
        "auto_followed_window": 0,
        "observed_significant_window": 0,
        "agent_records_window": 0,
    }
    _AUTO = ("(metadata_json LIKE '%\"source\":\"autowrite\"%' "
             "OR metadata_json LIKE '%\"source\": \"autowrite\"%')")
    try:
        with sqlite3.connect(str(db_path)) as c:
            def one(sql, params):
                try:
                    r = c.execute(sql, params).fetchone()
                    return r[0] if r and r[0] is not None else 0
                except Exception:
                    return 0
            out["ambient_writes"] = one(
                f"SELECT COUNT(*) FROM events WHERE workspace_id=? "
                f"AND archived_at IS NULL AND {_AUTO}", (ws,))
            out["ambient_writes_window"] = one(
                f"SELECT COUNT(*) FROM events WHERE workspace_id=? "
                f"AND archived_at IS NULL AND timestamp>=? AND {_AUTO}",
                (ws, cutoff))
            out["ambient_importance_avg"] = round(float(one(
                f"SELECT AVG(importance) FROM events WHERE workspace_id=? "
                f"AND archived_at IS NULL AND timestamp>=? AND {_AUTO}",
                (ws, cutoff)) or 0.0), 3)
            out["auto_surfaces"] = one(
                "SELECT COUNT(*) FROM lesson_surfaces WHERE workspace_id=? "
                "AND source LIKE 'hook.%'", (ws,))
            out["auto_surfaces_window"] = one(
                "SELECT COUNT(*) FROM lesson_surfaces WHERE workspace_id=? "
                "AND source LIKE 'hook.%' AND surfaced_at>=?", (ws, cutoff))
            out["auto_followed_window"] = one(
                "SELECT COUNT(*) FROM lesson_surfaces WHERE workspace_id=? "
                "AND followed=1 AND surfaced_at>=? "
                "AND follow_note LIKE '%auto-detected%'", (ws, cutoff))
            out["observed_significant_window"] = one(
                "SELECT COUNT(*) FROM agent_actions WHERE workspace_id=? "
                "AND significant=1 AND timestamp>=?", (ws, cutoff))
            out["agent_records_window"] = one(
                "SELECT COUNT(*) FROM mcp_calls WHERE workspace_id=? "
                "AND timestamp>=? AND (tool_name LIKE 'record_%' "
                "OR tool_name='remember' OR tool_name LIKE 'index_%')",
                (ws, cutoff))
    except Exception:
        pass
    return out
