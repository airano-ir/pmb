"""Ambient memory - the memory observes the agent and journals on its own.

The write side of PMB historically depended on the agent remembering to
call `record_batch`. It often doesn't. Ambient memory closes that gap the
same way auto-recall closed the read side: a PostToolUse hook streams the
agent's ACTIONS into a lightweight log, and a Stop hook decides - only if
the agent didn't journal anything itself this turn - whether to synthesize
an activity entry from those raw actions.

This mixin is the data + policy layer:

  • record_agent_action      - append one observed action (PostToolUse)
  • recent_agent_actions     - read the raw action log for a window
  • agent_wrote_recently     - did the agent call record_* this turn?
  • is_significant_action     - filter: a code edit counts, `ls` doesn't
  • forget_auto_written      - user control: drop memory the agent wrote
                                itself (kept honest + reversible)

No model, no network - pure SQLite. The synthesis + CLI live elsewhere
(pmb.hooks.autowrite, `pmb track-action`, `pmb autowrite`).
"""

from __future__ import annotations

import sqlite3
import time

# Single source of truth for the significance filter + writer lives in the
# dependency-light `ambient_log` module (so the PostToolUse hot path can use
# it without importing the engine). The mixin just delegates.
from pmb.core.ambient_log import (
    ensure_agent_actions_table as _ensure_table,
)
from pmb.core.ambient_log import (
    insert_agent_action as _insert_action,
)
from pmb.core.ambient_log import (
    is_significant_action as _is_significant,
)


class AmbientMixin:
    # ──────────────────────────────────────────────────────────────────
    # schema / significance - delegate to ambient_log
    # ──────────────────────────────────────────────────────────────────
    def _ensure_agent_actions_table(self, conn: sqlite3.Connection) -> None:
        _ensure_table(conn)

    @staticmethod
    def is_significant_action(tool: str, target: str = "", command: str = "") -> bool:
        """True if this observed action is worth journaling. (Edits/tests/
        commits yes; reads/`ls` no.) See pmb.core.ambient_log."""
        return _is_significant(tool, target, command)

    # ──────────────────────────────────────────────────────────────────
    # write
    # ──────────────────────────────────────────────────────────────────
    def record_agent_action(
        self,
        tool: str,
        target: str = "",
        status: str = "",
        command: str = "",
        session_id: str | None = None,
    ) -> int | None:
        """Append one observed agent action to the raw log. Returns row id."""
        return _insert_action(
            self.workspace.db_path, self.workspace.id,
            tool=tool, target=target, status=status, command=command,
            session_id=session_id,
        )

    # ──────────────────────────────────────────────────────────────────
    # read
    # ──────────────────────────────────────────────────────────────────
    def recent_agent_actions(
        self,
        minutes: float = 30.0,
        limit: int = 200,
        significant_only: bool = False,
    ) -> list[dict]:
        """Raw observed actions in the last `minutes`, newest first."""
        cutoff = time.time() - minutes * 60.0
        out: list[dict] = []
        try:
            with sqlite3.connect(self.workspace.db_path) as conn:
                self._ensure_agent_actions_table(conn)
                conn.row_factory = sqlite3.Row
                sql = (
                    "SELECT id, timestamp, tool, target, status, significant, "
                    "session_id, surface_ids FROM agent_actions "
                    "WHERE workspace_id = ? AND timestamp >= ?"
                )
                params: list = [self.workspace.id, cutoff]
                if significant_only:
                    sql += " AND significant = 1"
                sql += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                for r in conn.execute(sql, params).fetchall():
                    out.append({
                        "id": r["id"], "timestamp": r["timestamp"],
                        "tool": r["tool"], "target": r["target"],
                        "status": r["status"], "significant": bool(r["significant"]),
                        "session_id": r["session_id"],
                        "surface_ids": r["surface_ids"] or "",
                    })
        except Exception:
            pass
        return out

    def agent_wrote_recently(self, minutes: float = 30.0) -> bool:
        """Did the agent call a record_* MCP tool in the last `minutes`?

        This is the coordination gate: if the agent already journaled its
        own work this turn, ambient memory stays silent (the agent's own
        summary is richer than anything we'd synthesize). Reads the
        `mcp_calls` perf log written by the MCP timing wrapper.
        """
        cutoff = time.time() - minutes * 60.0
        try:
            with sqlite3.connect(self.workspace.db_path) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM mcp_calls "
                    "WHERE workspace_id = ? AND timestamp >= ? "
                    "AND (tool_name LIKE 'record_%' OR tool_name = 'remember' "
                    "     OR tool_name LIKE 'index_%')",
                    (self.workspace.id, cutoff),
                ).fetchone()
                return bool(row and row[0] > 0)
        except Exception:
            # mcp_calls table may not exist (non-MCP workspace). If we can't
            # tell, assume the agent did NOT write - ambient is opt-in anyway.
            return False

    # ──────────────────────────────────────────────────────────────────
    # control - user owns every byte
    # ──────────────────────────────────────────────────────────────────
    def forget_auto_written(self, minutes: float | None = None) -> int:
        """Archive activity events that ambient memory wrote itself
        (metadata.source == 'autowrite'). Reversible-ish: archived, not
        hard-deleted. `minutes=None` → all auto-written; else just recent.

        Lets the user drop anything ambient journaled without touching what
        they or the agent recorded explicitly."""
        cutoff = (time.time() - minutes * 60.0) if minutes else 0.0
        n = 0
        try:
            with sqlite3.connect(self.workspace.db_path) as conn:
                cur = conn.execute(
                    "UPDATE events SET archived_at = ? "
                    "WHERE workspace_id = ? AND archived_at IS NULL "
                    "AND event_type = 'activity' AND timestamp >= ? "
                    "AND (metadata_json LIKE '%\"source\":\"autowrite\"%' "
                    "     OR metadata_json LIKE '%\"source\": \"autowrite\"%')",
                    (time.time(), self.workspace.id, cutoff),
                )
                n = cur.rowcount
                conn.commit()
        except Exception:
            pass
        return n
