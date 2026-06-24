"""Memory Delta renderer: compact `format_context` output using the per-session
ledger. Items already seen this session and unchanged collapse to a one-line
"still active" reference; new items get a fresh handle; items previously seen
but absent from this render are surfaced under "Expired".

Wrap (don't replace) the existing renderer:
    full = format_context(res, ...)
    delta = apply_delta(engine, session_id, res, full)

When the ledger is empty for this session or the feature is off, returns the
input unchanged.
"""
from __future__ import annotations

import re
import sqlite3

from pmb.memo.ledger import (
    _KIND_DECISION,
    _KIND_FACT,
    _KIND_LESSON,
    active_handles,
    ensure_table,
    expired_since,
    register,
)

_SURFACE_RE = re.compile(r"\[surface_id=(\d+)\]")


def _content_for_lesson(L: dict) -> str:
    return (L.get("content") or "")[:200]


def _content_for_fact(d: dict) -> str:
    return (d.get("content") or "")[:200]


def _truncate(s: str, n: int = 80) -> str:
    s = (s or "").strip().replace("\r", " ").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def apply_delta(
    engine, session_id: str, res, rendered: str,
) -> str:
    """Compact `rendered` against this session's ledger.

    For each lesson with a `surface_id` already shown unchanged this session,
    replaces the full lesson line with "  ! [M07] still active: <preview>".
    Adds "Still active: M01, M07, …" and "Expired: M09" lines at the top.
    """
    if not (session_id and rendered):
        return rendered

    seen_now: set[tuple[str, str]] = set()
    handle_status: dict[int, dict] = {}   # surface_id -> {handle, status}

    db = engine.workspace.db_path
    ws = engine.workspace.id
    try:
        with sqlite3.connect(db) as conn:
            ensure_table(conn)
            # Lessons (top-level + nested in project_context) - keyed by surface_id.
            lessons = list(res.lessons or [])
            for L in (res.project or {}).get("lessons") or []:
                lessons.append(L)
            for L in lessons:
                sid = L.get("surface_id")
                if not sid:
                    continue
                row = register(conn, ws, session_id, _KIND_LESSON,
                               str(sid), _content_for_lesson(L))
                handle_status[int(sid)] = row
                seen_now.add((_KIND_LESSON, str(sid)))

            # Decisions / facts use ulid as ref_id when available; status is
            # tracked but the renderer doesn't compact them yet (rules are the
            # noisiest channel). We still register so "Still active" includes
            # them, which is the cheap, useful half.
            for d in (res.decisions or []) + ((res.project or {}).get("decisions") or []):
                rid = d.get("ulid") or ""
                if not rid:
                    continue
                register(conn, ws, session_id, _KIND_DECISION,
                         rid, _content_for_fact(d))
                seen_now.add((_KIND_DECISION, rid))

            for h in res.recall_hits or []:
                rid = h.get("ulid") or ""
                if not rid:
                    continue
                register(conn, ws, session_id, _KIND_FACT,
                         rid, _content_for_fact(h))
                seen_now.add((_KIND_FACT, rid))

            expired = expired_since(conn, ws, session_id, seen_now)
            actives = active_handles(conn, ws, session_id, limit=30)
            conn.commit()
    except Exception:
        return rendered  # never break the hook over a ledger error

    if not handle_status and not expired and not actives:
        return rendered

    # Rewrite each "  ! <lesson>... [surface_id=N]" line that came in
    # 'unchanged' from this session into a compact handle reference.
    out_lines: list[str] = []
    for raw in rendered.splitlines():
        m = _SURFACE_RE.search(raw)
        if not m:
            out_lines.append(raw)
            continue
        sid = int(m.group(1))
        info = handle_status.get(sid)
        if info and info["status"] == "unchanged" and info["handle"]:
            # Aggressive: drop the rule text entirely, just keep the handle ref.
            # The model has the full text from an earlier turn; restating buys
            # nothing within a chat (prompt cache) and the across-chat / post-
            # compact case is handled by rehydrate restating the full text.
            out_lines.append(f"  ! [{info['handle']}] still active")
        elif info and info["status"] == "updated" and info["handle"]:
            # Keep full text, tag as updated so the agent re-reads it.
            out_lines.append(raw + f"  [updated, was {info['handle']}]")
        elif info and info["handle"]:
            # New this session: tag with the freshly-assigned handle.
            out_lines.append(raw + f"  [{info['handle']}]")
        else:
            out_lines.append(raw)

    # Only the EXPIRED line is added unconditionally - it's a correctness signal
    # (don't keep relying on M07, it's gone). 'Still active' summary is cheap
    # redundancy with the inline tags and we keep it OUT to avoid bloating the
    # block. _ = actives is intentional (kept for future "list view" CLI).
    _ = actives
    if expired:
        line = (
            "Expired since last turn: "
            + ", ".join(e["handle"] for e in expired[:10])
            + " (no longer surfaced; treat as stale)"
        )
        insert_at = 2 if len(out_lines) >= 2 else len(out_lines)
        out_lines.insert(insert_at, line)

    return "\n".join(out_lines)
