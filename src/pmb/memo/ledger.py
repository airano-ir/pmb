"""Memory Delta Protocol - a per-session ledger of what PMB has already shown
the agent, so subsequent injections can render as deltas instead of full repeat.

Rationale: between chats, Anthropic's prompt cache does NOT survive - PMB pays
full price re-injecting auto-context every cold session. WITHIN a chat, the
cache helps for the parts that repeat, but the lesson/decision block changes per
message and is largely uncached. The ledger lets `format_context` emit
"[M12 still active]" instead of restating a 200-char rule that was just shown
two turns ago in the same session.

How:
  * Each item (lesson by surface_id, fact/decision by ulid) gets a stable
    short handle (M01, M02, ...) on first injection in a session.
  * The ledger stores (session_id, kind, ref_id, content_hash, handle, shown_at).
  * On the next turn, the renderer asks `lookup(session_id, kind, ref_id, sha)`:
      - hit + same content -> tell the renderer to compact it ("M12 still active")
      - hit + different content -> hand back the prior handle as STALE, render the
        new content + tag as "M12 updated"
      - miss -> assign a new handle and render the full item
  * SessionStart fires a `rehydrate(session_id)`: clears handles for that
    session so the next render restates full text (the "context rebase" step,
    which is critical because /compact loses the model's handle->text mapping).

Default OFF (config gate). Off the hot path until someone opts in; ledger
tables auto-create on first write.
"""
from __future__ import annotations

import hashlib
import sqlite3
import time

_KIND_LESSON = "lesson"
_KIND_DECISION = "decision"
_KIND_FACT = "fact"


def content_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="replace")).hexdigest()[:16]


def ensure_table(conn: sqlite3.Connection) -> None:
    """Idempotent. Per-session ledger; one row per (session, kind, ref_id)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_ledger (
            workspace_id  TEXT NOT NULL,
            session_id    TEXT NOT NULL,
            kind          TEXT NOT NULL,
            ref_id        TEXT NOT NULL,
            handle        TEXT NOT NULL,
            content_hash  TEXT NOT NULL,
            shown_at      REAL NOT NULL,
            last_seen_at  REAL NOT NULL,
            PRIMARY KEY (workspace_id, session_id, kind, ref_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_ledger_session "
        "ON memory_ledger(workspace_id, session_id)"
    )


def _next_handle(conn: sqlite3.Connection, workspace_id: str, session_id: str) -> str:
    """Allocate the next M-handle for this session: M01, M02, ..., M99, M100..."""
    row = conn.execute(
        "SELECT COUNT(*) FROM memory_ledger WHERE workspace_id=? AND session_id=?",
        (workspace_id, session_id),
    ).fetchone()
    n = (row[0] if row else 0) + 1
    return f"M{n:02d}"


def lookup(
    conn: sqlite3.Connection, workspace_id: str, session_id: str,
    kind: str, ref_id: str,
) -> dict | None:
    """Return the ledger row for this item this session, or None.
    Shape: {handle, content_hash, shown_at, last_seen_at}."""
    if not (workspace_id and session_id and ref_id):
        return None
    row = conn.execute(
        "SELECT handle, content_hash, shown_at, last_seen_at FROM memory_ledger "
        "WHERE workspace_id=? AND session_id=? AND kind=? AND ref_id=?",
        (workspace_id, session_id, kind, str(ref_id)),
    ).fetchone()
    if not row:
        return None
    return {
        "handle": row[0], "content_hash": row[1],
        "shown_at": row[2], "last_seen_at": row[3],
    }


def register(
    conn: sqlite3.Connection, workspace_id: str, session_id: str,
    kind: str, ref_id: str, content: str,
) -> dict:
    """Record/refresh an item in the ledger. Returns:
      {handle, status, prior_handle}
    status:
      - 'new'        -> first time this session, full render
      - 'unchanged'  -> same content_hash as before, compact "still active"
      - 'updated'    -> same item, content changed, render full + tag updated
    """
    now = time.time()
    chash = content_hash(content)
    if not (session_id and ref_id):
        # Without a session id we can't ledger; treat as fresh "new".
        return {"handle": "", "status": "new", "prior_handle": ""}
    prior = lookup(conn, workspace_id, session_id, kind, str(ref_id))
    if prior is None:
        handle = _next_handle(conn, workspace_id, session_id)
        conn.execute(
            "INSERT INTO memory_ledger "
            "(workspace_id, session_id, kind, ref_id, handle, content_hash, "
            " shown_at, last_seen_at) VALUES (?,?,?,?,?,?,?,?)",
            (workspace_id, session_id, kind, str(ref_id), handle, chash, now, now),
        )
        return {"handle": handle, "status": "new", "prior_handle": ""}
    if prior["content_hash"] == chash:
        conn.execute(
            "UPDATE memory_ledger SET last_seen_at=? WHERE workspace_id=? "
            "AND session_id=? AND kind=? AND ref_id=?",
            (now, workspace_id, session_id, kind, str(ref_id)),
        )
        return {"handle": prior["handle"], "status": "unchanged",
                "prior_handle": prior["handle"]}
    # Same item, content shifted -> keep same handle, refresh hash + ts.
    conn.execute(
        "UPDATE memory_ledger SET content_hash=?, last_seen_at=? "
        "WHERE workspace_id=? AND session_id=? AND kind=? AND ref_id=?",
        (chash, now, workspace_id, session_id, kind, str(ref_id)),
    )
    return {"handle": prior["handle"], "status": "updated",
            "prior_handle": prior["handle"]}


def rehydrate(conn: sqlite3.Connection, workspace_id: str, session_id: str) -> int:
    """Wipe the session's ledger (Context Rebase). Use after /compact, since the
    model has lost the handle->text mapping; the next render must restate full
    text. Returns the number of cleared rows."""
    cur = conn.execute(
        "DELETE FROM memory_ledger WHERE workspace_id=? AND session_id=?",
        (workspace_id, session_id),
    )
    return cur.rowcount or 0


def active_handles(
    conn: sqlite3.Connection, workspace_id: str, session_id: str, limit: int = 50,
) -> list[dict]:
    """All currently-known handles in this session, newest first - for the
    'Still active: M07, M12, M14' summary line and for `pmb memory ledger`."""
    rows = conn.execute(
        "SELECT kind, ref_id, handle, last_seen_at FROM memory_ledger "
        "WHERE workspace_id=? AND session_id=? "
        "ORDER BY last_seen_at DESC, rowid DESC LIMIT ?",
        (workspace_id, session_id, limit),
    ).fetchall()
    return [
        {"kind": r[0], "ref_id": r[1], "handle": r[2], "last_seen_at": r[3]}
        for r in rows
    ]


def expired_since(
    conn: sqlite3.Connection, workspace_id: str, session_id: str,
    seen_now: set[tuple[str, str]], stale_after_s: float = 120.0,
) -> list[dict]:
    """Items in the ledger NOT in `seen_now` and not touched recently -> the
    'Expired: M07, M09' line. `seen_now` is the (kind, ref_id) pairs in the
    current render. Conservative: we only flag items the renderer decided not
    to surface AND that have been quiet for `stale_after_s` seconds."""
    cutoff = time.time() - stale_after_s
    rows = conn.execute(
        "SELECT kind, ref_id, handle FROM memory_ledger "
        "WHERE workspace_id=? AND session_id=? AND last_seen_at < ?",
        (workspace_id, session_id, cutoff),
    ).fetchall()
    return [
        {"kind": r[0], "ref_id": r[1], "handle": r[2]}
        for r in rows if (r[0], r[1]) not in seen_now
    ]
