"""
Event Store - SQLite-based append-only event log.

Event types:
- "qa"      - Q/A pair from the agent
- "fact"    - extracted key=value fact
- "pin"     - explicit user pin
- "git"     - git event (commit, branch change)
- "file"    - file modification context
- "test"    - test result context

Schema (v1):
- events (id, ulid, workspace_id, event_type, content, metadata_json,
          timestamp, importance, access_count, last_accessed,
          archived_at, source_session_id)
- migrations (version, applied_at)

Indexes:
- workspace + recency
- event_type
- archived (to exclude from recall)
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 6


# Memory tiers (loose analogy to STM/MTM/LTM in human memory).
# Each event lives in exactly one tier at any time. Tiers differ only in
# how fast importance decays - recall reads all active tiers equally.
TIER_WORKING = "working"      # new memory, fast decay if not reinforced (~1 day half-life)
TIER_EPISODIC = "episodic"    # confirmed memory of a specific event (~46 day half-life)
TIER_SEMANTIC = "semantic"    # abstracted fact / decision / rule (~1 year half-life)


def default_tier_for_event_type(event_type: str) -> str:
    """Where a freshly-recorded event lands by default."""
    if event_type == "fact":
        return TIER_SEMANTIC
    return TIER_WORKING


# Per-tier daily decay multiplier. Compounded for `days_since_last_decay`.
TIER_DECAY_FACTORS: dict[str, float] = {
    TIER_WORKING: 0.70,   # half-life ≈ 1.94 days
    TIER_EPISODIC: 0.985, # half-life ≈ 46 days
    TIER_SEMANTIC: 0.998, # half-life ≈ 346 days
}


# Promotion thresholds (re-access counts) for moving up the tiers.
# Lowered from 3/10 -> 2/7 after dogfooding showed that even sustained recall
# loops rarely promote past the first threshold within a single session,
# because query phrasings vary too much for any single event to hit top-K
# three times. Two repeats is a more honest "this is a recurring topic"
# signal, and 7 is enough to mark something as semantic without letting
# one-off events drift up.
PROMOTE_WORKING_TO_EPISODIC_ACCESS = 2
PROMOTE_EPISODIC_TO_SEMANTIC_ACCESS = 7


@dataclass
class Event:
    """One record in memory."""

    id: int | None = None
    ulid: str = field(default_factory=lambda: _ulid())
    workspace_id: str = ""
    event_type: str = "qa"
    content: str = ""
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    archived_at: float | None = None
    source_session_id: str | None = None
    tier: str = TIER_WORKING

    def to_db_row(self) -> tuple:
        return (
            self.ulid,
            self.workspace_id,
            self.event_type,
            self.content,
            json.dumps(self.metadata, ensure_ascii=False),
            self.timestamp,
            self.importance,
            self.access_count,
            self.last_accessed,
            self.archived_at,
            self.source_session_id,
            self.tier,
        )

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> Event:
        # `tier` may be absent in pre-v3 rows; default to "working"
        try:
            tier_val = row["tier"]
        except (KeyError, IndexError):
            tier_val = TIER_WORKING
        return cls(
            id=row["id"],
            ulid=row["ulid"],
            workspace_id=row["workspace_id"],
            event_type=row["event_type"],
            content=row["content"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            timestamp=row["timestamp"],
            importance=row["importance"],
            access_count=row["access_count"],
            last_accessed=row["last_accessed"],
            archived_at=row["archived_at"],
            source_session_id=row["source_session_id"],
            tier=tier_val or TIER_WORKING,
        )

    def to_text(self) -> str:
        """Text representation for embedding."""
        if self.event_type == "qa":
            q = self.metadata.get("query", "")
            a = self.content
            return f"Q: {q}\nA: {a}"
        elif self.event_type == "fact":
            return f"Fact: {self.content}"
        elif self.event_type == "git":
            return f"Git event: {self.content}"
        elif self.event_type == "file":
            return f"File context: {self.content}"
        elif self.event_type == "test":
            return f"Test result: {self.content}"
        elif self.event_type == "pin":
            return f"Pinned: {self.content}"
        return self.content


def _ulid() -> str:
    """Simple ULID-like sortable ID."""
    return f"{int(time.time() * 1000):013x}_{uuid.uuid4().hex[:8]}"


_DDL = [
    """
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ulid TEXT UNIQUE NOT NULL,
        workspace_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        content TEXT NOT NULL,
        metadata_json TEXT,
        timestamp REAL NOT NULL,
        importance REAL DEFAULT 0.5,
        access_count INTEGER DEFAULT 0,
        last_accessed REAL NOT NULL,
        archived_at REAL,
        source_session_id TEXT,
        tier TEXT NOT NULL DEFAULT 'working'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_workspace_time ON events(workspace_id, timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_workspace_active ON events(workspace_id, archived_at) WHERE archived_at IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_event_type ON events(workspace_id, event_type)",
    # S7: expression index on metadata kind so find_lessons / find_decisions
    # become an indexed lookup instead of a full-table `metadata_json LIKE
    # '%"kind":"lesson"%'` scan on every recall + hook message. COALESCE folds
    # both spellings (lessons use metadata.kind; activity-style decisions use
    # metadata.activity_kind) into one indexed value. Pure additive DDL - no
    # column, no write-path change, no backfill; also whitespace-robust where
    # the old LIKE needed two spacing variants. The query expression MUST match
    # this one verbatim for the planner to use the index.
    "CREATE INDEX IF NOT EXISTS idx_meta_kind ON events(workspace_id, "
    "COALESCE(json_extract(metadata_json, '$.kind'), "
    "json_extract(metadata_json, '$.activity_kind')))",
    "CREATE INDEX IF NOT EXISTS idx_recency ON events(last_accessed DESC)",
    "CREATE INDEX IF NOT EXISTS idx_tier ON events(workspace_id, tier, archived_at) WHERE archived_at IS NULL",
    """
    CREATE TABLE IF NOT EXISTS migrations (
        version INTEGER PRIMARY KEY,
        applied_at REAL NOT NULL
    )
    """,
    # ------------------------------------------------------------------
    # v4: reasoning layer - direct event-to-event edges, narrative arcs.
    # Reflections live in `events` with event_type='reflection' + metadata
    # pointing to source ulid, so no new table for them.
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS event_edges (
        source_ulid TEXT NOT NULL,
        target_ulid TEXT NOT NULL,
        edge_type TEXT NOT NULL,
        confidence REAL NOT NULL DEFAULT 1.0,
        rationale TEXT,
        created_at REAL NOT NULL,
        PRIMARY KEY (source_ulid, target_ulid, edge_type)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_eedges_src ON event_edges(source_ulid, edge_type)",
    "CREATE INDEX IF NOT EXISTS idx_eedges_tgt ON event_edges(target_ulid, edge_type)",
    """
    CREATE TABLE IF NOT EXISTS arcs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workspace_id TEXT NOT NULL,
        title TEXT NOT NULL,
        summary TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'active',
        first_event_ulid TEXT,
        last_event_ulid TEXT,
        n_events INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL,
        last_updated REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_arcs_workspace ON arcs(workspace_id, status, last_updated DESC)",
    """
    CREATE TABLE IF NOT EXISTS arc_events (
        arc_id INTEGER NOT NULL,
        event_ulid TEXT NOT NULL,
        PRIMARY KEY (arc_id, event_ulid),
        FOREIGN KEY (arc_id) REFERENCES arcs(id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_arc_events_event ON arc_events(event_ulid)",
    # ------------------------------------------------------------------
    # v5: predictive cache (Improvement F).
    # LLM in sleep predicts likely user questions and pre-computes their
    # top-K results. At recall time, fuzzy match (cosine on embeddings)
    # → instant cache hit.
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS predictive_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workspace_id TEXT NOT NULL,
        query_text TEXT NOT NULL,
        query_embedding BLOB NOT NULL,
        top_ulids_json TEXT NOT NULL,
        created_at REAL NOT NULL,
        last_hit_at REAL,
        n_hits INTEGER NOT NULL DEFAULT 0,
        UNIQUE(workspace_id, query_text)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_pc_workspace ON predictive_cache(workspace_id, created_at DESC)",
    # ------------------------------------------------------------------
    # v6: lesson surface tracking.
    # Every time a lesson is surfaced to the agent (via recall, overview,
    # find_lessons, project_overview) we log one row. The agent can later
    # call mark_lesson_followed(surface_id, True/False) to confirm whether
    # the lesson actually changed its behaviour. Powers the self-improvement
    # loop ("of 12 lessons surfaced last week, 7 were followed").
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS lesson_surfaces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workspace_id TEXT NOT NULL,
        lesson_ulid TEXT NOT NULL,
        query TEXT NOT NULL,
        source TEXT NOT NULL,
        surfaced_at REAL NOT NULL,
        session_id TEXT,
        followed INTEGER DEFAULT NULL,
        follow_note TEXT,
        followed_at REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ls_lesson ON lesson_surfaces(lesson_ulid)",
    "CREATE INDEX IF NOT EXISTS idx_ls_ws_time ON lesson_surfaces(workspace_id, surfaced_at DESC)",
    # ------------------------------------------------------------------
    # v7: durable write outbox + error log (additive, IF NOT EXISTS so
    # existing DBs pick them up on next open; no schema-version bump needed).
    #
    # write_outbox: record_batch_async enqueues a row HERE synchronously
    # (~1ms) before returning, so a crash between accept and the background
    # write loses nothing - the drainer (and recover_on_start) replay pendings.
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS write_outbox (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workspace_id TEXT NOT NULL,
        created_at REAL NOT NULL,
        items_json TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        attempts INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        done_at REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_outbox_status ON write_outbox(status, id)",
    # error_log: one row per swallowed exception that used to be a bare
    # `except Exception: pass`. Surfaced by `pmb doctor` / the status panel.
    """
    CREATE TABLE IF NOT EXISTS error_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
        component TEXT NOT NULL,
        message TEXT,
        trace_head TEXT,
        note TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_errlog_ts ON error_log(ts DESC)",
]


def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    """Add `tier` column to existing v1/v2 databases without dropping data."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(events)").fetchall()]
    if "tier" not in cols:
        conn.execute("ALTER TABLE events ADD COLUMN tier TEXT NOT NULL DEFAULT 'working'")
    # Backfill: facts go to semantic; other content events stay 'working' then
    # get promoted naturally by recall reinforcement.
    conn.execute(
        "UPDATE events SET tier = 'semantic' WHERE tier IS NULL OR tier = 'working' "
        "  AND event_type = 'fact'"
    )
    conn.execute(
        "UPDATE events SET tier = 'episodic' WHERE tier IS NULL OR tier = 'working' "
        "  AND event_type IN ('git', 'qa') AND access_count >= 1"
    )


# Safe SQLite IN(...) chunk size. SQLite default
# `SQLITE_MAX_VARIABLE_NUMBER` is 999 pre-3.32, 32766 newer; we stay well
# under both. Used by `get_many` and any future bulk fetch.
_GET_MANY_CHUNK = 500


class EventStore:
    """SQLite-based event store. Thread-safe via the connection-per-call pattern."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        from pmb.core.sqlite_helper import apply_pragmas
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        apply_pragmas(conn)
        try:
            yield conn
        finally:
            conn.close()

    def _migrate(self):
        with self._conn() as conn:
            for ddl in _DDL:
                conn.execute(ddl)
            cur = conn.execute("SELECT MAX(version) FROM migrations")
            current = cur.fetchone()[0] or 0
            # Forward migrations for existing DBs
            if current < 3:
                _migrate_v2_to_v3(conn)
            if current < SCHEMA_VERSION:
                conn.execute(
                    "INSERT INTO migrations (version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, time.time()),
                )

    # -----------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------

    def append(self, event: Event) -> Event:
        """Persist an event. Returns the event with its id populated."""
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO events (
                    ulid, workspace_id, event_type, content, metadata_json,
                    timestamp, importance, access_count, last_accessed,
                    archived_at, source_session_id, tier
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                event.to_db_row(),
            )
            event.id = cur.lastrowid
        return event

    def append_many(self, events: list[Event]) -> list[Event]:
        with self._conn() as conn:
            for event in events:
                cur = conn.execute(
                    """
                    INSERT INTO events (
                        ulid, workspace_id, event_type, content, metadata_json,
                        timestamp, importance, access_count, last_accessed,
                        archived_at, source_session_id, tier
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    event.to_db_row(),
                )
                event.id = cur.lastrowid
        return events

    def update_tier(self, ulid: str, tier: str) -> None:
        """Move an event between memory tiers (working → episodic → semantic)."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE events SET tier = ? WHERE ulid = ?", (tier, ulid),
            )

    # -----------------------------------------------------------------
    # Read
    # -----------------------------------------------------------------

    def get_by_ulid(self, ulid: str) -> Event | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM events WHERE ulid = ?", (ulid,)
            ).fetchone()
            return Event.from_db_row(row) if row else None

    def get_many(
        self,
        ulids: list[str],
        workspace_id: str | None = None,
        only_active: bool = True,
    ) -> dict[str, Event]:
        """
        Batched fetch by ulid list. Returns {ulid: Event} map.

        Avoids the O(N) scan when ranking only needs a few candidate rows.

        Hardening (H6): chunks the IN(...) into batches of `_GET_MANY_CHUNK`
        so we never trip SQLITE_MAX_VARIABLE_NUMBER (default 999 pre-3.32,
        32766 newer). Recall on a workspace with a dense graph + large
        top_k can otherwise pile up >1000 candidate ulids in one shot.
        """
        if not ulids:
            return {}
        # Dedup while preserving order - caller might pass duplicates
        seen: set[str] = set()
        unique: list[str] = []
        for u in ulids:
            if u not in seen:
                seen.add(u)
                unique.append(u)

        out: dict[str, Event] = {}
        with self._conn() as conn:
            for chunk_start in range(0, len(unique), _GET_MANY_CHUNK):
                chunk = unique[chunk_start: chunk_start + _GET_MANY_CHUNK]
                placeholders = ",".join("?" * len(chunk))
                sql = f"SELECT * FROM events WHERE ulid IN ({placeholders})"
                params: list = list(chunk)
                if workspace_id is not None:
                    sql += " AND workspace_id = ?"
                    params.append(workspace_id)
                if only_active:
                    sql += " AND archived_at IS NULL"
                for r in conn.execute(sql, params).fetchall():
                    out[r["ulid"]] = Event.from_db_row(r)
        return out

    def list_active(self, workspace_id: str, limit: int = 100,
                    event_type: str | None = None) -> list[Event]:
        """Active (non-archived) events for the workspace."""
        with self._conn() as conn:
            if event_type:
                rows = conn.execute(
                    """
                    SELECT * FROM events
                    WHERE workspace_id = ? AND archived_at IS NULL AND event_type = ?
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (workspace_id, event_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM events
                    WHERE workspace_id = ? AND archived_at IS NULL
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (workspace_id, limit),
                ).fetchall()
            return [Event.from_db_row(r) for r in rows]

    def list_since(self, workspace_id: str, cutoff: float,
                   session_id: str | None = None,
                   limit: int = 5000) -> list[Event]:
        """Active events newer than `cutoff` (epoch seconds), OR explicitly
        tagged with `session_id`. Pushes the session_brief scope into SQL
        (uses idx_workspace_time) instead of loading the whole table and
        filtering in Python (S5)."""
        with self._conn() as conn:
            if session_id:
                rows = conn.execute(
                    """
                    SELECT * FROM events
                    WHERE workspace_id = ? AND archived_at IS NULL
                      AND (timestamp >= ? OR source_session_id = ?)
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (workspace_id, cutoff, session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM events
                    WHERE workspace_id = ? AND archived_at IS NULL
                      AND timestamp >= ?
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (workspace_id, cutoff, limit),
                ).fetchall()
            return [Event.from_db_row(r) for r in rows]

    def count(self, workspace_id: str, include_archived: bool = False) -> int:
        with self._conn() as conn:
            if include_archived:
                cur = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE workspace_id = ?",
                    (workspace_id,),
                )
            else:
                cur = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE workspace_id = ? AND archived_at IS NULL",
                    (workspace_id,),
                )
            return cur.fetchone()[0]

    def stats(self, workspace_id: str) -> dict:
        with self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM events WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM events WHERE workspace_id = ? AND archived_at IS NULL",
                (workspace_id,),
            ).fetchone()[0]
            by_type_rows = conn.execute(
                """
                SELECT event_type, COUNT(*) AS n FROM events
                WHERE workspace_id = ? AND archived_at IS NULL
                GROUP BY event_type
                """,
                (workspace_id,),
            ).fetchall()
            by_type = {r["event_type"]: r["n"] for r in by_type_rows}
            oldest = conn.execute(
                """
                SELECT MIN(timestamp) FROM events WHERE workspace_id = ?
                """,
                (workspace_id,),
            ).fetchone()[0]
            newest = conn.execute(
                """
                SELECT MAX(timestamp) FROM events WHERE workspace_id = ?
                """,
                (workspace_id,),
            ).fetchone()[0]

        return {
            "total": total,
            "active": active,
            "archived": total - active,
            "by_type": by_type,
            "oldest_timestamp": oldest,
            "newest_timestamp": newest,
        }

    # -----------------------------------------------------------------
    # Update / Mutate
    # -----------------------------------------------------------------

    def touch(self, ulid: str):
        """Mark an event as used (recall hit)."""
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE events
                SET access_count = access_count + 1, last_accessed = ?
                WHERE ulid = ?
                """,
                (time.time(), ulid),
            )

    def apply_recall_updates(
        self,
        touches: list[str],
        importance_updates: list[tuple[str, float]],
    ) -> None:
        """Batch the per-recall side effects into a single SQLite transaction.

        Saves p95 latency: per-event touch + importance update in two
        separate transactions adds ~5ms each. For top_k=5 that's ~50ms
        of avoidable SQLite open/close cost.
        """
        if not touches and not importance_updates:
            return
        with self._conn() as conn:
            now = time.time()
            conn.execute("BEGIN")
            try:
                if touches:
                    conn.executemany(
                        "UPDATE events SET access_count = access_count + 1, "
                        "last_accessed = ? WHERE ulid = ?",
                        [(now, u) for u in touches],
                    )
                if importance_updates:
                    conn.executemany(
                        "UPDATE events SET importance = ? WHERE ulid = ?",
                        [(max(0.0, min(1.0, imp)), u) for u, imp in importance_updates],
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def archive(self, ulid: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE events SET archived_at = ? WHERE ulid = ?",
                (time.time(), ulid),
            )

    def unarchive(self, ulid: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE events SET archived_at = NULL WHERE ulid = ?",
                (ulid,),
            )

    def purge(self, ulid: str) -> bool:
        """HARD delete - permanently remove the event row and its dangling
        graph links. Irreversible (unlike `archive`, which is reversible via
        `unarchive`). Returns True if a row was actually deleted.

        Does NOT touch the vector index - the engine layer calls
        `search.remove(ulid)` around this so the store stays self-contained
        (and a caller can purge from SQLite even if the vector store errors)."""
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM events WHERE ulid = ?", (ulid,))
            deleted = (cur.rowcount or 0) > 0
            # Drop graph rows that referenced this event so nothing points at a
            # gone ulid. Best-effort per table (older DBs may lack one).
            for sql, params in (
                ("DELETE FROM graph_event_entities WHERE event_ulid = ?", (ulid,)),
                ("DELETE FROM event_edges WHERE source_ulid = ? OR target_ulid = ?",
                 (ulid, ulid)),
            ):
                try:
                    conn.execute(sql, params)
                except Exception:
                    pass
        return deleted

    def pin(self, ulid: str, importance: float = 1.0):
        """Pin - high importance, never auto-archived."""
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE events
                SET importance = ?, archived_at = NULL
                WHERE ulid = ?
                """,
                (importance, ulid),
            )

    def update_importance(self, ulid: str, importance: float):
        with self._conn() as conn:
            conn.execute(
                "UPDATE events SET importance = ? WHERE ulid = ?",
                (max(0.0, min(1.0, importance)), ulid),
            )

    def set_metadata(self, ulid: str, metadata: dict) -> None:
        """Replace an event's metadata_json wholesale.

        Used by local-organization features (tags, TTL/expiry) that annotate
        an existing memory. Read-modify-write is done by the caller (via
        ``get_by_ulid``) so concurrent edits stay explicit. Does NOT touch
        content, embeddings, or the recall path.
        """
        with self._conn() as conn:
            conn.execute(
                "UPDATE events SET metadata_json = ? WHERE ulid = ?",
                (json.dumps(metadata, ensure_ascii=False), ulid),
            )

    def list_all(self, workspace_id: str, limit: int = 100_000,
                 event_type: str | None = None,
                 include_archived: bool = True) -> list[Event]:
        """Like ``list_active`` but can include archived rows.

        For export / analytics / timeline where we want the complete picture,
        not just the active set. Newest-first; pass a large ``limit`` to avoid
        a silent truncation.
        """
        clauses = ["workspace_id = ?"]
        params: list = [workspace_id]
        if not include_archived:
            clauses.append("archived_at IS NULL")
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM events WHERE {' AND '.join(clauses)} "
                f"ORDER BY timestamp DESC LIMIT ?",
                params,
            ).fetchall()
            return [Event.from_db_row(r) for r in rows]
