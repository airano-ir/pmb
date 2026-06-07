"""
Graph storage on top of the same SQLite file used for events.

Three tables (created by migration v2 on first open):

  graph_entities (id PK, workspace_id, kind, name, n_mentions, last_seen)
    UNIQUE(workspace_id, kind, name)

  graph_event_entities (event_ulid, entity_id, PRIMARY KEY)
    — many-to-many; lets us answer "which events mention entity X"

  graph_edges (workspace_id, entity_a, entity_b, weight, last_seen)
    PRIMARY KEY (workspace_id, entity_a, entity_b)
    — co-mention edges; weight increments on each fresh co-occurrence

Edges are undirected; we always store with `entity_a < entity_b` (by id)
to avoid double-counting.

Recall traversal (in engine.recall): query → extract entities from query →
find matching `graph_entities` rows → pull `graph_event_entities` to get
candidate ulids → boost their score in the final ranking.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator, Optional


SCHEMA_VERSION = 2


_DDL_V2 = [
    """
    CREATE TABLE IF NOT EXISTS graph_entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workspace_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        name TEXT NOT NULL,
        n_mentions INTEGER NOT NULL DEFAULT 0,
        last_seen REAL NOT NULL,
        UNIQUE(workspace_id, kind, name)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_entity_workspace_kind ON graph_entities(workspace_id, kind)",
    "CREATE INDEX IF NOT EXISTS idx_entity_name_lower ON graph_entities(workspace_id, lower(name))",
    """
    CREATE TABLE IF NOT EXISTS graph_event_entities (
        event_ulid TEXT NOT NULL,
        entity_id INTEGER NOT NULL,
        PRIMARY KEY (event_ulid, entity_id),
        FOREIGN KEY (entity_id) REFERENCES graph_entities(id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ee_entity ON graph_event_entities(entity_id)",
    """
    CREATE TABLE IF NOT EXISTS graph_edges (
        workspace_id TEXT NOT NULL,
        entity_a INTEGER NOT NULL,
        entity_b INTEGER NOT NULL,
        weight INTEGER NOT NULL DEFAULT 1,
        last_seen REAL NOT NULL,
        PRIMARY KEY (workspace_id, entity_a, entity_b)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_edges_a ON graph_edges(workspace_id, entity_a)",
    "CREATE INDEX IF NOT EXISTS idx_edges_b ON graph_edges(workspace_id, entity_b)",
]


@dataclass
class Entity:
    id: Optional[int]
    workspace_id: str
    kind: str
    name: str
    n_mentions: int = 0
    last_seen: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Edge:
    workspace_id: str
    entity_a: int
    entity_b: int
    weight: int
    last_seen: float

    def to_dict(self) -> dict:
        return asdict(self)


class GraphStore:
    """All graph operations live here. Shares the events.sqlite file."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._migrate()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
        finally:
            conn.close()

    def _migrate(self) -> None:
        with self._conn() as conn:
            for ddl in _DDL_V2:
                conn.execute(ddl)
            cur = conn.execute("SELECT MAX(version) FROM migrations")
            current = cur.fetchone()[0] or 0
            if current < SCHEMA_VERSION:
                conn.execute(
                    "INSERT OR IGNORE INTO migrations (version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, time.time()),
                )

    # ------------------------------------------------------------------
    # Entity upsert
    # ------------------------------------------------------------------

    def upsert_entity(self, workspace_id: str, kind: str, name: str) -> int:
        """Return the entity id, creating or bumping mention count."""
        now = time.time()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, n_mentions FROM graph_entities "
                "WHERE workspace_id = ? AND kind = ? AND name = ?",
                (workspace_id, kind, name),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE graph_entities SET n_mentions = n_mentions + 1, "
                    "last_seen = ? WHERE id = ?",
                    (now, row["id"]),
                )
                return int(row["id"])
            cur = conn.execute(
                "INSERT INTO graph_entities (workspace_id, kind, name, n_mentions, last_seen) "
                "VALUES (?, ?, ?, 1, ?)",
                (workspace_id, kind, name, now),
            )
            return int(cur.lastrowid)

    def link_event(self, event_ulid: str, entity_ids: Iterable[int]) -> None:
        with self._conn() as conn:
            for eid in entity_ids:
                conn.execute(
                    "INSERT OR IGNORE INTO graph_event_entities (event_ulid, entity_id) "
                    "VALUES (?, ?)",
                    (event_ulid, eid),
                )

    # ------------------------------------------------------------------
    # Edge bookkeeping
    # ------------------------------------------------------------------

    def bump_edges(self, workspace_id: str, entity_ids: list[int]) -> None:
        """For all pairs in entity_ids, increment co-occurrence weight."""
        if len(entity_ids) < 2:
            return
        ids = sorted(set(entity_ids))
        now = time.time()
        with self._conn() as conn:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    conn.execute(
                        "INSERT INTO graph_edges (workspace_id, entity_a, entity_b, weight, last_seen) "
                        "VALUES (?, ?, ?, 1, ?) "
                        "ON CONFLICT(workspace_id, entity_a, entity_b) DO UPDATE SET "
                        "weight = weight + 1, last_seen = excluded.last_seen",
                        (workspace_id, a, b, now),
                    )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def find_entities_by_name(
        self, workspace_id: str, names: Iterable[str], kinds: Iterable[str] = ()
    ) -> list[Entity]:
        names_l = [n.lower() for n in names if n]
        if not names_l:
            return []
        with self._conn() as conn:
            placeholders = ",".join("?" * len(names_l))
            query = (
                "SELECT * FROM graph_entities WHERE workspace_id = ? "
                f"AND lower(name) IN ({placeholders})"
            )
            params: list = [workspace_id, *names_l]
            if kinds:
                kinds_l = list(kinds)
                kph = ",".join("?" * len(kinds_l))
                query += f" AND kind IN ({kph})"
                params.extend(kinds_l)
            rows = conn.execute(query, params).fetchall()
            return [
                Entity(
                    id=r["id"], workspace_id=r["workspace_id"], kind=r["kind"],
                    name=r["name"], n_mentions=r["n_mentions"], last_seen=r["last_seen"],
                )
                for r in rows
            ]

    def event_ulids_for_entities(
        self, entity_ids: Iterable[int], limit: int = 500
    ) -> list[str]:
        ids = list(entity_ids)
        if not ids:
            return []
        with self._conn() as conn:
            placeholders = ",".join("?" * len(ids))
            rows = conn.execute(
                f"SELECT DISTINCT event_ulid FROM graph_event_entities "
                f"WHERE entity_id IN ({placeholders}) LIMIT ?",
                (*ids, limit),
            ).fetchall()
            return [r["event_ulid"] for r in rows]

    def event_entity_pairs(
        self, entity_ids: Iterable[int], limit: int = 2000
    ) -> list[tuple[str, int]]:
        """For each event-entity link in `entity_ids`, return (ulid, entity_id).

        Used by IDF-weighted graph boost: caller knows the IDF of each entity
        and needs to know which events match which entities to sum weights.
        """
        ids = list(entity_ids)
        if not ids:
            return []
        with self._conn() as conn:
            placeholders = ",".join("?" * len(ids))
            rows = conn.execute(
                f"SELECT event_ulid, entity_id FROM graph_event_entities "
                f"WHERE entity_id IN ({placeholders}) LIMIT ?",
                (*ids, limit),
            ).fetchall()
            return [(r["event_ulid"], r["entity_id"]) for r in rows]

    def neighbors(
        self, workspace_id: str, entity_id: int, top_k: int = 10
    ) -> list[tuple[Entity, int]]:
        """Find entities most strongly connected to the given one."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT e.*, edges.weight AS weight FROM graph_edges edges
                JOIN graph_entities e ON (
                  (edges.entity_a = ? AND e.id = edges.entity_b)
                  OR (edges.entity_b = ? AND e.id = edges.entity_a)
                )
                WHERE edges.workspace_id = ?
                ORDER BY edges.weight DESC, edges.last_seen DESC
                LIMIT ?
                """,
                (entity_id, entity_id, workspace_id, top_k),
            ).fetchall()
            return [
                (Entity(
                    id=r["id"], workspace_id=r["workspace_id"], kind=r["kind"],
                    name=r["name"], n_mentions=r["n_mentions"], last_seen=r["last_seen"],
                ), int(r["weight"]))
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Stats / listings
    # ------------------------------------------------------------------

    def stats(self, workspace_id: str) -> dict:
        with self._conn() as conn:
            n_entities = conn.execute(
                "SELECT COUNT(*) FROM graph_entities WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()[0]
            n_edges = conn.execute(
                "SELECT COUNT(*) FROM graph_edges WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()[0]
            by_kind = {}
            for r in conn.execute(
                "SELECT kind, COUNT(*) AS n FROM graph_entities "
                "WHERE workspace_id = ? GROUP BY kind", (workspace_id,),
            ).fetchall():
                by_kind[r["kind"]] = r["n"]
        return {"n_entities": n_entities, "n_edges": n_edges, "by_kind": by_kind}

    def top_entities(
        self, workspace_id: str, kind: Optional[str] = None, limit: int = 20
    ) -> list[Entity]:
        with self._conn() as conn:
            if kind:
                rows = conn.execute(
                    "SELECT * FROM graph_entities WHERE workspace_id = ? AND kind = ? "
                    "ORDER BY n_mentions DESC, last_seen DESC LIMIT ?",
                    (workspace_id, kind, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM graph_entities WHERE workspace_id = ? "
                    "ORDER BY n_mentions DESC, last_seen DESC LIMIT ?",
                    (workspace_id, limit),
                ).fetchall()
            return [
                Entity(
                    id=r["id"], workspace_id=r["workspace_id"], kind=r["kind"],
                    name=r["name"], n_mentions=r["n_mentions"], last_seen=r["last_seen"],
                )
                for r in rows
            ]

    def viz_graph(
        self, workspace_id: str, limit: int = 300, max_edges: int = 4000
    ) -> dict:
        """Nodes + edges for the memory visualization: the top `limit` entities
        by mention count and the edges among them. Read-only — never touches
        recall ranking. Returns plain JSON-ready dicts.
        """
        with self._conn() as conn:
            node_rows = conn.execute(
                "SELECT id, kind, name, n_mentions, last_seen FROM graph_entities "
                "WHERE workspace_id = ? ORDER BY n_mentions DESC, last_seen DESC "
                "LIMIT ?",
                (workspace_id, limit),
            ).fetchall()
            ids = [r["id"] for r in node_rows]
            edge_rows = []
            if ids:
                ph = ",".join("?" * len(ids))
                edge_rows = conn.execute(
                    f"SELECT entity_a, entity_b, weight FROM graph_edges "
                    f"WHERE workspace_id = ? AND entity_a IN ({ph}) "
                    f"AND entity_b IN ({ph}) ORDER BY weight DESC LIMIT ?",
                    (workspace_id, *ids, *ids, max_edges),
                ).fetchall()
        return {
            "nodes": [
                {
                    "id": r["id"],
                    "kind": r["kind"] or "other",
                    "name": r["name"],
                    "mentions": int(r["n_mentions"] or 0),
                    "last_seen": float(r["last_seen"] or 0.0),
                }
                for r in node_rows
            ],
            "edges": [
                {"a": r["entity_a"], "b": r["entity_b"], "w": int(r["weight"] or 1)}
                for r in edge_rows
            ],
        }
