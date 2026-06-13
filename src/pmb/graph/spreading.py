"""
Spreading activation - the priming side of the graph.

Cognitive analogy: when you recall something, neighbouring concepts in
your associative network become temporarily more accessible (more
'primed'). If you've just been thinking about Postgres, the next time
Redis is even slightly relevant it'll surface faster.

Implementation:
  - After a recall returns hits H, look at each hit's entities.
  - For each entity, find its strongest graph neighbours (event_entity links).
  - For every neighbour event that wasn't already in H, bump its importance
    by `boost * weight / max_weight` so the rarer associations matter more.
  - The boost is *temporary* - it relies on the existing access_count /
    last_accessed mechanism + tier decay to fade out within hours.

Cost is small: a single SQLite query per recall (one IN clause over entity
ids), and a single batched importance UPDATE.

Disabling: set `recall.spreading_activation = false`. The whole module
is a no-op then.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmb.core.engine import Engine
    from pmb.core.events import Event


def apply_spreading_activation(
    engine: Engine,
    hit_events: list[Event],
    boost: float = 0.05,
    half_life_hours: float = 2.0,
    max_primed: int = 30,
) -> dict:
    """Bump importance of graph neighbours of the recall hits.

    `half_life_hours` is the *effective* lifetime of the boost - we don't
    schedule a reverse-decay; instead we scale the boost so that even if
    the user never recalls again, normal decay neutralises it within
    `half_life_hours`.
    """
    if not hit_events or boost <= 0:
        return {"primed": 0, "skipped": "no_hits"}

    workspace_id = engine.workspace.id

    # 1. Collect entity ids of all hits (one round-trip)
    hit_ulids = [e.ulid for e in hit_events]
    import sqlite3
    with sqlite3.connect(engine.workspace.db_path) as conn:
        conn.row_factory = sqlite3.Row
        placeholders = ",".join("?" * len(hit_ulids))
        rows = conn.execute(
            f"SELECT DISTINCT entity_id FROM graph_event_entities "
            f"WHERE event_ulid IN ({placeholders})",
            hit_ulids,
        ).fetchall()
        entity_ids = [r["entity_id"] for r in rows]
        if not entity_ids:
            return {"primed": 0, "skipped": "no_entities"}

        # 2. Find events linked to those entities (1-hop neighbours)
        ent_ph = ",".join("?" * len(entity_ids))
        rows = conn.execute(
            f"SELECT event_ulid, entity_id FROM graph_event_entities "
            f"WHERE entity_id IN ({ent_ph}) LIMIT ?",
            (*entity_ids, max_primed * 5),
        ).fetchall()
        hit_set = set(hit_ulids)
        # Count how many hit-entities each candidate event shares - more
        # shared entities = stronger priming target.
        share: dict[str, int] = {}
        for r in rows:
            u = r["event_ulid"]
            if u in hit_set:
                continue
            share[u] = share.get(u, 0) + 1
        if not share:
            return {"primed": 0, "skipped": "no_neighbours"}

        # 3. Pick top-N by shared count
        primed = sorted(share.items(), key=lambda kv: -kv[1])[:max_primed]
        max_share = primed[0][1] if primed else 1

        # 4. Apply the boost (saturating, never overshoots 1.0 or pinned)
        # Scale: rare-association neighbour gets ~boost, dense-association
        # gets the full boost. We multiply by half_life-shrink factor so a
        # user-tunable lifetime makes sense in human terms.
        lifetime_scale = max(0.1, min(1.0, half_life_hours / 24.0))
        now = time.time()

        # Fetch current importances + tiers for these ulids in one go
        primed_ulids = [u for u, _ in primed]
        u_ph = ",".join("?" * len(primed_ulids))
        rows2 = conn.execute(
            f"SELECT ulid, importance, archived_at FROM events "
            f"WHERE workspace_id = ? AND ulid IN ({u_ph})",
            (workspace_id, *primed_ulids),
        ).fetchall()
        updates: list[tuple[float, str]] = []
        for r in rows2:
            if r["archived_at"] is not None:
                continue
            imp = r["importance"]
            if imp >= 0.99:  # pinned
                continue
            share_count = share[r["ulid"]]
            local_boost = boost * (share_count / max_share) * lifetime_scale
            new_imp = min(1.0, imp + local_boost * (1.0 - imp))
            if new_imp > imp + 0.0005:
                updates.append((new_imp, r["ulid"]))
        if updates:
            conn.executemany(
                "UPDATE events SET importance = ?, last_accessed = "
                + str(int(now))
                + " WHERE ulid = ?",
                updates,
            )

    return {
        "primed": len(updates),
        "candidates_considered": len(share),
        "entities_used": len(entity_ids),
    }
