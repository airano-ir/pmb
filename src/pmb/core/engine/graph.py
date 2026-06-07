from __future__ import annotations

from typing import Optional

from pmb.core.events import (
    Event,
)

from pmb.core.engine.types import (
    _dedupe_named_entities,
)


class GraphMixin:
    def _index_event_in_graph(self, ev: Event, full_text: str) -> list[int]:
        """Extract entities + upsert nodes + co-occurrence edges. Returns entity_ids."""
        files_hint = ev.metadata.get("files_changed") or []
        # Use the batch-pre-extracted result if record_batch pre-cached it
        # (one LLM call for N events instead of N calls). Falls through to
        # per-event extract() for solo writes and for batches that bypassed
        # pre-extraction (regex backend, or LLM batch failure).
        cache = getattr(self, '_extract_cache', None)
        ext = (cache or {}).get(full_text) if cache else None
        if ext is None:
            ext = self.entity_extractor.extract(full_text, files_hint=files_hint)
        named = ext.all_named()

        # Improvement H: person extraction (no-ML, regex + dict + speaker)
        if self.config.get("recall.person_extraction"):
            try:
                from pmb.graph.persons import (
                    extract_persons,
                    KnownPersons,
                )

                kp = KnownPersons(self.workspace.db_path, self.workspace.id)
                pres = extract_persons(
                    full_text,
                    metadata=ev.metadata,
                    known_persons=kp,
                )
                if pres.persons:
                    # Add person entities. Dedupe on (kind, name) — a name
                    # might already be a 'concept' but should ALSO be a
                    # 'person' (different graph node).
                    existing_pairs = {(k, n) for k, n in named}
                    for p in pres.persons:
                        if ("person", p) not in existing_pairs:
                            named.append(("person", p))
                    # Self-reinforce: bump mention counts in workspace dict
                    kp.bump(pres.persons)
            except Exception:
                pass

        # Improvement J (code half): Python AST symbol extraction.
        # If content looks like code, extract function/class/import entities
        # so the graph layer can answer "which code uses X" structurally.
        if self.config.get("recall.code_ast_extraction"):
            try:
                from pmb.reasoning.code_ast import (
                    extract_python_symbols,
                    symbols_to_entity_names,
                )

                syms = extract_python_symbols(full_text)
                if syms:
                    existing_pairs = {(k, n) for k, n in named}
                    for kn in symbols_to_entity_names(syms):
                        if kn not in existing_pairs:
                            named.append(kn)
            except Exception:
                pass

        if not named:
            return []

        # Improvement S: cross-kind dedup. A name that's been classified as
        # something specific (tech / file / function / class / import / person)
        # should NOT also live as a generic 'concept'. The specific kind wins.
        named = _dedupe_named_entities(named)

        entity_ids: list[int] = []
        for kind, name in named:
            eid = self.graph.upsert_entity(self.workspace.id, kind, name)
            entity_ids.append(eid)
        self.graph.link_event(ev.ulid, entity_ids)
        self.graph.bump_edges(self.workspace.id, entity_ids)
        return entity_ids

    def prune_graph(
        self,
        max_weight: int = 1,
        older_than_days: float = 30.0,
        also_drop_orphan_entities: bool = True,
    ) -> dict:
        """Prune the entity graph to keep it lean as workspace grows.

        Two cleanups:
          1. Drop edges with `weight <= max_weight` AND `last_seen` older
             than `older_than_days`. These are typically one-off co-mentions
             that won't help recall — keeping them just slows PPR.
          2. Drop entities with zero remaining edges AND zero event-links
             (orphans left after edge pruning).

        Run periodically (cron or after large ingests) on workspaces with
        10k+ events to keep recall fast. Reversible? No — but the underlying
        events/embeddings aren't touched, so a `pmb regraph` rebuilds.

        Returns: {n_edges_pruned, n_entities_pruned, edges_before, edges_after}
        """
        import sqlite3, time as _t

        cutoff = _t.time() - older_than_days * 86400.0
        with sqlite3.connect(self.workspace.db_path) as conn:
            edges_before = conn.execute(
                "SELECT COUNT(*) FROM graph_edges WHERE workspace_id = ?",
                (self.workspace.id,),
            ).fetchone()[0]
            cur = conn.execute(
                "DELETE FROM graph_edges WHERE workspace_id = ? AND weight <= ? AND last_seen < ?",
                (self.workspace.id, max_weight, cutoff),
            )
            n_edges_pruned = cur.rowcount
            edges_after = conn.execute(
                "SELECT COUNT(*) FROM graph_edges WHERE workspace_id = ?",
                (self.workspace.id,),
            ).fetchone()[0]

            n_entities_pruned = 0
            if also_drop_orphan_entities:
                # Find entities with no edges AND no event_link
                cur = conn.execute(
                    """
                    DELETE FROM graph_entities
                    WHERE workspace_id = ?
                      AND id NOT IN (
                        SELECT entity_a FROM graph_edges WHERE workspace_id = ?
                        UNION SELECT entity_b FROM graph_edges WHERE workspace_id = ?
                      )
                      AND id NOT IN (SELECT entity_id FROM graph_event_entities)
                    """,
                    (self.workspace.id, self.workspace.id, self.workspace.id),
                )
                n_entities_pruned = cur.rowcount
            conn.commit()

        return {
            "edges_before": edges_before,
            "edges_after": edges_after,
            "n_edges_pruned": n_edges_pruned,
            "n_entities_pruned": n_entities_pruned,
        }

    def regraph(self) -> dict:
        """Wipe `graph_entities`, `graph_event_entities`, `graph_edges` for
        the current workspace and re-extract entities from every active event.

        Use after upgrading the extractor (stop-lists, path guards, etc.) so
        old garbage nodes disappear without touching the event log itself.
        """
        import sqlite3

        ws = self.workspace.id
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.execute(
                "DELETE FROM graph_event_entities WHERE event_ulid IN "
                "(SELECT ulid FROM events WHERE workspace_id = ?)",
                (ws,),
            )
            conn.execute("DELETE FROM graph_edges WHERE workspace_id = ?", (ws,))
            conn.execute("DELETE FROM graph_entities WHERE workspace_id = ?", (ws,))
            conn.commit()

        # Also wipe the self-reinforcing known_persons dict — it learned
        # garbage names ("how", "appdata") that the old extractor allowed.
        try:
            with sqlite3.connect(self.workspace.db_path) as conn:
                conn.execute(
                    "DELETE FROM workspace_kv WHERE workspace_id = ? AND key = 'known_persons'",
                    (ws,),
                )
                conn.commit()
        except Exception:
            pass

        n_events = 0
        n_entities = 0
        for ev in self.events.list_active(ws, limit=100000):
            full_text = ev.content or ""
            try:
                ids = self._index_event_in_graph(ev, full_text)
                n_events += 1
                n_entities += len(ids)
            except Exception:
                continue
        return {"events_reindexed": n_events, "entities_created": n_entities}

    def graph_stats(self) -> dict:
        return self.graph.stats(self.workspace.id)

    def graph_top_entities(self, kind: Optional[str] = None, limit: int = 20) -> list[dict]:
        return [
            e.to_dict()
            for e in self.graph.top_entities(
                self.workspace.id,
                kind=kind,
                limit=limit,
            )
        ]

    def graph_neighbors(self, name: str, kind: Optional[str] = None, top_k: int = 10) -> dict:
        kinds = (kind,) if kind else ()
        ents = self.graph.find_entities_by_name(self.workspace.id, [name], kinds=kinds)
        if not ents:
            return {"entity": None, "neighbors": []}
        primary = ents[0]
        nbrs = self.graph.neighbors(self.workspace.id, primary.id, top_k=top_k)
        return {
            "entity": primary.to_dict(),
            "neighbors": [{"entity": e.to_dict(), "weight": w} for e, w in nbrs],
        }

    def graph_rebuild_from_events(self) -> dict:
        """One-shot reindex of the graph from all active events.

        Useful when migrating an existing PMB workspace from the
        pre-graph version, or after changing the entity extractor.
        """
        events = self.events.list_active(self.workspace.id, limit=100000)
        n_indexed = 0
        for ev in events:
            text_parts = [ev.content]
            q = ev.metadata.get("query") if isinstance(ev.metadata, dict) else None
            if q:
                text_parts.insert(0, q)
            self._index_event_in_graph(ev, full_text="\n".join(text_parts))
            n_indexed += 1
        return {"n_events_indexed": n_indexed, **self.graph.stats(self.workspace.id)}

    def _get_ppr_graph(self):
        """Lazy build + cache of the PPR graph snapshot. Rebuilt when the
        recall cache generation bumps (any write to events/edges). Returns
        None if the graph is empty (e.g. fresh workspace)."""
        gen = self.recall_cache._generation
        if self._ppr_graph is not None and self._ppr_graph_generation == gen:
            return self._ppr_graph
        try:
            from pmb.graph.ppr import build_ppr_graph

            self._ppr_graph = build_ppr_graph(self.workspace.db_path, self.workspace.id)
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning("PPR graph build failed: %s", e)
            self._ppr_graph = None
        self._ppr_graph_generation = gen
        return self._ppr_graph
