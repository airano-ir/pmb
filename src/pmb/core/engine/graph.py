from __future__ import annotations

from pmb.core.engine.types import (
    _dedupe_named_entities,
)
from pmb.core.events import (
    Event,
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
                    KnownPersons,
                    extract_persons,
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

    # ──────────────────────────────────────────────────────────────────
    # Async graph indexing — keep LLM extraction OFF the write hot-path
    # ──────────────────────────────────────────────────────────────────

    def _index_graph_or_defer(self, ev: Event, full_text: str) -> list[int]:
        """Graph-index an event, deferring the slow part off the write path.

        The regex (and spaCy) backends are fast and local — they run INLINE,
        exactly as before. But an LLM backend (`llm:claude` / `llm:ollama` /
        `llm:codex`) does a blocking CLI round-trip of up to
        `graph.llm_timeout_s` PER event. Running that inline violates PMB's
        core rule — "the write path does NO blocking LLM call" — and is what
        made records hang and, when an MCP request was serialized behind a
        stuck subprocess, could stall a following recall.

        So for LLM backends (when `graph.async_llm` is on, the default) we
        hand the event to a background worker and return immediately. The
        graph is eventually-consistent; `pmb regraph` is the backstop if the
        process dies before the worker drains. Writes stay ~instant.
        """
        backend = getattr(self.entity_extractor, "backend_name", "regex")
        is_llm = isinstance(backend, str) and backend.startswith("llm:")
        if not is_llm or not self.config.get("graph.async_llm"):
            return self._index_event_in_graph(ev, full_text)
        self._enqueue_graph(ev, full_text)
        return []

    def _enqueue_graph(self, ev: Event, full_text: str) -> None:
        """Append (ev, full_text) to the in-memory graph queue and ensure a
        single daemon worker is draining it. Lazy-inits its own state so we
        don't touch Engine.__init__."""
        import threading

        if getattr(self, "_graph_queue_lock", None) is None:
            self._graph_queue_lock = threading.Lock()
            self._graph_queue = []
            self._graph_worker_started = False
        with self._graph_queue_lock:
            self._graph_queue.append((ev, full_text))
            if not self._graph_worker_started:
                self._graph_worker_started = True
                threading.Thread(
                    target=self._drain_graph_queue,
                    daemon=True,
                    name="pmb-graph-defer",
                ).start()

    def _drain_graph_queue(self) -> None:
        """Background worker: index queued events through the (slow) LLM
        extractor one at a time, then exit. Re-spawned on the next enqueue.
        Each failure is swallowed (best-effort) — `pmb regraph` can rebuild
        any event the worker missed."""
        import logging

        while True:
            with self._graph_queue_lock:
                if not self._graph_queue:
                    self._graph_worker_started = False
                    self._graph_in_flight = False
                    return
                ev, full_text = self._graph_queue.pop(0)
                # Mark in-flight WHILE holding the lock so a concurrent
                # graph_queue_pending() count includes the item being worked.
                self._graph_in_flight = True
            try:
                self._index_event_in_graph(ev, full_text)
            except Exception:
                logging.getLogger(__name__).debug(
                    "deferred graph index failed for %s",
                    getattr(ev, "ulid", "?"), exc_info=True,
                )
            finally:
                with self._graph_queue_lock:
                    self._graph_in_flight = False

    def graph_queue_pending(self) -> int:
        """How many events still need deferred LLM graph indexing — queued
        PLUS the one currently being processed. Used by `wait_for_graph_queue`
        (tests/bulk flows) and diagnostics."""
        lock = getattr(self, "_graph_queue_lock", None)
        if lock is None:
            return 0
        with lock:
            n = len(getattr(self, "_graph_queue", []))
            if getattr(self, "_graph_in_flight", False):
                n += 1
            return n

    def wait_for_graph_queue(self, timeout_seconds: float = 120.0) -> dict:
        """Block until the deferred graph queue drains (or timeout). For
        tests / bulk ingests that need the graph consistent before asserting.
        Pure busy-wait poll — the worker runs in its own thread."""
        import time as _t

        deadline = _t.time() + timeout_seconds
        start_pending = self.graph_queue_pending()
        while _t.time() < deadline:
            if self.graph_queue_pending() == 0:
                break
            _t.sleep(0.05)
        return {
            "drained": self.graph_queue_pending() == 0,
            "start_pending": start_pending,
            "remaining": self.graph_queue_pending(),
        }

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
        import sqlite3
        import time as _t

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

    def graph_top_entities(self, kind: str | None = None, limit: int = 20) -> list[dict]:
        return [
            e.to_dict()
            for e in self.graph.top_entities(
                self.workspace.id,
                kind=kind,
                limit=limit,
            )
        ]

    def graph_neighbors(self, name: str, kind: str | None = None, top_k: int = 10) -> dict:
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
