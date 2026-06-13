from __future__ import annotations

import time

from pmb.core.engine.types import (
    _DUMMY_LOCK,
)


class EmbedMixin:
    def _enqueue_touches(
        self,
        touches: list[str],
        importance_updates: list[tuple[str, float]],
        tier_promotions: list[tuple[str, str]] | None = None,
    ) -> None:
        """Buffer recall side effects for the background flusher.

        Coalesces multiple touches of the same ulid (latest timestamp /
        importance wins) so a hot event accessed 16 times in 100ms
        produces ONE SQLite write, not 16. S9: tier promotions ride the SAME
        batch (was one connection+txn per promoted ulid per recall).
        """
        if not touches and not importance_updates and not tier_promotions:
            return
        now = time.time()
        with self._touch_lock:
            for u in touches:
                self._touch_buffer[u] = now
            for u, imp in importance_updates:
                self._touch_imp_buffer[u] = max(0.0, min(1.0, imp))
            for u, tier in (tier_promotions or []):
                self._touch_tier_buffer[u] = tier
            if not self._touch_flusher_started:
                self._touch_flusher_started = True
                import threading

                threading.Thread(
                    target=self._flush_touches_loop,
                    daemon=True,
                    name="pmb-touch-flusher",
                ).start()

    def _flush_touches_loop(self) -> None:
        """Daemon loop: every ~250ms drain the touch buffer to SQLite.

        Single connection per flush, single transaction. Multiple concurrent
        recalls coalesce into one write, so the SQLite write lock is held
        ~4 times per second regardless of recall traffic.
        """
        import time as _t

        while True:
            _t.sleep(0.25)
            if not self._drain_touch_buffer():
                # Buffer was empty for one tick; another empty tick and we
                # exit. Re-spawned on next enqueue. Saves an idle thread.
                _t.sleep(0.25)
                if not self._drain_touch_buffer():
                    with self._touch_lock:
                        self._touch_flusher_started = False
                    return

    def _drain_touch_buffer(self) -> bool:
        """Flush whatever's in the touch buffers right now. Returns True if
        anything was flushed. Safe to call from engine.close() too.
        """
        with self._touch_lock:
            if (not self._touch_buffer and not self._touch_imp_buffer
                    and not self._touch_tier_buffer):
                return False
            touches_snap = dict(self._touch_buffer)
            imp_snap = dict(self._touch_imp_buffer)
            tier_snap = dict(self._touch_tier_buffer)
            self._touch_buffer.clear()
            self._touch_imp_buffer.clear()
            self._touch_tier_buffer.clear()
        try:
            with self.events._conn() as conn:
                conn.execute("BEGIN")
                try:
                    if touches_snap:
                        conn.executemany(
                            "UPDATE events SET access_count = access_count + 1, "
                            "last_accessed = ? WHERE ulid = ?",
                            [(t, u) for u, t in touches_snap.items()],
                        )
                    if imp_snap:
                        conn.executemany(
                            "UPDATE events SET importance = ? WHERE ulid = ?",
                            [(i, u) for u, i in imp_snap.items()],
                        )
                    if tier_snap:
                        conn.executemany(
                            "UPDATE events SET tier = ? WHERE ulid = ?",
                            [(tier, u) for u, tier in tier_snap.items()],
                        )
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
            return True
        except Exception:
            # On failure, put the items back so they retry on next tick.
            with self._touch_lock:
                for u, t in touches_snap.items():
                    self._touch_buffer.setdefault(u, t)
                for u, i in imp_snap.items():
                    self._touch_imp_buffer.setdefault(u, i)
                for u, tier in tier_snap.items():
                    self._touch_tier_buffer.setdefault(u, tier)
            return False

    def warmup(self, with_first_query: bool = True) -> dict:
        """P1-1: Eagerly load model + BM25 + LanceDB so the first real
        recall is fast. Without this the first query pays the entire
        ~1-2s cold-start cost (model load + index open). Useful for:

        - MCP server boot
        - Voice assistant init (latency-sensitive)
        - CLI scripts that do one recall

        Returns timing breakdown for diagnostics.

        with_first_query=True also runs one no-op recall to warm the
        vector index handle and any per-query lazy state.
        """
        import time as _t

        t0 = _t.time()

        # 1. Load embedding model
        t_model_start = _t.time()
        try:
            _ = self.search.model
        except Exception:
            pass
        t_model = _t.time() - t_model_start

        # 2. Build BM25 from any persisted index
        t_bm25_start = _t.time()
        try:
            self.search.reload_bm25()
        except Exception:
            pass
        t_bm25 = _t.time() - t_bm25_start

        # 3. Open LanceDB connection
        t_lance_start = _t.time()
        try:
            _ = self.search._table  # triggers lazy lancedb.connect()
        except Exception:
            pass
        t_lance = _t.time() - t_lance_start

        # 4. Optional: fire one no-op recall to warm per-query state
        t_query_start = _t.time()
        if with_first_query:
            try:
                self.recall("warmup probe", top_k=1)
            except Exception:
                pass
        t_query = _t.time() - t_query_start

        # 5. Mark engine as warm so diagnostics tools can show state
        self._warmed_at = _t.time()
        self._is_warm = True

        # 6. Kick the embed-queue worker now that the model is ready. The
        # passive worker (see _drain_embed_queue) may have timed out and
        # exited while the process was cold; anything still queued (in-memory
        # or durable embed_queue_pending) gets drained here instead of waiting
        # for the next process restart.
        try:
            self._kick_embed_drain()
        except Exception:
            pass

        return {
            "total_ms": round((_t.time() - t0) * 1000, 1),
            "model_load_ms": round(t_model * 1000, 1),
            "bm25_load_ms": round(t_bm25 * 1000, 1),
            "lance_open_ms": round(t_lance * 1000, 1),
            "first_query_ms": round(t_query * 1000, 1),
            "with_first_query": with_first_query,
        }

    def is_warm(self) -> bool:
        """True after `warmup()` has been called successfully. Diagnostics
        (CLI `pmb stats`, MCP `health`) use this to show readiness state.
        """
        return bool(getattr(self, "_is_warm", False))

    def anchor_index(self):
        """v0.9 Semantic Anchor Engine - built lazily and ONLY when the engine
        is warm, so the cold path never loads the embedder for it. Returns the
        AnchorIndex or None (cold engine / anchors disabled). Cached per engine;
        the AnchorIndex itself caches embedded exemplars to disk."""
        if not getattr(self, "_is_warm", False):
            return None
        try:
            if not bool(self.config.get("lang.anchors")):
                return None
        except Exception:
            pass
        if self._anchor_index is None:
            try:
                from pmb.lang.anchors import ALL_ANCHORS, AnchorIndex
                self._anchor_index = AnchorIndex(
                    self.search.embed_batch,
                    getattr(self.search, "model_name", "default"),
                    ALL_ANCHORS,
                    self.workspace.storage_dir / "anchors",
                )
            except Exception:
                self._anchor_index = None
        return self._anchor_index

    def anchor_fires(self, text: str, name: str) -> bool:
        """B2 warm-only single-anchor check (`is_lesson_intent` /
        `looks_like_future_intent`). Returns False when cold, when anchors are
        disabled, or when the set doesn't fire - and NEVER loads the model
        (anchor_index() is warm-gated), so a cold write path stays lexical."""
        try:
            if not self.config.get("lang.anchors"):
                return False
        except Exception:
            return False
        idx = self.anchor_index()
        if idx is None:
            return False
        try:
            return bool(idx.fires(text, name))
        except Exception:
            return False

    def wait_for_embed_queue(self, timeout_seconds: float = 120.0) -> dict:
        """Block until the embed queue has drained (or timeout). Used by
        benchmarks and bulk-import flows that need the vector index
        consistent before measuring / querying.

        Returns counts for diagnostics.
        """
        import time as _t

        deadline = _t.time() + timeout_seconds
        # First wait for model load
        while not self.search.is_ready() and _t.time() < deadline:
            _t.sleep(0.2)
        # Then wait for in-memory queue to drain (worker keeps running
        # while items remain). Poll briefly.
        while _t.time() < deadline:
            with self._embed_queue_lock or _DUMMY_LOCK:
                in_mem = len(self._embed_queue)
            durable = 0
            if self._durable_embed_queue is not None:
                try:
                    durable = self._durable_embed_queue.pending_count()
                except Exception:
                    pass
            if in_mem == 0 and durable == 0:
                return {"in_memory_remaining": 0, "durable_remaining": 0, "timeout": False}
            _t.sleep(0.1)
        return {
            "in_memory_remaining": len(self._embed_queue),
            "durable_remaining": (
                self._durable_embed_queue.pending_count()
                if self._durable_embed_queue is not None
                else 0
            ),
            "timeout": True,
        }

    def _ensure_durable_embed_queue(self):
        """Lazy-init the SQLite-backed pending embeds table. Idempotent.

        Recovery thread spawns ONLY when there are pending rows from a
        previous process - avoids racing the model load on fresh / empty
        workspaces (which caused a Windows access violation in pytest
        teardown when the thread outlived the Engine).
        """
        if self._durable_embed_queue is not None:
            return self._durable_embed_queue
        try:
            from pmb.core.embed_queue import PersistentEmbedQueue

            self._durable_embed_queue = PersistentEmbedQueue(self.workspace.db_path)
            # Only spawn recovery if there's actually something to recover.
            try:
                pending = self._durable_embed_queue.pending_count()
            except Exception:
                pending = 0
            if pending > 0:
                import threading

                _engine_ref = self

                def _recover():
                    # Hold engine ref weakly via closure; if engine is GC'd
                    # before we run, bail out instead of segfaulting.
                    eng = _engine_ref
                    if eng is None or getattr(eng, "_closed", False):
                        return
                    try:
                        eng._durable_embed_queue.recover_on_start(
                            adder=lambda u, t: eng.search.add(u, t),
                            ready=eng.search.is_ready,
                        )
                    except Exception:
                        pass

                t = threading.Thread(
                    target=_recover,
                    daemon=True,
                    name="pmb-embed-recovery",
                )
                t.start()
        except Exception:
            self._durable_embed_queue = None
        return self._durable_embed_queue

    def _embed_or_defer(self, ulid: str, text: str) -> None:
        """Try to embed inline if the model is loaded; otherwise queue
        for the background worker. Writes always return immediately.

        Improvement X: when called from inside `record_batch` (signalled via
        `self._batch_defer`), do NOT embed inline - instead append to a
        per-batch buffer that gets drained as ONE batched encode at the end.
        This turns N sequential embed calls (~200ms each) into one batched
        encode (~500ms total for N≤16) via sentence-transformers native
        batching.
        """
        if getattr(self, "_batch_defer", False):
            self._batch_pending.append((ulid, text))
            return
        if self.search.is_ready():
            try:
                self.search.add(ulid, text)
                return
            except Exception:
                pass  # fall through to queue on failure
        self._enqueue_embed(ulid, text)

    def _enqueue_embed(self, ulid: str, text: str) -> None:
        import threading

        if self._embed_queue_lock is None:
            self._embed_queue_lock = threading.Lock()
        # Hardening H3: persist to durable queue first so the work
        # survives process restart. The in-memory queue is still the
        # hot path for fast drain in the same process.
        dq = self._ensure_durable_embed_queue()
        if dq is not None:
            try:
                dq.enqueue(ulid, text)
            except Exception:
                # Durable queue is best-effort; don't drop the work
                # if it fails - in-memory queue still tries to run.
                pass
        with self._embed_queue_lock:
            self._embed_queue.append((ulid, text))
            if not self._embed_worker_started:
                self._embed_worker_started = True
                threading.Thread(
                    target=self._drain_embed_queue,
                    daemon=True,
                    name="pmb-embed-drain",
                ).start()

    def _drain_embed_inline(self, max_items: int = 64) -> None:
        """Bounded SYNCHRONOUS drain - read-your-writes for recall. A recall
        is a legitimate model consumer (it embeds the query in a moment
        anyway), so flushing parked embeds here keeps 'record then recall'
        consistent in-process without reintroducing the cold-WRITE-path model
        load. Bounded so a giant backlog can't stall a recall; leftovers stay
        for the worker/warmup kick."""
        for _ in range(max_items):
            with self._embed_queue_lock or _DUMMY_LOCK:
                if not self._embed_queue:
                    return
                ulid, text = self._embed_queue.pop(0)
            try:
                self.search.add(ulid, text)
            except Exception:
                return  # stays in the durable queue for recover_on_start
            if self._durable_embed_queue is not None:
                try:
                    with self._durable_embed_queue._conn() as conn:
                        conn.execute(
                            "DELETE FROM embed_queue_pending WHERE ulid = ?",
                            (ulid,))
                except Exception:
                    pass

    def _kick_embed_drain(self) -> None:
        """Drain any embeds left queued while the process was cold. Called at
        the end of warmup(): the passive worker may have deadline-exited
        before the model became ready, so anything still in the in-memory
        queue (or the durable embed_queue_pending table) is picked up HERE
        rather than waiting for the next process restart."""
        import threading as _th
        spawn = False
        with self._embed_queue_lock or _DUMMY_LOCK:
            if self._embed_queue and not self._embed_worker_started:
                self._embed_worker_started = True
                spawn = True
        if spawn:
            _th.Thread(target=self._drain_embed_queue, daemon=True,
                       name="pmb-embed-drain").start()
        elif self._durable_embed_queue is not None:
            try:
                self._durable_embed_queue.drain_once(
                    lambda u, t: self.search.add(u, t), max_items=200)
            except Exception:
                pass

    def _drain_embed_queue(self) -> None:
        """Background worker: wait for the model to be ready (poll every
        500ms), then drain the queue. After fully draining, exits - re-spawned
        on next enqueue (or by warmup(), see below) if needed.

        The worker is PASSIVE by default: it WAITS for `is_ready()` but never
        triggers the model load itself. The old eager `_ = self.search.model`
        was the Codex 120s-timeout bomb: a write in a COLD stdio MCP server
        spawned this worker, the worker started the full torch model load in a
        background thread, and under memory pressure (a second ~400MB model
        next to the warm daemon → paging) the load's long GIL bursts starved
        the server's asyncio loop, so the NEXT tools/call wasn't even read
        before the client's 120s deadline. Writes are already durable without
        the model (SQLite row + embed_queue_pending), and the queue is drained
        by whoever legitimately warms the engine (prewarm thread, a recall,
        warmup(), or the next process via recover_on_start) - so a write-only
        cold process now NEVER pays (or inflicts) a model load. Set
        `embed.queue_autoload=true` to restore the old eager behavior."""
        import time as _t

        try:
            autoload = bool(self.config.get("embed.queue_autoload"))
        except Exception:
            autoload = False
        if autoload:
            # Old eager behavior (idempotent thanks to the lazy `model`
            # property + _ModelCache singleton).
            try:
                _ = self.search.model
            except Exception:
                pass
        # Eager mode: the load is in flight - wait it out (old semantics).
        # Passive mode: only a SHORT grace (covers a prewarm already in
        # flight). Parking for minutes is pointless AND harmful: the thread
        # pins the whole Engine (BM25/LanceDB handles) against GC, and a
        # test-suite-scale fleet of parked workers raises peak memory on the
        # already-constrained Windows box. If the model warms later, the
        # drain is re-triggered by warmup()'s _kick_embed_drain() or by the
        # next _enqueue_embed; a process that never warms leaves the durable
        # embed_queue_pending rows for recover_on_start.
        deadline = _t.time() + (600.0 if autoload else 15.0)
        while not self.search.is_ready() and _t.time() < deadline:
            # Attach instantly if ANOTHER consumer in this process already
            # constructed the model (process-wide _ModelCache) - a peek, never
            # a load. This is what the old eager `_ = self.search.model` was
            # implicitly doing mid-suite/mid-daemon; without it, vectors queued
            # before a recall-triggered lazy load would sit until warmup().
            try:
                if self.search.attach_cached_model():
                    break
            except Exception:
                pass
            _t.sleep(0.5)
        if not self.search.is_ready():
            # Re-arm the spawn flag so a later enqueue / warmup-kick can
            # start a fresh worker.
            with self._embed_queue_lock or _DUMMY_LOCK:
                self._embed_worker_started = False
            return
        # Drain (Hardening H3: failure no longer silently drops the
        # work - the row stays in `embed_queue_pending` and the next
        # process restart picks it up via `recover_on_start`).
        while True:
            with self._embed_queue_lock or _DUMMY_LOCK:
                if not self._embed_queue:
                    self._embed_worker_started = False
                    break
                ulid, text = self._embed_queue.pop(0)
            try:
                self.search.add(ulid, text)
            except Exception:
                # Stays in durable queue; retried via recover_on_start
                # or `pmb doctor` next time.
                continue
            # Success → drop the durable copy too
            if self._durable_embed_queue is not None:
                try:
                    with self._durable_embed_queue._conn() as conn:
                        conn.execute(
                            "DELETE FROM embed_queue_pending WHERE ulid = ?",
                            (ulid,),
                        )
                except Exception:
                    pass
        # After in-memory drain, also sweep any dead-letter recoveries
        if self._durable_embed_queue is not None:
            try:
                self._durable_embed_queue.drain_once(
                    lambda u, t: self.search.add(u, t),
                    max_items=200,
                )
            except Exception:
                pass

    def reindex_embeddings(self) -> dict:
        """Re-embed all active events with the current embedding model.

        Call after switching embedding models (e.g. from English-only to
        multilingual). The old vectors are deleted; new ones generated.

        WARNING: this hits the LLM model heavily. For 1000 events on CPU
        with all-MiniLM expects ~2-5 minutes. With multilingual model
        slightly more (~5%).
        """
        import time as _t

        events = self.events.list_active(self.workspace.id, limit=1_000_000)
        if not events:
            return {"n_events": 0, "elapsed_seconds": 0.0}
        t0 = _t.time()
        n = self.search.reindex_all(events)
        elapsed = _t.time() - t0
        self.recall_cache.bump_generation()
        return {
            "n_events": n,
            "elapsed_seconds": round(elapsed, 1),
            "model": self.search.model_name if hasattr(self.search, "model_name") else "unknown",
        }
