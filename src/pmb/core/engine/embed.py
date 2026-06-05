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
    ) -> None:
        """Buffer recall side effects for the background flusher.

        Coalesces multiple touches of the same ulid (latest timestamp /
        importance wins) so a hot event accessed 16 times in 100ms
        produces ONE SQLite write, not 16.
        """
        if not touches and not importance_updates:
            return
        now = time.time()
        with self._touch_lock:
            for u in touches:
                self._touch_buffer[u] = now
            for u, imp in importance_updates:
                self._touch_imp_buffer[u] = max(0.0, min(1.0, imp))
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
            if not self._touch_buffer and not self._touch_imp_buffer:
                return False
            touches_snap = dict(self._touch_buffer)
            imp_snap = dict(self._touch_imp_buffer)
            self._touch_buffer.clear()
            self._touch_imp_buffer.clear()
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
        previous process — avoids racing the model load on fresh / empty
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
        `self._batch_defer`), do NOT embed inline — instead append to a
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
                # if it fails — in-memory queue still tries to run.
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

    def _drain_embed_queue(self) -> None:
        """Background worker: wait for the model to load (poll every 500ms),
        then drain the queue. After fully draining, exits — re-spawned on
        next enqueue if needed.
        """
        import time as _t

        # Wait up to ~10 min for model load
        deadline = _t.time() + 600.0
        # Trigger model load if nothing else has yet (idempotent thanks to
        # the lazy `model` property + _ModelCache singleton)
        try:
            _ = self.search.model
        except Exception:
            pass
        while not self.search.is_ready() and _t.time() < deadline:
            _t.sleep(0.5)
        # Drain (Hardening H3: failure no longer silently drops the
        # work — the row stays in `embed_queue_pending` and the next
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
