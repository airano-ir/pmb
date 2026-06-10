from __future__ import annotations

import atexit as _atexit
import threading as _threading_mod

from pmb.core.engine.types import (
    _cap_batch_content,
)

# Interpreter-shutdown coordination for the outbox drainer threads. A daemon
# thread caught inside a SQLite/LanceDB C call while the interpreter finalizes
# can hard-crash the process (a native fault, not a Python exception). At exit
# we set the stop flag AND wake every drainer so they leave their loop BEFORE
# finalization instead of being killed mid-call.
_OUTBOX_STOP = _threading_mod.Event()
_OUTBOX_WAKES: list = []
_OUTBOX_WAKES_LOCK = _threading_mod.Lock()


def _outbox_shutdown() -> None:
    _OUTBOX_STOP.set()
    with _OUTBOX_WAKES_LOCK:
        for w in list(_OUTBOX_WAKES):
            try:
                w.set()
            except Exception:
                pass


_atexit.register(_outbox_shutdown)


class BatchMixin:
    def record_batch_bulk(self, items: list[dict]) -> dict:
        """Improvement #5: bulk-import mode for migrations / large imports.

        Skips ALL cross-cutting work that makes record_batch slow per item:
          - L1 + L2 dedup checks
          - graph entity indexing + edge upserts
          - temporal date parsing
          - causation edge insert
          - L2.5 borderline queue

        ONLY writes SQLite rows and queues embeddings. The graph layer
        will be missing until the caller runs `pmb regraph` to rebuild it
        from scratch.

        Use this when:
          - importing from another tool (mem0 export, JSONL file)
          - replaying a saved transcript
          - bulk-loading test data
          - any time you have ≥50 items and don't need dedup/graph immediately

        Returns the same shape as record_batch but with `bulk_mode=True` in
        the result so callers can branch on it. Typical speed-up: ~10-15×
        vs normal record_batch on 100 items.
        """
        if not items or not isinstance(items, list):
            return {
                "results": [],
                "n_ok": 0,
                "n_failed": 0,
                "errors": [{"index": 0, "error": "empty or invalid items"}],
                "bulk_mode": True,
            }
        self._bulk_collected_ulids = []
        self._bulk_mode = True
        try:
            result = self.record_batch(items)
        finally:
            self._bulk_mode = False
        result["bulk_mode"] = True
        result["bulk_ulids"] = list(self._bulk_collected_ulids)
        self._bulk_collected_ulids = []
        # Invalidate caches in one shot — record_batch's per-item bumps
        # were skipped in bulk mode.
        try:
            self.recall_cache.bump_generation()
        except Exception:
            pass
        return result

    def record_batch_async(self, items: list[dict]) -> dict:
        """Fire-and-forget batch write, now CRASH-SAFE via a durable outbox.

        The items are written to the `write_outbox` SQLite table SYNCHRONOUSLY
        (~1ms — this is the durability point), then a single background drainer
        replays them through the full `record_batch` pipeline (embedding,
        LanceDB, graph, dedup). If the process dies between this call and the
        write completing, the row stays `pending` and is replayed on the next
        drainer start / `recover_outbox()` — nothing is lost. The old
        fire-and-forget daemon-thread path lost items on process death; it is
        kept only behind `write.outbox=False` for bisecting.

        Returns IMMEDIATELY. The caller can't see resulting ULIDs (agents don't
        use them); a recall right after MAY miss the new events for ~100-1000ms
        while the drainer writes + embeds.
        """
        # Light validation only — heavy lifting in background
        if not items or not isinstance(items, list):
            return {"n_accepted": 0, "processing": "skipped", "errors": ["empty or invalid items"]}
        n_items = sum(1 for i in items if isinstance(i, dict) and i.get("type"))

        use_outbox = True
        try:
            use_outbox = bool(self.config.get("write.outbox"))
        except Exception:
            pass

        if use_outbox:
            outbox_id = self._outbox_enqueue(items)
            out = {"ok": True, "n": n_items}
            if outbox_id is not None:
                out["outbox_id"] = outbox_id
            else:
                # Durable enqueue failed (disk full / locked) — fall back to the
                # in-memory thread so the write still has a chance, and log it.
                out["outbox"] = "enqueue_failed_fell_back"
                self._spawn_legacy_async(items)
            try:
                n = self._adherence_nudge()
                if n:
                    out["_nudge"] = n
            except Exception:
                pass
            return out

        # ── legacy path (write.outbox=False): fire-and-forget daemon thread ──
        self._spawn_legacy_async(items)
        out = {"ok": True, "n": n_items}
        # Adherence nudge — short consequence-framed reminder when the
        # agent has been writing without reading. The agent SEES this in
        # the response payload and self-corrects. Quiet when adherence is
        # fine. This is the cheapest way to lift prepare-call rate on
        # agents that "forget" the READ-FIRST workflow.
        try:
            n = self._adherence_nudge()
            if n: out["_nudge"] = n
        except Exception:
            pass
        return out

    # ── durable write outbox (B4) ──────────────────────────────────────────

    def _spawn_legacy_async(self, items: list[dict]) -> None:
        """Old fire-and-forget path: a daemon thread runs record_batch. Used
        when write.outbox is off, or as a last resort if the durable enqueue
        itself fails. Increments the in-flight counter so wait_for_writes works."""
        import threading
        with self._async_writes_cv:
            self._async_writes_in_flight += 1

        def _process():
            try:
                self.record_batch(items)
            except Exception as e:
                try:
                    from pmb.core.errlog import log_error
                    log_error(self.workspace.db_path, "async_batch", e)
                except Exception:
                    pass
            finally:
                with self._async_writes_cv:
                    self._async_writes_in_flight -= 1
                    self._async_writes_cv.notify_all()

        threading.Thread(target=_process, daemon=True,
                         name="pmb-async-batch").start()

    def _outbox_enqueue(self, items: list[dict]) -> "int | None":
        """Synchronously persist a batch to write_outbox (the durability point)
        and wake the drainer. Returns the row id, or None on failure."""
        import json as _json
        import sqlite3 as _sql
        import time as _t
        try:
            payload = _json.dumps(items)
        except Exception:
            return None
        try:
            with _sql.connect(str(self.workspace.db_path)) as conn:
                cur = conn.execute(
                    "INSERT INTO write_outbox (workspace_id, created_at, "
                    "items_json, status) VALUES (?, ?, ?, 'pending')",
                    (self.workspace.id, _t.time(), payload),
                )
                row_id = cur.lastrowid
        except Exception as e:
            try:
                from pmb.core.errlog import log_error
                log_error(self.workspace.db_path, "write_outbox", e, "enqueue")
            except Exception:
                pass
            return None
        self._ensure_outbox_drainer()
        self._outbox_wake.set()
        return row_id

    def _ensure_outbox_drainer(self) -> None:
        """Start the single per-engine drainer thread if it isn't running."""
        import threading
        if _OUTBOX_STOP.is_set():
            return  # interpreter is shutting down — don't spawn new threads
        # Register this engine's wake so the atexit handler can interrupt it.
        with _OUTBOX_WAKES_LOCK:
            if self._outbox_wake not in _OUTBOX_WAKES:
                _OUTBOX_WAKES.append(self._outbox_wake)
        with self._outbox_lock:
            t = self._outbox_thread
            if t is not None and t.is_alive():
                return
            self._outbox_thread = threading.Thread(
                target=self._outbox_drain_loop, daemon=True, name="pmb-outbox")
            self._outbox_thread.start()

    def recover_outbox(self) -> int:
        """Replay any leftover rows from a previous (possibly crashed) process:
        reset orphaned 'processing' rows to 'pending' and start the drainer.
        Returns the number of pending rows found. Call once at the start of a
        long-lived process (MCP server / daemon). Best-effort."""
        import sqlite3 as _sql
        n = 0
        try:
            with _sql.connect(str(self.workspace.db_path)) as conn:
                conn.execute(
                    "UPDATE write_outbox SET status='pending' "
                    "WHERE status='processing' AND workspace_id=?",
                    (self.workspace.id,))
                row = conn.execute(
                    "SELECT COUNT(*) FROM write_outbox "
                    "WHERE status='pending' AND workspace_id=?",
                    (self.workspace.id,)).fetchone()
                n = int(row[0] or 0)
        except Exception as e:
            try:
                from pmb.core.errlog import log_error
                log_error(self.workspace.db_path, "write_outbox", e, "recover")
            except Exception:
                pass
            return 0
        self._outbox_recovered = True
        if n:
            self._ensure_outbox_drainer()
            self._outbox_wake.set()
        return n

    def _outbox_pending_count(self) -> int:
        import sqlite3 as _sql
        try:
            with _sql.connect(str(self.workspace.db_path)) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM write_outbox WHERE workspace_id=? "
                    "AND status IN ('pending','processing')",
                    (self.workspace.id,)).fetchone()
            return int(row[0] or 0)
        except Exception:
            return 0

    def _outbox_drain_loop(self) -> None:
        """Process pending rows oldest-first until the table is empty, then
        wait for a wake signal. Each row: mark processing → record_batch →
        done; on failure back off and retry up to 5 times, then mark failed.

        Exits cleanly the moment the workspace DB no longer exists (the dir was
        torn down — e.g. a temp test workspace), so finished-test drainer
        threads don't linger and race on the storage layer."""
        import os as _os
        import sqlite3 as _sql
        import time as _t
        while True:
            if _OUTBOX_STOP.is_set():
                return  # interpreter shutting down — leave before finalization
            try:
                if not _os.path.exists(str(self.workspace.db_path)):
                    return
            except Exception:
                return
            try:
                with _sql.connect(str(self.workspace.db_path)) as conn:
                    conn.row_factory = _sql.Row
                    row = conn.execute(
                        "SELECT id, items_json, attempts FROM write_outbox "
                        "WHERE workspace_id=? AND status='pending' "
                        "ORDER BY id LIMIT 1", (self.workspace.id,)).fetchone()
            except Exception:
                row = None
            if row is None:
                # Nothing pending. Wait briefly for a wake, then EXIT the
                # thread if still idle — the next _outbox_enqueue / recover
                # respawns it. This bounds lingering drainer threads (one per
                # engine would otherwise sleep forever and pile up across a
                # test session, raising the odds of a native fault under load).
                woken = self._outbox_wake.wait(timeout=20.0)
                self._outbox_wake.clear()
                if not woken:
                    # Decide to exit UNDER the lock with a final pending re-check
                    # so an enqueue racing here can't orphan its row: it either
                    # sees a live thread (we process below) or, after we clear
                    # _outbox_thread, _ensure_outbox_drainer respawns.
                    with self._outbox_lock:
                        if self._outbox_pending_count() == 0:
                            self._outbox_thread = None
                            return
                continue
            self._outbox_process_one(row["id"], row["items_json"],
                                     int(row["attempts"] or 0))
            # tiny yield so a burst of enqueues doesn't starve other threads
            _t.sleep(0)

    def _outbox_process_one(self, row_id: int, items_json: str,
                            attempts: int) -> None:
        import json as _json
        import sqlite3 as _sql
        import time as _t
        db = str(self.workspace.db_path)
        try:
            with _sql.connect(db) as conn:
                conn.execute("UPDATE write_outbox SET status='processing' "
                             "WHERE id=?", (row_id,))
            items = _json.loads(items_json)
            self.record_batch(items)
            with _sql.connect(db) as conn:
                conn.execute("UPDATE write_outbox SET status='done', done_at=? "
                             "WHERE id=?", (_t.time(), row_id))
        except Exception as e:
            attempts += 1
            terminal = attempts >= 5
            try:
                with _sql.connect(db) as conn:
                    conn.execute(
                        "UPDATE write_outbox SET status=?, attempts=?, "
                        "last_error=? WHERE id=?",
                        ("failed" if terminal else "pending", attempts,
                         f"{type(e).__name__}: {e}"[:300], row_id))
            except Exception:
                pass
            try:
                from pmb.core.errlog import log_error
                log_error(db, "write_outbox", e,
                          f"row={row_id} attempt={attempts}"
                          + (" TERMINAL" if terminal else ""))
            except Exception:
                pass
            if not terminal:
                _t.sleep(min(1.5 * attempts, 8.0))  # exponential-ish backoff

    def wait_for_writes(self, timeout: float = 120.0) -> bool:
        """Block the current thread until ALL in-flight async batches
        (queued via record_batch_async) have finished writing.

        Why you need this:
          record_batch_async spawns `daemon=True` threads. When the main
          process exits before they finish, Python tears them down and
          the queued items are silently lost. Long-running services (MCP
          server, dashboard) never hit this, but CLI scripts and
          fixtures absolutely do.

        Pattern for any seed / import / test script:
            for batch in batches:
                eng.record_batch_async(items=batch)
            eng.wait_for_writes()            # ← critical
            sys.exit(0)

        Args:
            timeout: max seconds to wait. Returns False if not drained
                in time (you should treat that as an error — investigate).

        Returns:
            True if drained cleanly, False on timeout.

        Waits for BOTH the durable outbox (pending+processing rows for this
        workspace) and the legacy in-flight counter, so it is correct whether
        write.outbox is on or off.
        """
        import time as _t
        deadline = _t.time() + timeout
        while True:
            with self._async_writes_cv:
                inflight = self._async_writes_in_flight
            pending = self._outbox_pending_count()
            if inflight == 0 and pending == 0:
                return True
            if _t.time() >= deadline:
                return False
            # nudge the drainer in case it is asleep, then re-check shortly
            try:
                self._outbox_wake.set()
            except Exception:
                pass
            _t.sleep(0.05)

    def record_batch(self, items: list[dict]) -> dict:
        """Improvement T: single-shot batch write across all record_* APIs.

        The agent extracts ALL atomic facts / goals / activities / milestones
        from a user turn in ONE thinking pass, then calls record_batch ONCE
        with the structured list. Eliminates the per-call LLM round-trip
        cost that's the real bottleneck (each round-trip ≈ 3-5s of LLM
        thinking; 11 calls = ~55s; 1 call = ~5s).

        Each item is a dict with a `type` discriminator and the same kwargs
        you'd pass to the matching record_* method:

          {"type": "fact", "content": "...", "importance": 0.7}
          {"type": "fact_tree", "main": "...", "subfacts": [...], "importance": 0.9}
          {"type": "goal", "title": "...", "status": "in_progress",
                            "due_at": <epoch>, "parent_goal_ulid": null}
          {"type": "activity", "content": "...", "kind": "edit", "actor": "agent"}
          {"type": "milestone", "chain_name": "...", "title": "...",
                                 "state": {...}, "triggered_by_ulid": null}

        Returns: {results: [...], n_ok, n_failed, errors}
        Each result mirrors what the corresponding record_* method returns,
        with `type` echoed back so the agent can stitch ULIDs back to inputs.

        Unknown / malformed items are skipped (logged in errors), the rest
        proceed — partial failure does NOT abort the batch.
        """
        # Improvement Z+AA: cap content size per item as a SAFETY guard
        # (agents occasionally dump 20KB+ blobs). The async wrapper handles
        # latency, so this can be generous — 5000 chars preserves nuance
        # while still preventing pathological inputs.
        MAX_CONTENT = 5000
        items = _cap_batch_content(items or [], MAX_CONTENT)

        results: list[dict] = []
        errors: list[dict] = []
        n_ok = 0

        # Improvement CC: serialize concurrent batches. Multiple async batch
        # threads must NOT overlap on self._batch_defer / self._batch_pending
        # or items get lost. The lock is per-engine, held only during one
        # batch — async callers still return instantly because each batch
        # is wrapped in its own thread by record_batch_async.
        with self._batch_lock:
            # Improvement X: defer all embeddings for the whole batch into one
            # native batched encode at the end. Without this, N items =
            # N sequential embed calls.
            self._batch_defer = True
            self._batch_pending = []

            # Pre-extract entities for the whole batch in ONE LLM call.
            # Only fires when (a) backend is non-regex (LLM/spaCy benefit from
            # batching) and (b) there are ≥2 items (no win for solo). Failures
            # silently fall through to per-event extract in _index_event_in_graph,
            # so this is a pure best-effort speedup.
            #
            # BUT this prefetch is a SYNCHRONOUS LLM call — it would block the
            # whole batch write. When graph.async_llm is on (default), we skip
            # it and let the background graph worker do per-event extraction
            # off the write path. We only run the synchronous batch prefetch
            # when async is explicitly disabled (deterministic tests) or the
            # backend is spaCy (local, fast, no CLI round-trip).
            self._extract_cache.clear()
            backend_nm = getattr(self.entity_extractor, "backend_name", "regex")
            _is_llm_backend = isinstance(backend_nm, str) and backend_nm.startswith("llm:")
            _async_on = bool(self.config.get("graph.async_llm"))
            try:
                if (len(items or []) >= 2
                        and backend_nm != "regex"
                        and hasattr(self.entity_extractor, "extract_batch")
                        and not (_is_llm_backend and _async_on)):
                    self._prefetch_batch_entities(items)
            except Exception:  # pragma: no cover - defensive
                self._extract_cache.clear()

            for idx, item in enumerate(items or []):
                if not isinstance(item, dict):
                    errors.append({"index": idx, "error": "not a dict"})
                    continue
                t = (item.get("type") or "").lower().strip()
                pin_after = bool(item.get("pin", False))
                try:
                    if t in ("lesson", "failure"):
                        # Procedural memory. "lesson" = a reusable
                        # correction/technique to apply going forward.
                        # "failure" = negative memory ("tried X, it did NOT
                        # work, do Y instead") - surfaced with a warning so
                        # the agent doesn't repeat it. Both stored as
                        # high-importance facts tagged with kind so recall,
                        # `pmb lessons`, and the audit can treat them specially.
                        content_in = item.get("content") or item.get(t) or ""
                        meta = dict(item.get("metadata") or {})
                        meta.setdefault("source", "lesson")
                        meta["kind"] = t
                        ulid = self.record_fact(
                            content_in,
                            importance=float(item.get("importance", 0.85)),
                            metadata=meta,
                        )
                        if pin_after:
                            try:
                                self.pin(ulid)
                            except Exception:
                                pass
                        results.append({"type": t, "ulid": ulid, "pinned": pin_after})
                        n_ok += 1
                    elif t == "fact":
                        content_in = item.get("content") or item.get("fact") or ""
                        ulid = self.record_fact(
                            content_in,
                            importance=float(item.get("importance", 0.7)),
                            metadata=item.get("metadata"),
                        )
                        if pin_after:
                            try:
                                self.pin(ulid)
                            except Exception:
                                pass
                        # Improvement WW: opportunistic atomic-fact extraction.
                        # When a long fact contains multiple sentences with
                        # pattern-recognisable atoms ("X lives in Y", "We use
                        # X", "X leads Y"), record those atoms as sibling
                        # events with metadata.parent_ulid → source. This
                        # lifts recall on questions targeting one atom inside
                        # a paragraph (mem0-style fact decomposition, no LLM).
                        atoms_created: list[str] = []
                        if self._atomic_extract_enabled:
                            try:
                                atoms_created = self._record_atomic_facts(
                                    content_in,
                                    parent_ulid=ulid,
                                    base_importance=float(item.get("importance", 0.7)),
                                )
                            except Exception:
                                pass
                        results.append(
                            {
                                "type": "fact",
                                "ulid": ulid,
                                "pinned": pin_after,
                                "atomic_facts": atoms_created,
                            }
                        )
                        n_ok += 1
                    elif t in ("fact_tree", "tree"):
                        res = self.record_fact_tree(
                            main=item.get("main") or item.get("content") or "",
                            subfacts=item.get("subfacts") or [],
                            importance=float(item.get("importance", 0.7)),
                            metadata=item.get("metadata"),
                        )
                        if pin_after and res.get("main_ulid"):
                            try:
                                self.pin(res["main_ulid"])
                            except Exception:
                                pass
                            res["pinned"] = True
                        res["type"] = "fact_tree"
                        results.append(res)
                        n_ok += 1
                    elif t in ("goal", "plan"):
                        # "plan" is an alias for a goal — the natural home for
                        # "remember, next we'll do X" / "запомни, что будем
                        # делать дальше". Tagged kind=plan so it's still a goal
                        # (surfaced by prepare / open-goals) but distinguishable.
                        goal_meta = dict(item.get("metadata") or {})
                        if t == "plan":
                            goal_meta.setdefault("kind", "plan")
                        ulid = self.record_goal(
                            title=item.get("title") or item.get("content") or "",
                            status=item.get("status", "pending"),
                            due_at=item.get("due_at"),
                            parent_goal_ulid=item.get("parent_goal_ulid"),
                            importance=float(item.get("importance", 0.7)),
                            metadata=goal_meta or None,
                        )
                        if pin_after:
                            try:
                                self.pin(ulid)
                            except Exception:
                                pass
                        results.append({"type": t, "ulid": ulid, "pinned": pin_after})
                        n_ok += 1
                    elif t == "activity":
                        ulid = self.record_activity(
                            summary=item.get("content") or item.get("summary") or "",
                            actor=item.get("actor", "agent"),
                            kind=item.get("kind", "action"),
                            importance=float(item.get("importance", 0.4)),
                        )
                        if pin_after:
                            try:
                                self.pin(ulid)
                            except Exception:
                                pass
                        results.append({"type": "activity", "ulid": ulid, "pinned": pin_after})
                        n_ok += 1
                    elif t == "milestone":
                        ulid = self.record_milestone(
                            chain_name=item.get("chain_name") or "",
                            title=item.get("title") or item.get("content") or "",
                            state=item.get("state"),
                            triggered_by_ulid=item.get("triggered_by_ulid"),
                            importance=float(item.get("importance", 0.6)),
                        )
                        if pin_after:
                            try:
                                self.pin(ulid)
                            except Exception:
                                pass
                        results.append({"type": "milestone", "ulid": ulid, "pinned": pin_after})
                        n_ok += 1
                    elif t == "preference":
                        ulid = self.record_preference(
                            preference=item.get("content") or item.get("preference") or "",
                            importance=float(item.get("importance", 0.7)),
                            metadata=item.get("metadata"),
                        )
                        if pin_after:
                            try:
                                self.pin(ulid)
                            except Exception:
                                pass
                        results.append({"type": "preference", "ulid": ulid, "pinned": pin_after})
                        n_ok += 1
                    elif t == "summary":
                        ulid = self.record_summary(
                            summary=item.get("content") or item.get("summary") or "",
                            importance=float(item.get("importance", 0.5)),
                            metadata=item.get("metadata"),
                        )
                        if pin_after:
                            try:
                                self.pin(ulid)
                            except Exception:
                                pass
                        results.append({"type": "summary", "ulid": ulid, "pinned": pin_after})
                        n_ok += 1
                    elif t in ("keyed_fact", "key_fact"):
                        # P0-2: upsert a (subject, attribute, value) triple.
                        # Archives prior facts with same key so the current
                        # value alone surfaces on recall.
                        res = self.record_keyed_fact(
                            subject=item.get("subject") or "user",
                            attribute=item.get("attribute") or "",
                            value=item.get("value") or item.get("content") or "",
                            importance=float(item.get("importance", 0.8)),
                            metadata=item.get("metadata"),
                        )
                        if pin_after and res.get("new_ulid"):
                            try:
                                self.pin(res["new_ulid"])
                            except Exception:
                                pass
                            res["pinned"] = True
                        res["type"] = "keyed_fact"
                        results.append(res)
                        n_ok += 1
                    else:
                        errors.append({"index": idx, "error": f"unknown type: {t!r}"})
                except Exception as e:
                    errors.append({"index": idx, "error": f"{type(e).__name__}: {e}"})

            # Improvement X: flush the per-batch embed buffer as ONE call.
            # If model is hot → batched encode (~500ms for 16 items vs 16×200ms).
            # If model not ready → spill into the async queue, write returns now.
            self._batch_defer = False
            pending = self._batch_pending
            self._batch_pending = []
            if pending:
                if self.search.is_ready():
                    try:
                        self.search.add_batch(pending)
                    except Exception:
                        for ulid, text in pending:
                            self._enqueue_embed(ulid, text)
                else:
                    for ulid, text in pending:
                        self._enqueue_embed(ulid, text)

        # Clear the batch-extract cache so a non-batch write right after this
        # one doesn't accidentally reuse stale entities.
        self._extract_cache.clear()

        return {
            "results": results,
            "n_ok": n_ok,
            "n_failed": len(errors),
            "errors": errors,
        }

    def _prefetch_batch_entities(self, items: list[dict]) -> None:
        """Pre-extract entities for the whole batch in ONE LLM call.

        Pulls the user-supplied text from each item (different field per type
        — content / fact / main / title / summary), redacts it the same way
        record_event will, then sends the whole list to
        `entity_extractor.extract_batch`. Results land in `self._extract_cache`
        keyed by the cleaned text so `_index_event_in_graph` (called later
        from inside record_event for each item) hits the cache instead of
        making N more LLM calls.

        N=5 events ≈ one 30 s LLM call vs five 30 s calls = ~5× speed-up.
        """
        from pmb.security.redact import redact

        def _text_of(it: dict) -> str:
            t = (it.get("type") or "").lower().strip()
            if t == "fact_tree" or t == "tree":
                return it.get("main") or it.get("content") or ""
            if t == "goal":
                return it.get("title") or it.get("content") or ""
            if t == "milestone":
                return it.get("title") or it.get("content") or ""
            if t in ("preference", "summary"):
                return it.get(t) or it.get("content") or ""
            if t in ("lesson", "failure"):
                return it.get("content") or it.get(t) or ""
            if t in ("keyed_fact", "key_fact"):
                return it.get("value") or it.get("content") or ""
            # fact / activity / unknown — content is the primary field.
            return it.get("content") or it.get("fact") or ""

        pairs: list[tuple[str, tuple]] = []
        cleaned_texts: list[str] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            raw = _text_of(item)
            if not raw or len(raw.strip()) < 4:
                continue
            clean, _ = redact(raw)
            cleaned_texts.append(clean)
            pairs.append((clean, ()))

        if len(pairs) < 2:
            return  # not worth the round-trip overhead

        results = self.entity_extractor.extract_batch(pairs)
        # Map cleaned-text → ExtractedEntities. Same `clean` string is passed
        # to record_event → _index_event_in_graph below, so lookups hit.
        for text, ext in zip(cleaned_texts, results):
            self._extract_cache[text] = ext

