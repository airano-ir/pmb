from __future__ import annotations

from pmb.core.engine.types import (
    _cap_batch_content,
)


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
        """Improvement AA: fire-and-forget batch write.

        Returns IMMEDIATELY (~5-50ms) after spawning a background thread that
        runs the full `record_batch` pipeline (embedding, LanceDB, graph,
        dedup). The MCP caller doesn't wait — no more 120s timeouts on big
        batches.

        Trade-off: the caller can't see the resulting ULIDs synchronously
        (just a count of accepted items). That's fine — agents almost never
        use ULIDs from record_batch's return, they just confirm the write
        happened.

        Recall called immediately after this MAY miss the new events for
        ~100-1000ms while embedding completes. For typical agent flow
        (write now, read on next user turn) this is invisible.
        """
        # Light validation only — heavy lifting in background
        if not items or not isinstance(items, list):
            return {"n_accepted": 0, "processing": "skipped", "errors": ["empty or invalid items"]}
        n_items = sum(1 for i in items if isinstance(i, dict) and i.get("type"))

        import threading

        # Counter+condition: wait_for_writes() blocks until this reaches 0.
        # Increment BEFORE spawning so a fast caller can still see "in flight".
        with self._async_writes_cv:
            self._async_writes_in_flight += 1

        def _process():
            try:
                self.record_batch(items)
            except Exception as e:
                # Last-resort log — silent failure is dangerous
                import logging
                logging.getLogger(__name__).exception(
                    "async batch processing failed: %s", e,
                )
            finally:
                with self._async_writes_cv:
                    self._async_writes_in_flight -= 1
                    self._async_writes_cv.notify_all()

        threading.Thread(
            target=_process,
            daemon=True,
            name="pmb-async-batch",
        ).start()

        # Improvement II: minimal response. Smaller payload, faster Codex
        # UI processing. Background flag suppressed because the caller
        # doesn't need it (we always run in background by default).
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
        """
        import time as _t
        deadline = _t.time() + timeout
        with self._async_writes_cv:
            while self._async_writes_in_flight > 0:
                remaining = deadline - _t.time()
                if remaining <= 0:
                    return False
                # `wait()` releases the lock while waiting; re-acquired on wake.
                self._async_writes_cv.wait(timeout=min(remaining, 1.0))
        return True

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

