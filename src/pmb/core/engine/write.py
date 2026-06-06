from __future__ import annotations

from pathlib import Path
from typing import Optional

from pmb.core.events import (
    Event,
    default_tier_for_event_type,
)
from pmb.security.redact import redact, redact_metadata

from pmb.core.engine.types import (
    _cap_batch_content,
)

import time

class WriteMixin:
    def _record_atomic_facts(
        self,
        text: str,
        parent_ulid: str,
        base_importance: float = 0.7,
    ) -> list[str]:
        """Extract atomic facts via fact_extract and record each as a
        sibling event linked to `parent_ulid` via metadata.parent_ulid.

        Returns the list of created child ulids.
        """
        try:
            from pmb.reasoning.fact_extract import extract_atomic_facts
        except Exception:
            return []
        atoms = extract_atomic_facts(text)
        if not atoms:
            return []
        out: list[str] = []
        # Slightly lower importance than the source so the original
        # paragraph still wins when the user asks the whole question.
        child_imp = max(0.1, min(0.9, base_importance - 0.05))
        for af in atoms:
            try:
                ulid = self.record_fact(
                    af.content,
                    importance=child_imp,
                    metadata={
                        "parent_ulid": parent_ulid,
                        "atomic_kind": af.kind,
                        "atomic_confidence": af.confidence,
                        "extracted": True,
                    },
                )
                out.append(ulid)
            except Exception:
                continue
        return out

    def remember(
        self,
        query: str,
        response: str,
        session_id: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[dict] = None,
    ) -> str:
        """Сохранить Q/A пару. Возвращает ulid."""
        # Auto-bind to current session if not provided
        if session_id is None:
            session_id = self.session_tracker.touch().id

        # Secret redaction at the persistence boundary
        clean_query, _ = redact(query)
        clean_response, _ = redact(response)
        clean_metadata, _ = redact_metadata({"query": clean_query, **(metadata or {})})

        ev = Event(
            workspace_id=self.workspace.id,
            event_type="qa",
            content=clean_response,
            metadata=clean_metadata,
            importance=importance,
            source_session_id=session_id,
            tier=default_tier_for_event_type("qa"),
        )
        ev = self.events.append(ev)
        # Index в hybrid search. Synchronous: the Python API contract is
        # "write returns when data is searchable". Use `_embed_or_defer`
        # only when we're inside `record_batch` (it sets _batch_defer).
        if getattr(self, "_batch_defer", False):
            self._embed_or_defer(ev.ulid, ev.to_text())
        else:
            self.search.add(ev.ulid, ev.to_text())
        # Index в graph
        self._index_event_in_graph(ev, full_text=f"{clean_query}\n{clean_response}")
        try:
            from pmb.reasoning.causation import add_temporal_next_edge

            add_temporal_next_edge(self, ev)
        except Exception:
            pass
        # Improvement C: parse event_time (date references) from content
        # and store in metadata. Enables temporal-proximity boost at recall.
        try:
            self._attach_event_time(ev)
        except Exception:
            pass
        self.recall_cache.bump_generation()
        return ev.ulid

    def record_fact(
        self,
        fact: str,
        importance: float = 0.7,
        metadata: Optional[dict] = None,
        session_id: Optional[str] = None,
    ) -> str:
        clean_fact, _ = redact(fact)
        clean_metadata, _ = redact_metadata(metadata or {})

        # Improvement #5 (bulk import): when self._bulk_mode is set,
        # skip ALL cross-cutting work (dedup, graph, temporal, causation,
        # L2.5 queue). Only the SQLite row + embedding land. The caller
        # must run `pmb regraph` afterwards to rebuild the graph.
        if getattr(self, "_bulk_mode", False):
            ev = Event(
                workspace_id=self.workspace.id,
                event_type="fact",
                content=clean_fact,
                metadata=clean_metadata,
                importance=importance,
                source_session_id=session_id,
                tier=default_tier_for_event_type("fact"),
            )
            ev = self.events.append(ev)
            self._embed_or_defer(ev.ulid, ev.to_text())
            self._bulk_collected_ulids.append(ev.ulid)
            return ev.ulid

        # Improvement U: write-time dedup (L1 exact + L2 semantic).
        # If we detect a duplicate, return the existing ULID instead of
        # writing a new event. The existing event gets its access_count
        # bumped so it surfaces faster on next recall.
        dup_hit, borderline = self._dedup_pre_write(
            content=clean_fact,
            event_type="fact",
        )
        if dup_hit is not None:
            self._bump_for_dup(dup_hit.canonical_ulid)
            return dup_hit.canonical_ulid

        ev = Event(
            workspace_id=self.workspace.id,
            event_type="fact",
            content=clean_fact,
            metadata=clean_metadata,
            importance=importance,
            source_session_id=session_id,
            tier=default_tier_for_event_type("fact"),
        )
        ev = self.events.append(ev)
        # Improvement W: embed inline if model loaded, else queue
        self._embed_or_defer(ev.ulid, ev.to_text())
        self._index_event_in_graph(ev, full_text=clean_fact)
        try:
            from pmb.reasoning.causation import add_temporal_next_edge

            add_temporal_next_edge(self, ev)
        except Exception:
            pass
        # Improvement C: parse event_time (date references) from content
        # and store in metadata. Enables temporal-proximity boost at recall.
        try:
            self._attach_event_time(ev)
        except Exception:
            pass

        # L2.5: borderline candidate detected — enqueue for async LLM verify
        if borderline is not None and self.config.get("dedup.async_verify"):
            try:
                from pmb.reasoning.dedup import enqueue_borderline

                enqueue_borderline(
                    self.workspace.db_path,
                    self.workspace.id,
                    new_ulid=ev.ulid,
                    candidate_ulid=borderline[0],
                    similarity=borderline[1],
                )
            except Exception:
                pass

        self.recall_cache.bump_generation()
        return ev.ulid

    def record_keyed_fact(
        self,
        subject: str,
        attribute: str,
        value: str,
        importance: float = 0.8,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Upsert a personal-attribute fact, archiving any prior fact with
        the same (subject, attribute) key.

        This is the missing piece for personal-assistant memory:
        "user lives in Kyiv" → "user lives in Warsaw" should not leave
        both active in recall. The old fact is archived with metadata
        `superseded_by=<new_ulid>` so history is preserved but only the
        current value surfaces by default.

        The key is `subject::attribute` (lowercased), stored as
        `metadata.keyed_fact_key`. Recall filters archived events out
        normally, so old values disappear from results without any
        extra work in the recall pipeline.

        Returns:
            {
                "new_ulid": str,
                "superseded_ulids": list[str],   # archived prior versions
                "key": str,
            }
        """
        if not subject or not attribute or not value:
            raise ValueError("subject, attribute, value all required")
        subject_norm = subject.strip().lower()
        attribute_norm = attribute.strip().lower()
        key = f"{subject_norm}::{attribute_norm}"

        # 1. Find any prior facts with the same key (active only)
        prior_ulids: list[str] = []
        try:
            import sqlite3 as _sql

            with _sql.connect(str(self.workspace.db_path)) as conn:
                conn.row_factory = _sql.Row
                rows = conn.execute(
                    "SELECT ulid, metadata_json FROM events "
                    "WHERE workspace_id = ? AND archived_at IS NULL "
                    "AND event_type = 'fact' "
                    "AND metadata_json LIKE ?",
                    (self.workspace.id, f'%"keyed_fact_key": "{key}"%'),
                ).fetchall()
            for r in rows:
                prior_ulids.append(r["ulid"])
        except Exception:
            pass

        # 2. Write the new fact with the key + supersedes pointer
        meta = dict(metadata or {})
        meta["keyed_fact_key"] = key
        meta["keyed_fact_subject"] = subject
        meta["keyed_fact_attribute"] = attribute
        meta["keyed_fact_value"] = value
        if prior_ulids:
            meta["supersedes"] = prior_ulids
        # Lever 3: stamp a validity window so a superseded value stays
        # queryable "as of" a past date (Zep-style). New value is valid from
        # now; each prior value's window is closed below. Additive metadata
        # only - recall behaviour is unchanged.
        now_ts = time.time()
        meta["valid_from"] = now_ts
        # Human-readable content for embedder/BM25
        content = f"{subject} {attribute}: {value}"
        new_ulid = self.record_fact(
            content,
            importance=importance,
            metadata=meta,
        )

        # 3. Archive priors — they stay in SQLite (queryable as history)
        # but `archived_at IS NULL` filter removes them from recall.
        for old_ulid in prior_ulids:
            try:
                self.events.archive(old_ulid)
                # Tag with the new pointer so callers can trace history.
                import sqlite3 as _sql, json as _json

                with _sql.connect(str(self.workspace.db_path)) as conn:
                    row = conn.execute(
                        "SELECT metadata_json FROM events WHERE ulid = ?",
                        (old_ulid,),
                    ).fetchone()
                    old_meta = _json.loads(row[0] or "{}") if row else {}
                    old_meta["superseded_by"] = new_ulid
                    old_meta["valid_to"] = now_ts  # close this value's window
                    conn.execute(
                        "UPDATE events SET metadata_json = ? WHERE ulid = ?",
                        (_json.dumps(old_meta), old_ulid),
                    )
            except Exception:
                continue

        self.recall_cache.bump_generation()
        return {
            "new_ulid": new_ulid,
            "superseded_ulids": prior_ulids,
            "key": key,
        }

    def keyed_fact_as_of(
        self, subject: str, attribute: str, at_time: float,
    ) -> Optional[dict]:
        """As-of temporal query (Zep-style): what was the value of
        (subject, attribute) at `at_time` (UTC epoch seconds)?

        Uses the valid_from / valid_to windows stamped by record_keyed_fact
        supersession, so "what city did I live in last March" returns the
        value that was current THEN, not the latest one.

        Standalone - NOT wired into the recall hot path, so it cannot affect
        recall ranking or latency. Returns {value, valid_from, valid_to,
        ulid, content} for the version valid at `at_time`, else None.
        """
        key = f"{subject.strip().lower()}::{attribute.strip().lower()}"
        versions: list[tuple] = []
        try:
            import sqlite3 as _sql, json as _json
            with _sql.connect(str(self.workspace.db_path)) as conn:
                conn.row_factory = _sql.Row
                rows = conn.execute(
                    "SELECT ulid, content, metadata_json, timestamp FROM events "
                    "WHERE workspace_id = ? AND event_type = 'fact' "
                    "AND metadata_json LIKE ? ORDER BY timestamp ASC",
                    (self.workspace.id, f'%"keyed_fact_key": "{key}"%'),
                ).fetchall()
            for r in rows:
                meta = _json.loads(r["metadata_json"] or "{}")
                versions.append((r["ulid"], r["content"], meta, r["timestamp"]))
        except Exception:
            return None
        if not versions:
            return None

        best = None
        for ulid, content, meta, ts in versions:
            vf = meta.get("valid_from", ts)
            vf = float(vf) if isinstance(vf, (int, float)) else float(ts)
            vt = meta.get("valid_to")
            if at_time < vf:
                continue
            if vt is None or at_time <= float(vt):
                # ascending order → keep updating, last match = most recent valid
                best = {
                    "value": meta.get("keyed_fact_value"),
                    "valid_from": vf,
                    "valid_to": (float(vt) if isinstance(vt, (int, float)) else None),
                    "ulid": ulid,
                    "content": content,
                }
        return best

    def topic_overview(self, topic: str, max_events: int = 40) -> dict:
        """Structured "what do I know about <topic>?" overview - no LLM.

        Recalls memories related to the topic and aggregates them into key
        facts & decisions, lessons, failures, open goals, a compact timeline,
        and related topics (entity-graph neighbours). Pure read + aggregation:
        it reuses recall() but writes nothing and changes no ranking. Exposed
        on the CLI (`pmb overview`) and over MCP so an agent can get up to
        speed on a project/feature in one call.
        """
        import time as _t
        topic = (topic or "").strip()
        if not topic:
            return {"topic": topic, "n_memories": 0, "empty": True}

        pack = self.recall(query=topic, top_k=max_events)
        results = list(pack.results)

        def _item(r) -> dict:
            meta = r.metadata or {}
            return {
                "date": getattr(r, "resolved_date", None),
                "content": (r.content or "")[:240],
                "score": round(float(r.score), 3),
                "kind": meta.get("kind") or r.event_type,
            }

        facts, lessons, failures, goals, other = [], [], [], [], []
        for r in results:
            meta = r.metadata or {}
            k = meta.get("kind")
            if k == "lesson":
                lessons.append(_item(r))
            elif k == "failure":
                failures.append(_item(r))
            elif r.event_type == "goal":
                goals.append(_item(r))
            elif r.event_type == "fact" or k == "decision":
                facts.append(_item(r))
            else:
                other.append(_item(r))

        timeline = sorted(
            ({"date": getattr(r, "resolved_date", None) or "",
              "content": (r.content or "")[:120]} for r in results),
            key=lambda x: x["date"],
        )[:12]

        ts = [r.timestamp for r in results if r.timestamp]
        span = None
        if ts:
            span = {"from": _t.strftime("%Y-%m-%d", _t.gmtime(min(ts))),
                    "to": _t.strftime("%Y-%m-%d", _t.gmtime(max(ts)))}

        related: list[str] = []
        try:
            for tok in topic.split()[:3]:
                gn = self.graph_neighbors(tok, top_k=5)
                if gn.get("entity"):
                    for nbr in gn.get("neighbors", []):
                        nm = (nbr.get("entity") or {}).get("name")
                        if nm and nm.lower() not in topic.lower() and nm not in related:
                            related.append(nm)
        except Exception:
            pass

        return {
            "topic": topic,
            "n_memories": len(results),
            "span": span,
            "facts": facts[:10],
            "lessons": lessons[:10],
            "failures": failures[:10],
            "goals": goals[:10],
            "other": other[:5],
            "timeline": timeline,
            "related_topics": related[:8],
            "empty": len(results) == 0,
        }

    def session_brief(self, session_id: Optional[str] = None,
                      minutes: Optional[float] = None, limit: int = 100) -> dict:
        """Compact digest of the CURRENT (or given) work session - what was
        decided / done / learned so far.

        Built so an agent can re-orient after its OWN context window compacts
        in a long session: PMB is the durable session memory, so instead of
        re-asking the user what it already did, the agent calls this and picks
        the thread back up. Pure read; off the recall ranking path.

        Scopes to events tagged with the active session; if none, falls back to
        the last `minutes` (config `session.brief_minutes`).
        """
        import time as _t
        sess = None
        if session_id is None:
            try:
                sess = self.session_tracker.current(auto_create=False)
            except Exception:
                sess = None
            session_id = getattr(sess, "id", None) if sess else None
        if minutes is None:
            try:
                minutes = float(self.config.get("session.brief_minutes"))
            except Exception:
                minutes = 180.0

        now = _t.time()
        # Scope to the session as a UNION: every event recorded since the
        # session began PLUS anything explicitly tagged with this session id.
        # Why both: only `record_activity` auto-binds events to the session;
        # facts / goals / lessons recorded during the session carry no
        # session id, so a tag-only filter silently drops them. Falling back to
        # the session's start time captures that work too. Without an active
        # session, use the last `minutes` window.
        sess_started = getattr(sess, "started_at", None) if sess else None
        cutoff = now - minutes * 60.0
        if session_id and isinstance(sess_started, (int, float)):
            cutoff = min(cutoff, sess_started - 1.0)
        events = self.events.list_active(self.workspace.id, limit=100000)
        if session_id:
            scoped = [e for e in events
                      if e.source_session_id == session_id or e.timestamp >= cutoff]
        else:
            scoped = [e for e in events if e.timestamp >= cutoff]
        scoped.sort(key=lambda e: e.timestamp)

        def _kind(meta) -> Optional[str]:
            # `record_activity` stores the kind under `activity_kind`; lessons /
            # failures use `kind`. Read both so decisions/done classify.
            return meta.get("kind") or meta.get("activity_kind")

        def _it(e) -> dict:
            meta = e.metadata or {}
            return {
                "when": _t.strftime("%m-%d %H:%M", _t.gmtime(e.timestamp)),
                "content": (e.content or "")[:200],
                "kind": _kind(meta) or e.event_type,
            }

        decisions, done, lessons, failures, goals, other = [], [], [], [], [], []
        for e in scoped:
            meta = e.metadata or {}
            k = _kind(meta)
            if k == "lesson":
                lessons.append(_it(e))
            elif k == "failure":
                failures.append(_it(e))
            elif e.event_type == "goal":
                goals.append(_it(e))
            elif k == "decision":
                decisions.append(_it(e))
            elif k == "completed":
                done.append(_it(e))
            else:
                other.append(_it(e))

        duration_min = None
        started = getattr(sess, "started_at", None) if sess else None
        if isinstance(started, (int, float)):
            duration_min = round((now - started) / 60.0)

        return {
            "session_id": session_id,
            "session_name": getattr(sess, "name", None) if sess else None,
            "duration_min": duration_min,
            "scope": "session" if (session_id and scoped) else f"last {minutes:g} min",
            "n_events": len(scoped),
            "decisions": decisions[:15],
            "done": done[:15],
            "lessons": lessons[:15],
            "failures": failures[:15],
            "goals": goals[:15],
            "other": other[:10],
            "empty": len(scoped) == 0,
        }

    def record_preference(
        self,
        preference: str,
        importance: float = 0.7,
        metadata: Optional[dict] = None,
    ) -> str:
        """User preference. Event_type='preference', tier='semantic'.
        Examples: "I prefer dark mode", "Я люблю спокойные игры".
        """
        meta = dict(metadata or {})
        meta["memory_type"] = "preference"
        return self.record_event(
            content=preference,
            event_type="preference",
            importance=importance,
            metadata=meta,
        )

    def record_summary(
        self,
        summary: str,
        importance: float = 0.5,
        metadata: Optional[dict] = None,
    ) -> str:
        """Conversation summary / digest. Event_type='summary', tier='episodic'.
        Marked so retrieval can prefer original facts over rephrased summaries.
        """
        meta = dict(metadata or {})
        meta["memory_type"] = "summary"
        return self.record_event(
            content=summary,
            event_type="summary",
            importance=importance,
            metadata=meta,
        )

    def get_keyed_fact_history(self, subject: str, attribute: str) -> list[dict]:
        """Return current + all prior values for a keyed fact, newest first.
        Useful for "what did I say about X before?" introspection.
        """
        key = f"{subject.strip().lower()}::{attribute.strip().lower()}"
        try:
            import sqlite3 as _sql, json as _json

            with _sql.connect(str(self.workspace.db_path)) as conn:
                conn.row_factory = _sql.Row
                rows = conn.execute(
                    "SELECT ulid, content, metadata_json, timestamp, "
                    "archived_at FROM events "
                    "WHERE workspace_id = ? AND event_type = 'fact' "
                    "AND metadata_json LIKE ? "
                    "ORDER BY timestamp DESC",
                    (self.workspace.id, f'%"keyed_fact_key": "{key}"%'),
                ).fetchall()
            out = []
            for r in rows:
                meta = _json.loads(r["metadata_json"] or "{}")
                out.append(
                    {
                        "ulid": r["ulid"],
                        "value": meta.get("keyed_fact_value"),
                        "content": r["content"],
                        "timestamp": r["timestamp"],
                        "is_current": r["archived_at"] is None,
                    }
                )
            return out
        except Exception:
            return []

    def record_fact_tree(
        self,
        main: str,
        subfacts: Optional[list[str]] = None,
        importance: float = 0.7,
        metadata: Optional[dict] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """Improvement P: store a main fact + multiple atomic subfacts.

        The main fact gets the requested importance. Subfacts inherit the
        main's importance × 0.85 (slightly lower — they're supplementary)
        and link back via metadata.parent_ulid.

        Use this when one event has multiple atomic data points worth
        remembering separately:

            record_fact_tree(
                "User fell down stairs and broke arm on May 23, 2026",
                subfacts=[
                    "Time of fall: 18:52",
                    "Recommended: visit ER for X-ray",
                    "First aid: ice 15-20min, remove rings, do not drive",
                    "Warning signs to call 911: numbness, bleeding, deformation",
                ],
                importance=0.9,
            )

        Each subfact is independently searchable. When recall finds the
        parent, callers can pull related subfacts via `get_subfacts(ulid)`.
        """
        # Store main first
        main_meta = dict(metadata or {})
        main_meta["has_subfacts"] = bool(subfacts)
        main_ulid = self.record_fact(
            main,
            importance=importance,
            metadata=main_meta,
            session_id=session_id,
        )

        sub_ulids: list[str] = []
        if subfacts:
            sub_importance = max(0.3, importance * 0.85)
            for s in subfacts:
                if not s or not s.strip():
                    continue
                # Each subfact links to parent via metadata
                sub_meta = {
                    "parent_ulid": main_ulid,
                    "is_subfact": True,
                }
                # Inherit parent's event_time if any
                sub_ulid = self.record_fact(
                    s.strip(),
                    importance=sub_importance,
                    metadata=sub_meta,
                    session_id=session_id,
                )
                sub_ulids.append(sub_ulid)
                # Also link main → subfact via event_edges for graph walks
                try:
                    from pmb.reasoning.causation import CausationEdge, upsert_edge

                    upsert_edge(
                        self.workspace.db_path,
                        CausationEdge(
                            source_ulid=main_ulid,
                            target_ulid=sub_ulid,
                            edge_type="references",
                            confidence=1.0,
                            rationale="subfact of main",
                        ),
                    )
                except Exception:
                    pass

        return {
            "main_ulid": main_ulid,
            "subfact_ulids": sub_ulids,
            "n_subfacts": len(sub_ulids),
        }

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

        def _process():
            try:
                self.record_batch(items)
            except Exception as e:
                # Last-resort log — silent failure is dangerous
                import logging

                logging.getLogger(__name__).exception(
                    "async batch processing failed: %s",
                    e,
                )

        threading.Thread(
            target=_process,
            daemon=True,
            name="pmb-async-batch",
        ).start()

        # Improvement II: minimal response. Smaller payload, faster Codex
        # UI processing. Background flag suppressed because the caller
        # doesn't need it (we always run in background by default).
        return {"ok": True, "n": n_items}

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
                    elif t == "goal":
                        ulid = self.record_goal(
                            title=item.get("title") or item.get("content") or "",
                            status=item.get("status", "pending"),
                            due_at=item.get("due_at"),
                            parent_goal_ulid=item.get("parent_goal_ulid"),
                            importance=float(item.get("importance", 0.7)),
                        )
                        if pin_after:
                            try:
                                self.pin(ulid)
                            except Exception:
                                pass
                        results.append({"type": "goal", "ulid": ulid, "pinned": pin_after})
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

        return {
            "results": results,
            "n_ok": n_ok,
            "n_failed": len(errors),
            "errors": errors,
        }

    def get_subfacts(self, parent_ulid: str) -> list[dict]:
        """Return all subfacts linked to a parent event."""
        import sqlite3, json as _j

        out = []
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ulid, content, importance, timestamp, metadata_json "
                "FROM events WHERE workspace_id = ? AND archived_at IS NULL "
                "AND json_extract(metadata_json, '$.parent_ulid') = ? "
                "ORDER BY timestamp ASC",
                (self.workspace.id, parent_ulid),
            ).fetchall()
            for r in rows:
                try:
                    meta = _j.loads(r["metadata_json"] or "{}")
                except Exception:
                    meta = {}
                out.append(
                    {
                        "ulid": r["ulid"],
                        "content": r["content"],
                        "importance": r["importance"],
                        "timestamp": r["timestamp"],
                        "metadata": meta,
                    }
                )
        return out

    def get_parent_fact(self, subfact_ulid: str) -> Optional[dict]:
        """If this event is a subfact, return its parent event."""
        ev = self.events.get_by_ulid(subfact_ulid)
        if not ev or not ev.metadata:
            return None
        parent_ulid = ev.metadata.get("parent_ulid")
        if not parent_ulid:
            return None
        parent = self.events.get_by_ulid(parent_ulid)
        if not parent:
            return None
        return {
            "ulid": parent.ulid,
            "content": parent.content,
            "event_type": parent.event_type,
            "importance": parent.importance,
            "timestamp": parent.timestamp,
            "metadata": parent.metadata,
        }

    def record_event(
        self,
        event_type: str,
        content: str,
        importance: float = 0.5,
        metadata: Optional[dict] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Generic event record (git, file, test, ...)."""
        clean_content, _ = redact(content)
        clean_metadata, _ = redact_metadata(metadata or {})
        ev = Event(
            workspace_id=self.workspace.id,
            event_type=event_type,
            content=clean_content,
            metadata=clean_metadata,
            importance=importance,
            source_session_id=session_id,
            tier=default_tier_for_event_type(event_type),
        )
        ev = self.events.append(ev)
        # Synchronous when called directly (Python API contract); deferred
        # only when invoked from inside record_batch (batch_defer set).
        if getattr(self, "_batch_defer", False):
            self._embed_or_defer(ev.ulid, ev.to_text())
        else:
            self.search.add(ev.ulid, ev.to_text())
        self._index_event_in_graph(ev, full_text=clean_content)
        # Cheap rule-based causation: temporal-next edge from the last event.
        # No LLM, just SQL. Fires only if the previous event is within minutes.
        try:
            from pmb.reasoning.causation import add_temporal_next_edge

            add_temporal_next_edge(self, ev)
        except Exception:
            pass
        # Improvement C: parse event_time (date references) from content
        # and store in metadata. Enables temporal-proximity boost at recall.
        try:
            self._attach_event_time(ev)
        except Exception:
            pass
        self.recall_cache.bump_generation()
        return ev.ulid

    def record_image(
        self,
        path: str,
        description: str = "",
        importance: float = 0.5,
        metadata: Optional[dict] = None,
        session_id: Optional[str] = None,
        encode_clip: bool = False,
    ) -> str:
        """Improvement J: record an image with optional CLIP embedding.

        The `description` is what's searchable via text — make it specific
        ('screenshot of auth flow diagram with arrows', not 'image').

        encode_clip=True triggers CLIP encoding (requires open_clip or
        sentence-transformers); silently degrades to text-only if missing.

        Returns the ulid of the image event.
        """
        from pmb.reasoning.images import attach_image

        att = attach_image(path, description=description, encode_clip=encode_clip)
        meta = dict(metadata or {})
        meta.update(att.to_metadata())
        # The event's content IS the description — that's what's indexed
        content = description or f"image: {Path(att.path).name}"
        return self.record_event(
            event_type="image",
            content=content,
            importance=importance,
            metadata=meta,
            session_id=session_id,
        )

    def search_images_by_text(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        """Cross-modal: encode query via CLIP text encoder, cosine against
        stored image CLIP embeddings. Falls back to plain text recall if
        CLIP unavailable."""
        import json as _j

        try:
            from pmb.reasoning.images import clip_encode_text
            import numpy as np

            q_emb = clip_encode_text(query)
            if q_emb is None:
                # Fallback: text-only recall, filtered to image events
                pack = self.recall(query, top_k=top_k * 2)
                return [r.to_dict() for r in pack.results if r.event_type == "image"][:top_k]
            # Iterate image events, compute cosine
            import sqlite3

            scored: list[tuple[str, float]] = []
            with sqlite3.connect(self.workspace.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT ulid, content, metadata_json FROM events "
                    "WHERE workspace_id = ? AND event_type = 'image' "
                    "AND archived_at IS NULL",
                    (self.workspace.id,),
                ).fetchall()
            for r in rows:
                try:
                    meta = _j.loads(r["metadata_json"] or "{}")
                    ce = meta.get("clip_embedding_json")
                    if not ce:
                        continue
                    img_emb = np.array(_j.loads(ce), dtype=np.float32)
                    if img_emb.size == 0:
                        continue
                    qe = np.array(q_emb, dtype=np.float32)
                    sim = float(
                        np.dot(img_emb, qe) / (np.linalg.norm(img_emb) * np.linalg.norm(qe) + 1e-9)
                    )
                    scored.append((r["ulid"], sim))
                except Exception:
                    continue
            scored.sort(key=lambda x: -x[1])
            top = scored[:top_k]
            out = []
            for ulid, sim in top:
                ev = self.events.get_by_ulid(ulid)
                if ev:
                    out.append(
                        {
                            "ulid": ev.ulid,
                            "score": sim,
                            "content": ev.content,
                            "metadata": ev.metadata,
                        }
                    )
            return out
        except Exception:
            return []
