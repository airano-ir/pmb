from __future__ import annotations

import time

from pmb.core.events import (
    Event,
)
from pmb.security.redact import redact, redact_metadata


class GoalsMixin:
    def record_goal(
        self,
        title: str,
        status: str = "pending",  # pending / in_progress / done / cancelled
        parent_goal_ulid: str | None = None,
        due_at: float | None = None,
        importance: float = 0.7,
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Create a goal/intent event. Goals have status + optional hierarchy.

        Use when user says they want / plan / intend to do something:
          "I want to learn Rust by year-end"
          "Need to ship v1.0 by Q3"
          "Plan: refactor auth module first, then frontend"

        Returns ulid.
        """
        clean_title, _ = redact(title)

        # Bulk-import shortcut: skip dedup + graph + L2.5 queue
        if getattr(self, "_bulk_mode", False):
            meta_bulk = {"goal_status": status, "goal_progress": 0}
            if parent_goal_ulid:
                meta_bulk["parent_goal_ulid"] = parent_goal_ulid
            if due_at is not None:
                meta_bulk["due_at"] = float(due_at)
            if metadata:
                meta_bulk.update(metadata)
            ev = Event(
                workspace_id=self.workspace.id,
                event_type="goal",
                content=clean_title,
                metadata=meta_bulk,
                importance=importance,
                source_session_id=session_id,
                tier="semantic",
            )
            ev = self.events.append(ev)
            self._embed_or_defer(ev.ulid, ev.to_text())
            self._bulk_collected_ulids.append(ev.ulid)
            return ev.ulid

        # Improvement U: dedup at write — most common case the user hits is
        # the AI writing the SAME goal in two languages (RU + EN) as two
        # separate goal events. L1 won't catch translations; L2 (cosine on
        # multilingual model) will.
        dup_hit, borderline = self._dedup_pre_write(
            content=clean_title,
            event_type="goal",
        )
        if dup_hit is not None:
            self._bump_for_dup(dup_hit.canonical_ulid)
            return dup_hit.canonical_ulid

        meta = {
            "goal_status": status,
            "goal_progress": 0,
        }
        if parent_goal_ulid:
            meta["parent_goal_ulid"] = parent_goal_ulid
        if due_at is not None:
            meta["due_at"] = float(due_at)
        if metadata:
            meta.update(metadata)
        ev = Event(
            workspace_id=self.workspace.id,
            event_type="goal",
            content=clean_title,
            metadata=meta,
            importance=importance,
            source_session_id=session_id,
            tier="semantic",  # goals are long-lived by default
        )
        ev = self.events.append(ev)
        # Improvement W: embed inline if model loaded, else queue
        self._embed_or_defer(ev.ulid, ev.to_text())
        try:
            self._index_event_in_graph(ev, full_text=clean_title)
        except Exception:
            pass

        # L2.5: borderline goal pair — enqueue for async LLM verify
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

    def update_goal(
        self,
        goal_ulid: str,
        status: str | None = None,
        progress: int | None = None,  # 0..100
        note: str | None = None,
    ) -> dict:
        """Update a goal's status / progress. Creates a linked update event
        recording the transition (so history of changes is preserved).
        """
        ev = self.events.get_by_ulid(goal_ulid)
        if ev is None or ev.event_type != "goal":
            return {"error": "goal not found"}
        meta = dict(ev.metadata or {})
        old_status = meta.get("goal_status")
        old_progress = meta.get("goal_progress", 0)
        if status is not None:
            meta["goal_status"] = status
        if progress is not None:
            meta["goal_progress"] = max(0, min(100, int(progress)))
        # Persist updated metadata in-place
        import json as _j
        import sqlite3

        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.execute(
                "UPDATE events SET metadata_json = ? WHERE ulid = ?",
                (_j.dumps(meta), goal_ulid),
            )

        # Append a separate "update" event in the chain so history exists
        upd_content = note or (
            f"Goal updated: {old_status}→{meta.get('goal_status')}, "
            f"{old_progress}→{meta.get('goal_progress', 0)}%"
        )
        upd_ev = Event(
            workspace_id=self.workspace.id,
            event_type="goal_update",
            content=upd_content,
            metadata={
                "goal_ulid": goal_ulid,
                "old_status": old_status,
                "new_status": meta.get("goal_status"),
                "old_progress": old_progress,
                "new_progress": meta.get("goal_progress", 0),
            },
            importance=0.5,
            tier="working",  # updates fade after a few days
        )
        upd_ev = self.events.append(upd_ev)
        # Link update → goal via event_edges
        try:
            from pmb.reasoning.causation import CausationEdge, upsert_edge

            upsert_edge(
                self.workspace.db_path,
                CausationEdge(
                    source_ulid=upd_ev.ulid,
                    target_ulid=goal_ulid,
                    edge_type="references",
                    confidence=1.0,
                    rationale="update of goal",
                ),
            )
        except Exception:
            pass
        self.recall_cache.bump_generation()
        return {
            "goal_ulid": goal_ulid,
            "update_ulid": upd_ev.ulid,
            "status": meta.get("goal_status"),
            "progress": meta.get("goal_progress", 0),
        }

    def list_goals(
        self,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """List goals (optionally filtered by status)."""
        import json as _j
        import sqlite3

        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            sql = (
                "SELECT ulid, content, timestamp, importance, metadata_json "
                "FROM events WHERE workspace_id = ? AND event_type = 'goal' "
                "AND archived_at IS NULL"
            )
            params: list = [self.workspace.id]
            if status:
                sql += " AND json_extract(metadata_json, '$.goal_status') = ?"
                params.append(status)
            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            try:
                meta = _j.loads(r["metadata_json"] or "{}")
            except Exception:
                meta = {}
            out.append(
                {
                    "ulid": r["ulid"],
                    "title": r["content"],
                    "status": meta.get("goal_status"),
                    "progress": meta.get("goal_progress", 0),
                    "parent_goal_ulid": meta.get("parent_goal_ulid"),
                    "due_at": meta.get("due_at"),
                    "timestamp": r["timestamp"],
                }
            )
        return out

    def record_milestone(
        self,
        chain_name: str,
        title: str,
        state: dict | None = None,
        triggered_by_ulid: str | None = None,
        importance: float = 0.6,
    ) -> str:
        """Record a milestone in a named state-chain.

        Each milestone references the previous one in the same chain
        (auto-linked) and optionally the event that triggered the change.

        Example:
          record_milestone(
              chain_name="architecture_layers",
              title="11 layers (added activity log)",
              state={"count": 11, "added": "activity"},
              triggered_by_ulid=<event ulid for the implementation>,
          )

        Later: `chain_history("architecture_layers")` returns the full
        sequence: 6 → 7 → ... → 11, with reasons.
        """
        clean_title, _ = redact(title)
        # Find the previous milestone in this chain (latest by timestamp)
        import sqlite3

        prev_ulid: str | None = None
        with sqlite3.connect(self.workspace.db_path) as conn:
            row = conn.execute(
                "SELECT ulid FROM events WHERE workspace_id = ? "
                "AND event_type = 'milestone' "
                "AND json_extract(metadata_json, '$.chain_name') = ? "
                "AND archived_at IS NULL "
                "ORDER BY timestamp DESC LIMIT 1",
                (self.workspace.id, chain_name),
            ).fetchone()
            if row:
                prev_ulid = row[0]

        meta: dict = {"chain_name": chain_name}
        if prev_ulid:
            meta["previous_milestone_ulid"] = prev_ulid
        if triggered_by_ulid:
            meta["triggered_by_ulid"] = triggered_by_ulid
        if state:
            clean_state, _ = redact_metadata(state)
            meta["state"] = clean_state

        ev = Event(
            workspace_id=self.workspace.id,
            event_type="milestone",
            content=clean_title,
            metadata=meta,
            importance=importance,
            tier="semantic",
        )
        ev = self.events.append(ev)
        # Synchronous when called directly (Python API contract); deferred
        # only when invoked from inside record_batch (batch_defer set).
        try:
            if getattr(self, "_batch_defer", False):
                self._embed_or_defer(ev.ulid, ev.to_text())
            else:
                self.search.add(ev.ulid, ev.to_text())
        except Exception:
            pass
        try:
            self._index_event_in_graph(ev, full_text=clean_title)
        except Exception:
            pass

        # Edges: this milestone → previous (chain link), and triggered_by → this
        try:
            from pmb.reasoning.causation import CausationEdge, upsert_edge

            if prev_ulid:
                upsert_edge(
                    self.workspace.db_path,
                    CausationEdge(
                        source_ulid=prev_ulid,
                        target_ulid=ev.ulid,
                        edge_type="temporal-next",
                        confidence=1.0,
                        rationale=f"chain {chain_name}",
                    ),
                )
            if triggered_by_ulid:
                upsert_edge(
                    self.workspace.db_path,
                    CausationEdge(
                        source_ulid=triggered_by_ulid,
                        target_ulid=ev.ulid,
                        edge_type="causes",
                        confidence=1.0,
                        rationale="triggered milestone",
                    ),
                )
        except Exception:
            pass
        self.recall_cache.bump_generation()
        return ev.ulid

    def chain_history(self, chain_name: str, limit: int = 100) -> list[dict]:
        """Full chronological history of a named state-chain.

        Returns oldest-first list of milestones with their state snapshots
        and trigger events. Reconstructs the evolution: 6 → 7 → ... → 11
        with the reason at each step.
        """
        import json as _j
        import sqlite3

        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ulid, content, timestamp, metadata_json "
                "FROM events WHERE workspace_id = ? AND event_type = 'milestone' "
                "AND json_extract(metadata_json, '$.chain_name') = ? "
                "AND archived_at IS NULL "
                "ORDER BY timestamp ASC LIMIT ?",
                (self.workspace.id, chain_name, limit),
            ).fetchall()
        out = []
        for r in rows:
            try:
                meta = _j.loads(r["metadata_json"] or "{}")
            except Exception:
                meta = {}
            out.append(
                {
                    "ulid": r["ulid"],
                    "title": r["content"],
                    "timestamp": r["timestamp"],
                    "state": meta.get("state", {}),
                    "triggered_by_ulid": meta.get("triggered_by_ulid"),
                    "previous_milestone_ulid": meta.get("previous_milestone_ulid"),
                }
            )
        return out

    def chain_current(self, chain_name: str) -> dict | None:
        """Latest milestone of a chain — the "current state"."""
        hist = self.chain_history(chain_name, limit=200)
        return hist[-1] if hist else None

    def record_activity(
        self,
        summary: str,
        actor: str = "agent",
        kind: str = "action",
        details: dict | None = None,
        importance: float = 0.4,
        session_id: str | None = None,
    ) -> str:
        """Log an action / activity. Used by the AI to record what it
        just did (made an edit, ran a tool, gave advice). Lighter than
        record_fact — these are session-scoped working memory.

        actor: 'agent' (AI's own action), 'user' (user did something),
               'system' (auto-generated event).
        kind:  'action' (default), 'edit', 'tool_call', 'recommendation',
               'plan', 'completed'.

        Returns ulid.
        """
        clean_summary, _ = redact(summary)
        meta = {"actor": actor, "activity_kind": kind}
        if details:
            clean_details, _ = redact_metadata(details)
            meta.update(clean_details)

        # E4: keep routine activities from crowding real facts in ranking.
        # Unless effectively pinned (>=0.99), cap activity importance at the
        # configured ceiling and leave a breadcrumb. Facts/lessons/goals are
        # written by other methods and never pass through here.
        try:
            ceiling = float(self.config.get("write.max_activity_importance") or 0.8)
            if importance < 0.99 and importance > ceiling:
                meta["importance_clamped"] = importance
                importance = ceiling
        except Exception:
            pass

        # R7: decisions are durable "why" memory. The documented agent pattern
        # records them as {"type":"activity","kind":"decision"}, but the working
        # tier decays with a ~2-day half-life, so a decision auto-archives within
        # a week — even though events.py reserves the SEMANTIC tier for
        # "decision / rule". Land kind=decision in the semantic tier so the
        # rationale survives; everything else stays working memory.
        _tier = "semantic" if kind == "decision" else "working"

        # 0.2 (former E6): exact-duplicate suppression for activities. Agents
        # re-log the same action seconds apart (live workspaces showed identical
        # activities ~60s apart). Within write.dedup_window_h, bump the existing
        # row instead of inserting a twin. Pure SQL probe — safe inside a batch.
        self._last_write_deduped = False
        if not getattr(self, "_bulk_mode", False):
            try:
                _win = float(self.config.get("write.dedup_window_h") or 0.0)
            except Exception:
                _win = 0.0
            if _win > 0:
                _hit = self._exact_dup_within_window(clean_summary, "activity", _win)
                if _hit is not None:
                    try:
                        self._bump_for_dup(_hit.canonical_ulid)
                    except Exception:
                        pass
                    self._last_write_deduped = True
                    return _hit.canonical_ulid

        # Bulk-import shortcut: skip graph + L2.5 queue, just persist
        if getattr(self, "_bulk_mode", False):
            ev = Event(
                workspace_id=self.workspace.id,
                event_type="activity",
                content=clean_summary,
                metadata=meta,
                importance=importance,
                source_session_id=session_id,
                tier=_tier,
            )
            ev = self.events.append(ev)
            self._embed_or_defer(ev.ulid, ev.to_text())
            self._bulk_collected_ulids.append(ev.ulid)
            return ev.ulid

        # Auto-bind to session
        if session_id is None:
            session_id = self.session_tracker.touch().id
        ev = Event(
            workspace_id=self.workspace.id,
            event_type="activity",
            content=clean_summary,
            metadata=meta,
            importance=importance,
            source_session_id=session_id,
            tier=_tier,  # working by default; semantic for kind=decision (R7)
        )
        ev = self.events.append(ev)
        # Synchronous unless inside batch.
        try:
            if getattr(self, "_batch_defer", False):
                self._embed_or_defer(ev.ulid, ev.to_text())
            else:
                self.search.add(ev.ulid, ev.to_text())
        except Exception:
            pass
        try:
            self._index_event_in_graph(ev, full_text=clean_summary)
        except Exception:
            pass
        try:
            from pmb.reasoning.causation import add_temporal_next_edge

            add_temporal_next_edge(self, ev)
        except Exception:
            pass
        self.recall_cache.bump_generation()
        return ev.ulid

    def recent_activity(
        self,
        minutes: float = 60.0,
        limit: int = 20,
        actor: str | None = None,
        kind: str | None = None,
    ) -> list[dict]:
        """Working memory dump — recent activity events, chronological.

        NO BM25/vector search — just SQL by timestamp. Instant.

        Use this BEFORE recall when answering "what did we just do",
        "what's the latest", "show recent changes" type questions.

        Filters:
          minutes: how far back (default 60)
          actor:   'agent' / 'user' / 'system' / None=all
          kind:    'action' / 'edit' / 'tool_call' / ... / None=all
        """
        import sqlite3

        cutoff = time.time() - minutes * 60.0
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            sql = (
                "SELECT ulid, content, timestamp, importance, metadata_json "
                "FROM events WHERE workspace_id = ? AND archived_at IS NULL "
                "AND event_type = 'activity' AND timestamp >= ?"
            )
            params: list = [self.workspace.id, cutoff]
            if actor:
                sql += " AND json_extract(metadata_json, '$.actor') = ?"
                params.append(actor)
            if kind:
                sql += " AND json_extract(metadata_json, '$.activity_kind') = ?"
                params.append(kind)
            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
        import json as _j

        out: list[dict] = []
        for r in rows:
            try:
                meta = _j.loads(r["metadata_json"] or "{}")
            except Exception:
                meta = {}
            out.append(
                {
                    "ulid": r["ulid"],
                    "content": r["content"],
                    "timestamp": r["timestamp"],
                    "actor": meta.get("actor"),
                    "kind": meta.get("activity_kind"),
                    "importance": r["importance"],
                }
            )
        return out

    def session_timeline(
        self,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Chronological events of a session. If session_id is None,
        uses the current session.

        Returns ALL event types (fact, qa, activity, ...) ordered by time.
        Use to summarize "what happened this session" or generate a
        post-mortem of what was done.
        """
        if session_id is None:
            sess = self.session_tracker.current()
            if not sess:
                return []
            session_id = sess.id
        import json as _j
        import sqlite3

        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ulid, event_type, content, timestamp, importance, metadata_json "
                "FROM events WHERE workspace_id = ? AND source_session_id = ? "
                "AND archived_at IS NULL "
                "ORDER BY timestamp ASC LIMIT ?",
                (self.workspace.id, session_id, limit),
            ).fetchall()
        out = []
        for r in rows:
            try:
                meta = _j.loads(r["metadata_json"] or "{}")
            except Exception:
                meta = {}
            out.append(
                {
                    "ulid": r["ulid"],
                    "event_type": r["event_type"],
                    "content": r["content"],
                    "timestamp": r["timestamp"],
                    "actor": meta.get(
                        "actor", "system" if r["event_type"] == "activity" else "user"
                    ),
                }
            )
        return out

    def what_just_happened(self, n: int = 5) -> list[dict]:
        """Last N events of ANY type, newest first.

        Used by AI to answer 'what did we just do?' without going through
        recall search.

        Returns activities AND facts AND any other event types — just the
        most recent stuff regardless of session binding. For session-only
        view use `session_timeline()`.
        """
        evs = self.events.list_active(self.workspace.id, limit=n)
        return [
            {
                "ulid": e.ulid,
                "event_type": e.event_type,
                "content": (e.content or "")[:300],
                "timestamp": e.timestamp,
                "actor": (e.metadata or {}).get("actor", "user"),
            }
            for e in evs[:n]
        ]
