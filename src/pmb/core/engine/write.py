from __future__ import annotations

import time
from pathlib import Path

from pmb.core.events import (
    Event,
    default_tier_for_event_type,
)
from pmb.reasoning.attributes import (
    canonicalize_attribute,
    detect_current_state,
    detect_negated_state,
    has_user_subject_cue,
    keyed_fact_key,
    looks_like_future_intent,
)
from pmb.security.redact import redact, redact_metadata


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
        try:
            _canon = bool(self.config.get("extract.canonical_atoms"))
        except Exception:
            _canon = False
        atoms = extract_atomic_facts(text, canonical=_canon)
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
        session_id: str | None = None,
        importance: float = 0.5,
        metadata: dict | None = None,
    ) -> str:
        """Save a Q/A pair. Returns the ulid."""
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
        # Index into hybrid search. Synchronous: the Python API contract is
        # "write returns when data is searchable". Use `_embed_or_defer`
        # only when we're inside `record_batch` (it sets _batch_defer).
        if getattr(self, "_batch_defer", False):
            self._embed_or_defer(ev.ulid, ev.to_text())
        else:
            self.search.add(ev.ulid, ev.to_text())
        # Index into graph (deferred to a worker for LLM backends)
        self._index_graph_or_defer(ev, full_text=f"{clean_query}\n{clean_response}")
        try:
            from pmb.reasoning.causation import add_temporal_next_edge

            add_temporal_next_edge(self, ev)
        except Exception as e:
            from pmb.core.errlog import log_error
            log_error(self.workspace.db_path, "temporal_next_edge", e)
        # Improvement C: parse event_time (date references) from content
        # and store in metadata. Enables temporal-proximity boost at recall.
        try:
            self._attach_event_time(ev)
        except Exception as e:
            from pmb.core.errlog import log_error
            log_error(self.workspace.db_path, "attach_event_time", e)
        self.recall_cache.bump_generation()
        return ev.ulid

    def record_fact(
        self,
        fact: str,
        importance: float = 0.7,
        metadata: dict | None = None,
        session_id: str | None = None,
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
        self._last_write_deduped = False
        dup_hit, borderline = self._dedup_pre_write(
            content=clean_fact,
            event_type="fact",
        )
        if dup_hit is not None:
            self._bump_for_dup(dup_hit.canonical_ulid)
            self._last_write_deduped = True
            return dup_hit.canonical_ulid

        # Write-time quality gate (#8, default OFF): FLAG (never reject)
        # suspected junk - cap importance and tag it so it can't be promoted to
        # a keyed fact and is a first-class declutter candidate. A memory
        # system must never silently drop user input, so we only down-weight.
        if self.config.get("write.quality_gate"):
            try:
                from pmb.maintenance.declutter import is_suspect_junk
                _mq = clean_metadata if isinstance(clean_metadata, dict) else {}
                if (
                    not _mq.get("keyed_fact_key")
                    and _mq.get("kind") != "lesson"
                    and _mq.get("source") != "lesson"
                    and is_suspect_junk(clean_fact)
                ):
                    clean_metadata = dict(_mq)
                    clean_metadata["quality_flag"] = "suspect_junk"
                    importance = min(float(importance), 0.2)
            except Exception:
                pass

        # R13: importance-spam containment. Agents habitually stamp 0.95 on
        # routine facts (the global CLAUDE.md even says importance=0.95 for
        # "remember"), and importance_factor = 0.5+0.5·imp gives a ~2× ranking
        # edge - so spam pollutes ranking. Cap how many >0.9 fact writes land
        # per rolling day; past the budget, clamp to a normal level and leave a
        # breadcrumb. Keyed/lesson facts are exempt; PINNED facts are exempt
        # automatically because pin() resets importance to 1.0 afterwards.
        try:
            budget = int(self.config.get("write.high_importance_daily_budget") or 0)
            _mq = clean_metadata if isinstance(clean_metadata, dict) else {}
            if (budget > 0 and float(importance) > 0.9
                    and not _mq.get("keyed_fact_key")
                    and _mq.get("kind") != "lesson"
                    and _mq.get("source") != "lesson"):
                import sqlite3 as _sql
                import time as _t
                cutoff = _t.time() - 86400.0
                with _sql.connect(str(self.workspace.db_path)) as _c:
                    n_hi = _c.execute(
                        "SELECT COUNT(*) FROM events WHERE workspace_id=? "
                        "AND event_type='fact' AND archived_at IS NULL "
                        "AND importance > 0.9 AND timestamp >= ?",
                        (self.workspace.id, cutoff),
                    ).fetchone()[0]
                if n_hi >= budget:
                    clean_metadata = dict(_mq)
                    clean_metadata["importance_budget_clamped"] = float(importance)
                    importance = min(float(importance), 0.7)
        except Exception:
            pass

        # Plan detector (#9): flag forward-looking "next we'll do X" facts so
        # the dashboard / `pmb goals` can suggest promoting them to a goal.
        # Non-destructive (a hint only) - we never auto-convert, since durable
        # preferences ("we will always use pnpm") would false-positive.
        try:
            _m = clean_metadata if isinstance(clean_metadata, dict) else {}
            if (
                _m.get("source") in self._CURRENT_STATE_SOURCES
                and not _m.get("keyed_fact_key")
                and not _m.get("kind")
                and not _m.get("is_subfact")
                and looks_like_future_intent(clean_fact, engine=self)
            ):
                clean_metadata = dict(_m)
                clean_metadata["suggest_goal"] = True
        except Exception:
            pass

        # B3: precompute the first-person flag at WRITE time so the per-candidate
        # PAMVR loop reads metadata.fp instead of re-deriving it. Lexical (free);
        # the warm `statement.about_self` anchor extends it to NON-ENGLISH personal
        # statements ("ich wohne in …") when anchor extraction is enabled (gated,
        # so a cold/default write pays no embed). Stored only when TRUE so
        # non-first-person facts stay metadata-clean and the lexical fallback
        # yields the same answer.
        try:
            if isinstance(clean_metadata, dict) and "fp" not in clean_metadata:
                from pmb.reasoning.pamvr import _has_first_person
                fp = _has_first_person(clean_fact)
                if (not fp and self.config.get("extract.anchor_keyed")
                        and getattr(self, "_is_warm", False)):
                    fp = self.anchor_fires(clean_fact[:120], "statement.about_self")
                if fp:
                    clean_metadata = dict(clean_metadata)
                    clean_metadata["fp"] = 1
        except Exception:
            pass

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
        # Keep the user-name cache fresh: a "My name is X" fact must take effect
        # on the very NEXT recall (mark dirty), not after 25 more events.
        try:
            self.note_user_names_write()
            from pmb.reasoning.user_names import looks_like_name_statement
            if looks_like_name_statement(clean_fact):
                self.mark_user_names_dirty()
        except Exception as e:
            from pmb.core.errlog import log_error
            log_error(self.workspace.db_path, "user_names_mark", e)
        # Improvement W: embed inline if model loaded, else queue
        self._embed_or_defer(ev.ulid, ev.to_text())
        self._index_graph_or_defer(ev, full_text=clean_fact)
        try:
            from pmb.reasoning.causation import add_temporal_next_edge

            add_temporal_next_edge(self, ev)
        except Exception as e:
            from pmb.core.errlog import log_error
            log_error(self.workspace.db_path, "temporal_next_edge", e)
        # Improvement C: parse event_time (date references) from content
        # and store in metadata. Enables temporal-proximity boost at recall.
        try:
            self._attach_event_time(ev)
        except Exception as e:
            from pmb.core.errlog import log_error
            log_error(self.workspace.db_path, "attach_event_time", e)

        # L2.5: borderline candidate detected - enqueue for async LLM verify
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
            except Exception as e:
                from pmb.core.errlog import log_error
                log_error(self.workspace.db_path, "dedup_borderline", e)

        # Current-state promotion (#9): if this plain user fact states a
        # CURRENT personal attribute ("I now live in Tampa"), also upsert the
        # matching keyed fact so the live value supersedes any stale one. The
        # plain fact above stays as history.
        self._maybe_promote_current_state(clean_fact, clean_metadata, importance)
        # Reverse direction (#E1): if this fact NEGATES an attribute the user
        # has a keyed value for ("I no longer live in Tampa"), close that value
        # so recall stops asserting it as current.
        self._maybe_close_on_negation(
            clean_fact, clean_metadata, ev.ulid,
            getattr(ev, "timestamp", None) or time.time())

        self.recall_cache.bump_generation()
        return ev.ulid

    # User-origin sources that may carry a personal current-state statement;
    # internal pipelines (reflection / project index / autowrite / …) excluded.
    _CURRENT_STATE_SOURCES = frozenset(
        {None, "", "cli", "cli-note", "mcp", "chat", "user", "note"}
    )

    def _maybe_promote_current_state(self, content, metadata, importance) -> None:
        """If CONTENT is a user statement of a current personal attribute,
        upsert the matching keyed fact so the live value supersedes the stale
        one. Best-effort + tightly gated; never breaks a normal record_fact."""
        try:
            meta = metadata if isinstance(metadata, dict) else {}
            if meta.get("keyed_fact_key"):
                return  # already a keyed fact - don't re-key (prevents recursion)
            if meta.get("quality_flag") == "suspect_junk":
                return  # write-time gate flagged this as junk - never promote it
            if not self.config.get("keyed.auto_detect_current_state"):
                return
            if meta.get("source") not in self._CURRENT_STATE_SOURCES:
                return  # only user-origin facts, never internal pipelines
            hit = detect_current_state(content)
            if not hit:
                # C2 (warm-only, gated): the regex/pack tier missed - try the
                # multilingual hypothesis-margin extractor before giving up.
                self._anchor_promote_keyed(content, meta, importance)
                return
            attribute, value = hit
            self.record_keyed_fact(
                subject=meta.get("keyed_fact_subject") or "user",
                attribute=attribute,
                value=value,
                importance=max(float(importance), 0.7),
                metadata={
                    "source": "current_state_auto",
                    "derived_from": content[:200],
                },
            )
        except Exception:
            pass  # promotion is best-effort

    def _anchor_promote_keyed(self, content, meta, importance) -> None:
        """C2 + F1: warm-only keyed extraction by hypothesis margin when the
        regex tier missed. Records each candidate with extraction-confidence
        metadata (`extract={tier,margin,pos,model}`). Gated on
        `extract.anchor_keyed`; best-effort, never raises, never loads the model
        on a cold engine (the extractor's anchor_index() is warm-gated)."""
        try:
            if not self.config.get("extract.anchor_keyed"):
                return
            from pmb.reasoning.extract_anchor import extract_keyed_anchor
            cands = extract_keyed_anchor(content, self)
            if not cands:
                return
            model = getattr(self.search, "model_name", "default")
            subject = (meta or {}).get("keyed_fact_subject") or "user"
            for c in cands:
                self.record_keyed_fact(
                    subject=subject, attribute=c.attr, value=c.value,
                    importance=max(float(importance), 0.7),
                    metadata={
                        "source": "anchor_extract",
                        "derived_from": content[:200],
                        "extract": {"tier": "anchor", "margin": c.margin,
                                    "pos": c.pos, "model": model},
                    },
                )
        except Exception:
            pass  # extraction augmentation is best-effort

    def _maybe_close_on_negation(self, content, metadata, new_ulid, now_ts) -> None:
        """If CONTENT is the user negating an attribute they currently have a
        keyed value for ("I no longer live in Tampa" while user::city=Tampa),
        CLOSE that value: archive the active keyed fact and stamp valid_to /
        closed_by / closed_reason. Recall stops asserting it; keyed_fact_as_of
        still sees it as history (it reads archived versions too). The negation
        fact itself stays. Gated by keyed.close_on_negation; best-effort."""
        try:
            meta = metadata if isinstance(metadata, dict) else {}
            if meta.get("keyed_fact_key"):
                return  # the keyed fact itself, not a negation OF one
            if meta.get("kind") == "lesson" or meta.get("source") == "lesson":
                return
            if meta.get("source") not in self._CURRENT_STATE_SOURCES:
                return  # user-origin only, never internal pipelines
            if not self.config.get("keyed.close_on_negation"):
                return
            attr = detect_negated_state(content)  # post-A1: user-subject only
            if not attr:
                # F2: the regex tier missed - try the anchor anti-hypothesis
                # against the user's CURRENT keyed values (warm-only, gated).
                self._anchor_close_on_negation(content, new_ulid, now_ts)
                return
            self._close_keyed_attr(attr, new_ulid, now_ts, "negated_by_user")
        except Exception as e:
            try:
                from pmb.core.errlog import log_error
                log_error(self.workspace.db_path, "keyed_close_negation", e)
            except Exception:
                pass

    def _close_keyed_attr(self, attr: str, new_ulid, now_ts,
                          reason: str) -> bool:
        """Archive the user's CURRENT keyed value for `attr`, stamping
        valid_to / closed_by / closed_reason. Returns True if it closed one.
        Shared by the regex (`_maybe_close_on_negation`) and anchor (F2) paths."""
        import json as _json
        import sqlite3 as _sql
        key = keyed_fact_key("user", attr)
        with _sql.connect(str(self.workspace.db_path)) as conn:
            conn.row_factory = _sql.Row
            rows = conn.execute(
                "SELECT ulid, metadata_json, timestamp FROM events "
                "WHERE workspace_id=? AND archived_at IS NULL "
                "AND event_type='fact' AND metadata_json LIKE ? "
                "ORDER BY timestamp DESC",
                (self.workspace.id, f'%"keyed_fact_key": "{key}"%'),
            ).fetchall()
        for r in rows:
            if r["timestamp"] >= now_ts:
                continue  # never close a value newer than the negation
            self.events.archive(r["ulid"])
            m = _json.loads(r["metadata_json"] or "{}")
            if not isinstance(m, dict):
                m = {}
            m["valid_to"] = now_ts
            m["closed_by"] = new_ulid
            m["closed_reason"] = reason
            with _sql.connect(str(self.workspace.db_path)) as conn:
                conn.execute("UPDATE events SET metadata_json=? WHERE ulid=?",
                             (_json.dumps(m), r["ulid"]))
            return True  # only the current value
        return False

    def _anchor_close_on_negation(self, content, new_ulid, now_ts) -> None:
        """F2: detect a contradiction of the user's CURRENT keyed value via the
        anti-hypothesis. For each live keyed attr, embed the pos/anti hypotheses
        of the CURRENT value against the new sentence in ONE batch; if the anti
        side clearly wins, the user is negating that value → close it. Warm-only,
        gated on extract.anchor_keyed; best-effort, never raises/loads cold."""
        try:
            if not self.config.get("extract.anchor_keyed"):
                return
            idx = self.anchor_index()
            if idx is None:
                return
            if not idx.fires(content, "statement.about_self"):
                return
            import numpy as np

            from pmb.reasoning.extract_anchor import HYPOTHESES
            targets: list[tuple[str, str]] = []
            for attr in ("city", "employer"):
                if attr not in HYPOTHESES:
                    continue
                cur = [h for h in self.get_keyed_fact_history("user", attr)
                       if h.get("is_current") and h.get("value")]
                if cur:
                    targets.append((attr, str(cur[0]["value"])))
            if not targets:
                return
            texts = [content]
            for attr, val in targets:
                pos_t, neg_t = HYPOTHESES[attr]
                texts.append(pos_t.format(v=val))
                texts.append(neg_t.format(v=val))
            vecs = np.asarray(self.search.embed_batch(texts), dtype=np.float32)
            vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
            s = vecs[0]
            for i, (attr, _val) in enumerate(targets):
                pos = float(vecs[1 + 2 * i] @ s)
                anti = float(vecs[2 + 2 * i] @ s)
                if anti - pos > 0.05 and anti > 0.45:
                    self._close_keyed_attr(attr, new_ulid, now_ts, "anchor_negation")
        except Exception:
            pass  # contradiction check is best-effort

    def record_keyed_fact(
        self,
        subject: str,
        attribute: str,
        value: str,
        importance: float = 0.8,
        metadata: dict | None = None,
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
        # Canonical key: synonymous attribute labels (city / current_city /
        # current_city_2026 / lives_in / and their localized aliases) collapse
        # to ONE key, so a new value supersedes the old instead of creating a
        # competing key.
        key = keyed_fact_key(subject, attribute)

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
            import json as _json
            prior_values: list[str] = []
            for r in rows:
                prior_ulids.append(r["ulid"])
                try:
                    pv = (_json.loads(r["metadata_json"] or "{}") or {}).get("keyed_fact_value")
                    if pv:
                        prior_values.append(str(pv))
                except Exception:
                    pass
        except Exception:
            prior_values = []

        # C2 cosine-merge: if the new value is an INFLECTION of a prior value
        # (RU "Kieve" vs "Kiev", a German dative), converge them - store the
        # shorter / nominative-looking form so one fact doesn't churn into two
        # competing keyed values. Warm-only + gated (extract.anchor_keyed);
        # byte-identical values already collapse for free upstream.
        try:
            if (prior_values and self.config.get("extract.anchor_keyed")
                    and getattr(self, "_is_warm", False)):
                from pmb.reasoning.extract_anchor import values_are_alias
                for pv in prior_values:
                    if pv.lower() != value.lower() and values_are_alias(self, value, pv):
                        value = min(value, pv, key=len)   # shorter ≈ nominative
                        break
                # If the merge converged onto an existing value, this is a pure
                # inflected RESTATEMENT - keep the current fact as-is instead of
                # churning it into an archived duplicate (identical content would
                # otherwise dedup-collide with the very row we'd archive).
                if any(pv.lower() == value.lower() for pv in prior_values):
                    cur = [h for h in self.get_keyed_fact_history(subject, attribute)
                           if h.get("is_current")]
                    if cur:
                        return {"ulid": cur[0].get("ulid"),
                                "superseded_ulids": [], "key": key,
                                "value": value, "merged_restatement": True}
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

        # 3. Archive priors - they stay in SQLite (queryable as history)
        # but `archived_at IS NULL` filter removes them from recall.
        for old_ulid in prior_ulids:
            try:
                self.events.archive(old_ulid)
                # Tag with the new pointer so callers can trace history.
                import json as _json
                import sqlite3 as _sql

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

        # Negation-tombstone cleanup (#5): now that a positive value exists,
        # retire older "user does NOT live in X / current city is unknown"
        # facts for the same attribute - they assert ignorance about a
        # now-known attribute and are pure stale noise. Archive-only.
        negated = self._archive_obsolete_negations(
            canonicalize_attribute(attribute), new_ulid, now_ts,
        )

        self.recall_cache.bump_generation()
        return {
            "new_ulid": new_ulid,
            "superseded_ulids": prior_ulids,
            "negation_ulids": negated,
            "key": key,
        }

    def _archive_obsolete_negations(
        self, canon_attribute: str, new_ulid: str, before_ts: float,
    ) -> list[str]:
        """Archive active plain facts that NEGATE / mark-unknown the given
        canonical attribute and predate `before_ts`, now that a positive keyed
        value exists. Archive-only (reversible), tagged superseded_by +
        superseded_reason. Skips pinned events, lessons, and keyed facts.
        Bounded scan (last 2000 active facts) so the write path stays fast.
        Gated by config `keyed.archive_obsolete_negations`. Returns archived
        ulids (empty list when disabled or nothing matched)."""
        if not self.config.get("keyed.archive_obsolete_negations"):
            return []
        archived: list[str] = []
        try:
            import json as _json
            import sqlite3 as _sql

            with _sql.connect(str(self.workspace.db_path)) as conn:
                conn.row_factory = _sql.Row
                rows = conn.execute(
                    "SELECT ulid, content, metadata_json, importance "
                    "FROM events WHERE workspace_id = ? AND archived_at IS NULL "
                    "AND event_type = 'fact' AND timestamp < ? "
                    "ORDER BY timestamp DESC LIMIT 2000",
                    (self.workspace.id, before_ts),
                ).fetchall()
            for r in rows:
                if r["ulid"] == new_ulid:
                    continue
                try:
                    meta = _json.loads(r["metadata_json"] or "{}")
                except Exception:
                    meta = {}
                if not isinstance(meta, dict):
                    meta = {}
                if meta.get("keyed_fact_key"):
                    continue  # keyed facts handled by supersession, not here
                if meta.get("kind") == "lesson" or meta.get("source") == "lesson":
                    continue  # lessons are instructions, not state
                if float(r["importance"] or 0.0) >= 0.99:
                    continue  # pinned - never auto-archive
                if detect_negated_state(r["content"] or "") != canon_attribute:
                    continue
                self.events.archive(r["ulid"])
                meta["superseded_by"] = new_ulid
                meta["superseded_reason"] = "negation_obsoleted_by_value"
                with _sql.connect(str(self.workspace.db_path)) as conn:
                    conn.execute(
                        "UPDATE events SET metadata_json = ? WHERE ulid = ?",
                        (_json.dumps(meta), r["ulid"]),
                    )
                archived.append(r["ulid"])
        except Exception as e:
            try:
                from pmb.core.errlog import log_error
                log_error(self.workspace.db_path, "negation_tombstone", e)
            except Exception:
                pass
        return archived

    def archive_negations_for_current_keys(self, dry_run: bool = True) -> dict:
        """Repair pass (#5): for every attribute that currently has a positive
        keyed value, archive older active negation/"unknown" facts about that
        same attribute. Retroactive version of the write-time cleanup, for
        corpora written before it existed. Archive-only; dry_run reports a plan.

        Returns {plan: [{attribute, ulid, content}], n, dry_run}.
        """
        import json as _json
        import sqlite3 as _sql

        # Which canonical attributes have a current positive keyed value?
        attrs_with_value: dict[str, tuple[str, float]] = {}  # canon -> (new_ulid, ts)
        try:
            with _sql.connect(str(self.workspace.db_path)) as conn:
                conn.row_factory = _sql.Row
                rows = conn.execute(
                    "SELECT ulid, metadata_json, timestamp FROM events "
                    "WHERE workspace_id = ? AND archived_at IS NULL "
                    "AND event_type = 'fact' "
                    "AND metadata_json LIKE '%\"keyed_fact_key\"%' "
                    "ORDER BY timestamp ASC",
                    (self.workspace.id,),
                ).fetchall()
            for r in rows:
                try:
                    meta = _json.loads(r["metadata_json"] or "{}")
                except Exception:
                    continue
                attr = meta.get("keyed_fact_attribute")
                if not attr:
                    continue
                canon = canonicalize_attribute(attr)
                attrs_with_value[canon] = (r["ulid"], r["timestamp"])
        except Exception:
            return {"plan": [], "n": 0, "dry_run": dry_run, "error": "scan_failed"}

        plan = []
        try:
            with _sql.connect(str(self.workspace.db_path)) as conn:
                conn.row_factory = _sql.Row
                rows = conn.execute(
                    "SELECT ulid, content, metadata_json, importance, timestamp "
                    "FROM events WHERE workspace_id = ? AND archived_at IS NULL "
                    "AND event_type = 'fact' ORDER BY timestamp DESC LIMIT 5000",
                    (self.workspace.id,),
                ).fetchall()
        except Exception:
            return {"plan": [], "n": 0, "dry_run": dry_run, "error": "scan_failed"}

        for r in rows:
            try:
                meta = _json.loads(r["metadata_json"] or "{}")
            except Exception:
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
            if meta.get("keyed_fact_key"):
                continue
            if meta.get("kind") == "lesson" or meta.get("source") == "lesson":
                continue
            if float(r["importance"] or 0.0) >= 0.99:
                continue
            attr = detect_negated_state(r["content"] or "")
            if attr is None or attr not in attrs_with_value:
                continue
            new_ulid, val_ts = attrs_with_value[attr]
            if r["timestamp"] >= val_ts:
                continue  # negation is newer than the positive value - leave it
            plan.append({"attribute": attr, "ulid": r["ulid"],
                         "content": (r["content"] or "")[:120],
                         "superseded_by": new_ulid})

        if not dry_run:
            for p in plan:
                try:
                    self.events.archive(p["ulid"])
                    with _sql.connect(str(self.workspace.db_path)) as conn:
                        row = conn.execute(
                            "SELECT metadata_json FROM events WHERE ulid = ?",
                            (p["ulid"],),
                        ).fetchone()
                        m = _json.loads(row[0] or "{}") if row else {}
                        m["superseded_by"] = p["superseded_by"]
                        m["superseded_reason"] = "negation_obsoleted_by_value"
                        conn.execute(
                            "UPDATE events SET metadata_json = ? WHERE ulid = ?",
                            (_json.dumps(m), p["ulid"]),
                        )
                except Exception:
                    continue
            self.recall_cache.bump_generation()

        return {"plan": plan, "n": len(plan), "dry_run": dry_run}

    def keyed_fact_as_of(
        self, subject: str, attribute: str, at_time: float,
    ) -> dict | None:
        """As-of temporal query (Zep-style): what was the value of
        (subject, attribute) at `at_time` (UTC epoch seconds)?

        Uses the valid_from / valid_to windows stamped by record_keyed_fact
        supersession, so "what city did I live in last March" returns the
        value that was current THEN, not the latest one.

        Standalone - NOT wired into the recall hot path, so it cannot affect
        recall ranking or latency. Returns {value, valid_from, valid_to,
        ulid, content} for the version valid at `at_time`, else None.
        """
        key = keyed_fact_key(subject, attribute)
        versions: list[tuple] = []
        try:
            import json as _json
            import sqlite3 as _sql
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

    # ──────────────────────────────────────────────────────────────────
    # Lesson surface tracking (self-improvement loop)
    # ──────────────────────────────────────────────────────────────────

    def record_preference(
        self,
        preference: str,
        importance: float = 0.7,
        metadata: dict | None = None,
    ) -> str:
        """User preference. Event_type='preference', tier='semantic'.
        Examples: "I prefer dark mode", "I like calm games".
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
        metadata: dict | None = None,
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
        key = keyed_fact_key(subject, attribute)
        try:
            import json as _json
            import sqlite3 as _sql

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

    def repair_keyed_facts(self, dry_run: bool = True) -> dict:
        """Collapse competing keyed facts onto ONE canonical value per
        (subject, canonical-attribute). For each group of ACTIVE keyed facts
        that canonicalize to the same key (e.g. ``user::city`` and the stale
        ``user::current_city_2026``), keep the newest value active, archive the
        rest (``superseded_by`` + ``valid_to``), and rewrite the survivor's key
        to canonical so future upserts supersede it correctly.

        Non-destructive - only archives + retags, never deletes. ``dry_run``
        (default True) reports the plan without writing.

        Returns {groups, n_archived, n_recanonicalized, dry_run}.
        """
        import json as _json
        import sqlite3 as _sql

        try:
            with _sql.connect(str(self.workspace.db_path)) as conn:
                conn.row_factory = _sql.Row
                rows = conn.execute(
                    "SELECT ulid, content, metadata_json, timestamp FROM events "
                    "WHERE workspace_id = ? AND archived_at IS NULL "
                    "AND event_type = 'fact' "
                    "AND metadata_json LIKE '%\"keyed_fact_key\"%' "
                    "ORDER BY timestamp ASC",
                    (self.workspace.id,),
                ).fetchall()
        except Exception:
            return {"groups": [], "n_archived": 0, "n_recanonicalized": 0,
                    "dry_run": dry_run, "error": "scan_failed"}

        groups: dict[str, list[dict]] = {}
        for r in rows:
            try:
                meta = _json.loads(r["metadata_json"] or "{}")
            except Exception:
                meta = {}
            old_key = meta.get("keyed_fact_key") or ""
            subject = meta.get("keyed_fact_subject")
            attribute = meta.get("keyed_fact_attribute")
            if subject and attribute:
                canon = keyed_fact_key(subject, attribute)
            elif "::" in old_key:
                s, a = old_key.split("::", 1)
                canon = keyed_fact_key(s, a)
            else:
                continue
            groups.setdefault(canon, []).append({
                "ulid": r["ulid"],
                "value": meta.get("keyed_fact_value"),
                "timestamp": r["timestamp"],
                "old_key": old_key,
            })

        plan: list[dict] = []
        for canon, members in groups.items():
            members.sort(key=lambda m: m["timestamp"])
            keep = members[-1]            # newest wins
            losers = members[:-1]
            needs_recanon = keep["old_key"] != canon
            if len(members) == 1 and not needs_recanon:
                continue                  # already clean
            plan.append({
                "canonical_key": canon,
                "keep_ulid": keep["ulid"],
                "keep_value": keep["value"],
                "archive_ulids": [m["ulid"] for m in losers],
                "archive_values": [m["value"] for m in losers],
                "recanonicalize": needs_recanon,
            })
            if dry_run:
                continue
            now_ts = time.time()
            for m in losers:
                try:
                    self.events.archive(m["ulid"])
                    with _sql.connect(str(self.workspace.db_path)) as conn:
                        row = conn.execute(
                            "SELECT metadata_json FROM events WHERE ulid = ?",
                            (m["ulid"],),
                        ).fetchone()
                        om = _json.loads(row[0] or "{}") if row else {}
                        om["superseded_by"] = keep["ulid"]
                        om["valid_to"] = now_ts
                        om["repaired_by"] = "repair_keyed_facts"
                        conn.execute(
                            "UPDATE events SET metadata_json = ? WHERE ulid = ?",
                            (_json.dumps(om), m["ulid"]),
                        )
                except Exception:
                    continue
            if needs_recanon:
                try:
                    with _sql.connect(str(self.workspace.db_path)) as conn:
                        row = conn.execute(
                            "SELECT metadata_json FROM events WHERE ulid = ?",
                            (keep["ulid"],),
                        ).fetchone()
                        km = _json.loads(row[0] or "{}") if row else {}
                        km["keyed_fact_key"] = canon
                        conn.execute(
                            "UPDATE events SET metadata_json = ? WHERE ulid = ?",
                            (_json.dumps(km), keep["ulid"]),
                        )
                except Exception:
                    pass

        if not dry_run and plan:
            self.recall_cache.bump_generation()
        return {
            "groups": plan,
            "n_archived": sum(len(p["archive_ulids"]) for p in plan),
            "n_recanonicalized": sum(1 for p in plan if p["recanonicalize"]),
            "dry_run": dry_run,
        }

    def backfill_keyed_from_facts(self, dry_run: bool = True) -> dict:
        """Retroactively promote current-state statements buried in plain facts
        into keyed facts (issue #9, for PRE-EXISTING corpora).

        Auto-promotion only fires on NEW writes, so a corpus written before it
        existed can have the truth ("the user currently lives in Tampa") sitting
        in a plain fact while a STALE keyed fact (user::city = Warsaw) wins
        recall. This scans active plain facts with the same conservative
        detector, takes the NEWEST current-state value per canonical attribute,
        and upserts it (superseding the stale keyed value) when it differs.

        Non-destructive: the old keyed value is archived (history), never
        deleted. ``dry_run`` (default) only reports the planned promotions.
        """
        import json as _json
        import sqlite3 as _sql

        best: dict[str, tuple] = {}  # canon_key -> (ts, attribute, value, ulid)
        try:
            with _sql.connect(str(self.workspace.db_path)) as conn:
                conn.row_factory = _sql.Row
                rows = conn.execute(
                    "SELECT ulid, content, metadata_json, timestamp FROM events "
                    "WHERE workspace_id = ? AND archived_at IS NULL "
                    "AND event_type = 'fact' ORDER BY timestamp ASC",
                    (self.workspace.id,),
                ).fetchall()
        except Exception:
            return {"promotions": [], "n": 0, "dry_run": dry_run, "error": "scan_failed"}

        for r in rows:
            try:
                meta = _json.loads(r["metadata_json"] or "{}")
            except Exception:
                meta = {}
            if isinstance(meta, dict) and (
                meta.get("keyed_fact_key")       # already a keyed fact
                or meta.get("is_subfact")         # supplementary atom, not the primary statement
                or meta.get("kind") == "lesson"   # an instruction, not a user state
                or meta.get("source") == "lesson"
            ):
                continue
            hit = detect_current_state(r["content"] or "")
            if not hit:
                continue
            attribute, value = hit
            ck = keyed_fact_key("user", attribute)
            prev = best.get(ck)
            if prev is None or r["timestamp"] > prev[0]:
                best[ck] = (r["timestamp"], attribute, value, r["ulid"])

        plan = []
        for ck, (_ts, attribute, value, ulid) in best.items():
            active = [h for h in self.get_keyed_fact_history("user", attribute)
                      if h["is_current"]]
            cur = active[0]["value"] if active else None
            if cur is not None and str(cur).strip().lower() == str(value).strip().lower():
                continue  # keyed value already correct
            plan.append({
                "attribute": attribute, "canonical_key": ck,
                "new_value": value, "old_value": cur, "from_ulid": ulid,
            })

        if not dry_run:
            for p in plan:
                self.record_keyed_fact(
                    "user", p["attribute"], p["new_value"], importance=0.85,
                    metadata={"source": "current_state_backfill",
                              "derived_from_ulid": p["from_ulid"]},
                )

        return {"promotions": plan, "n": len(plan), "dry_run": dry_run}

    def suggest_keyed_from_llm(
        self,
        dry_run: bool = True,
        limit: int = 40,
        backend: str | None = None,
        model: str | None = None,
    ) -> dict:
        """Offline LLM tier (#11): for recent plain facts the regex fast-path
        MISSED, ask a bounded LLM to extract a current-state
        {subject, attribute, value, negation, confidence}. A positive suggestion
        about the USER with confidence >= 0.8 is upserted via record_keyed_fact
        (which also runs the negation-tombstone cleanup); anything weaker is
        tagged metadata.suggested_key for review.

        Three independent gates keep a third-party fact ("Alice relocated to
        Berlin") from ever becoming user state: (1) a cheap first-person
        prefilter before the LLM is even called, (2) the LLM's own
        subject=="user" verdict, (3) flagged-junk facts are skipped.

        Open-ended understanding belongs OFFLINE - invoked by consolidation /
        `pmb consolidate`, NEVER on the recall hot path. Each call is
        timeout-clamped (≤15s) and behind the recall circuit breaker; the WHOLE
        pass is bounded by an LLMBudget (config llm.offline_max_calls /
        llm.offline_budget_s). `backend`/`model` override the config backend
        (so the consolidate CLI's --backend/--model reach this pass). Gated by
        config consolidate.suggest_keyed.
        """
        import json as _json
        import re as _re
        import sqlite3 as _sql

        out = {"suggestions": [], "applied": 0, "would_apply": 0, "tagged": 0,
               "dry_run": dry_run}
        if not self.config.get("consolidate.suggest_keyed"):
            out["skipped"] = "disabled"
            return out
        from pmb.core import circuit_breaker as _breaker
        if _breaker.is_open("llm"):
            out["skipped"] = "breaker_open"
            return out
        thr = self.config.get("recall.breaker_threshold") or 2
        cd = self.config.get("recall.breaker_cooldown_s") or 60.0

        cands: list[tuple[str, str]] = []
        try:
            with _sql.connect(str(self.workspace.db_path)) as conn:
                conn.row_factory = _sql.Row
                rows = conn.execute(
                    "SELECT ulid, content, metadata_json FROM events "
                    "WHERE workspace_id=? AND archived_at IS NULL "
                    "AND event_type='fact' ORDER BY timestamp DESC LIMIT ?",
                    (self.workspace.id, limit * 4),
                ).fetchall()
        except Exception:
            out["error"] = "scan_failed"
            return out
        for r in rows:
            try:
                meta = _json.loads(r["metadata_json"] or "{}")
            except Exception:
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
            if (meta.get("keyed_fact_key") or meta.get("suggested_key")
                    or meta.get("kind") == "lesson" or meta.get("source") == "lesson"):
                continue
            if meta.get("quality_flag") == "suspect_junk":
                continue  # gate 3: flagged junk never becomes user state
            content = r["content"] or ""
            # gate 1: third-party text ("Alice relocated to Berlin") never even
            # reaches the LLM - only facts that mention the user/first person.
            if not has_user_subject_cue(content):
                continue
            if detect_current_state(content):
                continue  # cheap regex already covers it - no LLM needed
            cands.append((r["ulid"], content))
            if len(cands) >= limit:
                break
        if not cands:
            return out

        try:
            from pmb.health.consolidate import resolve_llm_client
            llm = resolve_llm_client(
                backend=backend or self.config.get("consolidate.backend") or "auto",
                model=model,
            )
        except Exception:
            _breaker.record_failure("llm", thr, cd, "resolve_llm_client failed")
            out["skipped"] = "no_llm"
            return out
        if hasattr(llm, "timeout"):
            try:
                llm.timeout = max(1.0, min(float(llm.timeout), 15.0))
            except Exception:
                pass

        from pmb.core.llm_budget import budget_from_config
        budget = budget_from_config(self.config)

        _ALLOWED = {"city", "country", "employer", "job_title", "email", "phone",
                    "timezone", "relationship_status", "current_project"}
        for ulid, content in cands:
            if not budget.allow():
                out["stopped"] = "budget"
                break
            prompt = (
                "Does the text state a CURRENT, mutable personal attribute, "
                "and WHOSE attribute is it? Allowed attributes: city, country, "
                "employer, job_title, email, phone, timezone, "
                "relationship_status, current_project. (NOTE: hometown / "
                "birthplace is NOT current city.) Answer ONLY JSON "
                '{"subject": "user" | "other_person" | "unknown", '
                '"attribute": "<one or empty>", "value": "<value or empty>", '
                '"negation": true|false, "confidence": 0.0-1.0}. If the '
                'attribute belongs to anyone other than the speaker/user '
                '(e.g. "Alice relocated to Berlin"), subject MUST be '
                '"other_person".\n\n'
                "TEXT: " + content[:300]
            )
            try:
                resp = llm.complete(prompt, max_tokens=120)
            except Exception as e:
                _breaker.record_failure("llm", thr, cd, str(e))
                break
            budget.spend()
            _breaker.record_success("llm")
            try:
                m = _re.search(r"\{.*\}", resp or "", _re.DOTALL)
                v = _json.loads(m.group(0)) if m else {}
            except Exception:
                v = {}
            # gate 2: the LLM's own subject verdict must be the user.
            if str(v.get("subject") or "").strip().lower() != "user":
                continue
            attr = canonicalize_attribute(str(v.get("attribute") or ""))
            if not v.get("attribute") or attr not in _ALLOWED:
                continue
            out["suggestions"].append({
                "ulid": ulid, "attribute": attr,
                "value": v.get("value"), "negation": bool(v.get("negation")),
                "confidence": float(v.get("confidence") or 0.0),
            })

        for s in out["suggestions"]:
            if s["confidence"] >= 0.8 and not s["negation"] and s["value"]:
                if dry_run:
                    out["would_apply"] += 1      # nothing written in a dry run
                else:
                    self.record_keyed_fact(
                        "user", s["attribute"], str(s["value"]),
                        importance=0.8,
                        metadata={"source": "llm_keyed_suggestion",
                                  "derived_from_ulid": s["ulid"]},
                    )
                    out["applied"] += 1
            else:
                if not dry_run:
                    self._tag_suggested_key(s)
                out["tagged"] += 1
        out["budget"] = budget.report()
        return out

    def _tag_suggested_key(self, s: dict) -> None:
        """Stamp metadata.suggested_key on the source fact (below-threshold or
        negation LLM keyed suggestion) for dashboard review. Best-effort."""
        import json as _json
        import sqlite3 as _sql
        try:
            with _sql.connect(str(self.workspace.db_path)) as conn:
                row = conn.execute(
                    "SELECT metadata_json FROM events WHERE ulid=?",
                    (s["ulid"],)).fetchone()
                m = _json.loads(row[0] or "{}") if row else {}
                if not isinstance(m, dict):
                    m = {}
                m["suggested_key"] = {
                    "attribute": s["attribute"], "value": s.get("value"),
                    "negation": s.get("negation"), "confidence": s.get("confidence"),
                }
                conn.execute("UPDATE events SET metadata_json=? WHERE ulid=?",
                             (_json.dumps(m), s["ulid"]))
        except Exception as e:
            try:
                from pmb.core.errlog import log_error
                log_error(self.workspace.db_path, "tag_suggested_key", e)
            except Exception:
                pass

    def migrate_workspace_into(
        self,
        source: str,
        project: str | None = None,
        dry_run: bool = True,
    ) -> dict:
        """Merge a per-project workspace's memory INTO this one (issue #7).

        Copies active events from `source` (workspace id or name) into the
        current workspace, tagged ``project=<name>`` so they can be filtered
        with recall_scoped. The SOURCE is read-only here (raw SQL) and never
        modified - so this is fully reversible: the original workspace stays
        intact. Idempotent: events already migrated (matched by
        ``migrated_ulid``) are skipped. ``dry_run`` (default) only reports.

        Returns {source, source_name, project, n_source_active, n_already,
        n_to_migrate / n_migrated, dry_run}.
        """
        import json as _json
        import sqlite3 as _sql

        from pmb.core.workspace import Workspace, list_workspaces

        src = None
        for ws in list_workspaces(self.workspace.pmb_home):
            if ws.id == source or ws.name == source:
                src = ws
                break
        if src is None:
            cand = self.workspace.pmb_home / "workspaces" / source
            if (cand / "events.sqlite").exists():
                src = Workspace(id=source, name=source, root=cand,
                                pmb_home=self.workspace.pmb_home)
        if src is None:
            return {"error": f"source workspace {source!r} not found"}
        if src.id == self.workspace.id:
            return {"error": "source and target are the same workspace"}

        project_tag = project or src.name or src.id

        with _sql.connect(str(src.db_path)) as conn:
            conn.row_factory = _sql.Row
            rows = conn.execute(
                "SELECT ulid, event_type, content, metadata_json, importance, "
                "timestamp FROM events WHERE archived_at IS NULL "
                "ORDER BY timestamp ASC"
            ).fetchall()

        # Idempotency: which source ulids did a prior run already bring over?
        already: set[str] = set()
        try:
            with _sql.connect(str(self.workspace.db_path)) as conn:
                conn.row_factory = _sql.Row
                for r in conn.execute(
                    "SELECT metadata_json FROM events WHERE workspace_id = ? "
                    "AND metadata_json LIKE ?",
                    (self.workspace.id, f'%"migrated_from": "{src.id}"%'),
                ):
                    try:
                        m = _json.loads(r["metadata_json"] or "{}")
                        if m.get("migrated_ulid"):
                            already.add(m["migrated_ulid"])
                    except Exception:
                        continue
        except Exception:
            pass

        to_migrate = [r for r in rows if r["ulid"] not in already]

        if dry_run:
            return {
                "source": src.id, "source_name": src.name, "project": project_tag,
                "n_source_active": len(rows), "n_already": len(already),
                "n_to_migrate": len(to_migrate), "dry_run": True,
                "sample": [(r["content"] or "")[:80] for r in to_migrate[:5]],
            }

        n = 0
        for r in to_migrate:
            try:
                meta = _json.loads(r["metadata_json"] or "{}")
            except Exception:
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
            meta.setdefault("project", project_tag)
            meta["project_name"] = meta.get("project_name") or project_tag
            meta["migrated_from"] = src.id
            meta["migrated_ulid"] = r["ulid"]
            ev = Event(
                workspace_id=self.workspace.id,
                event_type=r["event_type"] or "fact",
                content=r["content"] or "",
                metadata=meta,
                importance=(r["importance"] if r["importance"] is not None else 0.5),
                timestamp=r["timestamp"],
                tier=default_tier_for_event_type(r["event_type"] or "fact"),
            )
            ev = self.events.append(ev)
            try:
                self._embed_or_defer(ev.ulid, ev.to_text())
            except Exception:
                pass
            n += 1
        self.recall_cache.bump_generation()
        return {
            "source": src.id, "source_name": src.name, "project": project_tag,
            "n_source_active": len(rows), "n_already": len(already),
            "n_migrated": n, "dry_run": False,
        }

    def record_fact_tree(
        self,
        main: str,
        subfacts: list[str] | None = None,
        importance: float = 0.7,
        metadata: dict | None = None,
        session_id: str | None = None,
    ) -> dict:
        """Improvement P: store a main fact + multiple atomic subfacts.

        The main fact gets the requested importance. Subfacts inherit the
        main's importance × 0.85 (slightly lower - they're supplementary)
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

    def get_subfacts(self, parent_ulid: str) -> list[dict]:
        """Return all subfacts linked to a parent event."""
        import json as _j
        import sqlite3

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

    def get_parent_fact(self, subfact_ulid: str) -> dict | None:
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
        metadata: dict | None = None,
        session_id: str | None = None,
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
        self._index_graph_or_defer(ev, full_text=clean_content)
        # Cheap rule-based causation: temporal-next edge from the last event.
        # No LLM, just SQL. Fires only if the previous event is within minutes.
        try:
            from pmb.reasoning.causation import add_temporal_next_edge

            add_temporal_next_edge(self, ev)
        except Exception as e:
            from pmb.core.errlog import log_error
            log_error(self.workspace.db_path, "temporal_next_edge", e)
        # Improvement C: parse event_time (date references) from content
        # and store in metadata. Enables temporal-proximity boost at recall.
        try:
            self._attach_event_time(ev)
        except Exception as e:
            from pmb.core.errlog import log_error
            log_error(self.workspace.db_path, "attach_event_time", e)
        self.recall_cache.bump_generation()
        return ev.ulid

    def record_image(
        self,
        path: str,
        description: str = "",
        importance: float = 0.5,
        metadata: dict | None = None,
        session_id: str | None = None,
        encode_clip: bool = False,
    ) -> str:
        """Improvement J: record an image with optional CLIP embedding.

        The `description` is what's searchable via text - make it specific
        ('screenshot of auth flow diagram with arrows', not 'image').

        encode_clip=True triggers CLIP encoding (requires open_clip or
        sentence-transformers); silently degrades to text-only if missing.

        Returns the ulid of the image event.
        """
        from pmb.reasoning.images import attach_image

        att = attach_image(path, description=description, encode_clip=encode_clip)
        meta = dict(metadata or {})
        meta.update(att.to_metadata())
        # The event's content IS the description - that's what's indexed
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
            import numpy as np

            from pmb.reasoning.images import clip_encode_text

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
