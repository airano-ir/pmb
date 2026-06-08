from __future__ import annotations

import json
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

class LessonsMixin:
    def _log_lesson_surfaces(
        self,
        lessons: list[dict],
        query: str,
        source: str,
        session_id: Optional[str] = None,
    ) -> list[dict]:
        """Log that these lessons were shown to the agent. Mutates the
        lesson dicts in place, adding a `surface_id` so the agent can later
        confirm follow-through via mark_lesson_followed. Returns the list."""
        if not lessons:
            return lessons
        import sqlite3, time as _t
        now = _t.time()
        ws = self.workspace.id
        try:
            with sqlite3.connect(self.workspace.db_path) as conn:
                for L in lessons:
                    ulid = L.get("ulid")
                    if not ulid:
                        continue
                    cur = conn.execute(
                        """
                        INSERT INTO lesson_surfaces
                        (workspace_id, lesson_ulid, query, source,
                         surfaced_at, session_id)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (ws, ulid, (query or "")[:500], source, now, session_id),
                    )
                    L["surface_id"] = cur.lastrowid
                conn.commit()
        except Exception:
            # Surface logging is best-effort — never break recall on it
            import logging
            logging.getLogger(__name__).debug(
                "lesson surface logging failed", exc_info=True
            )
        return lessons

    def mark_lesson_followed(
        self,
        surface_id: int,
        followed: bool = True,
        note: Optional[str] = None,
    ) -> dict:
        """Agent confirms whether a surfaced lesson actually changed its
        behaviour on the current task. Powers the dashboard "follow rate"
        and identifies dead lessons that always surface but never help."""
        import sqlite3, time as _t
        with sqlite3.connect(self.workspace.db_path) as conn:
            cur = conn.execute(
                """
                UPDATE lesson_surfaces
                SET followed = ?, follow_note = ?, followed_at = ?
                WHERE id = ? AND workspace_id = ?
                """,
                (1 if followed else 0, (note or "")[:500], _t.time(),
                 surface_id, self.workspace.id),
            )
            conn.commit()
        return {"ok": cur.rowcount > 0, "surface_id": surface_id,
                "followed": followed}

    def mark_lesson_not_applicable(
        self,
        surface_id: int,
        note: Optional[str] = None,
    ) -> dict:
        """Mark a surfaced lesson as NOT APPLICABLE to the turn it surfaced in.

        Used by the Stop-hook followcheck when a lesson shares zero topical
        overlap with everything the agent actually did this turn — the work
        simply wasn't about that lesson. Stored as `followed = -1` so it is
        excluded from the adherence denominator: a rule that never pertained
        to the work must not count as 'not followed'.

        Distinct from ignored (`followed = 0`), which means the lesson WAS
        relevant but the agent went against it. -1 is excluded from both the
        follow (✓) and ignored (✗) counts everywhere.
        """
        import sqlite3, time as _t
        with sqlite3.connect(self.workspace.db_path) as conn:
            cur = conn.execute(
                """
                UPDATE lesson_surfaces
                SET followed = -1, follow_note = ?, followed_at = ?
                WHERE id = ? AND workspace_id = ?
                """,
                ((note or "")[:500], _t.time(), surface_id, self.workspace.id),
            )
            conn.commit()
        return {"ok": cur.rowcount > 0, "surface_id": surface_id,
                "followed": -1, "not_applicable": True}

    def adherence_stats(self, days: float = 7.0) -> dict:
        """How well is the AI agent FOLLOWING the READ-FIRST workflow?

        Computes adherence metrics over a recent window:
          • prepare_rate — fraction of write-active days where prepare() was
            called at least once. The READ-FIRST rule says prepare() should
            run at the start of every substantive task.
          • lesson_followthrough — fraction of surfaced lessons the agent
            marked as followed via mark_lesson_followed.
          • read_write_ratio — read tool calls / write tool calls. A healthy
            memory tool sees ~2:1 reads-to-writes.

        Used by:
          • The _nudge field in record_batch_async responses — when scores
            are low, agent sees a one-line reminder.
          • The dashboard "Adherence" tab — surfaces leaking sessions.

        Returns floats in [0.0, 1.0] for the rate fields plus the raw
        counts so the caller can render however they want.
        """
        import sqlite3, time as _t
        cutoff = _t.time() - days * 86400.0
        ws = self.workspace.id
        out = {
            "days": days,
            "prepare_calls": 0,
            "write_calls": 0,
            "read_calls": 0,
            "write_active_days": 0,
            "prepare_days": 0,
            "prepare_rate": 0.0,
            "read_write_ratio": 0.0,
            "lesson_surfaces": 0,
            "lesson_followed": 0,
            "lesson_not_applicable": 0,
            "lesson_applicable": 0,
            "lesson_followthrough": 0.0,
        }
        read_tools = (
            "recall", "prepare", "project_overview", "find_lessons",
            "overview", "recent_activity", "what_just_happened",
            "session_brief", "list_goals", "recall_smart",
            "get_subfacts", "list_recent",
        )
        write_tools = (
            "record_batch", "record_fact", "record_fact_tree",
            "record_keyed_fact", "record_goal", "record_activity",
            "record_milestone", "remember", "index_pdf", "index_project",
        )
        with sqlite3.connect(self.workspace.db_path) as conn:
            # MCP-call metrics (prepare_rate / read_write_ratio) come from the
            # mcp_calls table, which only exists once the MCP server has run.
            # On a fresh / non-MCP workspace it's absent — that must NOT zero
            # out the lesson-surface metrics below, which live in their own
            # table. So each block gets its own try/except.
            try:
                rows = conn.execute(
                    "SELECT DATE(timestamp, 'unixepoch') AS d, tool_name, COUNT(*) "
                    "FROM mcp_calls WHERE workspace_id=? AND timestamp >= ? "
                    "GROUP BY d, tool_name",
                    (ws, cutoff),
                ).fetchall()
                write_days, prep_days = set(), set()
                prep_total = write_total = read_total = 0
                for d, tool, n in rows:
                    if tool in write_tools:
                        write_total += n
                        write_days.add(d)
                    if tool in read_tools:
                        read_total += n
                    if tool == "prepare":
                        prep_total += n
                        prep_days.add(d)
                out["write_calls"] = write_total
                out["read_calls"]  = read_total
                out["prepare_calls"] = prep_total
                out["write_active_days"] = len(write_days)
                out["prepare_days"] = len(prep_days & write_days)
                if write_days:
                    out["prepare_rate"] = len(prep_days & write_days) / len(write_days)
                if write_total > 0:
                    out["read_write_ratio"] = read_total / write_total
            except Exception:
                pass

            # Lesson follow-through — independent of mcp_calls.
            # Denominator is APPLICABLE surfaces, not all surfaces: a lesson
            # that surfaced but had zero topical overlap with what the agent
            # actually did this turn is marked not_applicable (followed = -1)
            # by the Stop-hook followcheck, and must NOT count as "not
            # followed". Otherwise the metric measures how broadly auto-recall
            # surfaces (noise), not how well relevant rules are followed.
            try:
                surf = conn.execute(
                    "SELECT COUNT(*) FROM lesson_surfaces WHERE workspace_id=? AND surfaced_at >= ?",
                    (ws, cutoff),
                ).fetchone()[0]
                flw = conn.execute(
                    "SELECT COUNT(*) FROM lesson_surfaces WHERE workspace_id=? AND surfaced_at >= ? AND followed=1",
                    (ws, cutoff),
                ).fetchone()[0]
                na = conn.execute(
                    "SELECT COUNT(*) FROM lesson_surfaces WHERE workspace_id=? AND surfaced_at >= ? AND followed=-1",
                    (ws, cutoff),
                ).fetchone()[0]
                out["lesson_surfaces"] = surf
                out["lesson_followed"] = flw
                out["lesson_not_applicable"] = na
                applicable = max(0, surf - na)
                out["lesson_applicable"] = applicable
                if applicable > 0:
                    out["lesson_followthrough"] = flw / applicable
            except Exception:
                pass
        return out

    def _adherence_nudge(self) -> Optional[str]:
        """One-line consequence-framed reminder when adherence is poor.

        Used by record_batch_async to inject a `_nudge` field in its
        response when the agent has been writing without reading.
        Returns None if adherence is fine — no need to nag.
        """
        try:
            s = self.adherence_stats(days=7.0)
        except Exception:
            return None
        prep_rate = s.get("prepare_rate", 0.0)
        rw       = s.get("read_write_ratio", 0.0)
        lt       = s.get("lesson_followthrough", 0.0)
        n_surf   = s.get("lesson_surfaces", 0)
        n_app    = s.get("lesson_applicable", 0)
        n_na     = s.get("lesson_not_applicable", 0)
        # Quiet on cold workspace: no surfaced lessons / no history.
        if s.get("write_calls", 0) < 5 and n_surf < 5:
            return None
        problems = []
        if prep_rate < 0.30:
            problems.append(
                f"prepare() rate {prep_rate*100:.0f}% this week (target ≥ 60%) — "
                f"you are writing without reading."
            )
        if rw < 0.50:
            problems.append(
                f"read/write ratio {rw:.2f} (target ≥ 0.80) — "
                f"the memory tool is being used as a logbook, not a memory."
            )
        if n_app >= 5 and lt < 0.10:
            problems.append(
                f"lesson follow-through {lt*100:.0f}% of {n_app} applicable"
                + (f" ({n_na} of {n_surf} surfaced weren't relevant)" if n_na else "")
                + " — ensure the lesson-followcheck Stop hook is active, or "
                "call mark_lesson_followed(surface_id, True/False) after acting."
            )
        if not problems:
            return None
        return "⚠ adherence: " + " · ".join(problems)

    def lesson_follow_stats(self, days: float = 7.0) -> dict:
        """Aggregate follow-rate stats over a recent window. Used by
        dashboard and `pmb lessons stats`."""
        import sqlite3, time as _t
        cutoff = _t.time() - days * 86400.0
        ws = self.workspace.id
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            tot = conn.execute(
                "SELECT COUNT(*) AS n FROM lesson_surfaces "
                "WHERE workspace_id = ? AND surfaced_at >= ?",
                (ws, cutoff),
            ).fetchone()
            followed = conn.execute(
                "SELECT COUNT(*) AS n FROM lesson_surfaces "
                "WHERE workspace_id = ? AND surfaced_at >= ? AND followed = 1",
                (ws, cutoff),
            ).fetchone()
            ignored = conn.execute(
                "SELECT COUNT(*) AS n FROM lesson_surfaces "
                "WHERE workspace_id = ? AND surfaced_at >= ? AND followed = 0",
                (ws, cutoff),
            ).fetchone()
            not_applicable = conn.execute(
                "SELECT COUNT(*) AS n FROM lesson_surfaces "
                "WHERE workspace_id = ? AND surfaced_at >= ? AND followed = -1",
                (ws, cutoff),
            ).fetchone()
            per_lesson = conn.execute(
                """
                SELECT ls.lesson_ulid,
                       COUNT(*) AS surfaces,
                       SUM(CASE WHEN ls.followed = 1 THEN 1 ELSE 0 END) AS followed,
                       SUM(CASE WHEN ls.followed = 0 THEN 1 ELSE 0 END) AS ignored,
                       SUM(CASE WHEN ls.followed = -1 THEN 1 ELSE 0 END) AS not_applicable,
                       e.content AS content
                FROM lesson_surfaces ls
                LEFT JOIN events e ON e.ulid = ls.lesson_ulid
                WHERE ls.workspace_id = ? AND ls.surfaced_at >= ?
                GROUP BY ls.lesson_ulid
                ORDER BY surfaces DESC
                LIMIT 30
                """,
                (ws, cutoff),
            ).fetchall()
        total = tot["n"] if tot else 0
        n_followed = followed["n"] if followed else 0
        n_ignored = ignored["n"] if ignored else 0
        n_na = not_applicable["n"] if not_applicable else 0
        # Follow-rate is over APPLICABLE surfaces (total minus not_applicable):
        # a lesson that never pertained to the work must not count against it.
        applicable = max(0, total - n_na)
        return {
            "days": days,
            "total_surfaces": total,
            "followed": n_followed,
            "ignored": n_ignored,
            "not_applicable": n_na,
            "applicable": applicable,
            "unknown": max(0, total - n_followed - n_ignored - n_na),
            "follow_rate": (n_followed / applicable) if applicable else 0.0,
            "per_lesson": [
                {"lesson_ulid": r["lesson_ulid"],
                 "surfaces": r["surfaces"],
                 "followed": r["followed"] or 0,
                 "ignored": r["ignored"] or 0,
                 "not_applicable": r["not_applicable"] or 0,
                 "content": (r["content"] or "")[:200]}
                for r in per_lesson
            ],
        }

    def find_lessons(self, query: str = "", limit: int = 5) -> list[dict]:
        """Return procedural lessons relevant to a query (or all recent
        lessons if query is empty). A "lesson" is an event with
        `metadata.kind == 'lesson'` or `event_type == 'lesson'` — it captures
        a project-specific rule ("we use pnpm, never npm") that should
        change agent behaviour.

        Used by:
          - MCP recall(): auto-attaches relevant lessons so the agent
            cannot miss them
          - topic_overview / project_overview: explicit lessons section
          - pmb overview CLI

        Implementation: scan recent active events on the lesson kind
        (cheap SQL filter), then for non-empty query rank by simple
        case-folded token-overlap with the query. We deliberately don't run
        the full recall pipeline — lessons are few (rarely >100) so a
        linear pass + token scoring is faster and avoids dragging in the
        whole embedding stack on tools that just want lessons.
        """
        import sqlite3
        ws = self.workspace.id
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT ulid, event_type, content, timestamp, importance,
                       metadata_json
                FROM events
                WHERE workspace_id = ?
                  AND archived_at IS NULL
                  AND (event_type = 'lesson'
                       OR (metadata_json LIKE '%"kind":"lesson"%'
                           OR metadata_json LIKE '%"kind": "lesson"%'))
                ORDER BY timestamp DESC
                LIMIT 500
                """,
                (ws,),
            ).fetchall()
        items: list[dict] = []
        for r in rows:
            try:
                md = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
            except Exception:
                md = {}
            items.append({
                "ulid": r["ulid"],
                "content": (r["content"] or "")[:300],
                "timestamp": r["timestamp"],
                "importance": r["importance"],
                "metadata": md,
            })
        if not query.strip():
            return items[:limit]
        # Relevance gate, using the SAME tokenizer + stopwords as followcheck
        # (pmb.core.text_match). The big precision win is the STOPWORD SET, not
        # a high count threshold: the old code split on \W+ with NO stopwords,
        # so a single generic word (code / test / file / pmb / a Windows-path
        # fragment like 'users'/'appdata') matched almost anything and flooded
        # surfacing with noise. distinctive_tokens strips all of that, so a
        # single *distinctive* overlap (pnpm, numpy, lancedb, record_batch) is
        # already a real signal. Keep a lesson sharing >= recall.lesson_min_overlap
        # distinctive tokens (default 1); rank strong (identifier-grade) matches
        # first. Raise the knob to 2 for stricter precision.
        from pmb.core.text_match import distinctive_tokens, is_strong
        q_tokens = distinctive_tokens(query)
        if not q_tokens:
            return items[:limit]
        try:
            min_ov = int(self.config.get("recall.lesson_min_overlap") or 1)
        except Exception:
            min_ov = 1

        # Lexical signal for every candidate (the always-on, model-free tier).
        for it in items:
            ov = q_tokens & distinctive_tokens(it.get("content") or "")
            it["_ov"] = len(ov)
            it["_strong"] = sum(1 for t in ov if is_strong(t))
            it["_sim"] = 0.0

        # Optional SEMANTIC tier (opt-in: recall.lesson_semantic). Reuses the
        # embeddings recall already computes to catch paraphrase / synonym /
        # cross-lingual matches the lexical gate structurally cannot — e.g. a
        # "каким пакетным менеджером собирать" query vs a "use pnpm not npm"
        # lesson shares ZERO tokens but is the same topic. NOT an LLM call (just
        # a vector cosine), so it doesn't break the no-LLM-on-read rule. OFF by
        # default so the per-turn hook stays model-free + instant; when on,
        # scoring a few hundred lesson vectors is cheap once the model is warm.
        sem_min = 1.1  # > 1.0 sentinel == disabled (cosine can't reach it)
        try:
            if self.config.get("recall.lesson_semantic"):
                sem_min = float(self.config.get("recall.lesson_semantic_min") or 0.45)
        except Exception:
            pass
        if sem_min <= 1.0 and items:
            try:
                import numpy as np
                from pmb.core.search import cosine_similarity
                qv = self.search.embed(query)
                arrow = self.search._table.to_arrow()
                want = {it["ulid"] for it in items}
                vec = {u: v for u, v in zip(
                    arrow.column("ulid").to_pylist(),
                    arrow.column("vector").to_pylist()) if u in want}
                order = [it["ulid"] for it in items if it["ulid"] in vec]
                if order:
                    mat = np.array([vec[u] for u in order], dtype=np.float32)
                    sims = cosine_similarity(qv, mat)
                    by_ulid = {u: float(s) for u, s in zip(order, sims)}
                    for it in items:
                        it["_sim"] = by_ulid.get(it["ulid"], 0.0)
            except Exception:
                pass  # best-effort — the lexical tier still stands on its own

        kept = [it for it in items
                if it["_ov"] >= min_ov or it["_sim"] >= sem_min]
        # Rank: strong lexical first, then semantic similarity, then raw overlap.
        kept.sort(key=lambda it: (it["_strong"], round(it["_sim"], 4), it["_ov"]),
                  reverse=True)
        for it in kept:  # strip scratch fields before returning
            for k in ("_ov", "_strong", "_sim"):
                it.pop(k, None)
        return kept[:limit]

    def find_decisions(self, query: str = "", limit: int = 5) -> list[dict]:
        """Return past DECISIONS (the "why we did X" rationale) relevant to a
        query. A "decision" is an event with `metadata.kind == 'decision'`
        (recorded via record_batch activity kind='decision' or record_activity
        kind='decision'). It captures a choice + reasoning ("chose Postgres
        over Mongo for JSONB") so the agent doesn't re-litigate settled calls.

        Mirrors find_lessons: cheap SQL scan filtered to the decision kind,
        then token-overlap ranking for non-empty queries. No embedding stack —
        decisions are few and a linear pass is faster + dependency-free.

        Used by the auto-recall hook to answer "before doing X, did we already
        decide something about X?" without the agent having to think to ask.
        """
        import sqlite3
        ws = self.workspace.id
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT ulid, event_type, content, timestamp, importance,
                       metadata_json
                FROM events
                WHERE workspace_id = ?
                  AND archived_at IS NULL
                  AND (metadata_json LIKE '%"kind":"decision"%'
                       OR metadata_json LIKE '%"kind": "decision"%'
                       OR metadata_json LIKE '%"activity_kind":"decision"%'
                       OR metadata_json LIKE '%"activity_kind": "decision"%')
                ORDER BY timestamp DESC
                LIMIT 500
                """,
                (ws,),
            ).fetchall()
        items: list[dict] = []
        seen_content: set[str] = set()
        for r in rows:
            try:
                md = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
            except Exception:
                md = {}
            content = (r["content"] or "")[:300]
            # Dedup near-identical decisions (the same call recorded across
            # sessions) by a normalized content key — surfacing the same
            # rationale 3× is noise.
            key = " ".join(content.lower().split())[:120]
            if key in seen_content:
                continue
            seen_content.add(key)
            items.append({
                "ulid": r["ulid"],
                "content": content,
                "timestamp": r["timestamp"],
                "importance": r["importance"],
                "metadata": md,
            })
        if not query.strip():
            return items[:limit]
        import re as _re
        q_tokens = set(t for t in _re.split(r"\W+", query.lower()) if len(t) >= 3)
        if not q_tokens:
            return items[:limit]
        def _score(item: dict) -> float:
            content = (item["content"] or "").lower()
            c_tokens = set(t for t in _re.split(r"\W+", content) if len(t) >= 3)
            return len(q_tokens & c_tokens)
        items.sort(key=_score, reverse=True)
        return [it for it in items if _score(it) >= 1][:limit]

    def recent_unconfirmed_surfaces(
        self, minutes: float = 30.0, limit: int = 50,
    ) -> list[dict]:
        """Lesson surfaces in the last `minutes` that have NO follow verdict
        yet (followed IS NULL). Used by the Stop-hook follow-through checker
        to decide which surfaced lessons to auto-assess for follow-through.

        Joins lesson_surfaces → events so the caller gets the lesson content
        (for token matching) without a second query. Returns newest first.
        """
        import sqlite3, time as _t
        cutoff = _t.time() - minutes * 60.0
        ws = self.workspace.id
        out: list[dict] = []
        try:
            with sqlite3.connect(self.workspace.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT s.id AS surface_id, s.lesson_ulid, s.surfaced_at,
                           s.source, e.content AS content
                    FROM lesson_surfaces s
                    LEFT JOIN events e ON e.ulid = s.lesson_ulid
                                      AND e.workspace_id = s.workspace_id
                    WHERE s.workspace_id = ?
                      AND s.surfaced_at >= ?
                      AND s.followed IS NULL
                    ORDER BY s.surfaced_at DESC
                    LIMIT ?
                    """,
                    (ws, cutoff, limit),
                ).fetchall()
            for r in rows:
                out.append({
                    "surface_id": r["surface_id"],
                    "lesson_ulid": r["lesson_ulid"],
                    "surfaced_at": r["surfaced_at"],
                    "source": r["source"],
                    "content": r["content"] or "",
                })
        except Exception:
            import logging
            logging.getLogger(__name__).debug(
                "recent_unconfirmed_surfaces failed", exc_info=True
            )
        return out

