from __future__ import annotations

import json


def _display_trim(text: str, cap: int = 600) -> str:
    """R2: lessons are SCORED on full content, but DISPLAYED trimmed - at a
    sentence boundary near `cap` so the actionable rule (often at the END of a
    long lesson) survives, instead of the old hard 300-char cut that both hid
    the rule AND ran token-overlap scoring on the truncated text."""
    t = text or ""
    if len(t) <= cap:
        return t
    cut = t[:cap]
    for sep in (". ", "! ", "? ", "\n"):
        i = cut.rfind(sep)
        if i >= cap * 0.6:
            return cut[: i + 1].rstrip()
    i = cut.rfind(" ")
    return (cut[:i] if i >= cap * 0.6 else cut).rstrip() + "…"


def _trim_item(it: dict) -> dict:
    """Display-trim a lesson/decision item's content in place (after scoring)."""
    it["content"] = _display_trim(it.get("content") or "")
    return it


class LessonsMixin:
    def _log_lesson_surfaces(
        self,
        lessons: list[dict],
        query: str,
        source: str,
        session_id: str | None = None,
    ) -> list[dict]:
        """Log that these lessons were shown to the agent. Mutates the
        lesson dicts in place, adding a `surface_id` so the agent can later
        confirm follow-through via mark_lesson_followed. Returns the list."""
        if not lessons:
            return lessons
        import sqlite3
        import time as _t
        now = _t.time()
        hour_start = (now // 3600.0) * 3600.0
        ws = self.workspace.id
        try:
            with sqlite3.connect(self.workspace.db_path) as conn:
                for L in lessons:
                    ulid = L.get("ulid")
                    if not ulid:
                        continue
                    # R1: dedup one surface per (lesson, session, hour). A single
                    # prepare()/recall() logs the SAME lesson on two paths
                    # (project_overview + find_lessons), and the same rule
                    # re-surfaces minutes later - both used to mint a NEW
                    # surface_id and inflate the adherence denominator. Reuse the
                    # existing id instead so the dashboard counts shows, not rows.
                    row = conn.execute(
                        "SELECT id FROM lesson_surfaces WHERE workspace_id=? "
                        "AND lesson_ulid=? "
                        "AND COALESCE(session_id,'')=COALESCE(?,'') "
                        "AND surfaced_at >= ? ORDER BY id LIMIT 1",
                        (ws, ulid, session_id, hour_start),
                    ).fetchone()
                    if row:
                        L["surface_id"] = row[0]
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
            # Surface logging is best-effort - never break recall on it
            import logging
            logging.getLogger(__name__).debug(
                "lesson surface logging failed", exc_info=True
            )
        return lessons

    def mark_lesson_followed(
        self,
        surface_id: int,
        followed: bool = True,
        note: str | None = None,
    ) -> dict:
        """Agent confirms whether a surfaced lesson actually changed its
        behaviour on the current task. Powers the dashboard "follow rate"
        and identifies dead lessons that always surface but never help."""
        import sqlite3
        import time as _t
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
        note: str | None = None,
    ) -> dict:
        """Mark a surfaced lesson as NOT APPLICABLE to the turn it surfaced in.

        Used by the Stop-hook followcheck when a lesson shares zero topical
        overlap with everything the agent actually did this turn - the work
        simply wasn't about that lesson. Stored as `followed = -1` so it is
        excluded from the adherence denominator: a rule that never pertained
        to the work must not count as 'not followed'.

        Distinct from ignored (`followed = 0`), which means the lesson WAS
        relevant but the agent went against it. -1 is excluded from both the
        follow (✓) and ignored (✗) counts everywhere.
        """
        import sqlite3
        import time as _t
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

    def capture_correction(
        self,
        message: str,
        *,
        severity: str = "weak",
        markers: list[str] | None = None,
        importance: float = 0.85,
        dedup_minutes: float = 90.0,
        project: str | None = None,
        session_id: str | None = None,
    ) -> dict:
        """Record a DRAFT lesson the moment the user pushes back, so the rule
        exists in memory at correction #1 instead of #7 (the RR failure).

        Hybrid mode: PMB saves the signal immediately (nothing lost), tagged
        source='correction-capture', draft=True - the agent is then nudged to
        refine it into a durable rule. Returns
        {lesson_ulid, surface_id, reused, severity}.

        Dedup: if a correction draft with strong token overlap was recorded in
        the last `dedup_minutes`, REUSE it (one angry topic = one draft, even
        across three angry messages) - we just mint a fresh surface on the
        existing draft so the agent still gets an id to confirm.
        """
        import sqlite3
        import time as _t

        from pmb.core.text_match import distinctive_tokens, is_strong

        msg = (message or "").strip()
        if not msg:
            return {"lesson_ulid": None, "surface_id": None,
                    "reused": False, "severity": severity, "skipped": "empty"}

        ws = self.workspace.id
        msg_tokens = distinctive_tokens(msg)

        # Resolve WHICH project this complaint is about, so the draft surfaces
        # only for that project (find_lessons project-scope) instead of leaking
        # across every other project in a shared store - and so the agent
        # immediately sees what the captured complaint relates to. Order: an
        # explicit hint, then detection from the message text, then a real
        # per-project workspace name (never the catch-all 'personal'/'default').
        proj = (project or "").strip() or None
        if not proj:
            try:
                det = self.detect_project_in_text(msg, min_mentions=1)
                if det and det.get("name"):
                    proj = str(det["name"]).strip() or None
            except Exception:
                proj = None
        if not proj:
            try:
                ws_name = (self.workspace.name or "").strip()
                # A path-like workspace name (no git repo -> the cwd path is used
                # as the name) is not a project label; reduce it to its last
                # component so the tag is 'myrepo', not 'C:\...\myrepo'. Skip the
                # catch-all multi-project stores.
                base = ws_name.replace("\\", "/").rstrip("/").split("/")[-1].strip()
                if base and base.lower() not in {
                        "personal", "default", "global", "main"}:
                    proj = base
            except Exception:
                proj = None

        # 1. Dedup against recent correction drafts.
        reuse_ulid = None
        try:
            cutoff = _t.time() - dedup_minutes * 60.0
            with sqlite3.connect(self.workspace.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT ulid, content FROM events
                    WHERE workspace_id = ? AND archived_at IS NULL
                      AND timestamp >= ?
                      AND json_extract(metadata_json, '$.source') = 'correction-capture'
                    ORDER BY timestamp DESC LIMIT 20
                    """,
                    (ws, cutoff),
                ).fetchall()
            for r in rows:
                prev = distinctive_tokens(r["content"] or "")
                overlap = msg_tokens & prev
                if len(overlap) >= 3 and sum(1 for t in overlap if is_strong(t)) >= 2:
                    reuse_ulid = r["ulid"]
                    break
        except Exception:
            reuse_ulid = None

        # 2. Record (or reuse).
        reused = reuse_ulid is not None
        if reused:
            lesson_ulid = reuse_ulid
        else:
            proj_tag = f"[project: {proj}] " if proj else ""
            content = (
                "[DRAFT lesson - auto-captured from a user correction] "
                f"{proj_tag}"
                f"User pushed back: \"{msg[:280]}\". "
                "Turn this into a durable rule: state plainly what to ALWAYS / "
                "NEVER do so this stops repeating, then it will surface next time."
            )
            try:
                lesson_ulid = self.record_fact(
                    content,
                    importance=importance,
                    metadata={
                        "kind": "lesson",
                        "source": "correction-capture",
                        "draft": True,
                        "severity": severity,
                        "markers": list(markers or []),
                        # Raw complaint, stored separately so the action-time
                        # guard matches on what the USER objected to - not on the
                        # draft's boilerplate ("durable rule", "repeating", ...)
                        # which would cause spurious fires.
                        "correction_text": msg[:280],
                        # The project this draft is scoped to (structured, so
                        # find_lessons(project=...) keys off it directly instead
                        # of guessing from the raw complaint text).
                        **({"project_name": proj} if proj else {}),
                    },
                    session_id=session_id,
                )
            except Exception as e:
                return {"lesson_ulid": None, "surface_id": None,
                        "reused": False, "severity": severity,
                        "skipped": f"record failed: {e}"}

        # 3. Surface it so the agent gets an id to confirm / refine.
        surface_id = None
        try:
            surfaced = self._log_lesson_surfaces(
                [{"ulid": lesson_ulid, "content": msg[:200]}],
                query=msg, source="correction-capture", session_id=session_id,
            )
            if surfaced:
                surface_id = surfaced[0].get("surface_id")
        except Exception:
            pass

        return {"lesson_ulid": lesson_ulid, "surface_id": surface_id,
                "reused": reused, "severity": severity, "project": proj}

    def adherence_stats(self, days: float = 7.0) -> dict:
        """How well is the AI agent FOLLOWING the READ-FIRST workflow?

        Computes adherence metrics over a recent window:
          • prepare_rate - fraction of write-active days where prepare() was
            called at least once. The READ-FIRST rule says prepare() should
            run at the start of every substantive task.
          • lesson_followthrough - fraction of surfaced lessons the agent
            marked as followed via mark_lesson_followed.
          • read_write_ratio - read tool calls / write tool calls. A healthy
            memory tool sees ~2:1 reads-to-writes.

        Used by:
          • The _nudge field in record_batch_async responses - when scores
            are low, agent sees a one-line reminder.
          • The dashboard "Adherence" tab - surfaces leaking sessions.

        Returns floats in [0.0, 1.0] for the rate fields plus the raw
        counts so the caller can render however they want.
        """
        import sqlite3
        import time as _t
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
            "lesson_ignored": 0,
            "lesson_not_applicable": 0,
            "lesson_applicable": 0,
            "lesson_decided": 0,
            "lesson_unknown": 0,
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
            # On a fresh / non-MCP workspace it's absent - that must NOT zero
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

            # Lesson follow-through - independent of mcp_calls.
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
                ign = conn.execute(
                    "SELECT COUNT(*) FROM lesson_surfaces WHERE workspace_id=? AND surfaced_at >= ? AND followed=0",
                    (ws, cutoff),
                ).fetchone()[0]
                out["lesson_surfaces"] = surf
                out["lesson_followed"] = flw
                out["lesson_ignored"] = ign
                out["lesson_not_applicable"] = na
                # R10: a surfaced lesson with NO verdict yet (followed IS NULL)
                # is UNKNOWN - it must NOT sit in the follow-through denominator
                # forever, or a frequently-surfaced-but-unmarked rule drags the
                # rate toward 0 even though nobody ever ignored it. The rate is
                # over DECIDED surfaces (followed + ignored); unknown is reported
                # separately so the dashboard can show "12 followed / 3 ignored
                # (40 awaiting a verdict)".
                decided = flw + ign
                unknown = max(0, surf - na - decided)
                out["lesson_decided"] = decided
                out["lesson_unknown"] = unknown
                out["lesson_applicable"] = max(0, surf - na)  # kept for back-compat
                if decided > 0:
                    out["lesson_followthrough"] = flw / decided
            except Exception:
                pass
        return out

    def _adherence_nudge(self) -> str | None:
        """One-line consequence-framed reminder when adherence is poor.

        S9: cached 60 s. The uncached form runs 4 aggregate queries via
        adherence_stats and sits on the SYNCHRONOUS record_batch response path,
        so back-to-back writes used to pay it every time. Adherence moves slowly
        over a 7-day window, so a 60 s cache is invisible to the signal."""
        import time as _t9
        cached = getattr(self, "_adh_nudge_cache", None)
        if cached is not None and (_t9.time() - cached[0]) < 60.0:
            return cached[1]
        nudge = self._adherence_nudge_uncached()
        self._adh_nudge_cache = (_t9.time(), nudge)
        return nudge

    def _adherence_nudge_uncached(self) -> str | None:
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
                f"prepare() rate {prep_rate*100:.0f}% this week (target ≥ 60%) - "
                f"you are writing without reading."
            )
        if rw < 0.50:
            problems.append(
                f"read/write ratio {rw:.2f} (target ≥ 0.80) - "
                f"the memory tool is being used as a logbook, not a memory."
            )
        if n_app >= 5 and lt < 0.10:
            problems.append(
                f"lesson follow-through {lt*100:.0f}% of {n_app} applicable"
                + (f" ({n_na} of {n_surf} surfaced weren't relevant)" if n_na else "")
                + " - ensure the lesson-followcheck Stop hook is active, or "
                "call mark_lesson_followed after acting."
            )
        if not problems:
            return None
        return "⚠ adherence: " + " · ".join(problems)

    def lesson_follow_stats(self, days: float = 7.0) -> dict:
        """Aggregate follow-rate stats over a recent window. Used by
        dashboard and `pmb lessons stats`."""
        import sqlite3
        import time as _t
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

    def find_negative_memories(self, query: str = "", limit: int = 6) -> list[dict]:
        """The "do NOT repeat this" corpus: failures + auto-captured correction
        drafts. Deliberately SEPARATE from find_lessons (which excludes
        correction drafts): this feeds the ACTION-TIME guard, which fires when
        the agent is about to do something it was corrected/burned on before.

        Matchable text is the raw complaint (`metadata.correction_text`) when
        present, else the content - so matching keys on what the USER objected
        to, not on draft boilerplate. Lean token-overlap ranking; the caller
        (pretool guard) re-applies a strong-match bar, so we don't need the full
        IDF machinery here. Returns [{ulid, content, kind, source, match_text}].
        """
        import sqlite3

        from pmb.core.text_match import distinctive_tokens
        ws = self.workspace.id
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT ulid, content, metadata_json, timestamp
                FROM events
                WHERE workspace_id = ? AND archived_at IS NULL
                  AND (
                    json_extract(metadata_json, '$.source') = 'correction-capture'
                    OR COALESCE(json_extract(metadata_json, '$.kind'),
                                json_extract(metadata_json, '$.activity_kind'))
                       = 'failure'
                  )
                ORDER BY timestamp DESC LIMIT 300
                """,
                (ws,),
            ).fetchall()
        items: list[dict] = []
        for r in rows:
            try:
                md = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
            except Exception:
                md = {}
            match_text = md.get("correction_text") or (r["content"] or "")
            items.append({
                "ulid": r["ulid"],
                "content": r["content"] or "",
                "match_text": match_text,
                "kind": "failure" if (md.get("kind") == "failure"
                                      or md.get("activity_kind") == "failure")
                        else "correction",
                "source": md.get("source") or "",
                "timestamp": r["timestamp"],
            })
        q = distinctive_tokens(query or "")
        if not q:
            return items[:limit]
        for it in items:
            ov = q & distinctive_tokens(it["match_text"])
            it["_ov"] = len(ov)
        ranked = sorted(items, key=lambda it: -it.get("_ov", 0))
        return [it for it in ranked if it.get("_ov", 0) > 0][:limit]

    def find_lessons(
        self,
        query: str = "",
        limit: int = 5,
        project: str | None = None,
    ) -> list[dict]:
        """Return procedural lessons relevant to a query (or all recent
        lessons if query is empty). A "lesson" is an event with
        `metadata.kind == 'lesson'` or `event_type == 'lesson'` - it captures
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
        the full recall pipeline - lessons are few (rarely >100) so a
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
                  AND COALESCE(json_extract(metadata_json, '$.kind'),
                               json_extract(metadata_json, '$.activity_kind'))
                      = 'lesson'
                """,
                (ws,),
            ).fetchall()
            # S7: served by idx_meta_kind (workspace_id, COALESCE(kind,
            # activity_kind)) instead of a full-table LIKE scan. We deliberately
            # OMIT `ORDER BY timestamp DESC LIMIT 500` from SQL: that clause lets
            # the planner pick the timestamp index for a free sort and skip the
            # selective kind index (stat1 has no per-value histogram to know
            # lessons are rare). Lessons are few, so we sort + cap in Python and
            # keep the indexed lookup. Behavior is identical (most-recent 500).
            # The COALESCE expression MUST stay identical to the index definition.
            rows = sorted(rows, key=lambda r: r["timestamp"], reverse=True)[:500]
        items: list[dict] = []
        for r in rows:
            try:
                md = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
            except Exception:
                md = {}
            # Correction-capture DRAFTS are scaffolding (the echoed complaint),
            # not rules yet. They're surfaced via the dedicated CORRECTION banner
            # with their own surface_id; keeping them out of general lesson
            # surfacing stops the raw echo from crowding out the real rule it's
            # about. Dedup still finds them (direct SQL, not this path).
            if md.get("source") == "correction-capture":
                continue
            items.append({
                "ulid": r["ulid"],
                "content": (r["content"] or ""),   # R2: FULL content for scoring
                "timestamp": r["timestamp"],
                "importance": r["importance"],
                "metadata": md,
            })
        if project:
            project_lc = project.strip().lower()
            scoped: list[dict] = []
            for it in items:
                md = it.get("metadata") or {}
                explicit = [
                    md.get("project_name"),
                    md.get("project"),
                    md.get("project_path"),
                ]
                explicit = [str(x).lower() for x in explicit if x]
                if explicit:
                    if any(project_lc in x for x in explicit):
                        scoped.append(it)
                    continue

                # Older lessons often predate structured project metadata but
                # carry a clear leading scope label: "LoadGuard ...",
                # "ApplyPilot ...", "PMB ...". Drop a lesson only when a
                # *different* known project is named near the start; generic
                # cross-project rules remain eligible.
                mentions = self.projects_in_text(
                    it.get("content") or "", min_mentions=1,
                )
                qualified = [
                    m for m in mentions
                    if (
                        bool(m.get("current_workspace"))
                        or bool(m.get("projectish_spelling"))
                        or int(m.get("n_mentions") or 0) >= 3
                        or (m.get("kind") or "").lower()
                        in {"project", "repo", "repository", "product", "app"}
                    )
                ]
                # The first qualified project mention is the lesson's scope.
                # Thus "PMB rule ... LoadGuard example" stays with PMB, while
                # "ApplyPilot ... PMB server" remains an ApplyPilot lesson.
                primary = min(
                    qualified,
                    key=lambda m: int(m.get("position") or 0),
                    default=None,
                )
                if primary and (primary.get("name") or "").lower() != project_lc:
                    continue
                scoped.append(it)
            items = scoped

        if not query.strip():
            return [_trim_item(it) for it in items[:limit]]
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
            return [_trim_item(it) for it in items[:limit]]
        try:
            min_ov = int(self.config.get("recall.lesson_min_overlap") or 1)
        except Exception:
            min_ov = 1

        # Corpus document-frequency over the candidate lessons (E1 / IDF,
        # computed PER CALL from the candidates - no global state, no
        # maintenance tick). In a workspace where most lessons are about the
        # same project, a token like "memory"/"recall"/"hook" is globally
        # distinctive but LOCALLY generic: it recurs across a large fraction of
        # lessons and carries no discriminating power, yet the old `_ov >= 1`
        # gate surfaced any lesson sharing one such token - the main off-topic
        # surface channel observed live. We measure each token's frequency in
        # THIS corpus and gate on CORPUS-DISTINCTIVE overlap (tokens rare here),
        # plus keep a rarity-weighted overlap for ranking. Follows lesson #942
        # (derive from DATA, not hardcoded lists) and #987 (precision via
        # stopwords) - this is the per-workspace, self-tuning generalisation.
        from collections import Counter as _Counter
        doc_tokens: list[set[str]] = [
            distinctive_tokens(it.get("content") or "") for it in items
        ]
        df: _Counter = _Counter()
        for _ct in doc_tokens:
            df.update(_ct)
        n_corpus = len(items)
        try:
            idf_gate = bool(self.config.get("recall.lesson_idf_gate"))
        except Exception:
            idf_gate = True
        try:
            idf_min_corpus = int(self.config.get("recall.lesson_idf_min_corpus") or 12)
        except Exception:
            idf_min_corpus = 12
        try:
            df_max_frac = float(self.config.get("recall.lesson_corpus_df_max") or 0.25)
        except Exception:
            df_max_frac = 0.25
        corpus_gate_on = idf_gate and n_corpus >= idf_min_corpus
        df_cut = df_max_frac * n_corpus  # token in <= this many docs == distinctive

        # Lexical signal for every candidate (the always-on, model-free tier).
        for it, _ct in zip(items, doc_tokens):
            ov = q_tokens & _ct
            it["_ov"] = len(ov)
            it["_strong"] = sum(1 for t in ov if is_strong(t))
            # corpus-distinctive overlap: shared tokens that are RARE here
            it["_dov"] = sum(1 for t in ov if df.get(t, 0) <= df_cut)
            # rarity-weighted overlap (corpus-rare tokens dominate the rank)
            it["_wov"] = (
                sum(1.0 - (df.get(t, 0) / n_corpus) for t in ov)
                if n_corpus else float(len(ov))
            )
            it["_sim"] = 0.0

        # Optional SEMANTIC tier (opt-in: recall.lesson_semantic). Reuses the
        # embeddings recall already computes to catch paraphrase / synonym /
        # cross-lingual matches the lexical gate structurally cannot - e.g. a
        # "which package manager to build with" query (in another language) vs a
        # "use pnpm not npm" lesson shares ZERO tokens but is the same topic.
        # NOT an LLM call (just
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
                pass  # best-effort - the lexical tier still stands on its own

        if corpus_gate_on:
            # Require a CORPUS-DISTINCTIVE overlap (a token rare in this
            # workspace) - not just any globally-distinctive token. This is the
            # precision fix: a lesson sharing only workspace-generic words is
            # dropped. The semantic tier (if on) can still admit a paraphrase.
            kept = [it for it in items
                    if it["_dov"] >= min_ov or it["_sim"] >= sem_min]
        else:
            kept = [it for it in items
                    if it["_ov"] >= min_ov or it["_sim"] >= sem_min]
        if self.config.get("lessons.rank_v2"):
            # R9: blend the signals the system ALREADY has into the rank, not
            # just lexical overlap - importance (a 0.97 rule should beat a stale
            # trivial one sharing a token) and the follow-history (a rule that
            # surfaces 10× and is NEVER followed fades). Recency breaks ties.
            import sqlite3 as _sql
            import time as _t9
            fol: dict[str, tuple[int, int]] = {}
            try:
                with _sql.connect(self.workspace.db_path) as _c9:
                    for u, surf, flw in _c9.execute(
                        "SELECT lesson_ulid, "
                        "SUM(CASE WHEN followed IS NULL OR followed != -1 "
                        "THEN 1 ELSE 0 END), "
                        "SUM(CASE WHEN followed=1 THEN 1 ELSE 0 END) "
                        "FROM lesson_surfaces WHERE workspace_id=? "
                        "GROUP BY lesson_ulid", (self.workspace.id,)):
                        fol[u] = (int(surf or 0), int(flw or 0))
            except Exception:
                pass
            _now9 = _t9.time()

            def _rank(it: dict):
                # rarity-weighted overlap (_wov) replaces raw count so a lesson
                # matching on a corpus-rare token outranks one matching on a
                # common word; identical to _ov when every token is unique.
                lex = it["_strong"] * 2.0 + it["_wov"] + it["_sim"]
                imp = float(it.get("importance") or 0.5)
                surf, flw = fol.get(it["ulid"], (0, 0))
                damp = 0.6 if (surf >= 10 and flw == 0) else 1.0
                age_d = max(0.0, _now9 - float(it.get("timestamp") or _now9)) / 86400.0
                rec = 1.0 / (1.0 + age_d / 30.0)
                return (lex * (0.5 + 0.5 * imp) * damp, rec)
            kept.sort(key=_rank, reverse=True)
        else:
            # legacy: strong lexical first, then semantic similarity, then overlap
            kept.sort(key=lambda it: (it["_strong"], round(it["_sim"], 4), it["_ov"]),
                      reverse=True)
        for it in kept:  # strip scratch fields before returning
            for k in ("_ov", "_strong", "_sim", "_dov", "_wov"):
                it.pop(k, None)
        return [_trim_item(it) for it in kept[:limit]]

    def find_decisions(self, query: str = "", limit: int = 5) -> list[dict]:
        """Return past DECISIONS (the "why we did X" rationale) relevant to a
        query. A "decision" is an event with `metadata.kind == 'decision'`
        (recorded via record_batch activity kind='decision' or record_activity
        kind='decision'). It captures a choice + reasoning ("chose Postgres
        over Mongo for JSONB") so the agent doesn't re-litigate settled calls.

        Mirrors find_lessons: cheap SQL scan filtered to the decision kind,
        then token-overlap ranking for non-empty queries. No embedding stack -
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
                  AND COALESCE(json_extract(metadata_json, '$.kind'),
                               json_extract(metadata_json, '$.activity_kind'))
                      = 'decision'
                """,
                (ws,),
            ).fetchall()
            # S7: indexed via idx_meta_kind (COALESCE of kind/activity_kind),
            # replacing the 4-variant LIKE scan. COALESCE prefers metadata.kind
            # (record_batch decisions) and falls back to activity_kind
            # (record_activity decisions) - both routes covered, whitespace-safe.
            # ORDER BY/LIMIT done in Python (see find_lessons) so the planner
            # keeps the selective kind index instead of the timestamp index.
            rows = sorted(rows, key=lambda r: r["timestamp"], reverse=True)[:500]
        items: list[dict] = []
        seen_content: set[str] = set()
        for r in rows:
            try:
                md = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
            except Exception:
                md = {}
            content = (r["content"] or "")   # R2: FULL content for scoring
            # Dedup near-identical decisions (the same call recorded across
            # sessions) by a normalized content key - surfacing the same
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
            return [_trim_item(it) for it in items[:limit]]
        # R12: use the SAME distinctive-token scorer as find_lessons - the old
        # bare `\W+` split with no stopwords let generic words (code/test/file/
        # the) count as relevance, so decisions surfaced noisily and
        # inconsistently with lessons.
        from pmb.core.text_match import distinctive_tokens
        q_tokens = distinctive_tokens(query)
        if not q_tokens:
            return [_trim_item(it) for it in items[:limit]]

        # Corpus-distinctive gate (the SAME E1/IDF precision fix as
        # find_lessons): a decision must share a token RARE in this workspace's
        # decision corpus, so it can't surface on one workspace-generic word.
        # One tokenization pass per item (the old code tokenized twice).
        from collections import Counter as _Counter
        doc_tokens = [distinctive_tokens(it.get("content") or "") for it in items]
        df: _Counter = _Counter()
        for _ct in doc_tokens:
            df.update(_ct)
        n_corpus = len(items)
        try:
            idf_gate = bool(self.config.get("recall.lesson_idf_gate"))
        except Exception:
            idf_gate = True
        try:
            idf_min_corpus = int(self.config.get("recall.lesson_idf_min_corpus") or 12)
        except Exception:
            idf_min_corpus = 12
        try:
            df_max_frac = float(self.config.get("recall.lesson_corpus_df_max") or 0.25)
        except Exception:
            df_max_frac = 0.25
        corpus_gate_on = idf_gate and n_corpus >= idf_min_corpus
        df_cut = df_max_frac * n_corpus

        scored: list[tuple[float, dict]] = []
        for it, _ct in zip(items, doc_tokens):
            ov = q_tokens & _ct
            if not ov:
                continue
            # corpus-distinctive overlap (tokens rare in this workspace)
            dov = sum(1 for t in ov if df.get(t, 0) <= df_cut)
            passes = (dov >= 1) if corpus_gate_on else (len(ov) >= 1)
            if not passes:
                continue
            # rarity-weighted overlap for ranking
            wov = (sum(1.0 - (df.get(t, 0) / n_corpus) for t in ov)
                   if n_corpus else float(len(ov)))
            scored.append((wov, it))
        scored.sort(key=lambda p: p[0], reverse=True)
        return [_trim_item(it) for _, it in scored][:limit]

    def recent_unconfirmed_surfaces(
        self, minutes: float = 30.0, limit: int = 50,
    ) -> list[dict]:
        """Lesson surfaces in the last `minutes` that have NO follow verdict
        yet (followed IS NULL). Used by the Stop-hook follow-through checker
        to decide which surfaced lessons to auto-assess for follow-through.

        Joins lesson_surfaces → events so the caller gets the lesson content
        (for token matching) without a second query. Returns newest first.
        """
        import sqlite3
        import time as _t
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
