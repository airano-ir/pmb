"""All MCP tool definitions, registered onto the server.

Extracted verbatim from build_server in pmb.mcp.server (no behavior
change). build_server creates the FastMCP `mcp` + `engine`, patches
mcp.tool with timing, then calls register_all(mcp, engine)."""

from __future__ import annotations

from typing import Optional

from pmb.mcp._toolspec import _maybe_tool  # noqa: F401


def register_all(mcp, engine):
    @mcp.tool()
    def recall(query: str, top_k: int = 5) -> dict:
        """Search PMB long-term memory. CALL THIS FIRST when user asks about:
        - past decisions (why did we choose X, what port did we set)
        - project history (when did we, what happened with)
        - people (who is X, what did Alice say)
        - any fact the user might have recorded earlier

        Trust results with score > 0.2 — they are the user's recorded history,
        more authoritative than inferring from code/docker/env.

        Returns structured pack with results, each containing event_type,
        content, metadata, score and ranking signals (bm25, vector, importance,
        recency).

        ALSO returns a top-level `lessons` field with up to 3 procedural
        lessons relevant to the query. Lessons are project-specific rules like
        "this repo uses pnpm, never npm" — READ THEM FIRST and FOLLOW them
        before acting on the regular results.
        """
        pack = engine.recall(query=query, top_k=top_k)
        out = pack.to_dict()
        # Auto-surface relevant lessons. The agent often ignores lessons
        # even though they're the most actionable memory. By piggy-backing
        # on every recall(), lessons become impossible to miss. Each
        # surfaced lesson gets a `surface_id` — agent can later call
        # mark_lesson_followed(surface_id) to confirm follow-through.
        try:
            lessons = engine.find_lessons(query=query, limit=3)
            if lessons:
                engine._log_lesson_surfaces(lessons, query=query, source="recall")
                out["lessons"] = lessons
        except Exception:
            pass
        # Auto-attach project_overview when the query mentions a known
        # project. ONE call gives the agent the full project context
        # without it having to think "should I call project_overview".
        try:
            det = engine.detect_project_in_text(query)
            if det:
                ov = engine.project_overview(det["name"])
                # Also surface-log any lessons inside the overview so
                # follow-tracking works for them too.
                ov_lessons = ov.get("lessons") or []
                if ov_lessons:
                    engine._log_lesson_surfaces(
                        ov_lessons, query=query, source="recall.project_context",
                    )
                out["project_context"] = ov
        except Exception:
            pass
        return out

    @mcp.tool()
    def overview(topic: str, max_events: int = 20) -> dict:
        """Get a structured OVERVIEW of everything memory knows about a topic.

        Call this when (re)starting work on a project, feature, person, or
        decision area to get the big picture in ONE call - instead of several
        recall() calls. Returns key facts & decisions, lessons, failures, open
        goals, a timeline, and related topics, all from stored memory. Use it
        at the start of a task to get up to speed on prior context.
        """
        return engine.topic_overview(topic, max_events=max_events)

    @mcp.tool()
    def project_overview(name: str) -> dict:
        """⭐ Use at the START of any project work — returns the FULL project
        context in ONE call, sourced from the entity graph (not recall):

          • All key facts about the project (top by importance)
          • LESSONS — procedural rules you MUST follow ("use pnpm not npm")
          • DECISIONS — past architectural / design choices
          • OPEN GOALS still in flight
          • Recent completed work + activity timeline
          • Related entities (tech stack, people, files mentioned with it)

        Faster + more complete than overview() because it walks the
        entity-graph directly. Best for known projects with a strong
        entity (e.g. project_overview("LoadGuard"), project_overview("LeanBoard")).

        Args:
            name: case-insensitive substring match against entity names.
                  Picks the highest-mention entity that matches.

        Returns:
            {entity, span, key_facts, lessons, decisions, open_goals,
             recent_completed, related_entities, n_total, empty}
        """
        ov = engine.project_overview(name)
        # Surface-log so follow-tracking works whenever the agent calls
        # this tool directly.
        try:
            lessons = ov.get("lessons") or []
            if lessons:
                engine._log_lesson_surfaces(
                    lessons, query=name, source="project_overview",
                )
        except Exception:
            pass
        return ov

    @mcp.tool()
    def prepare(message: str) -> dict:
        """⭐⭐⭐ Call this ONCE at the START of any substantive user task.

        Auto-detects what the task is about and pre-loads everything you'd
        need to act on it — in ONE call. Replaces 4-5 separate
        recall/overview/list_goals/recent_activity calls.

        Returns whatever's relevant for the message:
          • project_context — full project_overview if a known project is
            mentioned (LoadGuard / LeanBoard / PMB / …). Includes lessons
            (RULES to follow), decisions, open goals, recent activity,
            related entities. Surface-logged.
          • active_arcs — narrative arcs the project is currently living in
            (e.g. "Postgres adoption", "Auth refactor") with their member
            event ulids. Lets you understand the bigger story before acting.
          • lessons — top procedural rules matching the message, surface-
            logged so the self-improvement loop can track follow-through.
          • recent_activity — last 24h of decisions/edits/completions for
            session continuity.
          • open_goals — in-flight goals so you know what user is pursuing.

        WHEN to call:
          • User starts a task with a project name ("работаю над LoadGuard")
          • User says "fix bug in X / add feature to Y / refactor Z"
          • Any non-trivial coding/design task that touches stored memory
          • You feel uncertain about project conventions

        WHEN to SKIP:
          • Pure-knowledge questions ("what is X", "how does Y work")
          • Trivial one-liners (rename a var, change a const)
          • Continuation of a task already prepared in this session

        Fast path: ~10-20ms total (SQL only, no LLM, no embedding).
        """
        out: dict = {"message_excerpt": (message or "")[:120]}
        # 1. Project detection — gives the full project_overview + active
        #    narrative arcs (the "bigger story" the project is currently
        #    living in).
        try:
            det = engine.detect_project_in_text(message)
            if det:
                ov = engine.project_overview(det["name"])
                lessons_in_ov = ov.get("lessons") or []
                if lessons_in_ov:
                    engine._log_lesson_surfaces(
                        lessons_in_ov, query=message,
                        source="prepare.project_context",
                    )
                out["project_context"] = ov
                # Active narrative arcs that include events from this project.
                try:
                    arcs = engine.active_arcs_for_project(det["name"], limit=2)
                    if arcs:
                        out["active_arcs"] = arcs
                except Exception:
                    pass
        except Exception:
            pass
        # 2. Lessons matching the message — even without a clear project.
        try:
            ls = engine.find_lessons(query=message, limit=5)
            if ls:
                engine._log_lesson_surfaces(
                    ls, query=message, source="prepare.lessons",
                )
                out["lessons"] = ls
        except Exception:
            pass
        # 3. Recent activity for session continuity.
        try:
            act = engine.recent_activity(minutes=1440.0, limit=8)
            if act:
                out["recent_activity"] = act
        except Exception:
            pass
        # 4. Open goals.
        try:
            goals = engine.list_goals(status="in_progress", limit=5)
            if goals:
                out["open_goals"] = goals
        except Exception:
            pass
        # If literally nothing matched, signal it explicitly so the agent
        # doesn't sit there waiting for hidden context.
        if len(out) == 1:
            out["empty"] = True
            out["hint"] = (
                "No project / lesson / activity matched. Proceed with "
                "normal recall(query) only if user asks about past."
            )
        return out

    @mcp.tool()
    def record_keyed_fact(
        subject: str,
        attribute: str,
        value: str,
        importance: float = 0.85,
    ) -> dict:
        """⭐ Use for SINGULAR personal attributes that CHANGE over time.

        When the user states a fact about themselves (or a person/thing)
        where there's exactly ONE current value, and that value can
        change later — use this instead of record_fact.

        Examples that fit:
          • "I live in Warsaw" / "переехал в Варшаву"
            → record_keyed_fact("user", "city", "Warsaw")
          • "I work at Anthropic" / "теперь работаю в Anthropic"
            → record_keyed_fact("user", "employer", "Anthropic")
          • "my dog is now Pixel" (renamed)
            → record_keyed_fact("user_dog", "name", "Pixel")
          • "my phone number is +380..."
            → record_keyed_fact("user", "phone", "+380...")

        What happens automatically:
          1. Any prior fact with the same (subject, attribute) is
             ARCHIVED (not deleted) with `superseded_by` pointer and
             `valid_to` timestamp.
          2. Future recall returns ONLY the current value — old ones
             disappear from results.
          3. Historical lookup still works:
             engine.keyed_fact_as_of('user', 'city', past_timestamp)
             returns whichever value was current at that time
             (Zep-style time-travel).

        WRONG tool when:
          • User describes an EVENT/activity ("I moved to Warsaw last
            week" with date detail) — use record_activity
          • Multi-valued ("I speak Russian, English, Ukrainian") —
            use record_fact (each value is independently true)
          • Facts that won't change ("user's birthday is March 14") —
            record_fact is fine, or record_keyed_fact if you want the
            'I corrected my birthday' upsert behaviour.
          • You're not sure — use record_fact (safe default).

        Args:
            subject: who/what the fact is about ('user', 'user_dog',
                'company_xyz', etc.). Lowercased internally; spaces ok.
            attribute: the attribute name ('city', 'employer', 'phone',
                'name'). Lowercased internally.
            value: the current value as a short string ('Warsaw').
            importance: 0..1, default 0.85 (high because personal attrs
                are usually significant).

        Returns:
            {new_ulid, superseded_ulids: list[str], key: str}
        """
        return engine.record_keyed_fact(
            subject=subject,
            attribute=attribute,
            value=value,
            importance=importance,
        )

    @mcp.tool()
    def index_pdf(path: str, force: bool = False, importance: float = 0.6) -> dict:
        """📄 Extract text from a PDF and persist it as searchable memory.

        Each page is chunked (~1500 chars) and written as a fact. The
        agent can then recall any passage via the usual `recall()`. A
        re-ingest of the same file is a no-op (idempotent via SHA1).

        Use when the user says:
          • "read this PDF and remember it"
          • "запомни этот документ"
          • "index <some file>.pdf"
          • "summarise this paper" (call index_pdf first, then recall)

        Returns: {file, source_hash, n_pages, n_chunks, duration_ms, ...}
        """
        from pathlib import Path

        from pmb.ingest.pdf import ingest_pdf, ingest_pdfs
        p = Path(path)
        if p.is_dir():
            return ingest_pdfs(engine, p, recurse=False,
                               importance=importance, force=force)
        return ingest_pdf(engine, p, importance=importance, force=force)

    @mcp.tool()
    def index_project(
        path: str = ".",
        force: bool = False,
        max_files: int = 5000,
    ) -> dict:
        """📂 Index a code project's structure — per-file symbols, imports,
        languages — so the agent can recall things like "where is the auth
        flow", "which files import LanceDB", "show me the recall pipeline".

        Respects .gitignore. Idempotent per file (SHA1). Safe to re-run.

        Use when the user says:
          • "запомни структуру проекта"
          • "index this repo"
          • "scan the project"
          • "remember how the code is organised"

        Returns: {project_name, n_indexed, n_skipped, by_language, ...}
        """
        from pathlib import Path

        from pmb.ingest.project import index_project as _do_index
        return _do_index(engine, Path(path), force=force, max_files=max_files)

    @mcp.tool()
    def find_lessons(query: str = "", limit: int = 5) -> list[dict]:
        """Pull procedural lessons (project rules / gotchas) relevant to a
        topic. Use this BEFORE making a project-shaping choice — picking a
        library, setting up tooling, choosing an approach — to see what
        worked or failed before.

        Lessons are short rules captured from prior corrections / failures
        ("this repo uses pnpm, never npm", "Postgres pool size must stay
        below 30"). They override default behaviour.

        Each result includes a `surface_id`. After acting on a lesson,
        call mark_lesson_followed(surface_id) — the self-improvement loop
        uses follow-rate to prune dead lessons.

        Args:
            query: topic to filter by (empty = recent lessons across all projects)
            limit: max lessons to return (default 5)
        """
        lessons = engine.find_lessons(query=query, limit=limit)
        if lessons:
            engine._log_lesson_surfaces(
                lessons, query=query, source="find_lessons",
            )
        return lessons

    @mcp.tool()
    def mark_lesson_followed(
        surface_id: int,
        followed: bool = True,
        note: Optional[str] = None,
    ) -> dict:
        """Confirm whether a previously surfaced lesson actually changed
        your behaviour on the current task. Call this AFTER acting on a
        lesson — the self-improvement loop uses this signal to identify
        useful vs dead lessons.

        Args:
            surface_id: the `surface_id` field returned with the lesson
            followed: True if you followed the lesson, False if ignored
            note: optional one-line explanation (esp. useful for ignored)
        """
        return engine.mark_lesson_followed(
            surface_id=surface_id, followed=followed, note=note,
        )

    @mcp.tool()
    def session_brief(minutes: Optional[int] = None) -> dict:
        """Re-orient in a long session: a digest of what was decided / done /
        learned so far THIS session.

        Call this when you've lost the thread - after your own context window
        compacts, or many turns into a long task - instead of re-asking the
        user what you already did. Returns decisions, completed work, lessons,
        failures and goals from the current session (or the last `minutes`).
        PMB is your durable memory across your own context limits.
        """
        return engine.session_brief(minutes=minutes)

    @mcp.tool()
    def remember(query: str, response: str, importance: float = 0.5,
                 session_id: Optional[str] = None) -> dict:
        """Store a Q/A interaction in memory.

        Args:
            query: the user query / question
            response: the agent response / answer
            importance: 0..1 importance score (default 0.5)
            session_id: optional session identifier for grouping
        """
        ulid = engine.remember(
            query=query, response=response,
            importance=importance, session_id=session_id,
        )
        return {"ulid": ulid, "stored": True}

    @mcp.tool()
    def dedupe_sweep(threshold: float = 0.92, types: Optional[list[str]] = None) -> dict:
        """Improvement U: one-shot dedup over all active events.

        Clusters by cosine ≥ threshold within each event_type, archives
        losers (reversible via dedupe_undo). Use after the AI has
        accidentally written duplicate facts/goals across sessions, or
        periodically to keep the store clean.

        Conservative default (0.92). Lower to 0.85 if you want more aggressive
        merging — risks false merges of genuinely separate facts.
        """
        return engine.dedupe_sweep(threshold=threshold, event_types=types)

    @mcp.tool()
    def dedupe_list_pending(limit: int = 100) -> list[dict]:
        """List borderline duplicate pairs awaiting verdict (L2.5).
        These are pairs that scored cosine 0.80-0.92 — too risky to
        auto-merge, but worth review.
        """
        return engine.dedupe_list_pending(limit=limit)

    @mcp.tool()
    def dedupe_run_pending(backend: str = "auto", limit: int = 50) -> dict:
        """Drain the borderline pair queue via LLM verify.
        backend: 'auto' (try Ollama then Anthropic) | 'ollama' | 'anthropic'
        """
        return engine.dedupe_run_pending(backend=backend, limit=limit)

    @mcp.tool()
    def dedupe_undo() -> dict:
        """Restore events archived by previous dedup runs. Reversible."""
        return {"n_restored": engine.dedupe_undo()}

    @mcp.tool()
    def record_batch(items: list[dict]) -> dict:
        """⚡ PREFERRED for ANY message with multiple facts. Stores N atomic
        facts / goals / activities / milestones / fact_trees in a SINGLE call.

        WHY this matters: each separate MCP tool call costs the agent ~3-5
        seconds of LLM thinking (planning the next call). 11 separate
        record_fact calls = ~55 seconds. ONE record_batch call = ~5 seconds.
        ALWAYS prefer this over multiple record_fact/record_goal calls.

        Schema for `items` — list of operation dicts, each with a `type`:

          {"type": "fact",      "content": "...", "importance": 0.7}
          {"type": "fact_tree", "main": "...", "subfacts": ["...", "..."],
                                 "importance": 0.9}
          {"type": "goal",      "title": "...", "status": "in_progress",
                                 "due_at": <epoch_seconds_or_null>,
                                 "parent_goal_ulid": null}
          {"type": "activity",  "content": "...", "kind": "edit",
                                 "actor": "agent"}
          {"type": "milestone", "chain_name": "architecture_layers",
                                 "title": "11 layers (added activity)",
                                 "state": {"count": 11},
                                 "triggered_by_ulid": null}

        Example: user says "Сегодня пофиксил баг, к июню выкатить v1, завтра
        встреча с Максом из Grammarly, аллергия на арахис обострилась":

          record_batch(items=[
            {"type": "activity", "content": "Fixed JWT 24h validation bug, 3h",
             "kind": "edit"},
            {"type": "goal", "title": "Ship PMB v1.0 by end of June 2026",
             "status": "in_progress", "due_at": 1782000000},
            {"type": "fact_tree",
             "main": "Meeting Max on May 25 2026 at café on Podol",
             "subfacts": ["Max — ex-colleague from Grammarly",
                          "Topic: Rust startup idea"],
             "importance": 0.7},
            {"type": "fact_tree",
             "main": "User has peanut allergy (worsened May 24 2026)",
             "subfacts": ["Doctor advised: carry EpiPen always",
                          "Check expiry every 6 months"],
             "importance": 0.9},
          ])

        Returns: {n_accepted, processing} — fire-and-forget by default.
        The actual write happens in a background thread, so MCP returns in
        ~50ms even for very large batches. Items appear in recall within
        ~1-2 seconds.

        For synchronous semantics (waiting for full write to complete + ULIDs
        returned), set `mcp.record_batch_async=false` in PMB config.
        """
        if engine.config.get("mcp.record_batch_async"):
            return engine.record_batch_async(items=items)
        return engine.record_batch(items=items)

    @mcp.tool()
    def record_fact(fact: str, importance: float = 0.7) -> dict:
        """STORE a fact for long-term recall. CALL THIS PROACTIVELY whenever
        the user mentions:

        - Personal events ("I broke my arm", "вчера ел пиццу")
        - Health/medical ("у меня аллергия", "doctor said X")
        - Decisions ("we chose Postgres", "switched to Vite")
        - Preferences ("I prefer dark mode")
        - People ("my wife is Anna", "met Caroline at conf")
        - Schedule ("flying to Paris next week")
        - Anything the user might want to recall later

        Rules:
        - One atomic fact per call (separate calls for separate facts)
        - Use absolute dates derived from current session time, not "today"
        - importance: 0.9 health/medical, 0.7 events, 0.5 opinions
        - Better to over-store than miss — junk is cheap, gaps hurt

        DO NOT wait until end of conversation — call DURING the turn,
        as soon as user states the fact.

        Examples:
        - "Project uses Postgres 17 on port 5432"
        - "User prefers no comments in code"
        - "Decided to use Redis for rate limiting"
        """
        ulid = engine.record_fact(fact=fact, importance=importance)
        return {"ulid": ulid, "stored": True}

    @mcp.tool()
    def pin(ulid: str) -> dict:
        """Pin a memory: max importance, never auto-archived."""
        engine.pin(ulid)
        return {"ulid": ulid, "pinned": True}

    @mcp.tool()
    def forget(ulid: str) -> dict:
        """Archive a memory. Not deleted, can be restored."""
        engine.forget(ulid)
        return {"ulid": ulid, "archived": True}

    @mcp.tool()
    def stats() -> dict:
        """Get workspace and memory statistics."""
        return engine.stats()

    @mcp.tool()
    def list_recent(limit: int = 20, event_type: Optional[str] = None) -> list[dict]:
        """List the most recent active events in the workspace."""
        events = engine.events.list_active(
            engine.workspace.id, limit=limit, event_type=event_type,
        )
        return [
            {
                "ulid": e.ulid,
                "event_type": e.event_type,
                "content": e.content,
                "metadata": e.metadata,
                "timestamp": e.timestamp,
                "importance": e.importance,
                "access_count": e.access_count,
            }
            for e in events
        ]

    @mcp.tool()
    def sync_git(since_timestamp: Optional[float] = None) -> dict:
        """Capture recent git commits into memory.

        Pulls commits since last sync (or since the given timestamp), stores
        each as a 'git' event with full metadata (sha, author, files changed,
        diff stats).
        """
        return engine.sync_git(since_timestamp=since_timestamp)

    @mcp.tool()
    def session_start(name: Optional[str] = None) -> dict:
        """Start a new memory session. Used to group related events."""
        return engine.session_start(name)

    @mcp.tool()
    def session_end() -> Optional[dict]:
        """End the current session. Returns the closed session's info."""
        return engine.session_end()

    @mcp.tool()
    def session_current() -> Optional[dict]:
        """Get current active session info (or None)."""
        return engine.session_current()

    @mcp.tool()
    def apply_daily_decay(days_since: float = 1.0) -> dict:
        """Apply forgetting curve to all events.

        Decays importance, archives events that fall below threshold
        and are older than 90 days. Pinned events (importance 1.0) are skipped.
        """
        return engine.apply_daily_decay(days_since=days_since)

    @mcp.tool()
    def file_correlations(file_path: str, top_k: int = 10) -> list[dict]:
        """Find files that are often modified together with the given file.

        Based on git commit co-occurrence.
        """
        pairs = engine.file_correlations(file_path, top_k)
        return [{"file": f, "co_occur_count": c} for f, c in pairs]

    @mcp.tool()
    def file_history(file_path: str) -> list[dict]:
        """Get commit history for a specific file."""
        return engine.file_history(file_path)

    @mcp.tool()
    def run_self_test(n_samples: int = 20, apply_adaptive: bool = True) -> dict:
        """Quantify memory health: system asks itself questions from old memories.

        Returns accuracy metrics (acc@1, acc@3, acc@5) and list of failures.
        If apply_adaptive=True, failed events get importance boost.
        """
        return engine.run_self_test(n_samples=n_samples, apply_adaptive=apply_adaptive)

    @mcp.tool()
    def health_trend() -> dict:
        """Get trend of self-test accuracy over time.

        Returns 'stable' / 'degrading' / 'improving' / 'insufficient'.
        """
        return engine.health_trend()

    @mcp.tool()
    def detect_conflicts() -> list[dict]:
        """Find conflicting facts between different timestamps.

        Each conflict has 'suggested_resolution': supersede / concurrent / needs_review.
        """
        return engine.detect_conflicts()

    @mcp.tool()
    def auto_resolve_conflicts(dry_run: bool = True) -> dict:
        """Auto-archive obvious 'supersede' conflicts.

        Use dry_run=True to preview what would be done.
        """
        return engine.auto_resolve_conflicts(dry_run=dry_run)

    @mcp.tool()
    def compact_storage(dry_run: bool = False, age_days: int = 30) -> dict:
        """Compact storage: move old archived events to cold storage and VACUUM.

        Reduces main DB size and improves query performance.
        """
        return engine.compact(dry_run=dry_run, age_days=age_days)

    @mcp.tool()
    def cold_stats() -> dict:
        """Get info about cold storage (archived events database)."""
        return engine.cold_stats()

    @mcp.tool()
    def record_recall_feedback(
        ulid: str,
        verdict: str,
        query: Optional[str] = None,
        expected_ulid: Optional[str] = None,
    ) -> dict:
        """Record real recall feedback for an event.

        verdict: 'useful' | 'wrong' | 'irrelevant'.
        Useful events get importance boost, wrong/irrelevant get a small
        decrease, expected_ulid (if given) gets a stronger boost.
        This is the primary signal for adaptive importance — prefer it over
        the synthetic self-test.
        """
        return engine.record_recall_feedback(
            ulid, verdict, query=query, expected_ulid=expected_ulid,
        )

    @mcp.tool()
    def feedback_summary() -> dict:
        """Aggregate user recall feedback. Returns useful_rate and top
        flagged-wrong events. This is the real-user health metric."""
        return engine.feedback_summary()

    @mcp.tool()
    def graph_stats() -> dict:
        """Counts of entities and edges in this workspace's association graph."""
        return engine.graph_stats()

    @mcp.tool()
    def graph_top_entities(kind: Optional[str] = None, limit: int = 20) -> list[dict]:
        """Most-mentioned entities. kind = 'file' | 'tech' | 'concept' or None."""
        return engine.graph_top_entities(kind=kind, limit=limit)

    @mcp.tool()
    def graph_neighbors(name: str, kind: Optional[str] = None, top_k: int = 10) -> dict:
        """Co-occurrence neighbors of an entity. Useful for 'what relates to X'."""
        return engine.graph_neighbors(name=name, kind=kind, top_k=top_k)

    @mcp.tool()
    def consolidate_recent(
        dry_run: bool = False,
        since_days: float = 14.0,
        threshold: float = 0.5,
        min_size: int = 3,
        max_clusters: int = 10,
    ) -> dict:
        """LLM-based sleep-stage consolidation.

        Clusters related recent events and asks Claude Haiku to generalize
        each cluster into a single fact. Requires ANTHROPIC_API_KEY on the
        server side. Source events are archived (not deleted).

        Use dry_run=True to preview without storing.
        """
        return engine.consolidate(
            dry_run=dry_run, since_days=since_days,
            similarity_threshold=threshold,
            min_cluster_size=min_size,
            max_clusters=max_clusters,
        )

    # ------------------------------------------------------------------
    # PMB v2.5 reasoning layer (sleep operations)
    # ------------------------------------------------------------------

    @mcp.tool()
    def recall_smart(query: str, top_k: int = 5, confidence_threshold: float = 0.5) -> dict:
        """Auto-escalating recall (Improvement G). Starts cheap, retries
        with decomposition + bigger top_k + rerank if confidence < threshold.

        Use this for important / hard queries where you want best results
        rather than fastest. For routine lookups, use `recall` instead.
        """
        pack = engine.recall_smart(
            query, top_k=top_k,
            confidence_threshold=confidence_threshold,
        )
        return pack.to_dict()

    @mcp.tool()
    def reflect_batch(limit: int = 10, backend: str = "auto") -> dict:
        """PMB v2 sleep operation: LLM reflects on recent un-reflected events.

        For each event the LLM extracts:
          - significance (why it matters)
          - might_answer (what questions it helps answer)
          - causation edges to related recent events
          - bridge entities (linked back to source for graph search)

        Returns counts of reflections + edges added.
        """
        return engine.reflect_batch(limit=limit, backend=backend)

    @mcp.tool()
    def extract_facts(limit: int = 50, backend: str = "auto") -> dict:
        """Improvement D (mem0-style): LLM extracts atomic facts from recent
        events. Each fact becomes a new searchable event.

        Stronger than reflections for direct lookup queries. Idempotent —
        skips events that already have facts.
        """
        return engine.extract_facts_batch(limit=limit, backend=backend)

    @mcp.tool()
    def cluster_into_arcs(limit: int = 20, backend: str = "auto") -> dict:
        """PMB v2 Phase 3: LLM groups events into narrative arcs (story
        threads). Arcs help recall on 'tell me about X' style queries.

        Each event is either joined to an existing arc, used to create a
        new arc, or ignored.
        """
        return engine.cluster_events_into_arcs(limit=limit, backend=backend)

    @mcp.tool()
    def list_arcs(status: str = "active", limit: int = 50) -> list[dict]:
        """List narrative arcs in the current workspace."""
        return engine.list_arcs(status=status, limit=limit)

    @mcp.tool()
    def arc_detail(arc_id: int) -> Optional[dict]:
        """Get full detail of one arc: title, summary, member events."""
        return engine.arc_detail(arc_id)

    @mcp.tool()
    def precompute_predictive_cache(n_questions: int = 15, backend: str = "auto") -> dict:
        """Improvement F (PMB-unique): during sleep, LLM predicts likely
        questions and pre-runs recall. Future near-identical queries get
        instant (~3ms) responses via cosine match.

        Run periodically (e.g. once per day) for active workspaces.
        """
        return engine.precompute_predictive_cache(
            n_questions=n_questions, backend=backend,
        )

    @mcp.tool()
    def clear_predictive_cache() -> dict:
        """Clear all entries from the predictive cache."""
        n = engine.clear_predictive_cache()
        return {"n_cleared": n}

    @mcp.tool()
    def record_image(
        path: str,
        description: str = "",
        importance: float = 0.5,
        encode_clip: bool = False,
    ) -> dict:
        """Record an image (screenshot/diagram) in memory (Improvement J).

        `description` is the searchable text — make it specific.
        Set `encode_clip=True` for cross-modal text→image search (requires
        open_clip or sentence-transformers; gracefully degrades).

        Returns the ulid of the image event.
        """
        try:
            ulid = engine.record_image(
                path=path, description=description,
                importance=importance, encode_clip=encode_clip,
            )
            return {"ulid": ulid, "path": path}
        except FileNotFoundError as e:
            return {"error": str(e)}

    @mcp.tool()
    def search_images_by_text(query: str, top_k: int = 10) -> list[dict]:
        """Cross-modal image search. Encodes the query via CLIP text encoder
        and finds nearest stored image embeddings.

        Falls back to plain-text recall (filtered to image events) if CLIP
        is not installed.
        """
        return engine.search_images_by_text(query, top_k=top_k)

    @mcp.tool()
    def record_code(content: str, language: str = "python") -> dict:
        """Record code content. For Python, AST entities (functions, classes,
        imports) are auto-extracted into the graph layer (Improvement J).
        """
        ulid = engine.record_event(
            event_type="code", content=content,
            metadata={"language": language},
        )
        return {"ulid": ulid, "language": language}

    @mcp.tool()
    def record_goal(
        title: str,
        status: str = "pending",
        parent_goal_ulid: Optional[str] = None,
        due_at: Optional[float] = None,
        importance: float = 0.7,
    ) -> dict:
        """Improvement R: create a goal/intent (12-th semantic layer).

        Use when user states a goal, plan, or intention:
          "Хочу выучить Rust к концу года"
          "Need to ship v1.0 by Q3"
          "Plan: refactor auth first, then frontend"

        Goals have status (pending/in_progress/done/cancelled), optional
        hierarchy (parent_goal_ulid), and optional deadline.

        Returns {ulid}.
        """
        return {"ulid": engine.record_goal(
            title=title, status=status,
            parent_goal_ulid=parent_goal_ulid,
            due_at=due_at, importance=importance,
        )}

    @mcp.tool()
    def update_goal(
        goal_ulid: str,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        note: Optional[str] = None,
    ) -> dict:
        """Update goal status / progress. Creates a goal_update event so
        the full history of changes is preserved."""
        return engine.update_goal(
            goal_ulid=goal_ulid, status=status,
            progress=progress, note=note,
        )

    @mcp.tool()
    def list_goals(status: Optional[str] = None, limit: int = 50) -> list[dict]:
        """List goals (optionally filter by status)."""
        return engine.list_goals(status=status, limit=limit)

    @mcp.tool()
    def record_milestone(
        chain_name: str,
        title: str,
        state: Optional[dict] = None,
        triggered_by_ulid: Optional[str] = None,
        importance: float = 0.6,
    ) -> dict:
        """Improvement R: record a milestone in a named state-chain.

        Use to track EVOLUTION of any tracked thing — "X became Y because Z".

        Example:
          record_milestone(
              chain_name="architecture_layers",
              title="11 layers (added activity log)",
              state={"count": 11, "previous_count": 10, "added": "activity"},
              triggered_by_ulid=<ulid of the implementation event>,
          )

        Later `chain_history("architecture_layers")` returns the full
        sequence: 6 → 7 → ... → 11, with the reason at each step.

        Auto-links to the previous milestone in the same chain (no need
        to specify previous_ulid manually).

        Returns {ulid}.
        """
        return {"ulid": engine.record_milestone(
            chain_name=chain_name, title=title, state=state,
            triggered_by_ulid=triggered_by_ulid, importance=importance,
        )}

    @mcp.tool()
    def chain_history(chain_name: str, limit: int = 100) -> list[dict]:
        """Full chronological history of a state-chain.
        Reconstructs the evolution: state_1 → state_2 → ... → state_N
        with the reason at each step.
        """
        return engine.chain_history(chain_name=chain_name, limit=limit)

    @mcp.tool()
    def chain_current(chain_name: str) -> Optional[dict]:
        """Latest milestone of a chain — the 'current state'.
        Use to answer 'how many X do we have now?' / 'what's the latest?'."""
        return engine.chain_current(chain_name=chain_name)

    @mcp.tool()
    def record_activity(
        summary: str,
        actor: str = "agent",
        kind: str = "action",
        importance: float = 0.4,
    ) -> dict:
        """Improvement Q: log a working-memory activity.

        Use AFTER any significant action: you made an edit, ran a tool,
        gave a recommendation, decided something, completed a step.
        Lighter than record_fact — session-scoped, working tier (3-day decay).

        Examples:
          record_activity("Implemented typo correction with 5 algorithms", kind="edit")
          record_activity("Recommended ER visit for broken arm", kind="recommendation")
          record_activity("Ran test suite, 82/82 passing", kind="tool_call")
          record_activity("Finished v2.6 architecture review", kind="completed")

        actor:  'agent' (default — AI's own action) | 'user' | 'system'
        kind:   'action' | 'edit' | 'tool_call' | 'recommendation' | 'plan' | 'completed'

        Returns: {ulid}
        """
        ulid = engine.record_activity(
            summary=summary, actor=actor, kind=kind, importance=importance,
        )
        return {"ulid": ulid}

    @mcp.tool()
    def recent_activity(
        minutes: float = 60.0,
        limit: int = 20,
        actor: Optional[str] = None,
        kind: Optional[str] = None,
    ) -> list[dict]:
        """Improvement Q: working memory dump — instant, no search.

        Call THIS instead of recall for questions like:
          "что мы только что сделали?"
          "what did we just do?"
          "show me the last edits"
          "что было за последний час?"

        Returns chronological list (newest first) of activity events.
        Free of BM25/vector overhead — pure SQL by timestamp.

        Filters:
          minutes: lookback window (default 60)
          actor:   'agent' / 'user' / 'system' / None
          kind:    'action' / 'edit' / 'tool_call' / etc.
        """
        return engine.recent_activity(
            minutes=minutes, limit=limit, actor=actor, kind=kind,
        )

    @mcp.tool()
    def what_just_happened(n: int = 5) -> list[dict]:
        """Quick: last N events of any type from current session.
        Use to answer 'что мы делали?' / 'recap of this session'.
        Instant, no search."""
        return engine.what_just_happened(n=n)

    @mcp.tool()
    def session_timeline(limit: int = 100) -> list[dict]:
        """Chronological dump of CURRENT session events.
        Use for post-mortem 'summarize this session' style questions."""
        return engine.session_timeline(limit=limit)

    @mcp.tool()
    def record_fact_tree(
        main: str,
        subfacts: list[str],
        importance: float = 0.7,
    ) -> dict:
        """Improvement P: store a main fact + multiple linked sub-facts in ONE call.

        Use whenever ONE user statement contains MULTIPLE atomic data points
        worth remembering separately. Common cases:

        - User describes an event + you give advice:
            main: "On May 23, 2026, user fell and broke arm"
            subfacts: ["Time: 18:52", "Recommended ER visit",
                       "Avoid driving, ice 15min, remove rings",
                       "911 if: numbness, bleeding, deformation"]

        - User states a goal + you list steps:
            main: "User decided to migrate from MySQL to Postgres"
            subfacts: ["Reason: JSONB support", "Target: end of Q2",
                       "Need to convert auth schema first"]

        - Medical / health context:
            main: "User has high blood pressure per Dec 5 visit"
            subfacts: ["Reading: 145/95", "Doctor: Smith",
                       "Prescribed: lisinopril 10mg daily"]

        importance applies to the main; subfacts get importance × 0.85.
        Subfacts link to main via metadata.parent_ulid AND via event_edges.

        ALWAYS prefer this over multiple separate record_fact calls when
        the data points are RELATED to ONE event — preserves causal structure.

        Returns: {main_ulid, subfact_ulids, n_subfacts}
        """
        return engine.record_fact_tree(
            main=main, subfacts=subfacts, importance=importance,
        )

    @mcp.tool()
    def get_subfacts(parent_ulid: str) -> list[dict]:
        """Return all subfacts linked to a parent event (Improvement P).

        Useful after recall surfaces a main fact: call this to get the
        related atomic details (time, advice, warnings, etc.)."""
        return engine.get_subfacts(parent_ulid)

    @mcp.tool()
    def workspace_info() -> dict:
        """Identify the current PMB workspace.

        Useful when multiple MCP clients share one workspace — confirms which
        events.db file is in use.
        """
        return {
            "id": engine.workspace.id,
            "name": engine.workspace.name,
            "root": str(engine.workspace.root),
            "source": engine.workspace.source,
            "db_path": str(engine.workspace.db_path),
        }

