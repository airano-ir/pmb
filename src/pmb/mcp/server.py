"""
PMB MCP Server.

Tools:
- recall(query, top_k=5)            — поиск
- remember(query, response, ...)    — добавить Q/A
- record_fact(fact, ...)            — добавить factual statement
- pin(ulid)                         — закрепить
- forget(ulid)                      — архивировать
- stats()                           — статистика workspace
- list_recent(limit=20, event_type) — последние события

Engine инициализируется лениво один раз — workspace detection выполняется
на старте сервера, дальше используется один и тот же.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# UTF-8 на Windows. Guarded against stdout substitutions (Textual's
# _PrintCapture, pytest capture, etc.) that don't expose `.encoding`.
try:
    enc = getattr(sys.stdout, "encoding", None)
    if enc and enc.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from fastmcp import FastMCP

from pmb.core.engine import Engine
from pmb.core.workspace import detect_workspace


PMB_SYSTEM_INSTRUCTIONS = """\
PMB is OFF by default. Ignore PMB and answer normally for general questions
(general knowledge, coding help, debugging, "что такое X"). Engage only
on explicit triggers below.

CALL recall(q) ONLY if user asks about themselves/past/project:
  "когда я / что я / кто такой / почему мы выбрали / какой у меня"

CALL recent_activity / what_just_happened / list_goals for:
  "что я недавно спрашивал" → recent_activity(min=10080, kind="research")
  "что мы только что"        → what_just_happened(5)
  "какие у меня цели"        → list_goals(status="in_progress")

CALL record_batch(items=[…]) ONLY if user explicitly:
  1. Says "запомни / remember / это важно"
     → items with importance:0.95, pin:true
  2. Shares a personal fact ("я работаю над X", "вчера сделал Y")
     → items with importance:0.7
  3. You (agent) made a meaningful decision/edit/fix on user's behalf
     → {"type":"activity","kind":"decision"/"edit"/"completed", ...}
  4. User corrects you, or you find a reusable project gotcha/rule
     → {"type":"lesson","content":"This repo uses pnpm, never npm"}
       Lessons are procedural ("how to work here"), high-importance, and
       surface automatically on later recalls. Record on "no, do it this
       way", "we always/never …", "stop doing X".

RECALL lessons before a non-trivial coding task in a known project:
  recall("<task> conventions lessons") once; if a lesson returns, FOLLOW it.

For general questions answered from your training — DO NOT call PMB.
PMB is not a log of every interaction.

WHEN you do call record_batch:
  - Exactly ONE per turn (all items in one call)
  - Use pin:true field, NEVER separate pin()
  - NEVER recall after writing to verify

Item types: fact / fact_tree / goal / activity / milestone / lesson.

STYLE:
- Never say "в памяти / found in memory / согласно записям / я записал".
- Use recall results as your own knowledge, weave naturally.
- Don't narrate tool calls.
- ABSOLUTE dates ("On May 25, 2026"), not "today".
- The save-content rules apply to MEMORY only. Your answer length is
  NOT restricted — answer fully with whatever depth the question deserves.

PMB is local-only.
"""


# Kept for reference — the old verbose instructions. Not used.
PMB_SYSTEM_INSTRUCTIONS_OLD = """\
PMB (Personal Memory Brain) is the user's persistent long-term memory.
Treat it as the AUTHORITATIVE source for anything personal or project-specific.

═══════════════════════════════════════════════════════════════════════════
PART 0 — FIRST PRINCIPLE: SAVE GENEROUSLY, BATCH AGGRESSIVELY
═══════════════════════════════════════════════════════════════════════════

The rules below say BATCH writes and be QUIET. They do NOT say save LESS.

  ✅ user states 6 facts → ONE record_batch with 6 items
  ❌ user states 6 facts → "I'll save the important ones" → 2 items
  ❌ user states 6 facts → 6 separate record_fact calls

Junk is cheap. Gaps hurt the user. When in doubt — SAVE IT.

═══════════════════════════════════════════════════════════════════════════
PART 0.5 — EXPLICIT MEMORY TRIGGERS (pin to permanent)
═══════════════════════════════════════════════════════════════════════════

If the user uses any of these phrases — the fact is HIGH PRIORITY. After
saving, immediately call `pin(ulid)` so it never decays:

  • "запомни" / "запомни это" / "не забудь" / "сохрани"
  • "remember this" / "remember that" / "don't forget" / "save this"
  • "это важно" / "this is important" / "important:" / "для меня важно"
  • "for the record" / "на будущее" / "write this down"

Pattern: record_fact (or fact_tree) with importance=0.95, then pin.

  User: "Запомни — у меня день рождения 14 марта"
  → record_fact("User's birthday is March 14", importance=0.95)
  → pin(ulid)

═══════════════════════════════════════════════════════════════════════════
PART 1 — SPEED & STYLE RULES
═══════════════════════════════════════════════════════════════════════════

Each MCP call costs ~3-5 seconds of YOUR thinking time. User notices.

1. **BATCH ALL WRITES.** Call `record_batch` ONCE with all items per turn.
   Never make 5 separate record_* calls — 30s instead of 5s. See PART 2.

2. **ONE RECALL PER QUESTION.** Call `recall` ONCE with a well-chosen query.
   Don't loop. If top results aren't relevant, ONE rephrased follow-up is
   allowed; never more than 2.

3. **USE FACTS AS YOUR OWN KNOWLEDGE.** After recall, weave facts into the
   answer naturally. NEVER prefix with:
       ❌ "в памяти записано что..."
       ❌ "согласно записям..."
       ❌ "I found in memory that..."
       ❌ "Looking at my records..."
   Just answer as if you simply know:
       ✅ "Встреча с Максом завтра в кафе на Подоле, обсуждаете Rust-стартап."
       ✅ "Postgres 17 на порту 5433."

4. **NO PROCESS NARRATION.** Don't tell the user you're saving things,
   what tools you're calling, or how many records you stored. Just save
   silently and answer. The dashboard shows what was stored.

═══════════════════════════════════════════════════════════════════════════
PART 1 — WHEN TO `recall` (READ memory)
═══════════════════════════════════════════════════════════════════════════

ALWAYS call `recall` BEFORE answering, when the user asks about:
  • Past events:    "когда я...", "what did I do...", "когда последний раз..."
  • Personal data:  "какой у меня...", "what's my...", "где я живу"
  • Decisions:      "почему мы выбрали X?", "why did we choose..."
  • People:         "кто такой X?", "who is...", "что сказал X?"
  • Project state:  "какой port?", "what's the config?", "where's X?"
  • Health/life:    "когда я болел?", "what was my appointment?"

For "what did we just do?" / "что мы только что обсуждали?" — call
`what_just_happened(5)` or `recent_activity(minutes=60)` instead. Those
are instant (no vector search).

Trust results with score > 0.2 — that's the user's recorded reality, more
authoritative than your inferences from code / docker / env / web.

═══════════════════════════════════════════════════════════════════════════
PART 2 — WHEN TO WRITE memory  ← ALWAYS USE `record_batch`
═══════════════════════════════════════════════════════════════════════════

When user mentions multiple things worth saving, extract ALL of them in ONE
pass and call `record_batch` ONCE. Each item has a `type` discriminator:

  fact      → atomic single-statement fact
  fact_tree → main event + N linked subfacts (advice, time, details)
  goal      → a goal/intention with status (pending/in_progress/done) + due_at
  activity  → working-memory log entry (lighter, 3-day decay)
  milestone → checkpoint in a named state-chain (e.g. "layers: 6 → 7 → 11")

WHAT to extract from a typical user turn:

  PERSONAL EVENTS, HEALTH:                  → fact or fact_tree (importance 0.7-0.9)
  GOALS / PLANS / INTENTIONS with deadline: → goal (status="in_progress", due_at=…)
  ACTIONS the user just did:                → activity (kind="completed" / "edit")
  DECISIONS the user made:                  → fact (importance 0.7-0.8)
  RELATIONSHIPS:                             → fact ("User's wife is Anna")
  PROGRESS on a tracked metric:             → milestone (chain_name="…")

Example — user says one long paragraph:

  "Today fixed JWT bug (3h). Want v1.0 by June. Tomorrow meeting Max
   (ex-Grammarly) at Podol café — Rust startup. Peanut allergy worsened,
   doctor said carry EpiPen. Dropped LanceDB for SQLite-only. Finished
   async chapter in Rust book, 4 chapters left."

→ ONE call:

  record_batch(items=[
    {"type":"activity","content":"Fixed JWT 24h validation bug, 3h",
     "kind":"edit"},
    {"type":"goal","title":"Ship PMB v1.0 by end of June 2026",
     "status":"in_progress","due_at":<epoch for 2026-06-30>},
    {"type":"fact_tree",
     "main":"Meeting Max on May 25 2026 at café on Podol",
     "subfacts":["Max — ex-colleague from Grammarly",
                 "Topic: discussing Rust startup idea"],
     "importance":0.7},
    {"type":"fact_tree",
     "main":"User's peanut allergy worsened May 24 2026",
     "subfacts":["Doctor advised: carry EpiPen always",
                 "Check EpiPen expiry every 6 months"],
     "importance":0.9},
    {"type":"fact","content":"PMB dropped LanceDB, SQLite-only "
     "(LanceDB pulled ~200MB deps)","importance":0.8},
    {"type":"milestone","chain_name":"rust_book_progress",
     "title":"Finished async chapter, 4 chapters left",
     "state":{"chapters_left":4,"last_finished":"async"}},
  ])

WRITING RULES:
  • Use ABSOLUTE dates ("On May 24, 2026", never "today") — derived from
    the session date in the instructions header.
  • One atomic fact per item.
  • Use the user's primary language for the body when possible — multilingual
    embedding bridges RU↔EN automatically.
  • importance: 0.9 health/medical, 0.7 events/plans, 0.5 opinions.
  • Call `record_batch` DURING the turn, before you answer — not "at the end".

═══════════════════════════════════════════════════════════════════════════
PART 3 — AI-AGENT MODE (when YOU are doing the work)
═══════════════════════════════════════════════════════════════════════════

When YOU act on the user's behalf (writing code, refactoring, debugging,
deploying), YOUR actions also deserve memory. Don't only save what the user
said — save what YOU did.

  • Design decision you made    → activity(kind="decision", content="why X over Y")
  • Files you edited            → activity(kind="edit")
  • Tools you ran (tests, build, lint) → activity(kind="tool_call")
  • Step you finished           → activity(kind="completed") + maybe milestone
  • Multi-step plan you started → goal(status="in_progress")
  • Bug + fix + lesson learned  → fact_tree(main=bug, subfacts=[cause, fix, lesson])
  • High-level user instruction → fact(importance=0.9) + pin
  • Tracked metric changed (test count, build size, layer count) → milestone

Example. You fixed an auth bug:
  record_batch(items=[
    {"type": "activity", "kind": "edit",
     "content": "Extracted JWT validator into a separate module"},
    {"type": "fact_tree",
     "main": "JWT validation bug fixed on May 24 2026",
     "subfacts": ["Root cause: tokens older than 24h skipped exp check",
                  "Fix: added explicit exp comparison with 60s leeway",
                  "Lesson: always test with already-expired tokens"],
     "importance": 0.8},
    {"type": "activity", "kind": "completed",
     "content": "All 211 auth tests passing after refactor"},
  ])

Future "когда мы фиксили auth?" / "почему ты выбрал X?" instantly answered
without re-reading code.

═══════════════════════════════════════════════════════════════════════════
PART 4 — OTHER TOOLS (use when relevant)
═══════════════════════════════════════════════════════════════════════════

  recall_smart       — for important queries (escalates on low confidence)
  what_just_happened — instant: last N events of current session
  recent_activity    — instant: last X minutes of activity log
  list_goals         — open goals (status="in_progress")
  chain_history      — full evolution of a tracked metric
  pin                — pin a memory (use after "запомни"/"remember" triggers)
  dedupe_sweep       — one-shot dedup of duplicate facts
  workspace_info     — confirm which memory you're using

The single-item record_fact / record_goal / record_milestone tools still
exist for one-off cases, but for any multi-fact message PREFER record_batch.

═══════════════════════════════════════════════════════════════════════════
ARCHITECTURE NOTE — why writes stay fast
═══════════════════════════════════════════════════════════════════════════

Write-path (record_*) does NO LLM call — just embedding + SQLite insert
(~50ms). Deep semantic ops — atomic fact extraction, reflections, narrative
arcs, LLM-verify dedup — are SLEEP-MODE: run via separate commands
(`pmb reflect`, `pmb consolidate`, `pmb dedupe --run-pending`) when user is
idle. You don't trigger them. This keeps every turn fast.
"""


# Improvement Z: tool-profile gating. Set PMB_TOOL_PROFILE in the agent's
# config.toml env block to control which tools the LLM sees. Fewer tool
# definitions = faster LLM thinking + less choice confusion.
#
#   "minimal" — 10 tools, the absolute essentials
#   "default" — 14 tools, day-to-day usage (this is the default)
#   "full"    — all 55 tools incl. admin (consolidate, compact, run_self_test,
#               graph_stats, dedupe_run_pending, …). Use for debugging/dev.
#
# Even when an admin tool is HIDDEN from the agent, you can still call it
# via the CLI: `pmb consolidate`, `pmb dedupe`, `pmb prune-graph`, etc.

_TOOL_PROFILE = (os.environ.get("PMB_TOOL_PROFILE") or "default").lower().strip()

_MINIMAL_TOOLS = {
    "recall", "record_batch", "pin", "what_just_happened",
    "recent_activity", "list_goals", "update_goal", "workspace_info",
    "record_fact",          # one-off (when batch overkill)
    "record_fact_tree",     # one-off (event + subfacts)
    "session_brief",        # re-orient after context compaction (long sessions)
}
_DEFAULT_TOOLS = _MINIMAL_TOOLS | {
    "recall_smart",         # important queries with escalation
    "overview",             # structured "what do I know about <topic>"
    "record_goal",          # one-off goal
    "record_activity",      # one-off activity
    "record_milestone",     # one-off milestone
    "chain_history",        # query a state-chain
    "chain_current",        # latest milestone of a chain
    "get_subfacts",         # pull subfacts of a parent event
    "list_recent",          # last N events of any type
    "forget",               # archive an event
    "stats",                # workspace stats
}


def _should_register(tool_name: str) -> bool:
    """True if `tool_name` should be visible to the LLM under current profile."""
    if _TOOL_PROFILE == "full":
        return True
    if _TOOL_PROFILE == "minimal":
        return tool_name in _MINIMAL_TOOLS
    # default
    return tool_name in _DEFAULT_TOOLS


def _maybe_tool(mcp_instance, tool_name: str):
    """Return either @mcp.tool() decorator or a no-op (drops the tool).

    Usage:
        @_maybe_tool(mcp, "consolidate_recent")
        def consolidate_recent(...): ...
    """
    if _should_register(tool_name):
        return mcp_instance.tool()
    # No-op decorator — function body still defined but never registered
    def _noop(fn):
        return fn
    return _noop


def build_server(
    cwd: Optional[Path] = None,
    workspace_id: Optional[str] = None,
    name: str = "pmb",
    prewarm: bool = True,
) -> FastMCP:
    """Собрать MCP server с привязанным engine.

    prewarm=True: do a dummy embed during startup so sentence-transformers
    loads BEFORE the first tool call. Without this the first `recall` can
    take 30+s (model cold load) and Codex/Claude time out → Transport closed.
    """
    workspace = detect_workspace(cwd=cwd, explicit_id=workspace_id)
    engine = Engine(workspace=workspace)

    if prewarm:
        # Async prewarm with readiness state. The MCP transport (initialize
        # handshake) responds INSTANTLY because nothing blocks server boot.
        # Tools that need the heavy model (recall) check engine.is_warm()
        # at call time. If warming, they return BM25-only results with a
        # `warming_up: true` flag instead of timing out the client.
        #
        # This is the production-friendly version: a Codex/Claude client
        # with strict startup timeouts gets a snappy server, and the user
        # gets degraded-but-fast first results that the agent can show
        # while waiting for the full pipeline to be ready.
        import threading

        # Stage 1 — critical path (model + LanceDB import). Runs in BG.
        def _stage1_async():
            try:
                vec = engine.search.embed(
                    "warmup query for prewarm pipeline"
                )
            except Exception:
                return
            try:
                _ = (
                    engine.search._table.search(vec.tolist())
                    .metric("cosine")
                    .limit(3)
                    .to_list()
                )
            except Exception:
                pass
            try:
                engine.search.search(
                    "warmup hybrid search query", top_k=3,
                    importance_map={}, timestamp_map={},
                )
            except Exception:
                pass
            # Stage 2 — batch embed + reranker JIT
            try:
                engine.search.embed_batch([
                    "warmup batch item one",
                    "warmup batch item two",
                ])
            except Exception:
                pass
            try:
                if engine.config.get("recall.rerank"):
                    rr = engine.search.reranker
                    if rr is not None:
                        rr.predict([
                            ("query", "doc one"),
                            ("query", "doc two"),
                        ])
            except Exception:
                pass
            # Mark engine fully warm so tools stop returning the partial flag
            engine._is_warm = True
            engine._warmed_at = __import__("time").time()

        threading.Thread(
            target=_stage1_async, daemon=True, name="pmb-prewarm",
        ).start()

    # Personalize instructions with current date so the LLM knows what "today"
    # resolves to when storing facts with relative dates.
    import datetime as _dt
    today = _dt.datetime.now().strftime("%B %d, %Y")
    instructions = (
        f"Current session date: {today}.\n"
        f"When user says 'today', 'yesterday', 'last week' — resolve to absolute "
        f"dates relative to {today} when storing facts.\n\n"
        + PMB_SYSTEM_INSTRUCTIONS
    )

    mcp = FastMCP(name, instructions=instructions)

    # Improvement JJ: wrap mcp.tool() so every registered function is
    # automatically timed and logged to `mcp_calls` SQLite table. Adds
    # ~1ms per call; dashboard reads aggregates for the Performance tab.
    from pmb.mcp.perf import make_timing_wrapper as _make_timing
    _timing_decorator = _make_timing(
        db_path=engine.workspace.db_path,
        workspace_id=engine.workspace.id,
    )
    _original_tool = mcp.tool

    def _timed_tool(*t_args, **t_kwargs):
        original = _original_tool(*t_args, **t_kwargs)
        def _inner(fn):
            return original(_timing_decorator(fn))
        return _inner

    mcp.tool = _timed_tool

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
        """
        pack = engine.recall(query=query, top_k=top_k)
        return pack.to_dict()

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

    # Improvement Z: post-registration tool-profile filter.
    # All 55 tools registered above; now drop the ones not in active profile.
    # This reduces what Codex sees in the tool list — fewer descriptions to
    # parse each turn = faster + sharper LLM responses.
    if _TOOL_PROFILE != "full":
        allowed = _MINIMAL_TOOLS if _TOOL_PROFILE == "minimal" else _DEFAULT_TOOLS
        try:
            import asyncio
            # list_tools() is async. The stdio server path builds the server
            # before any event loop exists, so asyncio.run() works there and
            # gating applies. If we're already inside a running loop (in-memory
            # client / embedded host), asyncio.run() would raise and leave an
            # un-awaited coroutine — skip cleanly instead.
            try:
                asyncio.get_running_loop()
                in_loop = True
            except RuntimeError:
                in_loop = False
            if not in_loop:
                current = asyncio.run(mcp.list_tools())
                current_names = {t.name for t in current}
                remover = (
                    mcp.local_provider.remove_tool
                    if hasattr(mcp, "local_provider") and hasattr(mcp.local_provider, "remove_tool")
                    else mcp.remove_tool
                )
                for tname in current_names:
                    if tname not in allowed:
                        try:
                            remover(tname)
                        except Exception:
                            pass
        except Exception:
            pass  # filter best-effort; if FastMCP API changes, all tools stay

    return mcp


def main():
    """Entry point for `pmb-mcp` script."""
    workspace_id = os.environ.get("PMB_WORKSPACE")
    cwd_env = os.environ.get("PMB_CWD")
    cwd = Path(cwd_env) if cwd_env else None

    server = build_server(cwd=cwd, workspace_id=workspace_id)
    server.run()


if __name__ == "__main__":
    main()
