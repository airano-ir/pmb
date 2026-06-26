"""Tool-profile machinery for the MCP server (PMB_TOOL_PROFILE gating).

In its own module so per-area tool modules (pmb.mcp.tools) can import
_maybe_tool without importing pmb.mcp.server (which imports them)."""

import os

# Improvement Z: tool-profile gating. Set PMB_TOOL_PROFILE in the agent's
# config.toml env block to control which tools the LLM sees. Fewer tool
# definitions = faster LLM thinking + less choice confusion.
#
#   "minimal" - 10 tools, the core memory loop. THIS IS THE DEFAULT: a tight
#               read-first + single-write surface so a fresh agent isn't drowned
#               in 30+ tool choices on every turn. Everything else stays exactly
#               one env-flag away (PMB_TOOL_PROFILE=default / full).
#   "lean"    - the default set MINUS the pure read-status browse tools a host
#               HOOK already covers (what_just_happened, recent_activity,
#               list_recent, overview).
#   "default" - the fuller day-to-day set: core-10 plus recall escalation,
#               record_fact/_tree singles, ingestion, chains, browse and stats.
#               Opt in with PMB_TOOL_PROFILE=default.
#   "full"    - every tool incl. admin (consolidate, compact, run_self_test,
#               graph_stats, dedupe_run_pending, …). Use for debugging/dev.
#
# Even when a tool is HIDDEN from the agent, you can still call it via the CLI
# (`pmb consolidate`, `pmb dedupe`, …) and re-expose it with a wider profile.

_TOOL_PROFILE = (os.environ.get("PMB_TOOL_PROFILE") or "minimal").lower().strip()

# THE DEFAULT SURFACE (core-10): the smallest set that still runs the whole
# memory loop - read-first (prepare / recall / project_overview), the lesson
# loop (find_lessons -> mark_lesson_followed), the SINGLE write path
# (record_batch), mutable personal attrs (record_keyed_fact), goal tracking,
# and post-compaction re-orientation. Deliberately EXCLUDES the read-status
# browse tools a host hook already covers (what_just_happened / recent_activity)
# and the record_fact / record_fact_tree singles that record_batch subsumes -
# those live in `default`. Keep this list tight: every tool here is a choice
# paid by every agent on every turn.
_MINIMAL_TOOLS = {
    "prepare",              # ⭐ one-call READ-FIRST bundle at task start
    "recall",               # general semantic + lexical search over memory
    "project_overview",     # graph-driven full context for a NAMED project
    "find_lessons",         # procedural rules that apply to X (carry surface_id)
    "mark_lesson_followed", # close the self-improvement loop
    "session_brief",        # re-orient after context compaction (long sessions)
    "record_batch",         # ⭐ the ONE write path (facts / goals / activities)
    "record_keyed_fact",    # ⭐ upsert personal attributes (city, employer, ...)
    "list_goals",           # what's in flight
    "update_goal",          # move a goal's status / progress
}
# `default` = core-10 PLUS the fuller day-to-day surface (opt in with
# PMB_TOOL_PROFILE=default). The tools below were demoted from the core so a
# fresh agent isn't drowned, but stay one flag away. Membership here is the
# public "default surface" pinned by tests/eval/test_api_contract.py - adding
# is free, removing/renaming is a BREAKING change (note it in the changelog).
_DEFAULT_TOOLS = _MINIMAL_TOOLS | {
    "recall_smart",         # important queries with escalation (fast, bounded)
    "recall_deep",          # explicit slow/deep LLM-decomposition path
    "overview",             # structured "what do I know about <topic>"
    "what_just_happened",   # last-N events of the session (a host hook covers this)
    "recent_activity",      # last X minutes of activity (a host hook covers this)
    "list_recent",          # last N events of any type
    "pin",                  # pin an event against decay
    "workspace_info",       # which workspace / memory am I in
    "record_fact",          # one-off fact (when record_batch is overkill)
    "record_fact_tree",     # one-off event + subfacts
    "record_goal",          # one-off goal (record_batch also handles goals)
    "record_activity",      # one-off activity
    "record_milestone",     # one-off milestone
    "index_pdf",            # 📄 ingest PDF into memory
    "index_project",        # 📂 ingest code-project structure
    "project_structure",    # 🗺️ read a project's file/module map from memory
    "recall_exploration",   # 💡 reuse a past session's research conclusion (hash-gated)
    "record_exploration",   # 💡 memoize a research conclusion keyed to file hashes
    "lesson_impact",        # 📊 which lessons actually help outcomes (surface->outcome)
    "chain_history",        # query a state-chain
    "chain_current",        # latest milestone of a chain
    "get_subfacts",         # pull subfacts of a parent event
    "forget",               # archive an event
    "stats",                # workspace stats
}


# "lean" = the default set MINUS the read-status tools a host HOOK already
# delivers for free, AND that nothing else (rules / deliberate use) needs.
# We KEEP `prepare`, `recall`, `record_batch`, `project_overview`,
# `find_lessons`, goals, `mark_lesson_followed` - deliberate, agent-composed
# calls no hook can make - and `session_brief`, which the agent may invoke
# mid-session to re-orient (and which the CLAUDE.md rules reference). Only the
# pure-duplicate browse tools are trimmed. See `pmb connect claude-code`.
_LEAN_TOOLS = _DEFAULT_TOOLS - {
    "what_just_happened",   # auto-recall RECENT_QUERY + ambient cover it
    "recent_activity",      # same
    "list_recent",          # low-value browse; ambient/recent cover the need
    "overview",             # project_overview is the deliberate one to keep
}


# D1: compact tool descriptions. The full multi-paragraph docstrings (the
# read-before-write workflow, the write-triggers table, examples) are ALSO in
# the server `instructions` block, so repeating them per tool burns ~16KB of
# context every session. For non-"full" profiles we serve these short
# descriptions instead (one sentence: purpose + when-to-use + key args). The
# `full` profile keeps the long docstrings for debugging/dev. Tools not listed
# here keep their docstring unchanged.
_SHORT_DESC: dict[str, str] = {
    "record_batch": (
        "⚡ PREFERRED for any message with multiple memories - stores N atomic "
        "items in ONE call (each ~3-5s of agent thinking saved vs separate "
        "record_* calls). items: list of dicts, each with a `type`: "
        "fact{content,importance} | fact_tree{main,subfacts[],importance} | "
        "lesson{content,project?} | "
        "goal{title,status,due_at} | plan{title} (future intent) | "
        "activity{content,kind} | milestone{chain_name,title,state}. ONE "
        "record_batch per turn; use ABSOLUTE dates."
    ),
    "project_structure": (
        "🗺️ Read a project's file/module map from memory (no filesystem scan): "
        "languages, files grouped by top-level dir with their purpose + symbol "
        "count, key modules, and recent change intents. Run `index_project` "
        "first if empty; pair with `track modules` for per-file purpose. "
        "Args: name, max_files."
    ),
    "recall_exploration": (
        "💡 BEFORE re-exploring the codebase, reuse a PAST session's research. "
        "Returns memoized conclusions matching `intent`, each with a freshness "
        "check: fresh=true means all source files unchanged (trust it, skip "
        "re-reading); else stale_files lists what changed (re-check only those). "
        "Saves re-deriving from scratch. Args: intent, project, top_k."
    ),
    "record_exploration": (
        "💡 AFTER reading/grepping several files to reach a conclusion, memoize "
        "it so a future session reuses it instead of re-deriving. Stores intent "
        "+ conclusion + each source file's content hash (recall_exploration "
        "replays it with a freshness check). Record only grounded conclusions. "
        "Args: intent, conclusion, files, project."
    ),
    "lesson_impact": (
        "📊 Earned Memory: which lessons actually HELP outcomes, not just which "
        "were read/followed. Joins each surfaced lesson to the turn's outcome "
        "(tests pass/fail, red->green, build, deploy - no LLM) and returns "
        "per-lesson success_rate, lift vs the no-lesson baseline, and churn. "
        "Spot dead-weight or harmful rules (negative lift). Arg: window_days."
    ),
    "prepare": (
        "READ-FIRST bundle at the start of work on a known project. "
        "prepare(message=<the user's message>) returns project_context, "
        "surfaced lessons (each with surface_id - FOLLOW them, then "
        "mark_lesson_followed), recent_activity and open_goals in one ~10ms "
        "call. Replaces several recall() calls."
    ),
    "record_fact": (
        "Store ONE atomic fact. record_fact(fact, importance=0.7, metadata?). "
        "For several facts use record_batch instead. Future intent "
        "('we'll do X next') → record_goal, not a fact."
    ),
    "record_fact_tree": (
        "Store a main fact + linked subfacts in one call. "
        "record_fact_tree(main, subfacts=[...], importance=0.9). Each subfact "
        "becomes a sibling event searchable on its own."
    ),
    "recall": (
        "Search memory for anything about the user/past/project. "
        "recall(query, top_k=5). Returns results + auto-attached lessons "
        "(read & FOLLOW them) + project_context. Trust results with score>0.2 "
        "as the user's recorded reality."
    ),
    "recall_smart": (
        "recall for IMPORTANT queries - auto-escalates on low confidence within "
        "a bounded wall-clock budget. recall_smart(query, top_k=5). Returns "
        "escalation info so you don't fan out more recalls."
    ),
    "recall_deep": (
        "Explicit slow/deep recall with LLM query-decomposition. "
        "recall_deep(query). Opt-in - use only when recall/recall_smart aren't "
        "enough; may take seconds."
    ),
    "record_activity": (
        "Log ONE working-memory activity (lighter than a fact, 3-day decay). "
        "record_activity(content, kind='edit'|'completed'|'decision'|"
        "'tool_call', actor='agent'). For work you finished or a decision made."
    ),
    "record_goal": (
        "Record ONE goal/intention. record_goal(title, status='pending'|"
        "'in_progress'|'done', due_at=<epoch>, parent_goal_ulid?). Future "
        "intent ('next we'll do X') belongs here, not in a fact."
    ),
    "record_milestone": (
        "Record a checkpoint in a named state-chain. "
        "record_milestone(chain_name, title, state={...}, triggered_by_ulid?). "
        "For a metric's evolution (e.g. test count 6→7→11)."
    ),
    "record_keyed_fact": (
        "Upsert a mutable personal attribute. record_keyed_fact(subject, "
        "attribute, value) - e.g. user/city/Tampa. A new value SUPERSEDES the "
        "old under one canonical key instead of piling up."
    ),
    "project_overview": (
        "One-call full context for a NAMED project at the start of work. "
        "project_overview(name) → lessons (rules to follow), decisions, open "
        "goals, recent activity, related entities."
    ),
    "find_lessons": (
        "Standalone 'what procedural rules apply to X'. "
        "find_lessons(query, project?) → lessons with surface_id; project scope "
        "excludes explicit lessons from other projects while retaining generic "
        "rules. FOLLOW them, then mark_lesson_followed."
    ),
    "mark_lesson_followed": (
        "Report whether a surfaced lesson changed your behaviour. "
        "mark_lesson_followed(surface_id, followed=True|False, note='...', "
        "applicable=True|False). Use applicable=False when the lesson was "
        "irrelevant, not followed=False. Call after acting on a lesson - "
        "powers the self-improvement loop."
    ),
    "overview": (
        "Structured 'what do I know about <topic>'. overview(topic) → facts, "
        "entities and relations grouped for that topic."
    ),
    "session_brief": (
        "Re-orient after YOUR context compacted. session_brief() → what THIS "
        "session decided/built. Don't re-ask the user."
    ),
    "recent_activity": (
        "Instant (no vector search) list of the last X minutes of activity. "
        "recent_activity(minutes=60). For 'what did we just do'."
    ),
    "what_just_happened": (
        "Instant last-N events of the current session. "
        "what_just_happened(n=5). For 'what were we just doing'."
    ),
    "list_recent": (
        "List the last N events of any type. list_recent(limit=20, "
        "event_type?). A plain browse - recall/recent_activity are usually "
        "better."
    ),
    "list_goals": (
        "List open goals. list_goals(status='in_progress'). For 'what are my "
        "goals/what's in flight'."
    ),
    "update_goal": (
        "Move a goal's status/progress. update_goal(goal_ulid, "
        "status='in_progress'|'done', progress=0-100, note='...'). Records a "
        "goal_update event so the goal's history is preserved."
    ),
    "index_pdf": (
        "Ingest a PDF into memory. index_pdf(path) - extracts, chunks and "
        "embeds the document so its content is recallable."
    ),
    "index_project": (
        "Ingest a code-project's structure into memory. index_project(path) - "
        "indexes files/symbols for project-aware recall."
    ),
}


def short_description(tool_name: str) -> "str | None":
    """The compact description to serve for `tool_name` under non-full
    profiles, or None to keep the function's docstring."""
    if _TOOL_PROFILE == "full":
        return None
    return _SHORT_DESC.get(tool_name)


def _should_register(tool_name: str) -> bool:
    """True if `tool_name` should be visible to the LLM under current profile."""
    if _TOOL_PROFILE == "full":
        return True
    if _TOOL_PROFILE == "minimal":
        return tool_name in _MINIMAL_TOOLS
    if _TOOL_PROFILE == "lean":
        return tool_name in _LEAN_TOOLS
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
    # No-op decorator - function body still defined but never registered
    def _noop(fn):
        return fn
    return _noop
