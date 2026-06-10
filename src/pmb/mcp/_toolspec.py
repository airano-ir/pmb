"""Tool-profile machinery for the MCP server (PMB_TOOL_PROFILE gating).

In its own module so per-area tool modules (pmb.mcp.tools) can import
_maybe_tool without importing pmb.mcp.server (which imports them)."""

import os

# Improvement Z: tool-profile gating. Set PMB_TOOL_PROFILE in the agent's
# config.toml env block to control which tools the LLM sees. Fewer tool
# definitions = faster LLM thinking + less choice confusion.
#
#   "minimal" — 13 tools, the absolute essentials
#   "lean"    — 26 tools: default MINUS the pure read-status browse tools a
#               host HOOK already covers (what_just_happened, recent_activity,
#               list_recent, overview). KEEPS session_brief. Set by
#               `pmb connect claude-code` when it installs the hooks, so the
#               agent isn't offered a slow MCP version of what auto-recall /
#               session-restore already inject for free.
#   "default" — 30 tools, day-to-day usage (this is the default)
#   "full"    — all 65 tools incl. admin (consolidate, compact, run_self_test,
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
    "record_keyed_fact",    # ⭐ upsert personal attributes (city, employer, ...)
    "session_brief",        # re-orient after context compaction (long sessions)
    "prepare",              # ⭐ one-call READ-FIRST bundle at task start
}
_DEFAULT_TOOLS = _MINIMAL_TOOLS | {
    "recall_smart",         # important queries with escalation (fast, bounded)
    "recall_deep",          # explicit slow/deep LLM-decomposition path (#13)
    "overview",             # structured "what do I know about <topic>"
    "project_overview",     # graph-driven full context for a known project
    "find_lessons",         # standalone "what procedural rules apply to X"
    "mark_lesson_followed", # agent self-reports lesson follow-through
    "index_pdf",            # 📄 ingest PDF into memory
    "index_project",        # 📂 ingest code-project structure
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


# "lean" = the default set MINUS the read-status tools a host HOOK already
# delivers for free, AND that nothing else (rules / deliberate use) needs.
# We KEEP `prepare`, `recall`, `record_batch`, `project_overview`,
# `find_lessons`, goals, `mark_lesson_followed` — deliberate, agent-composed
# calls no hook can make — and `session_brief`, which the agent may invoke
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
        "⚡ PREFERRED for any message with multiple memories — stores N atomic "
        "items in ONE call (each ~3-5s of agent thinking saved vs separate "
        "record_* calls). items: list of dicts, each with a `type`: "
        "fact{content,importance} | fact_tree{main,subfacts[],importance} | "
        "goal{title,status,due_at} | plan{title} (future intent) | "
        "activity{content,kind} | milestone{chain_name,title,state}. ONE "
        "record_batch per turn; use ABSOLUTE dates."
    ),
    "prepare": (
        "READ-FIRST bundle at the start of work on a known project. "
        "prepare(message=<the user's message>) returns project_context, "
        "surfaced lessons (each with surface_id — FOLLOW them, then "
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
        "recall for IMPORTANT queries — auto-escalates on low confidence within "
        "a bounded wall-clock budget. recall_smart(query, top_k=5). Returns "
        "escalation info so you don't fan out more recalls."
    ),
    "recall_deep": (
        "Explicit slow/deep recall with LLM query-decomposition. "
        "recall_deep(query). Opt-in — use only when recall/recall_smart aren't "
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
        "attribute, value) — e.g. user/city/Tampa. A new value SUPERSEDES the "
        "old under one canonical key instead of piling up."
    ),
    "project_overview": (
        "One-call full context for a NAMED project at the start of work. "
        "project_overview(name) → lessons (rules to follow), decisions, open "
        "goals, recent activity, related entities."
    ),
    "find_lessons": (
        "Standalone 'what procedural rules apply to X'. find_lessons(query) → "
        "lessons with surface_id. FOLLOW them, then mark_lesson_followed."
    ),
    "mark_lesson_followed": (
        "Report whether a surfaced lesson changed your behaviour. "
        "mark_lesson_followed(surface_id, followed=True|False, note='...'). "
        "Call after acting on a lesson — powers the self-improvement loop."
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
        "event_type?). A plain browse — recall/recent_activity are usually "
        "better."
    ),
    "list_goals": (
        "List open goals. list_goals(status='in_progress'). For 'what are my "
        "goals/what's in flight'."
    ),
    "index_pdf": (
        "Ingest a PDF into memory. index_pdf(path) — extracts, chunks and "
        "embeds the document so its content is recallable."
    ),
    "index_project": (
        "Ingest a code-project's structure into memory. index_project(path) — "
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
    # No-op decorator — function body still defined but never registered
    def _noop(fn):
        return fn
    return _noop


