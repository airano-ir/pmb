"""Tool-profile machinery for the MCP server (PMB_TOOL_PROFILE gating).

In its own module so per-area tool modules (pmb.mcp.tools) can import
_maybe_tool without importing pmb.mcp.server (which imports them)."""

import os


# Improvement Z: tool-profile gating. Set PMB_TOOL_PROFILE in the agent's
# config.toml env block to control which tools the LLM sees. Fewer tool
# definitions = faster LLM thinking + less choice confusion.
#
#   "minimal" — 13 tools, the absolute essentials
#   "lean"    — 25 tools: default MINUS the pure read-status browse tools a
#               host HOOK already covers (what_just_happened, recent_activity,
#               list_recent, overview). KEEPS session_brief. Set by
#               `pmb connect claude-code` when it installs the hooks, so the
#               agent isn't offered a slow MCP version of what auto-recall /
#               session-restore already inject for free.
#   "default" — 29 tools, day-to-day usage (this is the default)
#   "full"    — all 64 tools incl. admin (consolidate, compact, run_self_test,
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
    "recall_smart",         # important queries with escalation
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


