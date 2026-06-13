"""X10 — public API stability contract.

Agents, hooks and user configs depend on a stable surface: a set of MCP tool
names and a set of Engine methods. This test PINS that surface as a
required-subset (new tools/methods may be added freely; renaming or removing a
pinned one is a BREAKING change and fails here). Bump deliberately + note it in
the changelog when you intend to break it.
"""
from __future__ import annotations

# MCP tools every client may rely on existing. Subset, not exact match.
STABLE_MCP_TOOLS = frozenset({
    "recall", "recall_smart", "recall_deep",
    "record_batch", "record_fact", "record_activity", "record_goal",
    "prepare", "project_overview", "session_brief",
    "find_lessons", "mark_lesson_followed", "list_goals",
    "pin", "forget",
})

# The absolute floor — must be served even in the 'minimal' profile.
CORE_MINIMAL_TOOLS = frozenset({"recall", "record_batch", "prepare"})

# Engine public methods callers depend on.
STABLE_ENGINE_METHODS = frozenset({
    "recall", "record_fact", "record_batch", "record_activity",
    "project_overview", "session_brief", "find_lessons", "find_decisions",
    "archive_cold", "detect_conflicts", "warmup",
})


def test_mcp_tool_surface_is_stable():
    from pmb.mcp import _toolspec as t
    default = set(getattr(t, "_DEFAULT_TOOLS", set()))
    missing = STABLE_MCP_TOOLS - default
    assert not missing, f"BREAKING: stable MCP tools missing from default surface: {missing}"


def test_minimal_profile_keeps_the_core_tools():
    from pmb.mcp import _toolspec as t
    minimal = set(getattr(t, "_MINIMAL_TOOLS", set()))
    missing = CORE_MINIMAL_TOOLS - minimal
    assert not missing, f"BREAKING: core tools dropped from minimal profile: {missing}"


def test_engine_public_methods_are_stable():
    from pmb.core.engine import Engine
    missing = {m for m in STABLE_ENGINE_METHODS if not callable(getattr(Engine, m, None))}
    assert not missing, f"BREAKING: Engine lost public methods: {missing}"
