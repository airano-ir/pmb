"""D1: tool-description token budget.

The full multi-paragraph docstrings live in the server `instructions` block,
so non-"full" profiles serve compact one-line descriptions instead. This test
pins the budget so a future docstring can't silently re-bloat every session's
context, and checks the short descriptions stay useful (non-empty, bounded).
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile

import pytest


def _profile_desc_chars(profile: str) -> tuple[int, int]:
    """(n_tools, total_description_chars) for a profile. Builds the server in a
    subprocess-free way by setting the env BEFORE importing the toolspec."""
    os.environ["PMB_TOOL_PROFILE"] = profile
    os.environ["PMB_HOME"] = tempfile.mkdtemp()
    # Re-import fresh so _TOOL_PROFILE picks up the env for this profile.
    for mod in ("pmb.mcp.server", "pmb.mcp._toolspec"):
        sys.modules.pop(mod, None)
    from pmb.mcp.server import build_server
    m = build_server(prewarm=False)
    tools = asyncio.run(m.list_tools())
    return len(tools), sum(len(t.description or "") for t in tools)


@pytest.fixture(autouse=True)
def _restore_env():
    keep = {k: os.environ.get(k) for k in ("PMB_TOOL_PROFILE", "PMB_HOME")}
    yield
    for k, v in keep.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    for mod in ("pmb.mcp.server", "pmb.mcp._toolspec"):
        sys.modules.pop(mod, None)


def test_default_profile_under_budget():
    n, chars = _profile_desc_chars("default")
    assert n >= 25, f"expected ~30 default tools, got {n}"
    # Target ≤ 12KB; we ship ~4.5KB. 8000 catches a real regression with slack.
    assert chars <= 8000, f"default tool descriptions = {chars} chars (>8000)"


def test_lean_profile_under_budget():
    n, chars = _profile_desc_chars("lean")
    assert chars <= 8000, f"lean tool descriptions = {chars} chars (>8000)"


def test_minimal_profile_smallest():
    n, chars = _profile_desc_chars("minimal")
    assert chars <= 4000, f"minimal tool descriptions = {chars} chars (>4000)"


def test_short_descriptions_present_and_bounded():
    # toolspec was reloaded by the helper; import fresh
    sys.modules.pop("pmb.mcp._toolspec", None)
    os.environ["PMB_TOOL_PROFILE"] = "default"
    from pmb.mcp._toolspec import _SHORT_DESC
    # the heaviest tools must have a compact override
    for name in ("record_batch", "prepare", "recall", "record_fact",
                 "record_fact_tree", "recall_smart", "project_overview"):
        assert name in _SHORT_DESC, f"{name} missing a short description"
    # each short description is non-empty and bounded
    for name, desc in _SHORT_DESC.items():
        assert 20 <= len(desc) <= 600, f"{name} short desc len={len(desc)}"
    # full profile keeps the docstring (returns None → use __doc__)
    os.environ["PMB_TOOL_PROFILE"] = "full"
    sys.modules.pop("pmb.mcp._toolspec", None)
    from pmb.mcp._toolspec import short_description as sd_full
    assert sd_full("record_batch") is None
