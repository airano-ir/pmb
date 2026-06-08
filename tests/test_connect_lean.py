"""Lean MCP profile + hooks-by-default in `pmb connect`.

Architecture: hooks = the involuntary floor (auto-recall / ambient / restore),
MCP = the deliberate ceiling. On a hook-enabled host the read-status tools the
hooks already cover are trimmed from the MCP surface ('lean' profile).

Safety: `install_hook` is mocked (no real ~/.claude/settings.json write) and
HOME is redirected to tmp (so rules don't land in the user's real CLAUDE.md).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USERPROFILE", str(home))   # Windows Path.home()
    monkeypatch.setenv("HOME", str(home))          # POSIX Path.home()
    return tmp_path


@pytest.fixture
def mock_hooks(monkeypatch):
    """Replace the real hook installer so tests never edit settings.json."""
    import pmb.cli.hooks as H
    calls: list[str] = []

    def fake(agent):
        calls.append(agent)
        return {"agent": agent, "action": "installed",
                "events": ["UserPromptSubmit", "SessionStart",
                           "PostToolUse", "Stop"]}

    monkeypatch.setattr(H, "install_hook", fake)
    return calls


# ─── server profile sets ────────────────────────────────────────────────


def test_lean_keeps_deliberate_drops_hook_covered():
    from pmb.mcp.server import _LEAN_TOOLS, _DEFAULT_TOOLS
    # deliberate, agent-composed tools the hook CANNOT make → kept. prepare +
    # session_brief stay too (rules reference them; deliberate mid-session use).
    for t in ("recall", "record_batch", "project_overview", "find_lessons",
              "list_goals", "mark_lesson_followed", "prepare", "session_brief"):
        assert t in _LEAN_TOOLS, f"{t} must stay in lean"
    # pure-duplicate browse tools a hook/recall already delivers → trimmed
    for t in ("what_just_happened", "recent_activity", "list_recent", "overview"):
        assert t in _DEFAULT_TOOLS, f"{t} should be in default"
        assert t not in _LEAN_TOOLS, f"{t} should be trimmed from lean"


def test_should_register_respects_lean(monkeypatch):
    # The profile machinery (_TOOL_PROFILE / _should_register) lives in
    # pmb.mcp._toolspec; _should_register reads _toolspec._TOOL_PROFILE, so
    # both the patch target and the function call go through _toolspec.
    import pmb.mcp._toolspec as TS
    monkeypatch.setattr(TS, "_TOOL_PROFILE", "lean")
    assert TS._should_register("recall") is True
    assert TS._should_register("session_brief") is True       # kept in lean
    assert TS._should_register("what_just_happened") is False  # trimmed
    # default still shows everything in the default set
    monkeypatch.setattr(TS, "_TOOL_PROFILE", "default")
    assert TS._should_register("what_just_happened") is True
    # full shows admin tools too
    monkeypatch.setattr(TS, "_TOOL_PROFILE", "full")
    assert TS._should_register("consolidate_recent") is True


# ─── connect wiring ─────────────────────────────────────────────────────


def test_connect_claude_sets_lean_and_installs_hooks(isolated_home, mock_hooks):
    from pmb.cli.connect import connect
    proj = isolated_home / "proj"; proj.mkdir()
    res = connect("claude-code", cwd=proj, scope="project", install_hooks=True)
    assert res["tool_profile"] == "lean"
    assert res["entry"]["env"]["PMB_TOOL_PROFILE"] == "lean"
    assert mock_hooks == ["claude-code"]
    assert res["hooks"] and not res["hooks"].get("error")


def test_connect_claude_no_hooks_keeps_full_surface(isolated_home, mock_hooks):
    from pmb.cli.connect import connect
    proj = isolated_home / "proj"; proj.mkdir()
    res = connect("claude-code", cwd=proj, scope="project", install_hooks=False)
    assert res["tool_profile"] is None
    assert "PMB_TOOL_PROFILE" not in res["entry"].get("env", {})
    assert mock_hooks == []          # installer not called
    assert res.get("hooks") is None


def test_connect_cursor_is_mcp_only(isolated_home, mock_hooks):
    from pmb.cli.connect import connect
    proj = isolated_home / "proj"; proj.mkdir()
    res = connect("cursor", cwd=proj, scope="project", install_hooks=True)
    assert res["tool_profile"] is None    # no hooks on cursor → full surface
    assert mock_hooks == []               # installer never called for cursor
    assert res.get("hooks") is None


def test_connect_codex_hooks_but_default_profile(isolated_home, mock_hooks):
    pytest.importorskip("tomli_w")
    from pmb.cli.connect import connect
    proj = isolated_home / "proj"; proj.mkdir()
    res = connect("codex", cwd=proj, scope="global", install_hooks=True)
    # codex installs ambient hooks but has NO per-turn read hook → keep prepare
    assert res["tool_profile"] is None
    assert "PMB_TOOL_PROFILE" not in res["entry"].get("env", {})
    assert mock_hooks == ["codex"]


def test_connect_remote_never_lean(isolated_home, mock_hooks):
    from pmb.cli.connect import connect
    proj = isolated_home / "proj"; proj.mkdir()
    res = connect("claude-code", cwd=proj, scope="project",
                  remote="alex@host:/srv/repo", install_hooks=True)
    # remote server manages its own profile; don't force lean / install hooks
    assert res["tool_profile"] is None
    assert mock_hooks == []


def test_connect_function_default_is_back_compat(isolated_home, mock_hooks):
    """connect() defaults install_hooks=False → unchanged from before."""
    from pmb.cli.connect import connect
    proj = isolated_home / "proj"; proj.mkdir()
    res = connect("claude-code", cwd=proj, scope="project")
    assert res["tool_profile"] is None
    assert "PMB_TOOL_PROFILE" not in res["entry"].get("env", {})
    assert mock_hooks == []
