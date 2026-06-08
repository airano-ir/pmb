"""Codex ambient memory — observe via the rollout log, journal via notify.

Codex has no PostToolUse hook, so the observer reads ~/.codex/sessions/.../
rollout-*.jsonl. These tests pin the parser (function_call → action,
status from output, record_* → coordination), the offset tracking (no dupes
across turns), the PowerShell significance filter, and the notify wiring.
"""

from __future__ import annotations

import json

import pytest

from pmb.hooks.codex_rollout import (
    find_latest_rollout,
    parse_rollout_actions,
)
from pmb.core.ambient_log import is_significant_action


def _write_rollout(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def _call(name, args, cid):
    return {"type": "function_call", "name": name,
            "arguments": json.dumps(args), "call_id": cid}


def _out(cid, text):
    return {"type": "function_call_output", "call_id": cid, "output": text}


# ─── parser ──────────────────────────────────────────────────────────────


def test_parse_maps_tools_and_status(tmp_path):
    roll = tmp_path / "rollout-x.jsonl"
    _write_rollout(roll, [
        _call("apply_patch", {"path": "src/auth.py"}, "c1"),
        _out("c1", "Exit code: 0"),
        _call("shell_command", {"command": "pytest tests/"}, "c2"),
        _out("c2", "Exit code: 0\nWall time: 3s"),
        _call("shell_command", {"command": "git commit -m fix"}, "c3"),
        _out("c3", "Exit code: 0"),
    ])
    scan = parse_rollout_actions(roll, 0)
    assert len(scan.actions) == 3
    tools = [a["tool"] for a in scan.actions]
    assert tools == ["Edit", "Bash", "Bash"]
    assert scan.actions[0]["target"] == "src/auth.py"
    assert all(a["status"] == "ok" for a in scan.actions)
    assert scan.agent_recorded is False


def test_parse_detects_nonzero_exit(tmp_path):
    roll = tmp_path / "rollout-e.jsonl"
    _write_rollout(roll, [
        _call("shell_command", {"command": "pytest"}, "c1"),
        _out("c1", "Exit code: 1\nFAILED"),
    ])
    scan = parse_rollout_actions(roll, 0)
    assert scan.actions[0]["status"] == "1"


def test_parse_detects_agent_record(tmp_path):
    """A record_batch call in the rollout = the agent journaled itself."""
    roll = tmp_path / "rollout-r.jsonl"
    _write_rollout(roll, [
        _call("apply_patch", {"path": "a.py"}, "c1"),
        _out("c1", "Exit code: 0"),
        _call("record_batch", {"items": []}, "c2"),
    ])
    scan = parse_rollout_actions(roll, 0)
    assert scan.agent_recorded is True


def test_parse_ignores_ui_tools(tmp_path):
    roll = tmp_path / "rollout-u.jsonl"
    _write_rollout(roll, [
        _call("render_chart", {}, "c1"),
        _call("update_plan", {}, "c2"),
        _call("web_search", {"query": "x"}, "c3"),
        _call("apply_patch", {"path": "real.py"}, "c4"),
    ])
    scan = parse_rollout_actions(roll, 0)
    # only the apply_patch is an action
    assert len(scan.actions) == 1
    assert scan.actions[0]["tool"] == "Edit"


def test_offset_tracking_no_dupes(tmp_path):
    roll = tmp_path / "rollout-o.jsonl"
    _write_rollout(roll, [
        _call("apply_patch", {"path": "a.py"}, "c1"),
        _out("c1", "Exit code: 0"),
    ])
    scan1 = parse_rollout_actions(roll, 0)
    assert len(scan1.actions) == 1
    off = scan1.new_offset
    # next turn appends more lines
    with open(roll, "a", encoding="utf-8") as f:
        f.write("\n" + json.dumps(_call("shell_command", {"command": "git commit -m y"}, "c2")))
        f.write("\n" + json.dumps(_out("c2", "Exit code: 0")))
    scan2 = parse_rollout_actions(roll, off)
    # only the NEW action, not the old apply_patch again
    assert len(scan2.actions) == 1
    assert "git commit" in scan2.actions[0]["target"]


# ─── significance (incl. PowerShell, which Codex on Windows uses) ────────


def test_powershell_reads_not_significant():
    assert not is_significant_action("Bash", command="Get-ChildItem -Force")
    assert not is_significant_action("Bash", command="Get-Content README.md")
    assert not is_significant_action("Bash", command="Select-String foo *.py")
    assert not is_significant_action("Bash", command="Test-Path src")


def test_real_work_is_significant():
    assert is_significant_action("Edit", "src/auth.py")
    assert is_significant_action("Bash", command="pytest tests/")
    assert is_significant_action("Bash", command="git commit -m fix")


# ─── find_latest_rollout ─────────────────────────────────────────────────


def test_find_latest_rollout(tmp_path):
    d = tmp_path / "sessions" / "2026" / "06" / "07"
    d.mkdir(parents=True)
    older = d / "rollout-a.jsonl"
    newer = d / "rollout-b.jsonl"
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")
    import os, time
    os.utime(older, (time.time() - 100, time.time() - 100))
    found = find_latest_rollout(tmp_path / "sessions")
    assert found == newer


def test_find_latest_rollout_none(tmp_path):
    assert find_latest_rollout(tmp_path / "nonexistent") is None


# ─── notify wiring (temp config, never touches real ~/.codex) ────────────


def test_notify_install_preserves_config(tmp_path, monkeypatch):
    pytest.importorskip("tomli_w")
    import pmb.cli.hooks as H
    cfg = tmp_path / "config.toml"
    cfg.write_text('model = "gpt-5.5"\n[plugins.foo]\nenabled = true\n',
                   encoding="utf-8")
    monkeypatch.setattr(H, "_codex_config_path", lambda: cfg)
    r = H._install_codex_notify()
    assert r["notify"] == "installed"
    text = cfg.read_text(encoding="utf-8")
    assert "codex-notify" in text
    assert "gpt-5.5" in text       # preserved
    assert "plugins" in text       # preserved


def test_notify_install_skips_foreign_notify(tmp_path, monkeypatch):
    pytest.importorskip("tomli_w")
    import pmb.cli.hooks as H
    cfg = tmp_path / "config.toml"
    cfg.write_text('notify = ["my-own-script"]\n', encoding="utf-8")
    monkeypatch.setattr(H, "_codex_config_path", lambda: cfg)
    r = H._install_codex_notify()
    assert r["notify"] == "skipped"
    # user's notify untouched
    assert "my-own-script" in cfg.read_text(encoding="utf-8")


# ─── capability registry (detect type → right mechanism) ─────────────────


def test_ambient_capability_per_agent():
    from pmb.cli.hooks import ambient_capability
    assert ambient_capability("claude-code") == "hooks"
    assert ambient_capability("codex") == "rollout"
    for mcp_only in ("cursor", "windsurf", "vscode", "zed", "gemini",
                     "opencode", "continue"):
        assert ambient_capability(mcp_only) == "mcp-only"
    assert ambient_capability("totally-unknown") == "unknown"


def test_install_dispatches_on_capability():
    from pmb.cli.hooks import install_hook
    # mcp-only agents get an honest 'mcp_only' result, not a crash
    r = install_hook("cursor")
    assert r["action"] == "mcp_only"
    # the reason now points the user at both connect + the project observer
    assert "ambient-watch" in r["reason"]
    assert "connect" in r["reason"]


def test_capability_report_shape():
    from pmb.cli.hooks import capability_report
    rep = capability_report()
    agents = {r["agent"] for r in rep}
    assert {"claude-code", "codex", "cursor"} <= agents
    cc = next(r for r in rep if r["agent"] == "claude-code")
    assert cc["ambient"] is True
    assert cc["ambient_mechanism"] == "hooks"
    # cursor now has ambient too — via the project observer.
    cur = next(r for r in rep if r["agent"] == "cursor")
    assert cur["ambient"] is True
    assert cur["ambient_mechanism"] == "project-observer"
