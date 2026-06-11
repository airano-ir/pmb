"""Tests for `pmb connect` config merging — no subprocesses, no I/O magic."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from pmb.cli.connect import (
    _load_json,
    _save_json,
    connect,
    make_local_entry,
    make_remote_entry,
    merge_entry,
)


@pytest.fixture
def tmp_cwd():
    with tempfile.TemporaryDirectory() as t:
        yield Path(t)


def test_make_remote_entry_rejects_missing_colon():
    with pytest.raises(ValueError):
        make_remote_entry("alex@server")  # missing :path


def test_make_remote_entry_shape():
    e = make_remote_entry("alex@server:/srv/repo")
    assert e["command"] == "ssh"
    assert e["args"][0] == "alex@server"
    assert "PMB_CWD" in e["args"][1]
    assert "/srv/repo" in e["args"][1]
    assert "pmb-mcp" in e["args"][1]


def test_make_local_entry_uses_pmb_cwd(tmp_cwd):
    e = make_local_entry(tmp_cwd)
    assert e["env"]["PMB_CWD"] == str(tmp_cwd)


def test_merge_entry_adds_new():
    existing = {"mcpServers": {"other": {"command": "x"}}}
    new_cfg, action = merge_entry(existing, "pmb", {"command": "y"})
    assert action == "added"
    assert "other" in new_cfg["mcpServers"]
    assert "pmb" in new_cfg["mcpServers"]
    assert new_cfg["mcpServers"]["pmb"]["command"] == "y"


def test_merge_entry_replaces_existing():
    existing = {"mcpServers": {"pmb": {"command": "old"}}}
    new_cfg, action = merge_entry(existing, "pmb", {"command": "new"})
    assert action == "replaced"
    assert new_cfg["mcpServers"]["pmb"]["command"] == "new"


def test_merge_entry_creates_mcpServers_when_missing():
    existing = {"other": "stuff"}
    new_cfg, action = merge_entry(existing, "pmb", {"command": "x"})
    assert action == "added"
    assert "other" in new_cfg  # preserved
    assert new_cfg["mcpServers"]["pmb"]["command"] == "x"


def test_load_save_roundtrip(tmp_cwd):
    p = tmp_cwd / "mcp.json"
    assert _load_json(p) == {}  # absent
    _save_json(p, {"mcpServers": {"pmb": {"command": "x"}}})
    assert _load_json(p) == {"mcpServers": {"pmb": {"command": "x"}}}


def test_load_corrupt_returns_empty(tmp_cwd):
    p = tmp_cwd / "mcp.json"
    p.write_text("not json", encoding="utf-8")
    assert _load_json(p) == {}


def test_connect_claude_code_project_writes_local_mcp_json(tmp_cwd):
    res = connect("claude-code", cwd=tmp_cwd, scope="project")
    assert res["agent"] == "claude-code"
    assert res["entry_name"] == "pmb"
    assert res["action"] == "added"
    mcp = tmp_cwd / ".mcp.json"
    assert mcp.exists()
    data = json.loads(mcp.read_text(encoding="utf-8"))
    assert "pmb" in data["mcpServers"]
    assert data["mcpServers"]["pmb"]["env"]["PMB_CWD"] == str(tmp_cwd)


def test_connect_preserves_other_entries(tmp_cwd):
    mcp = tmp_cwd / ".mcp.json"
    mcp.write_text(json.dumps({"mcpServers": {"github": {"command": "x"}}}), encoding="utf-8")
    connect("claude-code", cwd=tmp_cwd, scope="project")
    data = json.loads(mcp.read_text(encoding="utf-8"))
    assert "github" in data["mcpServers"]
    assert "pmb" in data["mcpServers"]


def test_connect_remote_uses_ssh(tmp_cwd):
    res = connect(
        "claude-code", cwd=tmp_cwd, scope="project",
        remote="alex@server:/srv/repo",
    )
    assert res["entry_name"] == "pmb-remote"
    assert res["entry"]["command"] == "ssh"


def test_connect_cursor_writes_cursor_dir(tmp_cwd):
    connect("cursor", cwd=tmp_cwd)
    mcp = tmp_cwd / ".cursor" / "mcp.json"
    assert mcp.exists()
    data = json.loads(mcp.read_text(encoding="utf-8"))
    assert "pmb" in data["mcpServers"]


def test_connect_name_override(tmp_cwd):
    res = connect(
        "claude-code", cwd=tmp_cwd, scope="project", name_override="my-mem",
    )
    assert res["entry_name"] == "my-mem"
    data = json.loads((tmp_cwd / ".mcp.json").read_text(encoding="utf-8"))
    assert "my-mem" in data["mcpServers"]


def test_unsupported_agent_raises(tmp_cwd):
    with pytest.raises(ValueError):
        connect("ide-of-doom", cwd=tmp_cwd)


def test_connect_replace_keeps_other_servers(tmp_cwd):
    """Run twice: second call should be 'replaced', not duplicate, other unchanged."""
    mcp = tmp_cwd / ".mcp.json"
    mcp.write_text(json.dumps({
        "mcpServers": {"pmb": {"command": "old"}, "other": {"command": "z"}}
    }), encoding="utf-8")
    res = connect("claude-code", cwd=tmp_cwd, scope="project")
    assert res["action"] == "replaced"
    data = json.loads(mcp.read_text(encoding="utf-8"))
    assert "other" in data["mcpServers"]
    assert data["mcpServers"]["other"]["command"] == "z"
