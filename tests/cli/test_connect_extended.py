"""Tests for the Sprint-1 extended `pmb connect` agents:
windsurf / gemini / vscode / zed / opencode / continue.

Each agent has a different config format; these tests pin the exact
on-disk shape so a refactor can't silently break an integration.
No subprocesses - pure config-file merging.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from pmb.cli.connect import (
    JSON_AGENT_SPECS,
    connect,
    merge_continue_entry,
    merge_keyed_entry,
    shape_entry,
    supported_agents,
)


@pytest.fixture
def tmp_cwd():
    with tempfile.TemporaryDirectory() as t:
        yield Path(t)


# ----------------------------------------------------------------------
# Registry sanity
# ----------------------------------------------------------------------

def test_supported_agents_includes_big_three_and_extended():
    agents = supported_agents()
    for must in ("claude-code", "cursor", "codex"):
        assert must in agents
    for ext in ("windsurf", "gemini", "vscode", "zed", "opencode", "continue"):
        assert ext in agents
    # 3 core + 6 extended = at least 9
    assert len(agents) >= 9


def test_every_spec_has_a_config_path():
    for name, spec in JSON_AGENT_SPECS.items():
        assert spec.project_path or spec.global_path, f"{name} has no config path"


# ----------------------------------------------------------------------
# Entry shaping - the part most likely to drift from each agent's schema
# ----------------------------------------------------------------------

def test_shape_claude_flat():
    e = shape_entry("claude", "pmb-mcp", [], {"PMB_CWD": "/x"})
    assert e == {"command": "pmb-mcp", "env": {"PMB_CWD": "/x"}}


def test_shape_claude_with_args():
    e = shape_entry("claude", "python", ["-m", "pmb.mcp.server"], {})
    assert e == {"command": "python", "args": ["-m", "pmb.mcp.server"]}


def test_shape_zed_wraps_command_object():
    e = shape_entry("zed", "pmb-mcp", ["-x"], {"PMB_CWD": "/x"})
    assert e == {"command": {"path": "pmb-mcp", "args": ["-x"], "env": {"PMB_CWD": "/x"}}}


def test_shape_opencode_command_is_list_env_is_environment():
    e = shape_entry("opencode", "pmb-mcp", ["-m", "x"], {"PMB_CWD": "/x"})
    assert e["type"] == "local"
    assert e["command"] == ["pmb-mcp", "-m", "x"]
    assert e["enabled"] is True
    assert e["environment"] == {"PMB_CWD": "/x"}
    assert "env" not in e  # opencode uses 'environment', not 'env'


def test_shape_unknown_raises():
    with pytest.raises(ValueError):
        shape_entry("nonsense", "x", [], {})


# ----------------------------------------------------------------------
# Merge helpers preserve siblings
# ----------------------------------------------------------------------

def test_merge_keyed_preserves_other_servers():
    existing = {"servers": {"github": {"command": "gh"}}}
    new_cfg, action = merge_keyed_entry(existing, "servers", "pmb", {"command": "pmb-mcp"})
    assert action == "added"
    assert "github" in new_cfg["servers"]
    assert new_cfg["servers"]["pmb"]["command"] == "pmb-mcp"


def test_merge_keyed_replaces():
    existing = {"context_servers": {"pmb": {"command": {"path": "old"}}}}
    new_cfg, action = merge_keyed_entry(
        existing, "context_servers", "pmb", {"command": {"path": "new"}}
    )
    assert action == "replaced"
    assert new_cfg["context_servers"]["pmb"]["command"]["path"] == "new"


def test_merge_continue_list_form_adds():
    existing = {"mcpServers": [{"name": "other", "command": "x"}]}
    new_cfg, action = merge_continue_entry(existing, "pmb", "pmb-mcp", [], {"PMB_CWD": "/x"})
    assert action == "added"
    names = {s["name"] for s in new_cfg["mcpServers"]}
    assert names == {"other", "pmb"}


def test_merge_continue_list_form_replaces():
    existing = {"mcpServers": [{"name": "pmb", "command": "old"}]}
    new_cfg, action = merge_continue_entry(existing, "pmb", "new", [], {})
    assert action == "replaced"
    assert len(new_cfg["mcpServers"]) == 1
    assert new_cfg["mcpServers"][0]["command"] == "new"


# ----------------------------------------------------------------------
# End-to-end connect() per agent - verify the file lands correctly
# ----------------------------------------------------------------------

def test_connect_vscode_project_writes_servers_key(tmp_cwd):
    res = connect("vscode", cwd=tmp_cwd, scope="project")
    assert res["agent"] == "vscode"
    assert res["action"] == "added"
    cfg_file = Path(res["config_path"])
    assert cfg_file == tmp_cwd / ".vscode" / "mcp.json"
    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert "servers" in data            # VS Code uses 'servers', not 'mcpServers'
    assert "pmb" in data["servers"]
    assert data["servers"]["pmb"]["env"]["PMB_CWD"] == str(tmp_cwd)


def test_connect_zed_uses_config_path_override(tmp_cwd):
    target = tmp_cwd / "zed_settings.json"
    res = connect("zed", cwd=tmp_cwd, config_path=str(target))
    assert Path(res["config_path"]) == target
    data = json.loads(target.read_text(encoding="utf-8"))
    assert "context_servers" in data
    assert "command" in data["context_servers"]["pmb"]
    assert data["context_servers"]["pmb"]["command"]["path"]  # wrapped object


def test_connect_opencode_override(tmp_cwd):
    target = tmp_cwd / "opencode.json"
    res = connect("opencode", cwd=tmp_cwd, config_path=str(target))
    data = json.loads(target.read_text(encoding="utf-8"))
    assert "mcp" in data
    entry = data["mcp"]["pmb"]
    assert entry["type"] == "local"
    assert isinstance(entry["command"], list)
    assert entry["enabled"] is True


def test_connect_continue_yaml_list(tmp_cwd):
    target = tmp_cwd / "continue.yaml"
    res = connect("continue", cwd=tmp_cwd, config_path=str(target))
    import yaml
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert isinstance(data["mcpServers"], list)
    assert data["mcpServers"][0]["name"] == "pmb"


def test_connect_windsurf_override_preserves_others(tmp_cwd):
    target = tmp_cwd / "mcp_config.json"
    target.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}), encoding="utf-8")
    connect("windsurf", cwd=tmp_cwd, config_path=str(target))
    data = json.loads(target.read_text(encoding="utf-8"))
    assert "other" in data["mcpServers"]
    assert "pmb" in data["mcpServers"]


def test_connect_extended_shared_workspace_name(tmp_cwd):
    target = tmp_cwd / "gemini.json"
    res = connect("gemini", cwd=tmp_cwd, workspace_id="personal", config_path=str(target))
    assert res["entry_name"] == "pmb-shared"
    data = json.loads(target.read_text(encoding="utf-8"))
    assert "pmb-shared" in data["mcpServers"]
    assert data["mcpServers"]["pmb-shared"]["env"]["PMB_WORKSPACE"] == "personal"


def test_connect_extended_remote_ssh(tmp_cwd):
    target = tmp_cwd / "vscode.json"
    res = connect("vscode", cwd=tmp_cwd, remote="alex@server:/srv/repo",
                  config_path=str(target))
    assert res["entry_name"] == "pmb-remote"
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["servers"]["pmb-remote"]["command"] == "ssh"


def test_connect_replace_is_idempotent(tmp_cwd):
    target = tmp_cwd / "mcp.json"
    connect("vscode", cwd=tmp_cwd, config_path=str(target))
    res2 = connect("vscode", cwd=tmp_cwd, config_path=str(target))
    assert res2["action"] == "replaced"
    data = json.loads(target.read_text(encoding="utf-8"))
    # exactly one pmb entry, not duplicated
    assert list(data["servers"].keys()).count("pmb") == 1
