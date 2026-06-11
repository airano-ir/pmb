"""Tests for Codex CLI support in pmb connect."""
from __future__ import annotations

from pathlib import Path

from pmb.cli.connect import (
    _load_toml,
    _save_toml,
    codex_paths,
    connect,
    make_local_entry,
    merge_codex_entry,
)


def test_codex_paths():
    target = codex_paths(Path.cwd())
    assert target.name == "codex"
    assert ".codex" in str(target.config_paths[0])
    assert target.config_paths[0].name == "config.toml"


def test_merge_codex_entry_adds_to_empty_config():
    existing = {}
    entry = {"command": "pmb-mcp", "env": {"PMB_CWD": "/path/to/proj"}}
    new_cfg, action = merge_codex_entry(existing, "pmb", entry)
    assert action == "added"
    assert "mcp_servers" in new_cfg
    assert "pmb" in new_cfg["mcp_servers"]
    assert new_cfg["mcp_servers"]["pmb"]["command"] == "pmb-mcp"
    assert new_cfg["mcp_servers"]["pmb"]["args"] == []
    assert new_cfg["mcp_servers"]["pmb"]["env"]["PMB_CWD"] == "/path/to/proj"
    assert new_cfg["mcp_servers"]["pmb"]["startup_timeout_sec"] == 120


def test_merge_codex_entry_preserves_other_config():
    """Existing Codex config (marketplaces, plugins, projects) must NOT be touched."""
    existing = {
        "model": "gpt-5",
        "marketplaces": {"openai-bundled": {"source": "..."}},
        "plugins": {"documents": {"enabled": True}},
        "projects": {"/some/path": {"trust_level": "trusted"}},
        "mcp_servers": {
            "node_repl": {"command": "node_repl.exe", "args": []},
        },
    }
    entry = make_local_entry(Path("/proj"))
    new_cfg, action = merge_codex_entry(existing, "pmb", entry)
    assert action == "added"
    # Other sections untouched
    assert new_cfg["model"] == "gpt-5"
    assert new_cfg["marketplaces"] == existing["marketplaces"]
    assert new_cfg["plugins"] == existing["plugins"]
    assert new_cfg["projects"] == existing["projects"]
    # Existing MCP server preserved
    assert "node_repl" in new_cfg["mcp_servers"]
    # New PMB server added
    assert "pmb" in new_cfg["mcp_servers"]


def test_merge_codex_entry_replaces_existing():
    existing = {
        "mcp_servers": {"pmb": {"command": "old-pmb", "args": []}},
    }
    entry = {"command": "new-pmb"}
    new_cfg, action = merge_codex_entry(existing, "pmb", entry)
    assert action == "replaced"
    assert new_cfg["mcp_servers"]["pmb"]["command"] == "new-pmb"


def test_toml_roundtrip(tmp_path):
    """Make sure we can write TOML and read it back identically."""
    config = {
        "model": "gpt-5",
        "mcp_servers": {
            "pmb": {
                "command": "pmb-mcp",
                "args": [],
                "env": {"PMB_CWD": "/proj"},
                "startup_timeout_sec": 120,
            },
        },
    }
    p = tmp_path / "config.toml"
    _save_toml(p, config)
    loaded = _load_toml(p)
    assert loaded == config


def test_codex_connect_end_to_end(tmp_path, monkeypatch):
    """End-to-end: pre-existing Codex config + pmb connect codex → entry added."""
    # Set up a fake $HOME pointing to tmp_path so Codex paths land there
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    codex_config = tmp_path / ".codex" / "config.toml"
    codex_config.parent.mkdir(parents=True, exist_ok=True)
    # Pre-existing TOML with marketplaces (must be preserved)
    codex_config.write_text(
        'model = "gpt-5"\n\n'
        '[marketplaces.openai-bundled]\n'
        'source = "/some/path"\n\n'
        '[mcp_servers.node_repl]\n'
        'command = "node_repl.exe"\n'
        'args = []\n',
        encoding="utf-8",
    )

    result = connect("codex", cwd=tmp_path / "myproject")
    assert result["agent"] == "codex"
    assert result["action"] == "added"

    # Verify the TOML is valid and preserves everything
    loaded = _load_toml(codex_config)
    assert loaded["model"] == "gpt-5"
    assert "marketplaces" in loaded
    assert "node_repl" in loaded["mcp_servers"]
    assert "pmb" in loaded["mcp_servers"]
    pmb_entry = loaded["mcp_servers"]["pmb"]
    assert "command" in pmb_entry
    assert pmb_entry["env"]["PMB_CWD"].endswith("myproject")


def test_codex_with_shared_workspace(tmp_path, monkeypatch):
    """`pmb connect codex --workspace personal` forces a shared workspace."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    (tmp_path / ".codex").mkdir(parents=True, exist_ok=True)

    result = connect(
        "codex", cwd=tmp_path / "anywhere",
        workspace_id="personal",
    )
    loaded = _load_toml(tmp_path / ".codex" / "config.toml")
    pmb = loaded["mcp_servers"][result["entry_name"]]
    assert pmb["env"]["PMB_WORKSPACE"] == "personal"
