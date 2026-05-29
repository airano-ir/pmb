"""Tests for auto-installed agent instruction rules (Improvement O)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmb.cli.connect import (
    install_agent_rules, instruction_paths_for_agent, connect,
    PMB_AGENT_RULES_START, PMB_AGENT_RULES_END,
)


def test_create_new_instructions_file(tmp_path):
    f = tmp_path / "AGENTS.md"
    action = install_agent_rules(f)
    assert action == "created"
    body = f.read_text(encoding="utf-8")
    assert PMB_AGENT_RULES_START in body
    assert PMB_AGENT_RULES_END in body
    assert "record_batch" in body
    assert "запомни" in body  # RU trigger example present


def test_append_to_existing_instructions(tmp_path):
    f = tmp_path / "AGENTS.md"
    f.write_text("# User Custom Rules\n\nDon't overwrite me.\n", encoding="utf-8")
    action = install_agent_rules(f)
    assert action == "added"
    body = f.read_text(encoding="utf-8")
    assert "User Custom Rules" in body          # preserved
    assert "Don't overwrite me" in body         # preserved
    assert PMB_AGENT_RULES_START in body        # added
    assert "record_batch" in body


def test_update_existing_rules_block(tmp_path):
    """Re-running pmb connect should UPDATE the existing block, not duplicate."""
    f = tmp_path / "AGENTS.md"
    f.write_text(
        "# Custom\n\n"
        f"{PMB_AGENT_RULES_START}\nold content\n{PMB_AGENT_RULES_END}\n"
        "\nother content\n",
        encoding="utf-8",
    )
    action = install_agent_rules(f)
    assert action == "updated"
    body = f.read_text(encoding="utf-8")
    # Should appear EXACTLY once
    assert body.count(PMB_AGENT_RULES_START) == 1
    assert body.count(PMB_AGENT_RULES_END) == 1
    # Old custom content preserved
    assert "Custom" in body
    assert "other content" in body
    # New rules present
    assert "record_batch" in body
    assert "old content" not in body


def test_paths_resolution():
    cwd = Path("/some/project")
    claude = instruction_paths_for_agent("claude-code", cwd)
    assert any("CLAUDE.md" in str(p) for p in claude)
    codex = instruction_paths_for_agent("codex", cwd)
    assert any("AGENTS.md" in str(p) for p in codex)
    cursor = instruction_paths_for_agent("cursor", cwd)
    assert any(".cursorrules" in str(p) for p in cursor)
    unknown = instruction_paths_for_agent("xyz", cwd)
    assert unknown == []


def test_connect_codex_writes_agents_md(tmp_path, monkeypatch):
    """End-to-end: pmb connect codex creates ~/.codex/AGENTS.md with rules."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    (tmp_path / ".codex").mkdir(parents=True, exist_ok=True)

    result = connect("codex", cwd=tmp_path / "proj")
    assert result["agent"] == "codex"
    # AGENTS.md should have been created in ~/.codex/
    agents_md = tmp_path / ".codex" / "AGENTS.md"
    assert agents_md.exists()
    body = agents_md.read_text(encoding="utf-8")
    assert "record_batch" in body
    assert PMB_AGENT_RULES_START in body
    # Result reports it too
    assert result.get("instruction_rules"), "should report rules written"


def test_connect_claude_writes_claude_md(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    (tmp_path / ".claude").mkdir(parents=True, exist_ok=True)

    result = connect("claude-code", cwd=tmp_path / "proj", scope="global")
    claude_md = tmp_path / ".claude" / "CLAUDE.md"
    assert claude_md.exists()
    body = claude_md.read_text(encoding="utf-8")
    assert "record_batch" in body


def test_connect_idempotent_no_dup(tmp_path, monkeypatch):
    """Calling connect twice should NOT duplicate the rules block."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    (tmp_path / ".codex").mkdir(parents=True, exist_ok=True)

    connect("codex", cwd=tmp_path / "proj")
    connect("codex", cwd=tmp_path / "proj")  # second call

    body = (tmp_path / ".codex" / "AGENTS.md").read_text(encoding="utf-8")
    assert body.count(PMB_AGENT_RULES_START) == 1
    assert body.count(PMB_AGENT_RULES_END) == 1
