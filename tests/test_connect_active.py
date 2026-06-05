"""Tests for `pmb connect --active` (proactive-logging agent rules).

The default rules are deliberately conservative (PMB OFF until a trigger).
`--active` appends an addendum that makes the agent log its OWN work
(decisions / lessons / done) during coding, while keeping recall lazy.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmb.cli.connect import (  # noqa: E402
    _build_agent_rules_block, install_agent_rules,
    PMB_AGENT_RULES_START, PMB_AGENT_RULES_END,
)


def test_default_rules_are_conservative():
    block = _build_agent_rules_block(active=False)
    assert "PMB is OFF by default" in block
    assert "ACTIVE MODE" not in block


def test_active_rules_add_proactive_logging():
    block = _build_agent_rules_block(active=True)
    assert "ACTIVE MODE" in block
    assert '"kind":"decision"' in block      # proactive write examples present
    assert '"type":"lesson"' in block
    assert "Recall stays lazy" in block       # recall side unchanged
    assert "PMB is OFF by default" in block   # appended, not replaced


def test_install_active_writes_addendum(tmp_path):
    p = tmp_path / "AGENTS.md"
    assert install_agent_rules(p, active=True) == "created"
    body = p.read_text(encoding="utf-8")
    assert "ACTIVE MODE" in body
    assert PMB_AGENT_RULES_START in body and PMB_AGENT_RULES_END in body


def test_install_default_no_addendum(tmp_path):
    p = tmp_path / "CLAUDE.md"
    install_agent_rules(p, active=False)
    body = p.read_text(encoding="utf-8")
    assert "ACTIVE MODE" not in body
    assert "PMB is OFF by default" in body


def test_toggle_active_off_removes_addendum(tmp_path):
    """Re-running connect WITHOUT --active drops the active addendum (the
    BEGIN/END markers replace the whole block, they don't accumulate)."""
    p = tmp_path / "AGENTS.md"
    install_agent_rules(p, active=True)
    assert "ACTIVE MODE" in p.read_text(encoding="utf-8")
    install_agent_rules(p, active=False)
    body = p.read_text(encoding="utf-8")
    assert "ACTIVE MODE" not in body
    assert "PMB is OFF by default" in body


# ----------------------------------------------------------------------
# pmb setup - agent auto-detection
# ----------------------------------------------------------------------

def test_detect_installed_agents_empty(tmp_path, monkeypatch):
    from pmb.cli.connect import detect_installed_agents
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "noxdg"))
    assert detect_installed_agents(home=tmp_path, cwd=tmp_path) == []


def test_detect_installed_agents_finds_codex(tmp_path, monkeypatch):
    from pmb.cli.connect import detect_installed_agents
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "noxdg"))
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex" / "config.toml").write_text("", encoding="utf-8")
    assert "codex" in detect_installed_agents(home=tmp_path, cwd=tmp_path)


def test_detect_installed_agents_finds_cursor_project(tmp_path, monkeypatch):
    from pmb.cli.connect import detect_installed_agents
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "noxdg"))
    (tmp_path / ".cursor").mkdir()
    (tmp_path / ".cursor" / "mcp.json").write_text("{}", encoding="utf-8")
    assert "cursor" in detect_installed_agents(home=tmp_path, cwd=tmp_path)


def test_setup_help_runs():
    from typer.testing import CliRunner
    from pmb.cli.main import app
    r = CliRunner().invoke(app, ["setup", "--help"])
    assert r.exit_code == 0
    assert "Guided first-time setup" in r.output


# ----------------------------------------------------------------------
# Active mode - per-category toggles (pro config) + self-improvement loop
# ----------------------------------------------------------------------

def test_active_addendum_all_on_by_default():
    from pmb.cli.connect import build_active_addendum
    a = build_active_addendum(None)
    for needle in ('"kind":"decision"', '"kind":"completed"', '"type":"lesson"',
                   '"type":"failure"', '"type":"goal"'):
        assert needle in a, needle
    assert "Self-improvement loop" in a            # apply_lessons on by default


def test_active_addendum_toggles_drop_categories():
    from pmb.cli.connect import build_active_addendum
    a = build_active_addendum({"log_failures": False, "log_goals": False})
    assert '"type":"failure"' not in a
    assert '"type":"goal"' not in a
    assert '"kind":"decision"' in a                # others remain


def test_active_addendum_apply_lessons_off():
    from pmb.cli.connect import build_active_addendum
    a = build_active_addendum({"apply_lessons": False})
    assert "Self-improvement loop" not in a
    assert '"type":"lesson"' in a                  # logging lessons still on


def test_agent_config_keys_exist():
    from pmb.config import SCHEMA
    for k in ("agent.log_decisions", "agent.log_completed", "agent.log_lessons",
              "agent.log_failures", "agent.log_goals", "agent.apply_lessons"):
        assert k in SCHEMA
        assert SCHEMA[k].type is bool
        assert SCHEMA[k].default is True


def test_active_mode_and_overview_config_keys():
    from pmb.config import SCHEMA
    assert SCHEMA["agent.active_mode"].type is bool
    assert SCHEMA["agent.active_mode"].default is False     # opt-in auto-logging
    assert SCHEMA["overview.max_events"].type is int
    assert SCHEMA["overview.max_events"].default == 40


def test_overview_help_runs():
    from typer.testing import CliRunner
    from pmb.cli.main import app
    r = CliRunner().invoke(app, ["overview", "--help"])
    assert r.exit_code == 0
    assert "overview" in r.output.lower()


def test_topic_overview_empty_topic(tmp_path, monkeypatch):
    """Empty topic returns early (no recall / no model load)."""
    monkeypatch.setenv("PMB_HOME", str(tmp_path / "h"))
    monkeypatch.setenv("PMB_WORKSPACE", "ovtest")
    from pmb.core.engine import Engine
    eng = Engine()
    ov = eng.topic_overview("")
    assert ov["empty"] is True
    assert ov["n_memories"] == 0


# ----------------------------------------------------------------------
# Session continuity - session_brief + context-continuity rules
# ----------------------------------------------------------------------

def test_continuity_and_session_config_keys():
    from pmb.config import SCHEMA
    assert SCHEMA["agent.context_continuity"].type is bool
    assert SCHEMA["agent.context_continuity"].default is True
    assert SCHEMA["session.brief_minutes"].type is int
    assert SCHEMA["session.brief_minutes"].default == 180


def test_active_addendum_has_continuity_section():
    from pmb.cli.connect import build_active_addendum
    a = build_active_addendum(None)
    assert "session_brief" in a
    assert "lose the thread" in a.lower()


def test_active_addendum_continuity_off():
    from pmb.cli.connect import build_active_addendum
    a = build_active_addendum({"context_continuity": False})
    assert "session_brief" not in a


def test_session_brief_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("PMB_HOME", str(tmp_path / "h"))
    monkeypatch.setenv("PMB_WORKSPACE", "sbtest")
    from pmb.core.engine import Engine
    eng = Engine()
    b = eng.session_brief()
    assert b["empty"] is True
    assert b["n_events"] == 0


def test_session_brief_groups_recent_events(tmp_path, monkeypatch):
    """No model load needed - session_brief reads events directly."""
    monkeypatch.setenv("PMB_HOME", str(tmp_path / "h"))
    monkeypatch.setenv("PMB_WORKSPACE", "sbtest2")
    from pmb.core.engine import Engine
    eng = Engine()
    eng.record_fact("Chose Postgres over Mongo", metadata={"kind": "decision"})
    eng.record_fact("Fixed the JWT expiry bug", metadata={"kind": "completed"})
    eng.record_fact("use pnpm not npm",
                    metadata={"source": "lesson", "kind": "lesson"})
    b = eng.session_brief(minutes=180)
    assert b["n_events"] >= 3
    assert any("Postgres" in d["content"] for d in b["decisions"])
    assert any("JWT" in d["content"] for d in b["done"])
    assert any("pnpm" in l["content"] for l in b["lessons"])
