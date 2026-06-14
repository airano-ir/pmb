"""Setup-time offline-brain (LLM tier) choice: pick the background LLM backend
(auto/claude/ollama/anthropic) with plain plus/minus; persists to global config.
Only schema-supported backends are offered (no codex, no fake 'none')."""
from __future__ import annotations

import io

from rich.console import Console

from pmb.cli.commands.manage import _LLM_TIER_PROFILES, _choose_llm_tier
from pmb.config import SCHEMA, Config


def _con():
    return Console(file=io.StringIO(), force_terminal=False)


def test_profiles_wellformed():
    keys = {"key", "label", "backend", "needs", "pros", "cons"}
    assert _LLM_TIER_PROFILES
    for p in _LLM_TIER_PROFILES:
        assert keys <= set(p), f"profile missing keys: {p}"


def test_only_schema_supported_backends_offered():
    allowed = set(getattr(SCHEMA["consolidate.backend"], "choices", ()) or ())
    assert allowed, "consolidate.backend should declare choices"
    for p in _LLM_TIER_PROFILES:
        assert p["backend"] in allowed, f"{p['backend']} not in {allowed}"


def test_yes_keeps_current_and_never_prompts(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("setup --yes must not prompt for the LLM tier")
    monkeypatch.setattr("typer.prompt", _boom)
    assert _choose_llm_tier(_con(), "auto", yes=True) == "auto"


def test_pick_number_returns_that_backend(monkeypatch):
    monkeypatch.setattr("typer.prompt", lambda *a, **k: "3")
    assert _choose_llm_tier(_con(), "auto", yes=False) == _LLM_TIER_PROFILES[2]["backend"]


def test_blank_keeps_current(monkeypatch):
    monkeypatch.setattr("typer.prompt", lambda *a, **k: k.get("default", "1"))
    assert _choose_llm_tier(_con(), "auto", yes=False) == "auto"


def test_choice_persists_to_global_config(tmp_path):
    cfg = Config(pmb_home=tmp_path)
    cfg.set_global("consolidate.backend", "ollama")
    assert Config(pmb_home=tmp_path).get("consolidate.backend") == "ollama"
