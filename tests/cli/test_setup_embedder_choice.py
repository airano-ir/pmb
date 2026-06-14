"""Setup-time embedder choice: a human picks the memory model (light/balanced/
best) with plain-language plus/minus; the pick persists to the global config.
Tests the pure chooser logic (stubbed prompt) + config persistence."""
from __future__ import annotations

import io

from rich.console import Console

from pmb.cli.commands.manage import _EMBEDDER_PROFILES, _choose_embedder
from pmb.config import Config

DEFAULT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _con():
    return Console(file=io.StringIO(), force_terminal=False)


def test_profiles_wellformed_and_include_current_default():
    keys = {"key", "label", "model", "ram", "rec", "pros", "cons"}
    assert _EMBEDDER_PROFILES
    for p in _EMBEDDER_PROFILES:
        assert keys <= set(p), f"profile missing keys: {p}"
    assert any(p["model"] == DEFAULT for p in _EMBEDDER_PROFILES), \
        "the shipped default model must be one of the offered options"


def test_yes_keeps_current_and_never_prompts(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("setup --yes must not prompt for the embedder")
    monkeypatch.setattr("typer.prompt", _boom)
    assert _choose_embedder(_con(), DEFAULT, yes=True) == DEFAULT


def test_pick_number_returns_that_model(monkeypatch):
    monkeypatch.setattr("typer.prompt", lambda *a, **k: "3")
    assert _choose_embedder(_con(), DEFAULT, yes=False) == _EMBEDDER_PROFILES[2]["model"]


def test_blank_keeps_current(monkeypatch):
    # Enter -> typer.prompt returns its default (the current model's index)
    monkeypatch.setattr("typer.prompt", lambda *a, **k: k.get("default", "1"))
    assert _choose_embedder(_con(), DEFAULT, yes=False) == DEFAULT


def test_garbage_input_keeps_current(monkeypatch):
    monkeypatch.setattr("typer.prompt", lambda *a, **k: "not-a-number")
    assert _choose_embedder(_con(), DEFAULT, yes=False) == DEFAULT


def test_choice_persists_to_global_config(tmp_path):
    cfg = Config(pmb_home=tmp_path)
    cfg.set_global("embedding.model", "BAAI/bge-m3")
    # a fresh Config (as the daemon / next run would build) reads the new value
    assert Config(pmb_home=tmp_path).get("embedding.model") == "BAAI/bge-m3"
