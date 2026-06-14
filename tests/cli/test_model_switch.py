"""`pmb model` - switch the embedder later (download + reindex + restart in one
step). Tests the target resolution + the guard paths (cancel / no-op) without
triggering the heavy reindex."""
from __future__ import annotations

from pmb.cli.commands.manage import (
    _EMBEDDER_PROFILES,
    _resolve_model_target,
    model,
)
from pmb.config import Config

DEFAULT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def test_resolve_profile_keys():
    assert _resolve_model_target("light", DEFAULT) == _EMBEDDER_PROFILES[0]["model"]
    assert _resolve_model_target("best", DEFAULT) == "BAAI/bge-m3"
    assert _resolve_model_target("BALANCED", DEFAULT) == _EMBEDDER_PROFILES[1]["model"]


def test_resolve_raw_id_passthrough():
    raw = "intfloat/multilingual-e5-base"
    assert _resolve_model_target(raw, DEFAULT) == raw


def test_resolve_empty_keeps_current():
    assert _resolve_model_target("", DEFAULT) == DEFAULT
    assert _resolve_model_target(None, DEFAULT) == DEFAULT


def test_model_cancel_leaves_config_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv("PMB_HOME", str(tmp_path))
    monkeypatch.setattr("typer.confirm", lambda *a, **k: False)  # user declines
    model(name="best", yes=False, no_reindex=False)
    assert "embedding.model" not in Config(pmb_home=tmp_path)._global


def test_model_noop_when_same_does_not_confirm(monkeypatch, tmp_path):
    monkeypatch.setenv("PMB_HOME", str(tmp_path))

    def _boom(*a, **k):
        raise AssertionError("a no-op switch must not prompt or change anything")
    monkeypatch.setattr("typer.confirm", _boom)
    model(name="light", yes=False, no_reindex=False)  # 'light' == current default
    assert "embedding.model" not in Config(pmb_home=tmp_path)._global
