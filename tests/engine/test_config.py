"""Tests for the console-configurable settings layer."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from pmb.config import SCHEMA, Config, _coerce, _flatten, _unflatten


@pytest.fixture
def tmp_dirs():
    with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as home:
        yield Path(ws), Path(home)


def test_get_returns_default_for_unknown_workspace(tmp_dirs):
    ws, home = tmp_dirs
    cfg = Config(workspace_dir=ws, pmb_home=home)
    assert cfg.get("recall.top_k") == SCHEMA["recall.top_k"].default
    assert cfg.source_of("recall.top_k") == "default"


def test_unknown_key_raises(tmp_dirs):
    ws, home = tmp_dirs
    cfg = Config(workspace_dir=ws, pmb_home=home)
    with pytest.raises(KeyError):
        cfg.get("nope.not.real")
    with pytest.raises(KeyError):
        cfg.set_workspace("nope", 1)


def test_set_workspace_persists_and_reads_back(tmp_dirs):
    ws, home = tmp_dirs
    cfg = Config(workspace_dir=ws, pmb_home=home)
    cfg.set_workspace("recall.bm25_weight", "0.7")
    assert cfg.get("recall.bm25_weight") == 0.7
    assert cfg.source_of("recall.bm25_weight") == "workspace"

    # Re-open from disk
    cfg2 = Config(workspace_dir=ws, pmb_home=home)
    assert cfg2.get("recall.bm25_weight") == 0.7
    assert cfg2.source_of("recall.bm25_weight") == "workspace"


def test_workspace_overrides_global(tmp_dirs):
    ws, home = tmp_dirs
    cfg = Config(workspace_dir=ws, pmb_home=home)
    cfg.set_global("recall.bm25_weight", "0.3")
    cfg.set_workspace("recall.bm25_weight", "0.8")
    assert cfg.get("recall.bm25_weight") == 0.8
    assert cfg.source_of("recall.bm25_weight") == "workspace"


def test_override_kwarg_wins_over_workspace(tmp_dirs):
    ws, home = tmp_dirs
    Config(workspace_dir=ws, pmb_home=home).set_workspace("recall.bm25_weight", "0.8")
    cfg = Config(workspace_dir=ws, pmb_home=home, overrides={"recall.bm25_weight": 0.1})
    assert cfg.get("recall.bm25_weight") == 0.1
    assert cfg.source_of("recall.bm25_weight") == "override"


def test_reset_workspace_returns_to_default(tmp_dirs):
    ws, home = tmp_dirs
    cfg = Config(workspace_dir=ws, pmb_home=home)
    cfg.set_workspace("recall.top_k", "12")
    assert cfg.get("recall.top_k") == 12
    cfg.reset_workspace("recall.top_k")
    assert cfg.get("recall.top_k") == SCHEMA["recall.top_k"].default
    assert cfg.source_of("recall.top_k") == "default"


def test_reset_all_workspace(tmp_dirs):
    ws, home = tmp_dirs
    cfg = Config(workspace_dir=ws, pmb_home=home)
    cfg.set_workspace("recall.top_k", "12")
    cfg.set_workspace("recall.bm25_weight", "0.3")
    cfg.reset_workspace(None)
    assert cfg.source_of("recall.top_k") == "default"
    assert cfg.source_of("recall.bm25_weight") == "default"


def test_coerce_boolean_strings():
    s = SCHEMA["recall.rerank"]
    assert _coerce("true", s) is True
    assert _coerce("FALSE", s) is False
    assert _coerce(1, s) is True
    assert _coerce(0, s) is False


def test_coerce_choices_enforced():
    s = SCHEMA["consolidate.backend"]
    assert _coerce("ollama", s) == "ollama"
    with pytest.raises(ValueError):
        _coerce("gpt-9000", s)


def test_coerce_range_enforced():
    s = SCHEMA["recall.bm25_weight"]
    assert _coerce("0.4", s) == 0.4
    with pytest.raises(ValueError):
        _coerce("1.5", s)
    with pytest.raises(ValueError):
        _coerce("-0.1", s)


def test_flatten_unflatten_roundtrip():
    flat = {"recall.bm25_weight": 0.7, "decay.factor_per_day": 0.99}
    nested = _unflatten(flat)
    assert nested == {"recall": {"bm25_weight": 0.7}, "decay": {"factor_per_day": 0.99}}
    assert _flatten(nested) == flat


def test_corrupt_yaml_does_not_crash(tmp_dirs):
    ws, home = tmp_dirs
    (ws / "config.yaml").write_text("this is: not: valid: yaml: maybe", encoding="utf-8")
    # Should silently fall back to defaults rather than raise
    cfg = Config(workspace_dir=ws, pmb_home=home)
    assert cfg.get("recall.bm25_weight") == SCHEMA["recall.bm25_weight"].default


def test_effective_returns_all_keys(tmp_dirs):
    ws, home = tmp_dirs
    cfg = Config(workspace_dir=ws, pmb_home=home)
    eff = cfg.effective()
    assert set(eff.keys()) == set(SCHEMA.keys())
