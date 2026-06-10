"""C5: semantic intent fallback (opt-in, default OFF).

When lexical detection misses AND the engine is warm AND the feature is on,
the message is classified by embedding cosine against per-intent exemplars.
Tested with a stubbed multilingual embedder (no real model load).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmb.core.engine import Engine
from pmb.hooks import Intent
from pmb.hooks.semantic_intent import classify_semantic_intent


@pytest.fixture
def tmp_pmb_home():
    import gc
    import shutil
    import time as _t
    tmp = tempfile.mkdtemp()
    home = Path(tmp) / "pmb_home"
    os.environ["PMB_HOME"] = str(home)
    try:
        yield home
    finally:
        os.environ.pop("PMB_HOME", None)
        gc.collect()
        for _ in range(3):
            try:
                shutil.rmtree(tmp, ignore_errors=False)
                break
            except (OSError, PermissionError):
                _t.sleep(0.2)
                gc.collect()
        else:
            shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def tmp_workspace_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def _engine(ws, home, **over):
    cfg = {"recall.cache_size": 0}
    cfg.update(over)
    return Engine(cwd=ws, pmb_home=home, config_overrides=cfg)


def _basis(i):
    v = np.zeros(4, dtype="float32")
    v[i] = 1.0
    return v


def _fake_multilingual_embed(text):
    """A stand-in for the multilingual model: maps same-meaning phrases (in any
    language) to the same basis direction by keyword."""
    t = text.lower()
    if "nomatch" in t:
        return np.ones(4, dtype="float32") / 2.0          # cos 0.5 to each
    if any(k in t for k in ("goal", "in progress", "working towards", "ziele", "цели")):
        return _basis(0)   # GOALS
    if any(k in t for k in ("just", "moment ago", "discussing")):
        return _basis(1)   # RECENT
    if any(k in t for k in ("rule", "convention", "lesson", "work in this project")):
        return _basis(2)   # LESSONS
    return _basis(3)       # PAST / default


def _warm_engine_with_stub(eng, monkeypatch):
    monkeypatch.setattr(eng.search, "embed", _fake_multilingual_embed)
    monkeypatch.setattr(eng, "is_warm", lambda: True)


def test_classify_cross_lingual_goals(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    _warm_engine_with_stub(eng, monkeypatch)
    # German "what are my goals" → GOALS centroid via the multilingual stub
    assert classify_semantic_intent(eng, "Was sind meine Ziele?") == Intent.GOALS_QUERY
    assert classify_semantic_intent(eng, "какие у меня цели") == Intent.GOALS_QUERY


def test_classify_below_threshold_returns_none(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    _warm_engine_with_stub(eng, monkeypatch)
    # the NOMATCH vector is cos 0.5 to every centroid → rejected at threshold 0.6
    assert classify_semantic_intent(eng, "nomatch query", threshold=0.6) is None


def test_run_auto_context_uses_semantic_when_enabled(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    from pmb.hooks import run_auto_context
    eng = _engine(tmp_workspace_dir, tmp_pmb_home,
                  **{"hooks.semantic_intents": True})
    _warm_engine_with_stub(eng, monkeypatch)
    # a non-trivial query lexical patterns don't match (German, no '?')
    res = run_auto_context(eng, "Was sind meine offenen Ziele und Vorhaben")
    assert "SEMANTIC_INTENT" in res.intents
    assert Intent.GOALS_QUERY in res.intents


def test_disabled_by_default_stays_skip(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    from pmb.hooks import run_auto_context
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)  # hooks.semantic_intents off
    _warm_engine_with_stub(eng, monkeypatch)

    def _boom(*a, **k):
        raise AssertionError("semantic classify must not run when disabled")
    monkeypatch.setattr("pmb.hooks.semantic_intent.classify_semantic_intent", _boom)
    res = run_auto_context(eng, "Was sind meine offenen Ziele und Vorhaben")
    assert res.skipped and "SEMANTIC_INTENT" not in res.intents


def test_not_tried_when_cold(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    from pmb.hooks import run_auto_context
    eng = _engine(tmp_workspace_dir, tmp_pmb_home,
                  **{"hooks.semantic_intents": True})
    monkeypatch.setattr(eng.search, "embed", _fake_multilingual_embed)
    monkeypatch.setattr(eng, "is_warm", lambda: False)  # cold → never classify

    def _boom(*a, **k):
        raise AssertionError("semantic classify must not run on a cold engine")
    monkeypatch.setattr("pmb.hooks.semantic_intent.classify_semantic_intent", _boom)
    res = run_auto_context(eng, "Was sind meine offenen Ziele und Vorhaben")
    assert res.skipped
