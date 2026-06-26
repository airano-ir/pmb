"""Semantic intent fallback for the auto-recall hook.

Two tiers (see pmb/hooks/semantic_intent.py):
  * B1 anchors  — the calibrated Semantic Anchor Engine, DEFAULT ON
                  (`lang.anchors`). Hermetic wiring tests stub the index; one
                  eval test exercises the REAL multilingual embedder.
  * C5 centroid — legacy raw-cosine exemplars, only when `lang.anchors` is off.
                  Kept green here with a stubbed multilingual embedder.
"""
from __future__ import annotations

import numpy as np
import pytest

from pmb.core.engine import Engine
from pmb.hooks import Intent
from pmb.hooks.semantic_intent import (
    classify_anchor_intent,
    classify_semantic_intent,
)
from pmb.lang.anchors import AnchorHit


def _engine(ws, home, **over):
    cfg = {"recall.cache_size": 0}
    cfg.update(over)
    return Engine(cwd=ws, pmb_home=home, config_overrides=cfg)


# ── B1: anchor tier (hermetic — stub the index, no model) ─────────────────────


class _FakeIndex:
    """Keyword→anchor stub so we can test the mapping + dispatch without a
    model. Returns hits best-margin-first, like the real AnchorIndex."""

    def __init__(self, mapping: dict[str, str]):
        self._m = mapping

    def classify(self, text: str):
        t = (text or "").lower()
        hits = [AnchorHit(name, 0.9, 0.5)
                for sub, name in self._m.items() if sub in t]
        return hits


def _warm_with_index(eng, monkeypatch, mapping):
    monkeypatch.setattr(eng, "is_warm", lambda: True)
    monkeypatch.setattr(eng, "anchor_index", lambda: _FakeIndex(mapping))


def test_anchor_intent_maps_each_set(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    cases = {
        "goals here": ("goals here", "intent.goals_query", Intent.GOALS_QUERY),
        "past here": ("past here", "intent.past_query", Intent.PAST_QUERY),
        "recent here": ("recent here", "intent.recent_query", Intent.RECENT_QUERY),
        "lessons here": ("lessons here", "intent.lessons_query", Intent.LESSONS_QUERY),
        "work here": ("work here", "intent.work_request", Intent.WORK_REQUEST),
        "self here": ("self here", "intent.self_intent", Intent.PAST_QUERY),
    }
    for text, (sub, anchor, expected) in cases.items():
        monkeypatch.setattr(eng, "anchor_index", lambda a=anchor, s=sub: _FakeIndex({s: a}))
        monkeypatch.setattr(eng, "is_warm", lambda: True)
        assert classify_anchor_intent(eng, text) == expected


def test_anchor_trivial_ack_stays_silent(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    _warm_with_index(eng, monkeypatch, {"ack": "intent.trivial_ack"})
    # A foreign-language ack as the strongest hit must NOT surface memory.
    assert classify_anchor_intent(eng, "some ack text") is None


def test_anchor_no_hit_returns_none(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    _warm_with_index(eng, monkeypatch, {"goalz": "intent.goals_query"})
    assert classify_anchor_intent(eng, "totally unrelated") is None


def test_classify_semantic_prefers_anchors(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)   # lang.anchors default ON
    _warm_with_index(eng, monkeypatch, {"ziele": "intent.goals_query"})
    # Even with a centroid stub present, the anchor tier is authoritative.
    monkeypatch.setattr(eng.search, "embed",
                        lambda t: np.ones(4, dtype="float32"))
    assert classify_semantic_intent(eng, "Was sind meine Ziele?") == Intent.GOALS_QUERY


def test_run_auto_context_fires_anchors_by_default(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    from pmb.hooks import run_auto_context
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)   # NO opt-in flag needed now
    _warm_with_index(eng, monkeypatch, {"ziele": "intent.goals_query"})
    res = run_auto_context(eng, "Was sind meine offenen Ziele und Vorhaben")
    assert "SEMANTIC_INTENT" in res.intents
    assert Intent.GOALS_QUERY in res.intents


def test_run_auto_context_skips_when_both_tiers_off(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    from pmb.hooks import run_auto_context
    eng = _engine(tmp_workspace_dir, tmp_pmb_home,
                  **{"lang.anchors": False})   # both anchors + centroids off
    monkeypatch.setattr(eng, "is_warm", lambda: True)

    def _boom(*a, **k):
        raise AssertionError("no semantic tier may run when both are disabled")
    monkeypatch.setattr("pmb.hooks.semantic_intent.classify_semantic_intent", _boom)
    res = run_auto_context(eng, "Was sind meine offenen Ziele und Vorhaben")
    assert res.skipped and "SEMANTIC_INTENT" not in res.intents


def test_anchor_tier_not_tried_when_cold(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    from pmb.hooks import run_auto_context
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    monkeypatch.setattr(eng, "is_warm", lambda: False)   # cold → never classify

    def _boom(*a, **k):
        raise AssertionError("semantic tier must not run on a cold engine")
    monkeypatch.setattr("pmb.hooks.semantic_intent.classify_semantic_intent", _boom)
    res = run_auto_context(eng, "Was sind meine offenen Ziele und Vorhaben")
    assert res.skipped


# ── C5 legacy centroid tier (only when lang.anchors is off) ───────────────────


def _basis(i):
    v = np.zeros(4, dtype="float32")
    v[i] = 1.0
    return v


def _fake_multilingual_embed(text):
    t = text.lower()
    if "nomatch" in t:
        return np.ones(4, dtype="float32") / 2.0
    if any(k in t for k in ("goal", "in progress", "working towards", "ziele", "цели")):
        return _basis(0)
    if any(k in t for k in ("just", "moment ago", "discussing")):
        return _basis(1)
    if any(k in t for k in ("rule", "convention", "lesson", "work in this project")):
        return _basis(2)
    return _basis(3)


def _warm_centroid(eng, monkeypatch):
    monkeypatch.setattr(eng.search, "embed", _fake_multilingual_embed)
    monkeypatch.setattr(eng, "is_warm", lambda: True)


def test_centroid_cross_lingual_goals(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home, **{"lang.anchors": False})
    _warm_centroid(eng, monkeypatch)
    assert classify_semantic_intent(eng, "Was sind meine Ziele?") == Intent.GOALS_QUERY
    assert classify_semantic_intent(eng, "какие у меня цели") == Intent.GOALS_QUERY


def test_centroid_below_threshold_returns_none(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home, **{"lang.anchors": False})
    _warm_centroid(eng, monkeypatch)
    assert classify_semantic_intent(eng, "nomatch query", threshold=0.6) is None


# ── B1: real multilingual embedder (eval — loads the model once) ──────────────


# platform_sensitive: 1-2 of 12 multilingual intents flip with embedder float
# math on macOS arm64; gates on the Linux reference runner.
@pytest.mark.platform_sensitive
def test_anchor_intent_real_multilingual(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    """End-to-end: real anchors + calibrated taus classify languages the lexical
    patterns (en/ru/uk only) don't cover — with ZERO per-language data anywhere.

    Cases span six European languages × six intents and were measured to fire
    (2026-06-12). The known weak corners are NOT asserted here: distant-script
    imperatives (JA/ZH `work_request`, the project's lowest-recall set at 0.70)
    and a few borderline Spanish phrasings still miss — that gap is exactly what
    Phase D (Anchor→Lexicon Distillation) and a future embedder upgrade close."""
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    # anchor_index() gates on the _is_warm ATTRIBUTE (not the method) so the cold
    # hook path never builds it; set it directly to allow the real index build.
    eng._is_warm = True
    cases = [
        ("Was muss noch erledigt werden?", Intent.GOALS_QUERY),          # DE goals
        ("qu'est-ce qu'il reste à faire", Intent.GOALS_QUERY),           # FR goals
        ("quali sono i miei obiettivi aperti", Intent.GOALS_QUERY),      # IT goals
        ("warum haben wir diese Datenbank gewählt", Intent.PAST_QUERY),  # DE past
        ("por qué elegimos esta base de datos", Intent.PAST_QUERY),      # ES past
        ("wer bin ich", Intent.PAST_QUERY),                              # DE self→past
        ("gdzie mieszkam", Intent.PAST_QUERY),                           # PL self→past
        ("worüber haben wir gerade gesprochen", Intent.RECENT_QUERY),    # DE recent
        ("où en étais-je", Intent.RECENT_QUERY),                         # FR recent
        ("refaktoriere diese Funktion", Intent.WORK_REQUEST),            # DE work
        ("arregla el módulo de autenticación", Intent.WORK_REQUEST),     # ES work
        ("was sind die Projektkonventionen", Intent.LESSONS_QUERY),      # DE lessons
    ]
    got = {text: classify_anchor_intent(eng, text) for text, _ in cases}
    hits = sum(1 for text, exp in cases if got[text] == exp)
    # 12 cases across 6 languages and 6 intents — allow one embedder flake.
    assert hits >= 11, f"multilingual anchor intents regressed: {got}"
