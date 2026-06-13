"""B2 — statement-level binary detectors (warm-only anchor fallback).

`is_lesson_intent` (recall boost) and `looks_like_future_intent` (write-time
suggest-goal flag) keep their lexical floor and gain a WARM-ONLY anchor tier so
they also fire for languages the markers/regex don't cover. The cold path (no
engine / cold engine) must stay purely lexical and never load the model.
"""
from __future__ import annotations

from pmb.core.engine import Engine
from pmb.memory_quality import is_lesson_intent
from pmb.reasoning.attributes import looks_like_future_intent


def _engine(ws, home, **over):
    cfg = {"recall.cache_size": 0}
    cfg.update(over)
    return Engine(cwd=ws, pmb_home=home, config_overrides=cfg)


# ── lexical floor unchanged (no engine) ───────────────────────────────────────


def test_lesson_intent_lexical_floor_no_engine():
    assert is_lesson_intent("how should I structure this") is True   # marker
    assert is_lesson_intent("the build is green") is False
    assert is_lesson_intent("random unrelated text", engine=None) is False


def test_future_intent_lexical_floor_no_engine():
    assert looks_like_future_intent("next we'll add the export feature") is True
    assert looks_like_future_intent("we added the export feature yesterday") is False
    assert looks_like_future_intent("plain settled fact", engine=None) is False


# ── warm anchor fallback (stub the engine's anchor_fires — hermetic) ──────────


def test_lesson_intent_warm_anchor_fallback(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    calls = {}

    def _fires(text, name):
        calls["name"] = name
        return name == "query.lesson_intent"
    monkeypatch.setattr(eng, "anchor_fires", _fires)
    # No lexical marker (non-English) → falls through to the anchor.
    assert is_lesson_intent("völlig andere formulierung", engine=eng) is True
    assert calls["name"] == "query.lesson_intent"


def test_future_intent_warm_anchor_fallback(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    monkeypatch.setattr(eng, "anchor_fires",
                        lambda text, name: name == "statement.future_intent")
    # No English future regex match → the anchor decides.
    assert looks_like_future_intent("zrobimy to w przyszłym tygodniu", engine=eng) is True


def test_consumers_stay_lexical_when_anchor_silent(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    monkeypatch.setattr(eng, "anchor_fires", lambda text, name: False)
    assert is_lesson_intent("etwas völlig anderes", engine=eng) is False
    assert looks_like_future_intent("etwas völlig anderes", engine=eng) is False


def test_anchor_error_falls_back_to_lexical(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)

    def _boom(text, name):
        raise RuntimeError("anchor index exploded")
    monkeypatch.setattr(eng, "anchor_fires", _boom)
    # A raising anchor tier must not crash the consumer — lexical result stands.
    assert is_lesson_intent("how should I do this", engine=eng) is True   # lexical
    assert is_lesson_intent("kein marker hier", engine=eng) is False      # safe


def test_anchor_fires_is_warm_only(tmp_pmb_home, tmp_workspace_dir):
    # A COLD engine (never warmed) must return False without loading the model.
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    assert eng.anchor_fires("how should I do this", "query.lesson_intent") is False


# ── real multilingual embedder (eval — loads the model once) ──────────────────


def test_lesson_intent_real_multilingual(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng._is_warm = True   # allow the warm-only anchor index to build
    # Polish/German guidance queries — no English markers, no Latin pack active.
    assert is_lesson_intent("jak powinienem to tutaj zrobić", engine=eng) is True
    assert is_lesson_intent("was ist die Konvention für Commits", engine=eng) is True
    # A plain non-guidance statement must NOT trip it.
    assert is_lesson_intent("die Katze schläft auf dem Sofa", engine=eng) is False
