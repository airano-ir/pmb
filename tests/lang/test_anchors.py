"""A1 — Semantic Anchor Engine. The proof: intents classify across languages
with ZERO language-specific data in the system (English exemplars only; the
multilingual embedder transfers). Borderline cases (some German/Russian
phrasings sit just under the PLACEHOLDER threshold) are calibration targets for
A2, not asserted here — A1 pins the architecture + the reliable cross-lingual
wins at the default tau.
"""
from __future__ import annotations

import pytest

from pmb.core.engine import Engine


def test_cold_engine_has_no_anchor_index(tmp_pmb_home, tmp_workspace_dir):
    # No warmup → engine is cold → anchor_index() must NOT load the model.
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
                 config_overrides={"recall.cache_size": 0})
    assert eng.is_warm() is False
    assert eng.anchor_index() is None


def test_anchors_disabled_switch(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
                 config_overrides={"recall.cache_size": 0, "lang.anchors": False})
    eng._is_warm = True   # pretend warm without loading a model
    assert eng.anchor_index() is None   # kill switch wins


def _fired(idx, text):
    return {s.name.split(".")[-1] for (s, p, m) in idx._scores(text)
            if p >= s.floor and m >= s.tau}


@pytest.mark.eval
def test_multilingual_intent_classification(tmp_pmb_home, tmp_workspace_dir):
    # Warm engine (loads the real multilingual embedder).
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
                 config_overrides={"recall.cache_size": 0})
    eng.warmup()
    idx = eng.anchor_index()
    assert idx is not None

    # cache round-trips: a second index for the same model+anchors loads the npz
    cache = next((eng.workspace.storage_dir / "anchors").glob("anchors-*.npz"), None)
    assert cache is not None and cache.exists()

    # goals_query fires across languages with NO goals data per language
    for lang_text in ["what's left to do",          # EN
                      "co mi zostało do zrobienia",  # PL
                      "次に何をすべきか"]:             # JA
        assert "goals_query" in _fired(idx, lang_text), f"goals miss: {lang_text!r}"

    # work_request across EN + DE
    for lang_text in ["refactor the auth module", "repariere das Auth-Modul"]:
        assert "work_request" in _fired(idx, lang_text), f"work miss: {lang_text!r}"

    # self / past / ack transfer to Russian (no RU data)
    assert "self_intent" in _fired(idx, "где я живу")
    assert "past_query" in _fired(idx, "почему мы выбрали постгрес")
    assert "trivial_ack" in _fired(idx, "спасибо")

    # hard negatives must NOT fire goals (statement / theory, not a goals ask)
    assert "goals_query" not in _fired(idx, "I just finished that task")
    assert "goals_query" not in _fired(idx, "what is a goal in OKR methodology")
