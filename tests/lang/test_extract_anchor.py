"""C2 — keyed-fact extraction by hypothesis margin (real embedder).

Measured 2026-06-12. The point: city/employer/name extracted across EN/RU/DE/FR
with ZERO per-language regex, negation suppressed by the anti-hypothesis, and
non-self sentences produce nothing. The decision is precision-first — an
ambiguous span yields NO keyed fact rather than a wrong one.
"""
from __future__ import annotations

import pytest

from pmb.core.engine import Engine
from pmb.reasoning.extract_anchor import extract_keyed_anchor


def _warm_engine(ws, home):
    eng = Engine(cwd=ws, pmb_home=home, config_overrides={"recall.cache_size": 0})
    eng._is_warm = True   # allow the warm-only anchor index to build
    return eng


def _has(cands, attr, value):
    return any(c.attr == attr and c.value == value for c in cands)


# platform_sensitive: a 1-of-8 embedder margin flip on macOS arm64; this same
# file ALSO gates (blocking) on the Linux packs-off-ratchet job.
@pytest.mark.platform_sensitive
def test_keyed_extraction_multilingual(tmp_pmb_home, tmp_workspace_dir):
    eng = _warm_engine(tmp_workspace_dir, tmp_pmb_home)
    must = [
        ("I live in Tampa", "city", "Tampa"),
        ("Я живу в Киеве", "city", "Киеве"),            # RU inflected, raw
        ("Ich wohne in Berlin", "city", "Berlin"),
        ("I work at Stripe", "employer", "Stripe"),
        ("ich arbeite bei Google", "employer", "Google"),
        ("my name is Alex", "name", "Alex"),
        ("меня зовут Алексей", "name", "Алексей"),
        ("je m'appelle Pierre", "name", "Pierre"),
    ]
    hits = sum(1 for sent, a, v in must if _has(extract_keyed_anchor(sent, eng), a, v))
    # 8 strong cases across 4 languages — allow one embedder flake.
    assert hits >= 7, f"keyed extraction regressed: {hits}/8"


def test_negation_does_not_assert(tmp_pmb_home, tmp_workspace_dir):
    eng = _warm_engine(tmp_workspace_dir, tmp_pmb_home)
    # The anti-hypothesis ("no longer lives in") wins → no positive assertion.
    assert extract_keyed_anchor("I no longer live in Tampa", eng) == []


def test_non_self_sentences_extract_nothing(tmp_pmb_home, tmp_workspace_dir):
    eng = _warm_engine(tmp_workspace_dir, tmp_pmb_home)
    assert extract_keyed_anchor("the cat is sleeping on the sofa", eng) == []
    assert extract_keyed_anchor("we decided to use Postgres", eng) == []


def test_cold_engine_extracts_nothing(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
                 config_overrides={"recall.cache_size": 0})   # never warmed
    assert extract_keyed_anchor("I live in Tampa", eng) == []


def test_confidence_fields_present(tmp_pmb_home, tmp_workspace_dir):
    eng = _warm_engine(tmp_workspace_dir, tmp_pmb_home)
    cands = extract_keyed_anchor("I live in Tampa", eng)
    assert cands and cands[0].attr == "city"
    assert 0.0 < cands[0].pos <= 1.0       # F1 confidence input
    assert cands[0].margin > 0
