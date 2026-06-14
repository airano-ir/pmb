"""Query-worthiness gate (_looks_conversational): a GENERIC_FACTUAL message that
is really conversational/meta ('is it better now?') should surface nothing from
the Context recall channel. Pure function, deterministic. Thresholds calibrated
on a labelled set: gap<0.08 & conf<0.70 & zero lexical overlap == conversational,
catching the same-domain noise the cosine specificity gate cannot.
"""
from __future__ import annotations

from pmb.hooks.auto_recall import _looks_conversational

GAP, CONF = 0.08, 0.70


def _h(score, content):
    return {"score": score, "content": content}


def test_diffuse_low_conf_no_overlap_is_conversational():
    # the real case: "качество впрысков лучше стало?" -> off-topic LoadGuard hit,
    # tiny gap, low confidence, zero lexical overlap -> suppress.
    hits = [_h(0.66, "LoadGuard learning-loop rule: predicted NET vs gross"),
            _h(0.66, "another unrelated english lesson")]
    assert _looks_conversational("качество впрысков лучше стало?", hits, 0.56, GAP, CONF) is True


def test_lexical_anchor_keeps_it():
    hits = [_h(0.70, "we picked lancedb for the vector store"), _h(0.50, "x")]
    assert _looks_conversational("which lancedb settings matter", hits, 0.55, GAP, CONF) is False


def test_clear_winner_gap_keeps_it():
    hits = [_h(1.00, "some english fact"), _h(0.20, "x")]  # gap 0.80 >= 0.08
    assert _looks_conversational("совсем другой запрос", hits, 0.55, GAP, CONF) is False


def test_high_confidence_keeps_it():
    hits = [_h(0.70, "some english fact"), _h(0.69, "x")]  # tiny gap but confident
    assert _looks_conversational("совсем другой запрос", hits, 0.85, GAP, CONF) is False


def test_disabled_when_threshold_zero():
    hits = [_h(0.66, "english"), _h(0.66, "y")]
    assert _looks_conversational("болтовня", hits, 0.40, 0.0, CONF) is False
    assert _looks_conversational("болтовня", hits, 0.40, GAP, 0.0) is False


def test_no_hits_is_not_conversational():
    assert _looks_conversational("anything", [], 0.40, GAP, CONF) is False
