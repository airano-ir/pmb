"""Specificity gate for GENERIC_FACTUAL recall surfacing (_specificity_ok).

Separates real matches from 'same-domain but unhelpful' hits: a hit that
cleared the evidence floor is surfaced only if it shares a distinctive token
with the message OR is a strong absolute embedding match. Pure function, no
model - deterministic. Thresholds/behaviour were calibrated on the real corpus
(genuine hits: lexical overlap or raw_cosine>=~0.08; same-domain noise: neither).
"""
from __future__ import annotations

from pmb.hooks.auto_recall import _specificity_ok


def _hit(content, raw_cosine):
    return {"content": content, "signals": {"raw_cosine": raw_cosine}}


def test_disabled_when_strong_cosine_zero():
    # gate off -> everything passes (back-compat with the pre-gate behaviour)
    assert _specificity_ok("some message", _hit("unrelated", 0.0), 0.0) is True


def test_lexical_overlap_passes_even_at_low_cosine():
    # shares the distinctive token 'lancedb' -> specific, keep it
    h = _hit("we picked lancedb for the vector store", 0.05)
    assert _specificity_ok("which lancedb settings matter", h, 0.08) is True


def test_strong_cosine_passes_without_lexical_overlap():
    # no shared token, but the embedding match is strong -> trust it
    h = _hit("совершенно другой язык без общих слов", 0.12)
    assert _specificity_ok("strong semantic match only here", h, 0.08) is True


def test_same_domain_noise_is_suppressed():
    # the real failure mode: moderate cosine, ZERO distinctive overlap
    # (a Russian conversational message vs an English lesson) -> drop
    h = _hit("PMB on Windows: claude.CMD shim points at a deleted version", 0.060)
    assert _specificity_ok("это не причина в коде верно", h, 0.08) is False


def test_borderline_below_strong_and_no_overlap_suppressed():
    h = _hit("unrelated english lesson about daemons restarting", 0.079)
    assert _specificity_ok("почему долго как-то", h, 0.08) is False


def test_only_stopword_overlap_is_not_specific():
    # sharing only generic/stopword tokens is not real specificity
    h = _hit("the code and the file for this test", 0.05)
    assert _specificity_ok("the code and the file", h, 0.08) is False
