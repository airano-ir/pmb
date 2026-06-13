"""Tests for memory-quality signals: freshness (staleness) + confidence.

Pure functions, no Engine. These are display/capture signals - they must NOT
affect ranking, so tests here only assert the signal values.
"""
from __future__ import annotations

from pmb.memory_quality import (
    confidence_from,
    confidence_label,
    days_old,
    freshness_label,
    is_lesson_intent,
    is_stale,
)

DAY = 86400.0
NOW = 1_000_000_000.0


def test_days_old():
    assert days_old(NOW - 10 * DAY, NOW) == 10.0
    assert days_old(NOW + 100, NOW) == 0.0  # future clamps to 0


def test_fresh_recent_not_stale():
    assert is_stale(NOW - 10 * DAY, NOW, access_count=0) is False
    assert freshness_label(NOW - 10 * DAY, NOW) is None


def test_old_and_unused_is_stale():
    assert is_stale(NOW - 300 * DAY, NOW, access_count=0) is True
    lbl = freshness_label(NOW - 300 * DAY, NOW, access_count=0)
    assert lbl and "stale" in lbl


def test_old_but_frequently_used_not_stale():
    # accessed a lot -> still relevant even if old
    assert is_stale(NOW - 300 * DAY, NOW, access_count=10) is False


def test_freshness_age_formatting():
    assert "mo" in freshness_label(NOW - 200 * DAY, NOW, access_count=0)
    assert "y" in freshness_label(NOW - 400 * DAY, NOW, access_count=0)


def test_confidence_by_source():
    assert confidence_from({"source": "lesson"}) == 0.9
    assert confidence_from({"source": "cli-note"}) == 0.85
    assert confidence_from({"source": "chatgpt"}) == 0.65
    assert confidence_from({"actor": "agent"}) == 0.5
    assert confidence_from({}) == 0.6  # unknown default


def test_confidence_kind_failure_high():
    # a recorded failure is a reliable "don't do this"
    assert confidence_from({"source": "lesson", "kind": "failure"}) == 0.9


def test_confidence_explicit_override():
    assert confidence_from({"confidence": 0.33}) == 0.33


def test_confidence_label_buckets():
    assert confidence_label(0.9) == "high"
    assert confidence_label(0.65) == "med"
    assert confidence_label(0.4) == "low"


def test_lesson_intent_detection():
    assert is_lesson_intent("how should I set up the build") is True
    assert is_lesson_intent("what's the convention for imports here") is True
    # G3: RU lesson-intent ("какое правило …") is the warm query.lesson_intent
    # anchor now (test_statement_anchors); cold is_lesson_intent is EN-only.
    assert is_lesson_intent("what should we avoid in deploys") is True
    # NOT lesson-intent - plain factual/personal queries
    assert is_lesson_intent("where do I live") is False
    assert is_lesson_intent("why did we pick Postgres") is False
    assert is_lesson_intent("who is Alice") is False
