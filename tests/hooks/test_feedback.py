"""Tests for recall feedback log and feedback-driven adaptive boost."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from pmb.core.engine import Engine
from pmb.health.adaptive import apply_feedback_adaptive
from pmb.health.feedback import (
    expected_ulid_boost_history,
    history,
    record_feedback,
    summary,
)


def test_record_useful_boosts_importance(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ulid = eng.remember("Q", "Postgres setup", importance=0.5)

    before = eng.events.get_by_ulid(ulid).importance
    record_feedback(eng, ulid, "useful")
    after = eng.events.get_by_ulid(ulid).importance
    assert after > before


def test_record_wrong_demotes_importance(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ulid = eng.remember("Q", "noise", importance=0.5)
    record_feedback(eng, ulid, "wrong")
    after = eng.events.get_by_ulid(ulid).importance
    assert after < 0.5


def test_record_with_expected_boosts_expected(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    wrong_ulid = eng.remember("Q", "wrong hit", importance=0.5)
    expected_ulid = eng.remember("Q2", "right answer", importance=0.5)

    before = eng.events.get_by_ulid(expected_ulid).importance
    record_feedback(eng, wrong_ulid, "wrong", query="postgres", expected_ulid=expected_ulid)
    after = eng.events.get_by_ulid(expected_ulid).importance
    assert after > before


def test_record_invalid_verdict_raises(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ulid = eng.remember("Q", "x")
    with pytest.raises(ValueError):
        record_feedback(eng, ulid, "maybe")


def test_record_nonexistent_ulid_raises(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    with pytest.raises(LookupError):
        record_feedback(eng, "0000000000000_baadf00d", "useful")


def test_record_nonexistent_expected_ulid_raises(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ulid = eng.remember("Q", "x")
    with pytest.raises(LookupError):
        record_feedback(eng, ulid, "wrong", expected_ulid="0000000000000_baadf00d")


def test_record_other_workspace_ulid_rejected(tmp_pmb_home):
    """ULID from another workspace must not be acceptable."""
    with tempfile.TemporaryDirectory() as a:
        with tempfile.TemporaryDirectory() as b:
            eng_a = Engine(cwd=Path(a), pmb_home=tmp_pmb_home)
            eng_b = Engine(cwd=Path(b), pmb_home=tmp_pmb_home)
            ulid_a = eng_a.remember("Q", "from A")
            # eng_b should not accept ulid_a as feedback target
            with pytest.raises(LookupError):
                record_feedback(eng_b, ulid_a, "useful")


def test_history_returns_entries(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    u1 = eng.remember("Q", "a")
    u2 = eng.remember("Q", "b")
    record_feedback(eng, u1, "useful", query="alpha")
    record_feedback(eng, u2, "wrong", query="beta")

    h = history(eng)
    assert len(h) == 2
    assert h[0].verdict == "useful"
    assert h[1].verdict == "wrong"


def test_summary_no_data(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    s = summary(eng)
    assert s["verdict"] == "no_data"
    assert s["total"] == 0


def test_summary_mixed(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    u = eng.remember("Q", "x")
    for _ in range(7):
        record_feedback(eng, u, "useful")
    for _ in range(3):
        record_feedback(eng, u, "wrong")

    s = summary(eng)
    assert s["total"] == 10
    assert s["useful"] == 7
    assert s["wrong"] == 3
    assert 0.6 < s["useful_rate"] < 0.8
    assert s["verdict"] == "healthy"


def test_feedback_adaptive_promotes_repeatedly_useful(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    u = eng.remember("Q", "stable fact", importance=0.5)
    for _ in range(3):
        record_feedback(eng, u, "useful")

    # apply_feedback_adaptive: 3 useful → promote to target 0.85
    result = apply_feedback_adaptive(eng)
    assert result["n_promoted_useful"] >= 1
    assert eng.events.get_by_ulid(u).importance >= 0.85


def test_feedback_adaptive_promotes_expected_ulid(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    wrong_u = eng.remember("Q", "noise", importance=0.5)
    expected_u = eng.remember("Q", "right answer", importance=0.5)
    record_feedback(eng, wrong_u, "wrong", expected_ulid=expected_u)
    record_feedback(eng, wrong_u, "wrong", expected_ulid=expected_u)

    result = apply_feedback_adaptive(eng)
    assert result["n_promoted_expected"] >= 1
    assert eng.events.get_by_ulid(expected_u).importance >= 0.90


def test_feedback_adaptive_demotes_repeatedly_wrong(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    u = eng.remember("Q", "noise", importance=0.6)
    # Record wrong feedback 3 times (each call also drops importance inline)
    for _ in range(3):
        record_feedback(eng, u, "wrong")

    # Now apply_feedback_adaptive should demote further
    before = eng.events.get_by_ulid(u).importance
    result = apply_feedback_adaptive(eng)
    after = eng.events.get_by_ulid(u).importance
    assert result["n_demoted_wrong"] >= 1
    assert after <= before


def test_feedback_adaptive_skips_pinned(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    u = eng.remember("Q", "important", importance=0.5)
    eng.pin(u)  # importance = 1.0
    for _ in range(5):
        record_feedback(eng, u, "wrong")

    apply_feedback_adaptive(eng)
    assert eng.events.get_by_ulid(u).importance >= 0.99


def test_expected_ulid_boost_history(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    wrong = eng.remember("Q", "wrong")
    expected = eng.remember("Q", "expected")
    record_feedback(eng, wrong, "wrong", expected_ulid=expected)
    record_feedback(eng, wrong, "wrong", expected_ulid=expected)
    counts = expected_ulid_boost_history(eng)
    assert counts[expected] == 2
