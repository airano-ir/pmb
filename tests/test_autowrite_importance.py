"""Importance-aware ambient auto-write: record IMPORTANT work, not just any.

The earlier ambient writer journaled any turn with >=2 significant actions and
tagged every entry a flat importance 0.4 — so 'edited two files' landed in
memory looking exactly as important as 'shipped and verified a fix'. These
tests pin the fix:

  • a git-free importance score (outcome signals drive it, raw edits barely
    move it; commits/git are deliberately NOT a factor),
  • a quality bar in the gate (mechanical churn is skipped),
  • outcome-first synthesis ('fixed a failing run; edited ...' not 'edited ...'),
  • the recorded entry carries its REAL importance so recall ranks it right.

Pure SQL + template, no model.
"""

from __future__ import annotations

import pytest

from pmb.hooks.autowrite import (
    classify_actions,
    score_turn_importance,
    synthesize_template,
    autowrite_gate,
    run_autowrite,
)

_BAR = 0.45  # the default autowrite.min_importance


def _act(tool, target="", status="ok", significant=True):
    return {"tool": tool, "target": target, "status": status,
            "significant": significant, "timestamp": 0.0}


# ── scoring: outcome drives importance, edits barely do ───────────────────


def test_edit_only_is_below_the_bar():
    score, sig = score_turn_importance([_act("Edit", "a.py"), _act("Edit", "b.py")])
    assert score < _BAR            # 'just info' → would be skipped
    assert sig["n_files"] == 2
    assert not sig["tests_passed"]


def test_edit_plus_tests_passed_clears_bar():
    score, sig = score_turn_importance(
        [_act("Bash", "pytest tests/", "ok"), _act("Edit", "a.py")])
    assert score >= _BAR
    assert sig["tests_passed"]


def test_deploy_is_important():
    score, sig = score_turn_importance(
        [_act("Bash", "kubectl apply -f k8s/", "ok"), _act("Edit", "a.py")])
    assert score >= _BAR
    assert sig["deploy"]


def test_red_then_green_is_resolution_and_high():
    """pytest fails → edit → pytest passes (newest-first order)."""
    actions = [
        _act("Bash", "pytest", "ok"),      # newest: passing re-run
        _act("Edit", "fix.py", "ok"),
        _act("Bash", "pytest", "1"),       # oldest: the failure
    ]
    score, sig = score_turn_importance(actions)
    assert sig["error_resolution"] is True
    assert sig["tests_passed"] is True
    assert score >= 0.6                     # fix + green is clearly important


def test_single_failure_is_not_resolution():
    """One failing test and no re-run is NOT a resolved failure."""
    actions = [_act("Bash", "pytest", "1"), _act("Edit", "x.py", "ok")]
    _, sig = score_turn_importance(actions)
    assert sig["error_resolution"] is False
    assert sig["tests_failed"] is True
    assert sig["tests_passed"] is False


def test_last_run_wins_for_pass_fail():
    """Ends red even though it was green earlier."""
    actions = [
        _act("Bash", "pytest", "1"),       # newest: failing
        _act("Bash", "pytest", "ok"),      # earlier: passing
    ]
    _, sig = score_turn_importance(actions)
    assert sig["tests_failed"] is True
    assert sig["tests_passed"] is False


def test_magnitude_adds_a_little():
    big = [_act("Edit", f"f{i}.py") for i in range(6)]
    small = [_act("Edit", "a.py"), _act("Edit", "b.py")]
    assert score_turn_importance(big)[0] > score_turn_importance(small)[0]


def test_commit_does_not_raise_importance():
    """git is git-free here: a commit is significant but adds no importance."""
    with_commit = [_act("Edit", "a.py"),
                   _act("Bash", "git commit -m wip", "ok")]
    without = [_act("Edit", "a.py")]
    assert score_turn_importance(with_commit)[0] == score_turn_importance(without)[0]
    assert score_turn_importance(with_commit)[0] < _BAR


def test_empty_is_zero():
    assert score_turn_importance([])[0] == 0.0
    assert score_turn_importance([_act("Read", "x", significant=False)])[0] == 0.0


# ── synthesis leads with the outcome ──────────────────────────────────────


def test_synthesis_leads_with_passed_tests():
    s = synthesize_template([_act("Bash", "pytest", "ok"), _act("Edit", "auth.py")])
    assert s.startswith("Ran tests (passed)")
    assert "auth.py" in s


def test_synthesis_leads_with_fix():
    s = synthesize_template([
        _act("Bash", "pytest", "ok"),
        _act("Edit", "fix.py", "ok"),
        _act("Bash", "pytest", "1"),
    ])
    assert "fixed a failing run" in s.lower()
    # no contradictory 'still failing' / redundant 'passed' alongside the fix
    assert "still failing" not in s.lower()


def test_synthesis_edit_only_fallback():
    s = synthesize_template([_act("Edit", "a.py"), _act("Edit", "b.py")])
    assert "file(s)" in s


# ── gate + run: the bar is enforced, real importance is recorded ──────────


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("PMB_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PMB_WORKSPACE", "imp_e2e")
    from pmb.core.engine import Engine
    return Engine()


def test_gate_skips_mechanical_churn(engine):
    engine.record_agent_action("Edit", "a.py", "ok")
    engine.record_agent_action("Edit", "b.py", "ok")
    # count passes (2 >= 2) but the importance bar (0.45) does not (~0.30)
    reason = autowrite_gate(engine, window_minutes=60, min_actions=2,
                            min_importance=_BAR)
    assert reason and "low importance" in reason


def test_gate_passes_with_outcome(engine):
    engine.record_agent_action("Edit", "a.py", "ok")
    engine.record_agent_action("Bash", "pytest tests/", "ok",
                               command="pytest tests/")
    assert autowrite_gate(engine, window_minutes=60, min_actions=2,
                          min_importance=_BAR) is None


def test_run_records_real_importance(engine):
    engine.record_agent_action("Edit", "a.py", "ok")
    engine.record_agent_action("Bash", "pytest tests/", "ok",
                               command="pytest tests/")
    res = run_autowrite(engine, window_minutes=60, min_actions=2,
                        min_importance=_BAR, synthesizer="template", apply=True)
    assert res.wrote is True
    assert res.importance >= _BAR          # not the old flat 0.4
    assert res.importance != 0.4
    acts = engine.recent_activity(minutes=60, limit=5)
    assert any((a.get("content") or "").startswith("Ran tests") for a in acts)


def test_run_skips_low_importance_turn(engine):
    engine.record_agent_action("Edit", "a.py", "ok")
    engine.record_agent_action("Edit", "b.py", "ok")
    res = run_autowrite(engine, window_minutes=60, min_actions=2,
                        min_importance=_BAR, synthesizer="template", apply=True)
    assert res.wrote is False
    assert "low importance" in (res.skipped_reason or "")


def test_bar_zero_keeps_old_permissive_behaviour(engine):
    """min_importance=0 (or unset) → count gate only, mechanical turns write."""
    engine.record_agent_action("Edit", "a.py", "ok")
    engine.record_agent_action("Edit", "b.py", "ok")
    res = run_autowrite(engine, window_minutes=60, min_actions=2,
                        synthesizer="template", apply=True)  # no min_importance
    assert res.wrote is True
