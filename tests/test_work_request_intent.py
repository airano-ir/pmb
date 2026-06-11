"""R4: a WORK REQUEST statement (no '?', no project) still surfaces lessons.

"tighten the retry logic" / "refactor the auth module" used to classify as
[SKIP] (no question mark, no known project), so the agent did real work with
ZERO surfaced lessons or decisions — the exact moment a rule like "use pnpm,
never npm" should fire. Now a work-verb / imperative fires Intent.WORK_REQUEST,
a non-SKIP intent, so the always-on lessons + decisions side-dish runs.
"""
from __future__ import annotations

import pytest

from pmb.core.engine import Engine
from pmb.hooks.auto_recall import Intent, detect_intents


def test_work_statement_is_not_skip():
    for msg in ("refactor the auth module",
                "tighten up the retry logic in the client",
                "implement the new pricing rules"):
        intents = detect_intents(msg, known_projects=set())
        assert Intent.WORK_REQUEST in intents, msg
        assert intents != [Intent.SKIP]


def test_chitchat_still_skips():
    intents = detect_intents("anyway that was a nice chat earlier", known_projects=set())
    assert intents == [Intent.SKIP]


@pytest.fixture
def eng(tmp_pmb_home, tmp_workspace_dir):
    return Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
                  config_overrides={"recall.cache_size": 0})


def test_work_request_surfaces_lessons(eng):
    from pmb.hooks import run_auto_context
    eng.record_fact("this repo uses pnpm, never npm for the auth module",
                    metadata={"kind": "lesson", "source": "lesson"})
    res = run_auto_context(eng, "refactor the auth module package install")
    assert not res.skipped
    assert Intent.WORK_REQUEST in res.intents
    assert res.lessons, "a work request must surface the relevant lesson"
    assert any("pnpm" in (L.get("content") or "") for L in res.lessons)
