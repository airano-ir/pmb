"""Action-time REPEAT guard: fire "you were corrected on this before" at the
MOMENT the agent is about to do it - the close on 'guard fired != agent obeyed'.

Built on the correction-capture corpus (drafts + failures), which find_lessons
deliberately excludes. The guard matches the about-to-run tool excerpt against
that corpus with the same strong bar as the lesson guard, once per session.
"""
from __future__ import annotations

import pytest

from pmb.core.engine import Engine
from pmb.mcp.daemon import pretool_negatives


@pytest.fixture
def eng(tmp_pmb_home, tmp_workspace_dir):
    e = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
               config_overrides={"recall.cache_size": 0})
    # A captured correction (the locate-me pain) + an explicit failure.
    e.capture_correction(
        "снова location field заполнил руками вместо locate me на greenhouse",
        severity="strong")
    e.record_fact(
        "tried submitting the greenhouse form before validating fields, it failed",
        metadata={"kind": "failure", "source": "lesson"})
    # A plain lesson that must NOT show up in the negative corpus.
    e.record_fact("commit messages in imperative mood",
                  metadata={"kind": "lesson", "source": "lesson"})
    return e


# ── find_negative_memories ──────────────────────────────────────────────────

def test_negative_corpus_has_correction_and_failure_not_plain_lesson(eng):
    out = eng.find_negative_memories("", limit=20)
    kinds = {it["kind"] for it in out}
    assert "correction" in kinds
    assert "failure" in kinds
    # the plain lesson ("imperative mood") must not be in the negative corpus
    assert not any("imperative mood" in it["content"] for it in out)


def test_negative_corpus_matches_on_complaint_text_not_boilerplate(eng):
    # querying with the user's own words should surface the correction draft
    out = eng.find_negative_memories("location field locate me greenhouse", limit=5)
    assert out and out[0]["kind"] == "correction"
    # matched on the raw complaint, exposed as match_text
    assert "locate me" in out[0]["match_text"].lower()


# ── pretool_negatives (the action-time guard) ───────────────────────────────

def test_guard_fires_when_about_to_repeat_locate_me(eng):
    excerpt = "Bash: fill the location field by typing the city on greenhouse"
    fired = pretool_negatives(eng, excerpt, set())
    assert fired, "the guard must fire when the action matches a past correction"
    assert any("locate me" in (L.get("match_text") or "").lower() for L in fired)


def test_guard_fires_on_failure_match(eng):
    excerpt = "click submit on the greenhouse form now"
    fired = pretool_negatives(eng, excerpt, set())
    assert any(L["kind"] == "failure" for L in fired)


def test_guard_once_per_session(eng):
    seen: set = set()
    first = pretool_negatives(eng, "type the city into the location field greenhouse", seen)
    assert first
    again = pretool_negatives(eng, "again location field greenhouse city typing", seen)
    assert not any(L["ulid"] in {f["ulid"] for f in first} for L in again), \
        "a fired item must not re-fire in the same session"


def test_guard_silent_on_unrelated_action(eng):
    assert not pretool_negatives(eng, "Read the project README", set())
    assert not pretool_negatives(eng, "Bash ls -la /tmp", set())
