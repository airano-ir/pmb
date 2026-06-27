"""auto-recall intent routing: a project-named QUESTION must also fact-recall."""
from __future__ import annotations

from pmb.hooks.auto_recall import Intent, detect_intents


def test_project_named_question_also_triggers_fact_recall():
    """Regression: 'what is <project>'s X?' used to classify ONLY as
    PROJECT_OVERVIEW (project context), so the specific fact was never
    recalled. It must now ALSO fire GENERIC_FACTUAL so the answer surfaces."""
    intents = detect_intents("what is the pmb default timezone?", known_projects={"pmb"})
    assert Intent.PROJECT_OVERVIEW in intents
    assert Intent.GENERIC_FACTUAL in intents


def test_project_overview_without_question_stays_overview_only():
    """A project mention with NO question must NOT add a fact-recall."""
    intents = detect_intents("pmb project overview please", known_projects={"pmb"})
    assert Intent.PROJECT_OVERVIEW in intents
    assert Intent.GENERIC_FACTUAL not in intents


def test_plain_question_still_generic_factual():
    intents = detect_intents("what port did we pick for the database?", known_projects={"pmb"})
    assert Intent.GENERIC_FACTUAL in intents
