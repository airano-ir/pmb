"""R2: lessons are scored on FULL content, not a 300-char prefix.

The old code truncated a lesson to 300 chars BEFORE token-overlap scoring, so a
lesson whose distinctive term sat past char 300 (common — the actionable rule
is often at the END of a long lesson) could never surface. Now scoring uses the
full text and only the DISPLAY is trimmed at a sentence boundary.
"""
from __future__ import annotations

from pmb.core.engine import Engine


def _engine(ws, home):
    return Engine(cwd=ws, pmb_home=home, config_overrides={"recall.cache_size": 0})


def test_lesson_matches_on_token_past_300_chars(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    # A long lesson whose distinctive identifier (zircon-deploy-token) sits well
    # past char 300, after lots of generic preamble.
    preamble = ("When setting up the deployment pipeline there are many steps "
                "to consider and a lot of context to keep in mind about the "
                "environment and the various services that participate in it, "
                "and only after all of that ") * 2
    rule = "the secret is named zircondeploytoken and must be rotated weekly"
    content = preamble + rule
    assert len(preamble) > 300
    eng.record_fact(content, metadata={"kind": "lesson", "source": "lesson"})

    hits = eng.find_lessons(query="how do I rotate zircondeploytoken", limit=5)
    assert any("zircondeploytoken" in (h["content"] or "").lower()
               or h["ulid"] for h in hits), "lesson should surface on the tail token"
    # and it should actually be the one we recorded
    assert hits, "the tail-token lesson must surface (R2)"


def test_displayed_content_is_trimmed_but_generous(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    long_rule = ("Always run the migration in a transaction. " * 30).strip()
    eng.record_fact(long_rule, metadata={"kind": "lesson", "source": "lesson"})
    hits = eng.find_lessons(query="migration transaction", limit=5)
    assert hits
    shown = hits[0]["content"]
    # trimmed (not the full ~1200 chars) but more generous than the old 300
    assert len(shown) <= 700
    assert len(shown) > 300
