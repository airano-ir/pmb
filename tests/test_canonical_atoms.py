"""C3 — language-neutral 'attr: value' atomic facts for the user's own
current-state statements (gated; third parties keep the descriptive template)."""
from __future__ import annotations

from pmb.reasoning.fact_extract import extract_atomic_facts


def _contents(text, canonical):
    return [a.content for a in extract_atomic_facts(text, canonical=canonical)]


def test_canonical_off_keeps_localized_template():
    text = "I live in Tampa. I really enjoy the warm weather here every day."
    out = _contents(text, canonical=False)
    assert any("lives in tampa" in c.lower() for c in out)
    assert not any(c.startswith("city:") for c in out)


def test_canonical_on_user_location():
    text = "I live in Tampa. I really enjoy the warm weather here every day."
    assert "city: Tampa" in _contents(text, canonical=True)


def test_canonical_skips_third_party():
    text = "Alice lives in Paris. She really enjoys the museums there a lot."
    out = _contents(text, canonical=True)
    assert not any(c.startswith("city:") for c in out)
    assert any("Paris" in c for c in out)   # kept as a descriptive atom
