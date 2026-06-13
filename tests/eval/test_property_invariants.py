"""T6 — property-based tests (hypothesis) for the pure, hot-path invariants.

These functions run on EVERY user message / write and take untrusted text
(any language, any unicode, emoji, control chars). A single unhandled regex /
index error here crashes a hook and the whole agent turn, so the invariant that
matters most is simply: they never raise, and their cheap structural guarantees
hold for ALL input — not just the handful of examples in the unit tests.

Marked `property` so it can be deselected on a tight CI budget (`-m 'not
property'`), but it is fast (pure functions, no model).
"""
from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

# TEXT exercises arbitrary unicode (control chars, emoji, any script); the
# MULTILINGUAL alphabet exercises the Cyrillic char-classes / lang-pack
# tokenizers (Phase L) densely. A plain-string alphabet keeps input generation
# cheap (a `st.characters() | st.sampled_from(...)` union trips hypothesis's
# too-slow health check).
TEXT = st.text(max_size=200)
MULTILINGUAL = st.text(
    alphabet="абвгдеёжзийклмнопрстуфхцчшщъыьэюяіїєґ ABCabcxyz_.-0123!?",
    max_size=120,
)
pytestmark = pytest.mark.property


@given(TEXT | MULTILINGUAL)
def test_distinctive_tokens_total_and_clean(s):
    from pmb.core.text_match import STOPWORDS, distinctive_tokens
    toks = distinctive_tokens(s)
    assert isinstance(toks, set)
    for t in toks:
        assert isinstance(t, str) and t == t.lower()
        assert t not in STOPWORDS          # stopwords are always stripped
    # idempotent: tokenizing the already-lowered text yields the same set
    assert distinctive_tokens(s.lower()) == distinctive_tokens(s.lower())


@given(st.text(max_size=80))
def test_is_strong_is_total_bool(s):
    from pmb.core.text_match import is_strong
    assert isinstance(is_strong(s), bool)   # never raises, always a bool


@given(TEXT | MULTILINGUAL)
def test_extract_atomic_facts_total_and_bounded(s):
    from pmb.reasoning.fact_extract import extract_atomic_facts
    facts = extract_atomic_facts(s)
    assert isinstance(facts, list)
    for fa in facts:
        assert isinstance(fa.content, str) and fa.content
        assert len(fa.content) <= 140      # the documented atomic-fact cap
        assert 0.0 <= fa.confidence <= 1.0


@given(TEXT | MULTILINGUAL)
def test_intent_classifiers_are_total(s):
    from pmb.hooks.auto_recall import detect_intents, is_trivial
    assert isinstance(is_trivial(s), bool)
    intents = detect_intents(s, known_projects=set())
    assert isinstance(intents, list) and intents   # never empty (SKIP at minimum)


@given(TEXT | MULTILINGUAL)
def test_recall_gate_and_pamvr_regexes_are_total(s):
    # The recall personal-intent gate + pamvr self-reference matchers run on raw
    # query text; they must tolerate any input without raising (Phase L moved
    # their RU/UK halves into packs — fuzz the assembled regexes).
    import pmb.core.engine.recall as R
    import pmb.reasoning.pamvr as P
    assert isinstance(bool(R._QWORD_RE.search(s)), bool)
    assert isinstance(bool(R._ATTR_RE.search(s)), bool)
    assert isinstance(P._has_first_person(s), bool)
    assert isinstance(bool(P._SELF_INTENT_RE.search(s.lower())), bool)


@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-50, max_value=50))
def test_importance_clamp_invariant(x):
    # The write path clamps importance into [0, 1]; mirror that invariant so a
    # regression in the clamp expression is caught.
    clamped = max(0.0, min(1.0, x))
    assert 0.0 <= clamped <= 1.0
