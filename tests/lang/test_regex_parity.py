"""L1 regex-relocation parity gate.

Relocating RU/UK regex fragments out of the .py modules into the built-in lang
packs must keep the PUBLIC matching behavior identical. The baseline
(`_regex_parity_baseline.json`) was captured from the modules BEFORE any regex
moved to a pack. These tests replay the same probes and assert the same
outputs — so a botched relocation fails loudly instead of silently breaking
Russian/Ukrainian recall.
"""
from __future__ import annotations

import json
from pathlib import Path

_FIXTURES = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()
) / "tests" / "fixtures"

_BASE = json.loads(
    (_FIXTURES / "_regex_parity_baseline.json").read_text(encoding="utf-8")
)


def _norm(v):
    """Round-trip through JSON so tuples (e.g. detect_current_state's return)
    compare equal to the list form stored in the baseline."""
    return json.loads(json.dumps(v, ensure_ascii=False))


def test_user_names_behavior_unchanged():
    import pmb.reasoning.user_names as U
    b = _BASE["user_names"]
    assert {p: sorted(U.detect_user_names([p])) for p in b["detect"]} == b["detect"]
    assert {p: U.looks_like_name_statement(p) for p in b["looks"]} == b["looks"]
    assert {p: bool(U._SELF_RE.search(p)) for p in b["self_re"]} == b["self_re"]
    assert sorted(U.SELF_MARKERS) == b["self_markers"]


def test_query_split_behavior_unchanged():
    import pmb.reasoning.query_split as Q
    b = _BASE["query_split"]
    fn = getattr(Q, "split_compound_query", None) or getattr(Q, "split_query", None)
    assert fn is not None
    for probe, expected in b.items():
        assert fn(probe) == expected, f"query_split drift on {probe!r}"


def test_intents_unchanged():
    from pmb.hooks.auto_recall import detect_intents
    b = _BASE["intents"]
    for probe, expected in b.items():
        got = detect_intents(probe, known_projects=set())
        assert got == expected, f"intent drift on {probe!r}: {got} != {expected}"


def test_fact_extract_behavior_unchanged():
    import json as _json
    base = _json.loads((_FIXTURES / "_fact_extract_baseline.json")
                       .read_text(encoding="utf-8"))
    import pmb.reasoning.fact_extract as FE
    for probe, expected in base.items():
        facts = FE.extract_atomic_facts(probe)
        got = sorted([f.content, f.kind] for f in facts)
        assert got == expected, f"fact_extract drift on {probe!r}: {got} != {expected}"


def test_future_intent_unchanged():
    # G3: _FUTURE_INTENT_RE is EN-inline now (RU markers lived in the deleted
    # pack); RU future-intent is the warm statement.future_intent anchor
    # (test_statement_anchors). Pin the EN cold behavior here.
    from pmb.reasoning.attributes import looks_like_future_intent
    yes = ["next we'll refactor auth", "plan: ship v1 by June", "to-do: write docs",
           "we will migrate the db", "let's add caching"]
    no = ["I live in Tampa", "the api runs on port 5432", "we use pnpm not npm"]
    for s in yes:
        assert looks_like_future_intent(s), f"should be future intent: {s!r}"
    for s in no:
        assert not looks_like_future_intent(s), f"should NOT be future intent: {s!r}"


def test_pamvr_self_reference_matchers_unchanged():
    # G3: _FIRST_PERSON / _SELF_INTENT_RE / _RELATION_MARKERS are the EN inline
    # floor now (RU/UK fragments lived in the deleted packs); RU/UK self-reference
    # rides the warm anchor tier. Pin the EN cold behavior here.
    import pmb.reasoning.pamvr as P
    fp_yes = ["I live here", "my city", "myself"]
    fp_no = ["the server runs", "Alice lives in Berlin", "weather is nice"]
    for s in fp_yes:
        assert P._has_first_person(s), f"first-person miss: {s!r}"
    for s in fp_no:
        assert not P._has_first_person(s), f"first-person false hit: {s!r}"
    si_yes = ["where do i live", "what's my name", "what do i prefer"]
    si_no = ["where does Alice live", "what is the capital of France"]
    for s in si_yes:
        assert P._SELF_INTENT_RE.search(s.lower()), f"self-intent miss: {s!r}"
    for s in si_no:
        assert not P._SELF_INTENT_RE.search(s.lower()), f"self-intent false hit: {s!r}"
    for w in ["friend", "wife", "brother"]:
        assert w in P._RELATION_MARKERS, f"relation marker missing: {w!r}"


def test_recall_personal_intent_gate_unchanged():
    # G3: _QWORD_RE / _ATTR_RE are the EN inline floor now (RU qwords/first-person
    # lived in the deleted packs); RU personal-attribute queries ride the warm
    # anchor tier. Pin the EN cold R6 gate here.
    import pmb.core.engine.recall as R
    for w in ["where", "when", "how"]:
        assert R._QWORD_RE.search(w), f"qword miss: {w!r}"
    for w in ["i", "my", "user"]:
        assert R._ATTR_RE.search(w), f"attr miss: {w!r}"
    assert R._QWORD_RE.search("where do i live") and R._ATTR_RE.search("where do i live")
    # topical query: no first-person/user cue -> attr gate must NOT fire
    assert not R._ATTR_RE.search("how does postgres handle vacuum")


def test_attributes_behavior_unchanged():
    import pmb.reasoning.attributes as A
    b = _BASE["attributes"]
    assert _norm({p: A.detect_current_state(p) for p in b["detect_current_state"]}) \
        == b["detect_current_state"]
    assert _norm({p: A.detect_negated_state(p) for p in b["detect_negated_state"]}) \
        == b["detect_negated_state"]
    assert _norm({p: A.has_user_subject_cue(p) for p in b["has_user_subject_cue"]}) \
        == b["has_user_subject_cue"]
