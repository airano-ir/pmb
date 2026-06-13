"""C1 — universal value-span detector (no model, no language data)."""
from __future__ import annotations

from pmb.reasoning.spans import Span, looks_like_identifier, value_spans


def _texts(sentence):
    return {(s.text, s.kind) for s in value_spans(sentence)}


def test_proper_spans_multiscript_case_preserved():
    assert ("Tampa", "proper") in _texts("I live in Tampa now")
    assert ("Киеве", "proper") in _texts("Я живу в Киеве")        # inflected, raw
    assert ("München", "proper") in _texts("Ich wohne in München")


def test_acronyms_and_function_words_excluded():
    out = _texts("NASA is great")
    assert ("NASA", "proper") not in out          # all-caps acronym
    assert not any(t.lower() == "the" for t, _ in _texts("The cat sat"))


def test_number_and_date_spans():
    assert ("10:30", "number") in _texts("the meeting is at 10:30")
    assert ("1990", "number") in _texts("I was born in 1990")


def test_identifiers_never_become_values():
    out = _texts("I fixed record_batch using qwen2.5 and gpt4")
    assert not any("record_batch" in t for t, _ in out)
    assert not any(t == "qwen2.5" for t, _ in out)
    assert not any(t == "gpt4" for t, _ in out)
    assert looks_like_identifier("record_batch")
    assert looks_like_identifier("qwen2.5")
    assert looks_like_identifier("gpt4")
    assert not looks_like_identifier("Tampa")
    assert not looks_like_identifier("2026")


def test_dedupe_and_cap():
    spans = value_spans("Tampa Tampa Tampa", max_spans=8)
    assert sum(1 for s in spans if s.text == "Tampa") == 1
    assert all(isinstance(s, Span) for s in spans)
    assert len(value_spans(" ".join(f"Word{i}" for i in range(20)))) <= 8
