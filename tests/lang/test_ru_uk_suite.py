"""
P0-2 Item 2: comprehensive RU/UK regression suite for the personal-
assistant scenarios from the Alternix reviewer.

Coverage:
  A. Name declensions       (Олексій / Олексія / Олексію, Алексей / Алексея)
  B. Mixed RU+UK in one set (cross-lingual recall)
  C. Multiple friends       (similar facts, must not confuse)
  D. Fact corrections       (keyed-upsert end-to-end)
  E. "Don't answer"         (off-topic questions must not surface
                              irrelevant personal facts)

Run:
    pytest tests/test_ru_uk_suite.py -v
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest


@pytest.fixture
def voice_engine():
    """Fresh Engine configured the way a voice-assistant integrator would:
    multilingual embedder, atomic extract on, PAMVR on, no cache."""
    from pmb.core.engine import Engine
    tmp_home = Path(tempfile.mkdtemp())
    tmp_ws = Path(tempfile.mkdtemp())
    os.environ["PMB_HOME"] = str(tmp_home)
    eng = Engine(
        cwd=tmp_ws, pmb_home=tmp_home, rerank_model=None,
        config_overrides={
            "recall.cache_size": 0,
            "recall.pamvr_enabled": True,
            "write.atomic_fact_extract": True,
        },
    )
    _ = eng.search.model
    eng.warmup(with_first_query=False)
    yield eng
    try:
        eng.close()
    except Exception:
        pass


# ------------------------------------------------------------------
# A. NAME DECLENSIONS
# ------------------------------------------------------------------


def test_ru_name_declensions(voice_engine):
    """User refers to 'Алексей' by different cases. Recall must find
    the same fact via any declension."""
    eng = voice_engine
    eng.record_batch([{
        "type": "fact",
        "content": "Мой друг Алексей работает программистом в Stripe.",
    }])
    time.sleep(0.6)
    queries = [
        "Кто такой Алексей",          # nominative
        "Что ты знаешь про Алексея",  # accusative
        "Расскажи об Алексее",        # prepositional
    ]
    hits = 0
    for q in queries:
        pack = eng.recall(q, top_k=3)
        contents = " ".join(r.content for r in pack.results).lower()
        if "алексей" in contents:
            hits += 1
    assert hits >= 2, (
        f"only {hits}/3 declensions surfaced Алексей. Multilingual model "
        f"should handle Russian case endings."
    )


def test_uk_name_declensions(voice_engine):
    """Ukrainian declensions: Олексій / Олексія / Олексію."""
    eng = voice_engine
    eng.record_batch([{
        "type": "fact",
        "content": "Друг користувача Олексій працює програмістом.",
    }])
    time.sleep(0.6)
    queries = [
        "Хто такий Олексій",
        "Що знаєш про Олексія",
        "Розкажи про Олексія",
    ]
    hits = 0
    for q in queries:
        pack = eng.recall(q, top_k=3)
        contents = " ".join(r.content for r in pack.results).lower()
        if "олексій" in contents:
            hits += 1
    assert hits >= 2, f"only {hits}/3 UK declensions hit"


# ------------------------------------------------------------------
# B. MIXED RU+UK CROSS-LINGUAL
# ------------------------------------------------------------------


def test_ru_query_finds_uk_fact(voice_engine):
    """User stored fact in UK, asks in RU. Multilingual embedder bridges."""
    eng = voice_engine
    eng.record_batch([{
        "type": "fact",
        "content": "Мій день народження 7 червня.",
    }])
    time.sleep(0.6)
    pack = eng.recall("когда у меня день рождения", top_k=3)
    contents = " ".join(r.content for r in pack.results).lower()
    assert "червня" in contents or "июня" in contents or "7" in contents, (
        f"RU query didn't find UK birthday fact. Contents: {contents[:100]}"
    )


def test_uk_query_finds_ru_fact(voice_engine):
    """Stored in RU, asked in UK."""
    eng = voice_engine
    eng.record_batch([{
        "type": "fact",
        "content": "Я работаю инженером в компании Acme.",
    }])
    time.sleep(0.6)
    pack = eng.recall("де я працюю", top_k=3)
    contents = " ".join(r.content for r in pack.results).lower()
    assert "инженер" in contents or "acme" in contents or "работа" in contents, (
        f"UK query missed RU work fact: {contents[:100]}"
    )


def test_mixed_language_query(voice_engine):
    """One query mixes RU and UK words — should still work."""
    eng = voice_engine
    eng.record_batch([
        {"type": "fact", "content": "Меня зовут Алексей."},
        {"type": "fact", "content": "Я живу у Києві."},
    ])
    time.sleep(0.6)
    # Mixed: "кто" = RU, "Києві" = UK
    pack = eng.recall("кто живёт у Києві", top_k=3)
    contents = " ".join(r.content for r in pack.results).lower()
    # Should surface at least one of the two facts
    assert "алексей" in contents or "києв" in contents, (
        f"mixed-language query failed: {contents[:100]}"
    )


# ------------------------------------------------------------------
# C. MULTIPLE FRIENDS — disambiguation
# ------------------------------------------------------------------


def test_two_friends_not_confused(voice_engine):
    """Two friends with similar facts. Query about one must not surface
    the other's fact at top-1."""
    eng = voice_engine
    eng.record_batch([
        {"type": "fact", "content": "Мой друг Алексей работает программистом."},
        {"type": "fact", "content": "Мой друг Борис работает врачом."},
    ])
    time.sleep(0.6)
    pack_a = eng.recall("Кем работает Алексей", top_k=3)
    pack_b = eng.recall("Кем работает Борис", top_k=3)
    top_a = pack_a.results[0].content.lower() if pack_a.results else ""
    top_b = pack_b.results[0].content.lower() if pack_b.results else ""
    assert "алексей" in top_a and "программист" in top_a, (
        f"Алексей's top-1 wrong: {top_a}"
    )
    assert "борис" in top_b and "врач" in top_b, (
        f"Борис's top-1 wrong: {top_b}"
    )


def test_multiple_residence_facts(voice_engine):
    """Different people at different cities. Each query finds the right
    person, not a random one."""
    eng = voice_engine
    eng.record_batch([
        {"type": "fact", "content": "Я живу в Киеве."},
        {"type": "fact", "content": "Мой друг Алексей живёт в Варшаве."},
        {"type": "fact", "content": "Моя сестра живёт в Берлине."},
    ])
    time.sleep(0.6)
    pack = eng.recall("где живёт Алексей", top_k=3)
    top1 = pack.results[0].content.lower() if pack.results else ""
    assert "алексей" in top1 and "варшав" in top1, (
        f"Алексей residence not at top-1: {top1}"
    )


# ------------------------------------------------------------------
# D. FACT CORRECTIONS — keyed-upsert end-to-end
# ------------------------------------------------------------------


def test_residence_correction_keyed_only(voice_engine):
    """Using the keyed-upsert API: only current value surfaces."""
    eng = voice_engine
    eng.record_keyed_fact("user", "residence", "Киев")
    eng.record_keyed_fact("user", "residence", "Варшава")
    time.sleep(0.6)
    pack = eng.recall("где живёт пользователь", top_k=5)
    contents = " ".join(r.content for r in pack.results).lower()
    assert "варшав" in contents, f"current 'Варшава' missing: {contents[:100]}"
    assert "киев" not in contents, f"archived 'Киев' surfaced: {contents[:100]}"


def test_history_preserves_old_value(voice_engine):
    """Old fact is archived but recoverable via get_keyed_fact_history."""
    eng = voice_engine
    eng.record_keyed_fact("user", "job", "developer")
    eng.record_keyed_fact("user", "job", "manager")
    hist = eng.get_keyed_fact_history("user", "job")
    assert len(hist) == 2
    current = [h for h in hist if h["is_current"]][0]
    archived = [h for h in hist if not h["is_current"]][0]
    assert current["value"] == "manager"
    assert archived["value"] == "developer"


def test_three_step_correction(voice_engine):
    """Three sequential updates. Each step archives the active prior;
    history grows to 3 entries (2 archived + 1 current)."""
    eng = voice_engine
    eng.record_keyed_fact("user", "residence", "Киев")
    eng.record_keyed_fact("user", "residence", "Варшава")
    r3 = eng.record_keyed_fact("user", "residence", "Берлин")
    # Each upsert archives the active prior (1 prior at a time).
    assert len(r3["superseded_ulids"]) == 1, (
        f"third upsert archives the one active prior (Варшава), got "
        f"{len(r3['superseded_ulids'])}"
    )
    # But the history endpoint sees ALL three values
    hist = eng.get_keyed_fact_history("user", "residence")
    assert len(hist) == 3, f"history should have 3 entries: {hist}"
    current = [h for h in hist if h["is_current"]]
    archived = [h for h in hist if not h["is_current"]]
    assert len(current) == 1
    assert current[0]["value"] == "Берлин"
    assert len(archived) == 2
    assert {a["value"] for a in archived} == {"Киев", "Варшава"}


# ------------------------------------------------------------------
# E. "DON'T ANSWER" — off-topic questions
# ------------------------------------------------------------------
# Reviewer note: an assistant must distinguish "answer from memory" vs
# "answer from general knowledge". The retrieval layer can't fully do
# this, but it should NOT promote unrelated personal facts to top-1
# when the query is generic.


def test_off_topic_does_not_promote_personal_fact(voice_engine):
    """Personal facts must not surface at top-1 for general questions."""
    eng = voice_engine
    eng.record_batch([
        {"type": "fact", "content": "Меня зовут Алексей."},
        {"type": "fact", "content": "Я живу в Киеве."},
        {"type": "fact", "content": "Мне 30 лет."},
    ])
    time.sleep(0.6)
    # Generic factual question — gold answer is "Earth orbits the Sun"
    # NOT in our memory. Top-1 should be low-confidence or unrelated.
    pack = eng.recall("сколько планет в солнечной системе", top_k=3)
    if not pack.results:
        return  # acceptable — no match
    top_score = pack.results[0].score
    # If we surface ANY personal fact, its score should be LOW
    # (heuristic threshold; the goal is "don't fabricate confidence")
    if top_score > 0.6:
        # If high confidence, it must at least share content tokens
        # with the query — not just be the most-recent personal fact
        top_content = pack.results[0].content.lower()
        query_tokens = {"планет", "солнечной", "системе", "сколько"}
        if not any(t[:5] in top_content for t in query_tokens):
            pytest.fail(
                f"off-topic query promoted irrelevant personal fact at "
                f"top-1 with score {top_score:.2f}: {top_content}"
            )


# ------------------------------------------------------------------
# F. ATOMIC EXTRACTION COVERAGE (RU + UK in one go)
# ------------------------------------------------------------------


@pytest.mark.skip(reason="G3: RU/UK atomic-fact extraction removed with the "
                  "packs; RU/UK facts recalled whole via vector, keyed attrs via "
                  "warm C2 anchors.")
def test_paragraph_extracts_ru_uk_atoms():
    """Single function-level test (no engine) — sanity check that the
    pattern bank handles a representative mix."""
    from pmb.reasoning.fact_extract import extract_atomic_facts
    text = (
        # Russian
        "Меня зовут Алексей. Я живу в Киеве. Мой день рождения 7 июня. "
        "Я люблю спокойные игры. У меня кошка Мурка."
        # Ukrainian
        " Мене звати Олексій. Я живу у Києві. Мій день народження 7 червня."
    )
    facts = extract_atomic_facts(text, min_len=20, min_sentences=1)
    kinds = {f.kind for f in facts}
    # At least one RU + one UK pattern fires
    assert any(k.startswith("ru_") for k in kinds)
    assert any(k.startswith("uk_") for k in kinds)
    # Identity in both languages
    assert "ru_identity" in kinds or "uk_identity" in kinds
    # Locations cover Cyrillic place names
    assert any("location" in k for k in kinds)
    # Birthday recognised
    assert any("birthday" in k for k in kinds)
