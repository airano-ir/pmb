"""Tests for typo-tolerant query correction (Improvement K)."""
from __future__ import annotations

from pmb.core.engine import Engine
from pmb.reasoning.typo_fix import (
    correct_query,
    levenshtein,
)

# ----------------------------------------------------------------------
# Levenshtein
# ----------------------------------------------------------------------

def test_levenshtein_identical_is_zero():
    assert levenshtein("alice", "alice") == 0


def test_levenshtein_single_typo():
    # "aliceee" → "alice" needs 2 deletions
    assert levenshtein("aliceee", "alice") == 2


def test_levenshtein_too_far_exits_early():
    # very different — should return > max_edits cheaply
    d = levenshtein("xenobiology", "alice", max_edits=2)
    assert d > 2


def test_levenshtein_typos():
    assert levenshtein("postgers", "postgres") == 2  # transposition: 2 edits
    assert levenshtein("clude", "claude") == 1
    assert levenshtein("docke", "docker") == 1


# ----------------------------------------------------------------------
# Query correction
# ----------------------------------------------------------------------

def test_correct_proper_noun_typo():
    known = [("person", "alice"), ("person", "bob"), ("tech", "postgres")]
    corrected, fixes = correct_query("who is Aliceee", known)
    assert "alice" in corrected.lower()
    assert "Aliceee" not in corrected
    assert len(fixes) == 1
    assert fixes[0].original == "Aliceee"
    assert fixes[0].corrected == "alice"


def test_correct_tech_typo():
    known = [("tech", "postgres"), ("tech", "redis")]
    corrected, fixes = correct_query("Postgers port 5433", known)
    assert "postgres" in corrected.lower()
    assert len(fixes) == 1


def test_no_correction_when_exact_match():
    known = [("person", "alice")]
    corrected, fixes = correct_query("who is alice", known)
    assert fixes == []
    assert corrected == "who is alice"


def test_no_correction_when_no_close_entity():
    """'unrelatedword' has no entity within edit distance 2."""
    known = [("person", "alice")]
    corrected, fixes = correct_query("what is unrelatedword", known)
    assert fixes == []


def test_multiple_corrections_in_one_query():
    known = [("person", "alice"), ("tech", "postgres")]
    corrected, fixes = correct_query("did Aliceee use Postgers?", known)
    assert "alice" in corrected.lower()
    assert "postgres" in corrected.lower()
    assert len(fixes) == 2


def test_short_tokens_skipped():
    """Tokens shorter than min_token_len (default 3) are skipped to avoid
    accidentally matching 'is' → 'it' style noise."""
    known = [("person", "alice")]
    corrected, fixes = correct_query("Is it OK?", known)
    assert fixes == []  # 'Is' is too short


def test_file_paths_in_entities_skipped():
    """Entities like 'src/api/main.py' shouldn't be candidates — they need
    exact match, not fuzzy."""
    known = [("file", "src/api/main.py"), ("person", "alice")]
    corrected, fixes = correct_query("Aliceee runs main.py", known)
    # Should match Alice but not even consider main.py
    fix_names = [f.corrected for f in fixes]
    assert "alice" in fix_names
    assert not any("/" in n for n in fix_names)


# ----------------------------------------------------------------------
# Engine integration
# ----------------------------------------------------------------------

def test_engine_recall_corrects_typo_in_query(tmp_pmb_home, tmp_workspace_dir):
    """End-to-end: store event about Alice, query 'Aliceee', still finds it."""
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={
            "recall.cache_size": 0,
            "recall.typo_correction": True,
            "recall.spreading_activation": False,
        },
    )
    # Alice gets added as a person entity via Improvement H
    target = eng.record_event(
        event_type="qa",
        content="I had coffee with Alice yesterday in Paris",
        metadata={"speaker": "user"},
    )
    eng.record_fact("Unrelated note about lunch")
    eng.record_fact("Some other distractor")

    # Query with double-typo — without correction this returns nothing relevant
    pack = eng.recall("who is Aliceee", top_k=5)
    ulids = [r.ulid for r in pack.results]
    assert target in ulids, (
        f"Aliceee should be corrected → alice → finds the event; got {ulids}"
    )


def test_engine_typo_correction_disabled(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={"recall.typo_correction": False},
    )
    eng.record_event(
        event_type="qa",
        content="I had coffee with Alice yesterday",
        metadata={"speaker": "user"},
    )
    # Without correction, "Aliceee" → no match → recall returns lower confidence
    pack = eng.recall("who is Aliceee", top_k=3)
    # Just verify it doesn't crash
    assert pack is not None
