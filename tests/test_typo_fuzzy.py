"""Tests for multi-algorithm fuzzy matcher (Improvement L)."""
from __future__ import annotations

from pmb.reasoning.typo_fix import (
    correct_query,
    find_best_match,
    soundex,
    substring_score,
    trigram_jaccard,
)

# ----------------------------------------------------------------------
# Individual algorithms
# ----------------------------------------------------------------------

def test_substring_catches_extra_chars():
    """Aliceee contains 'alice' as substring → score ~0.95."""
    assert substring_score("aliceee", "alice") > 0.9


def test_substring_catches_short_query():
    """alic is substring of alice → score ~0.85."""
    assert substring_score("alic", "alice") > 0.8


def test_substring_no_false_positive():
    """Random unrelated → 0."""
    assert substring_score("postgres", "alice") == 0.0


def test_substring_too_short_rejected():
    """'is' inside 'list' shouldn't fire."""
    assert substring_score("is", "list") == 0.0


def test_trigram_finds_partial_overlap():
    """Postgresql vs Postgres — should be high but < 1."""
    j = trigram_jaccard("postgresql", "postgres")
    assert 0.6 < j < 0.95


def test_trigram_distant_words():
    """Unrelated words → near 0."""
    assert trigram_jaccard("postgres", "kubernetes") < 0.2


def test_soundex_alice():
    """'Alice' Soundex code."""
    assert soundex("alice")[0] == "A"
    assert len(soundex("alice")) == 4


def test_soundex_phonetic_match():
    """'Carolyn' and 'Caroline' should soundex-match (both K-R-L-N)."""
    assert soundex("carolyn") == soundex("caroline")


def test_soundex_different():
    assert soundex("postgres") != soundex("alice")


# ----------------------------------------------------------------------
# Best-match cascade
# ----------------------------------------------------------------------

def test_cascade_picks_exact_over_fuzzy():
    candidates = [("alice", "person"), ("alike", "concept")]
    m = find_best_match("alice", candidates)
    assert m.confidence == 1.0
    assert m.method == "exact"


def test_cascade_substring_catches_extra_chars():
    candidates = [("alice", "person")]
    m = find_best_match("aliceee", candidates)
    assert m is not None
    assert m.name == "alice"
    assert m.method == "substring"


def test_cascade_levenshtein_for_small_typos():
    candidates = [("postgres", "tech")]
    m = find_best_match("posgres", candidates)
    assert m is not None
    assert m.name == "postgres"
    assert m.method == "lev"


def test_cascade_trigram_for_long_typo():
    """Postgresql → postgres via trigram (Lev distance = 2 should also catch
    actually; verifying multiple algorithms work)."""
    candidates = [("postgres", "tech")]
    m = find_best_match("postgresql", candidates)
    assert m is not None
    assert m.name == "postgres"
    # Could be 'substring' (postgres in postgresql) OR 'trigram'
    assert m.method in ("substring", "trigram", "lev")


def test_cascade_radical_typo_handled():
    """3 char difference — should still be caught by trigram or substring."""
    candidates = [("caroline", "person")]
    m = find_best_match("carolyne", candidates)
    assert m is not None
    assert m.name == "caroline"


def test_cascade_phonetic_match():
    """Karolyn → Caroline via Soundex (when other layers miss)."""
    candidates = [("caroline", "person")]
    m = find_best_match("karolyne", candidates)
    # Could match via trigram or soundex
    assert m is not None
    assert m.name == "caroline"


def test_cascade_no_false_positive_on_unrelated():
    candidates = [("alice", "person"), ("postgres", "tech")]
    m = find_best_match("xenomorph", candidates)
    # No real match — confidence should be low if anything
    assert m is None or m.confidence < 0.6


# ----------------------------------------------------------------------
# Query-level corrections (full pipeline)
# ----------------------------------------------------------------------

def test_query_4_char_typo():
    """Aliceeee (3 extra) handled by substring or trigram."""
    known = [("person", "alice")]
    out, fixes = correct_query("who is Aliceeee", known)
    assert "alice" in out.lower()
    assert len(fixes) == 1


def test_query_transposition():
    """Postgers → Postgres (single transposition = 2 edits)."""
    known = [("tech", "postgres")]
    out, fixes = correct_query("Postgers port 5433", known)
    assert "postgres" in out.lower()


def test_query_long_word_typo():
    """'aithentication' → 'authentication' (Lev=2 or trigram)."""
    known = [("function", "authentication")]
    out, fixes = correct_query("How does aithentication work?", known)
    assert "authentication" in out.lower()


def test_query_protects_function_words():
    """'who' must NOT be 'corrected' just because it's close to entity 'how'."""
    known = [("concept", "how"), ("person", "alice")]
    out, fixes = correct_query("who is alice", known)
    assert "who" in out  # preserved
    # Only Aliceee-like typos should fire; alice is exact, no change


def test_query_multiple_corrections():
    """Multiple typos in one query — all fixed."""
    known = [("person", "alice"), ("tech", "postgres")]
    out, fixes = correct_query("Aliceee uses Postgres-ql", known)
    assert "alice" in out.lower()
    assert "postgres" in out.lower()
    assert len(fixes) >= 2


def test_query_multi_word_entity():
    """Two-word entity 'mary anne' should match consecutive query tokens."""
    known = [("person", "mary anne"), ("person", "bob")]
    out, fixes = correct_query("did mary anne come?", known)
    # Exact match shouldn't produce corrections
    assert fixes == []


def test_query_no_typos_no_corrections():
    known = [("person", "alice")]
    out, fixes = correct_query("alice came home", known)
    assert fixes == []
    assert out == "alice came home"


def test_query_handles_capitalization():
    known = [("person", "alice")]
    out, fixes = correct_query("ALICEEEE went there", known)
    assert "alice" in out.lower()


# ----------------------------------------------------------------------
# Backward compat — existing 'Correction' field signature
# ----------------------------------------------------------------------

def test_correction_has_method_field():
    known = [("person", "alice")]
    _, fixes = correct_query("Aliceee", known)
    assert len(fixes) == 1
    assert hasattr(fixes[0], "method")
    assert fixes[0].method in ("exact", "substring", "lev", "trigram", "soundex")
