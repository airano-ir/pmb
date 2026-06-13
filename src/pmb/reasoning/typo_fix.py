"""
Multi-algorithm typo-tolerant query correction (Improvements K + L).

5-layer cascade - each layer catches a different type of typo:

  1. EXACT match              - fast path, query token == entity name
  2. SUBSTRING containment    - "Aliceee" contains "alice", or "alic" ⊂ "alice"
  3. LEVENSHTEIN ≤ 2          - 1-2 char edits (insert/delete/substitute/transpose)
  4. TRIGRAM Jaccard ≥ 0.55   - catches 3-4 char mistakes, robust on long words
  5. SOUNDEX                  - phonetic match for radical mis-spellings

Algorithms are ordered cheapest → most expensive. Cascade stops at first
strong match. If multiple algorithms find candidates, we pick the one with
the highest confidence score.

Cost on 100 entities: ~0.5-1.5ms per query (microseconds per algorithm × 5
× 100 entities). Negligible.

For 10k+ entities we'd index trigrams in a inverted index - TODO. For
typical user workspaces (100-5000 entities) linear is fine.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass

log = logging.getLogger(__name__)


# A token to consider for correction - looks like a name (capitalized in
# query) or 3+ char word.
_TOKEN_RE = re.compile(r"\b[A-Za-z][a-zA-Z'\-]{2,29}\b")


# Function words we should never auto-correct - they're query scaffolding,
# not entity references. Stops things like 'who' → 'how' just because they
# share 2 letters with some entity.
_QUERY_FUNCTION_WORDS = {
    "what", "where", "when", "why", "how", "who", "whom", "whose", "which",
    "the", "and", "but", "for", "with", "from", "into", "onto", "upon",
    "this", "that", "these", "those", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "can", "shall",
    "any", "all", "some", "many", "much", "few", "more", "most", "less",
    "least", "than", "then", "now", "here", "there", "yes", "no", "not",
    "also", "only", "very", "just", "even", "still", "yet", "again",
    "ever", "never", "always", "often", "sometimes", "usually",
    "tell", "give", "show", "find", "make", "take", "see", "look",
    "about", "above", "below", "after", "before", "during", "since",
    "until", "without", "within", "between", "among",
    "you", "your", "yours", "they", "them", "their", "theirs",
    "his", "her", "hers", "its", "ours", "mine", "myself", "yourself",
}


@dataclass
class Correction:
    original: str
    corrected: str
    edits: int                  # Levenshtein distance for backward compat
    entity_kind: str | None  # which entity kind matched
    method: str = "lev"         # which algorithm produced the match

    def __str__(self) -> str:
        return f"{self.original} -> {self.corrected} ({self.method}, d={self.edits})"


# ======================================================================
# Algorithm 1: Levenshtein
# ======================================================================

def levenshtein(a: str, b: str, max_edits: int = 3) -> int:
    """Compute Levenshtein distance with early-exit if it exceeds max_edits.

    Returns the distance, or max_edits + 1 if larger. Standard DP, O(len(a)*len(b)).
    """
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_edits:
        return max_edits + 1
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr[0] = i
        row_min = curr[0]
        for j in range(1, len(b) + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j-1] + 1,
                prev[j-1] + cost,
            )
            if curr[j] < row_min:
                row_min = curr[j]
        if row_min > max_edits:
            return max_edits + 1
        prev, curr = curr, prev
    return prev[len(b)]


# ======================================================================
# Algorithm 2: Substring containment
# ======================================================================

def substring_score(query_tok: str, entity: str) -> float:
    """1.0 if one contains the other (case-insensitive); 0 otherwise.
    Asymmetric weighted - longer-contains-shorter is stronger signal."""
    q, e = query_tok.lower(), entity.lower()
    if not q or not e:
        return 0.0
    if q == e:
        return 1.0
    # Require shorter side >= 4 chars to avoid "is" ⊂ "list" false positives
    short = min(len(q), len(e))
    if short < 4:
        return 0.0
    if e in q:
        return 0.95  # entity inside query token (Aliceee contains alice)
    if q in e:
        return 0.85  # query token inside entity (alic inside alice)
    return 0.0


# ======================================================================
# Algorithm 3: Trigram Jaccard similarity
# ======================================================================

def trigrams(s: str) -> set:
    """Generate character trigrams of s, with boundary markers."""
    s = "_" + s.lower() + "_"
    return {s[i:i+3] for i in range(len(s) - 2)}


def trigram_jaccard(a: str, b: str) -> float:
    """Set Jaccard over trigrams. 1.0 = identical; 0.0 = no overlap."""
    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


# ======================================================================
# Algorithm 4: Soundex (phonetic)
# ======================================================================

_SOUNDEX_MAP = {
    'b': '1', 'f': '1', 'p': '1', 'v': '1',
    'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
    'd': '3', 't': '3',
    'l': '4',
    'm': '5', 'n': '5',
    'r': '6',
}


def soundex(s: str) -> str:
    """Russell Soundex code. Returns 4-char code like 'A420' for 'Alice'."""
    if not s:
        return ""
    s = s.lower()
    first = s[0].upper()
    digits = []
    prev = _SOUNDEX_MAP.get(s[0], '0')
    for ch in s[1:]:
        d = _SOUNDEX_MAP.get(ch, '0')
        if d == '0':
            # Vowel or h/w → reset previous digit
            prev = '0'
            continue
        if d != prev:
            digits.append(d)
            prev = d
    code = first + ''.join(digits)
    # Pad / truncate to 4 chars
    return (code + "0000")[:4]


def soundex_match(a: str, b: str) -> bool:
    return soundex(a) == soundex(b)


# ======================================================================
# Combined matcher
# ======================================================================

@dataclass
class FuzzyMatch:
    name: str
    kind: str
    confidence: float      # 0..1
    method: str            # which algorithm produced it
    edits: int = 0


def find_best_match(
    query_tok: str,
    candidates: list[tuple[str, str]],  # (name_lower, kind)
    levenshtein_budget: int = 2,
    trigram_threshold: float = 0.55,
) -> FuzzyMatch | None:
    """Run 5-layer cascade and return the single best match (if any).

    Layer priority (lower number = stronger signal):
       1 exact          confidence 1.00
       2 substring      confidence 0.85-0.95
       3 Levenshtein    confidence 1.0 - edits*0.15
       4 trigram        confidence == jaccard score
       5 soundex        confidence 0.55 (weakest)
    """
    q = query_tok.lower()
    if not q:
        return None

    best: FuzzyMatch | None = None

    def _maybe_update(name: str, kind: str, conf: float, method: str, edits: int = 0):
        nonlocal best
        if best is None or conf > best.confidence:
            best = FuzzyMatch(name=name, kind=kind, confidence=conf, method=method, edits=edits)

    # Layer 1: exact
    for name, kind in candidates:
        if name == q:
            return FuzzyMatch(name=name, kind=kind, confidence=1.0, method="exact", edits=0)

    # Layer 2: substring
    for name, kind in candidates:
        s = substring_score(q, name)
        if s > 0:
            _maybe_update(name, kind, s, "substring", edits=abs(len(q) - len(name)))
    if best and best.confidence >= 0.95:
        return best

    # Layer 3: Levenshtein
    for name, kind in candidates:
        d = levenshtein(q, name, max_edits=levenshtein_budget)
        if d <= levenshtein_budget and d > 0:
            conf = max(0.6, 1.0 - d * 0.15)
            _maybe_update(name, kind, conf, "lev", edits=d)
    if best and best.confidence >= 0.85:
        return best

    # Layer 4: trigram Jaccard (good for longer words with multiple errors)
    if len(q) >= 5:
        for name, kind in candidates:
            if len(name) < 4:
                continue
            jac = trigram_jaccard(q, name)
            if jac >= trigram_threshold:
                _maybe_update(name, kind, jac, "trigram", edits=999)
    if best and best.confidence >= 0.7:
        return best

    # Layer 5: Soundex (last resort for radical phonetic mismatches)
    if len(q) >= 4:
        q_sx = soundex(q)
        for name, kind in candidates:
            if len(name) < 4:
                continue
            if soundex(name) == q_sx:
                # Demand at least SOME char overlap to avoid total nonsense
                if len(set(q) & set(name)) >= 2:
                    _maybe_update(name, kind, 0.55, "soundex", edits=999)

    return best


# ======================================================================
# Multi-word entity matching helper
# ======================================================================

def _build_candidate_index(
    known_entities: Iterable[tuple[str, str]],
) -> tuple[list[tuple[str, str]], dict]:
    """Return (single_word_candidates, multi_word_lookup).

    Multi-word lookup maps (word1, word2) → (full_name, kind) so a query
    pair like 'mary anne' can match the entity 'mary anne'.
    """
    single: list[tuple[str, str]] = []
    multi: dict = {}
    for kind, name in known_entities:
        if not name or len(name) < 3:
            continue
        if "/" in name or "\\" in name:
            continue  # skip file paths
        low = name.lower().strip()
        single.append((low, kind))
        parts = low.split()
        if len(parts) == 2:
            multi[(parts[0], parts[1])] = (low, kind)
        elif len(parts) > 2:
            # Index first two words too - partial matching
            multi[(parts[0], parts[1])] = (low, kind)
    return single, multi


# ======================================================================
# Top-level: correct_query
# ======================================================================

def correct_query(
    query: str,
    known_entities: Iterable[tuple[str, str]],
    max_edits: int = 2,
    min_token_len: int = 3,
    trigram_threshold: float = 0.55,
) -> tuple[str, list[Correction]]:
    """Apply typo corrections to a query. Returns (new_query, corrections).

    5-algorithm cascade (see find_best_match). Handles:
      - 1-2 char typos (Levenshtein)
      - Substring (Aliceee, alic, ALICE)
      - 3-4 char mistakes on long words (trigram)
      - Phonetic mismatch (soundex)
      - Multi-word entities (mary anne)

    Function words (who/what/the/etc) are protected.
    """
    if not query:
        return query, []

    candidates, multi_word = _build_candidate_index(known_entities)
    if not candidates:
        return query, []

    canonical_set = {n for n, _ in candidates}

    # Find tokens with their positions (for multi-word matching)
    tokens = []
    for m in _TOKEN_RE.finditer(query):
        tokens.append((m.start(), m.end(), m.group(0)))

    corrections: list[Correction] = []
    # Map of original token → replacement (apply at end to avoid index shifting)
    replacements: list[tuple[str, str]] = []

    # First pass: multi-word entity matching (consecutive pairs)
    matched_indices = set()
    for i in range(len(tokens) - 1):
        a_text = tokens[i][2].lower()
        b_text = tokens[i+1][2].lower()
        if (a_text, b_text) in multi_word:
            # Exact multi-word match - nothing to correct
            matched_indices.add(i)
            matched_indices.add(i + 1)

    # Second pass: per-token fuzzy matching
    for idx, (start, end, orig) in enumerate(tokens):
        if idx in matched_indices:
            continue
        low = orig.lower()
        if len(low) < min_token_len:
            continue
        if low in _QUERY_FUNCTION_WORDS:
            continue
        if low in canonical_set:
            continue  # already matches exactly
        match = find_best_match(
            low, candidates,
            levenshtein_budget=max_edits,
            trigram_threshold=trigram_threshold,
        )
        if match and match.confidence >= 0.55:
            corrections.append(Correction(
                original=orig,
                corrected=match.name,
                edits=match.edits,
                entity_kind=match.kind,
                method=match.method,
            ))
            replacements.append((orig, match.name))

    # Apply all replacements (whole-word boundary)
    new_query = query
    for orig, repl in replacements:
        pattern = re.compile(r"\b" + re.escape(orig) + r"\b")
        new_query = pattern.sub(repl, new_query, count=1)

    return new_query, corrections
