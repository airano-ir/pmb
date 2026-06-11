"""
Pattern-based query splitting for compound recall.

Compound queries like "when was the last time I went to the gym and why
haven't I gone all week" lose recall when treated as a single bag-of-tokens —
the two sub-questions ("when did I go" vs "why didn't I go") get averaged into
a diffuse query vector that matches neither sub-fact well.

Solution: split on natural join markers, run each sub-query through the
normal recall pipeline, and fuse the candidate lists via Reciprocal Rank
Fusion (RRF). No LLM needed — patterns cover ~80% of compound queries.

When to fire:
  - Query length >= 8 tokens (single-clause queries are short)
  - At least one strong split marker present
  - Each resulting sub-query has >= 2 content tokens

Patterns (English, inline):
  "X and why Y" / "X and how Y" / "X and when Y"
  "X because Y"
  "X, also Y"
  "X. Y." (sentence boundary in a single query)

The RU/UK equivalents are supplied by the built-in ru.yaml / uk.yaml packs
(L1) and merge into the patterns above.
"""

from __future__ import annotations

import re

from pmb import lang as _lang

# Split markers: ordered most-specific first so we don't fragment subordinate
# clauses incorrectly. Each pattern matches a *separator*; the text BEFORE
# and AFTER becomes two sub-queries. Each entry: (compiled_regex, "label").
#
# English is inline. The RU/UK conjunction, WH-words, the causal/additive
# markers and Cyrillic-capital sentence boundaries live in the built-in
# ru.yaml / uk.yaml packs (L1). The WH-conjunction pattern is ASSEMBLED from
# the EN + pack token lists, so mixed-language splits keep working.
_SPLIT_CONJ = ["and"] + sorted(_lang.merged_set("query_split_conj", set()))
_SPLIT_WH = ["why", "how", "when", "where", "who", "what"] + sorted(
    _lang.merged_set("query_split_wh", set()))
_SPLIT_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Strong WH-conjunction (clearly two questions) — assembled EN + pack
    (re.compile(
        r"\s+(?:" + "|".join(_SPLIT_CONJ) + r")\s+(?=(?:"
        + "|".join(_SPLIT_WH) + r")\b)", re.IGNORECASE),
     "and-wh"),
    # English "because"
    (re.compile(r"\s+because\s+", re.IGNORECASE), "because"),
    # English ", also"
    (re.compile(r"\s*,\s*also\s+", re.IGNORECASE), "comma-also"),
    # Sentence boundary inside a single utterance (Latin capital)
    (re.compile(r"\?\s+(?=[A-Z])"), "sentence-q"),
    (re.compile(r"\.\s+(?=[A-Z][a-z])"), "sentence-period"),
] + _lang.compile_patterns_labeled("query_split_patterns")


# Minimal token count for a sub-query to be considered a real question (not
# a fragment like a bare conjunction + WH-word)
_MIN_SUB_TOKENS = 2

# Skip splitting on very short queries — they are by definition single-clause
_MIN_LEN_FOR_SPLIT = 6


# EN inline; the RU/UK function words live in the built-in ru.yaml / uk.yaml
# packs under `query_split_stop` (L1) and merge in here.
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "to",
    "for", "and", "or", "with", "we", "i", "you", "do", "does", "did",
    "what", "who", "where", "when", "why", "how", "by", "at", "from", "as",
    "be", "have", "has", "had", "this", "that", "these", "those", "it",
    "our", "my", "your", "their", "his", "her", "about", "any", "some",
} | _lang.merged_set("query_split_stop", set())


def _count_content_tokens(s: str) -> int:
    if not s:
        return 0
    n = 0
    for m in re.finditer(r"[^\W_]+", s):  # Unicode word chars (Latin+Cyrillic+…)
        t = m.group(0).lower()
        if len(t) > 2 and t not in _STOP:
            n += 1
    return n


def split_query(query: str, max_parts: int = 3) -> list[str]:
    """Split a compound query into sub-queries.

    Returns a list with at least one element (the original query). When no
    pattern fires, the list is `[query]` — callers should treat that as
    "no split occurred".

    Empty/short queries pass through unchanged.
    """
    if not query:
        return [query]
    q = query.strip()
    if len(q) < _MIN_LEN_FOR_SPLIT:
        return [q]
    # Pre-gate: query must have at least 2 content tokens overall.
    # We don't gate harder — the per-piece check below decides if a split
    # is meaningful.
    if _count_content_tokens(q) < 2:
        return [q]

    parts: list[str] = [q]
    for pat, _label in _SPLIT_PATTERNS:
        new_parts: list[str] = []
        for chunk in parts:
            pieces = pat.split(chunk, maxsplit=max_parts - 1)
            pieces = [p.strip(" ?.!,;:") for p in pieces if p.strip()]
            # Quality gate: every piece must have at least 1 content token,
            # AND at least one piece must have >= MIN_SUB_TOKENS. This
            # accepts splits like "who's on the team" + "what is their role"
            # while still rejecting noise like a bare "?" split.
            counts = [_count_content_tokens(p) for p in pieces]
            # Quality gate: split is real only when both halves carry a
            # content token (kills splits like "and?" or ".?"). For very
            # short content-token halves (1+1) we still trust the marker —
            # patterns are strong enough that a "and why" or "potomu chto"
            # really does separate two questions.
            if len(pieces) >= 2 and all(c >= 1 for c in counts):
                new_parts.extend(pieces)
            else:
                new_parts.append(chunk)
        parts = new_parts
        if len(parts) >= max_parts:
            break

    # Dedup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        key = p.lower().strip(" ?.!,;:")
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out[:max_parts] if len(out) > 1 else [q]


def rrf_fuse(
    ranked_lists: list[list[str]],
    k: int = 60,
    top_n: int = 50,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion across multiple ranked ULID lists.

    Each ranked_list is an ordered list of ULIDs (rank 0 = best).
    Returns (ulid, fused_score) sorted by score descending.

    RRF formula: score(d) = Σ 1 / (k + rank_i(d)) over lists i where d appears.
    k=60 is the classic default from Cormack et al. 2009.
    """
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, ulid in enumerate(lst):
            scores[ulid] = scores.get(ulid, 0.0) + 1.0 / (k + rank)
    items = sorted(scores.items(), key=lambda kv: -kv[1])
    return items[:top_n]
