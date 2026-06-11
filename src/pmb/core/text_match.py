"""Shared lexical-matching primitives for lesson relevance.

Two places judge whether a lesson relates to a piece of text:
  • `find_lessons` (surface side) — should this lesson surface for this message?
  • `hooks.followcheck` (confirm side) — did the agent's work follow the lesson?

They must use the SAME yardstick, or we get the worst case: a lesson surfaces
on a loose match (one common word) but then can never be confirmed followed,
so it just drags the adherence denominator down. Centralising the tokenizer,
the stopword set, and the "strong token" test here keeps both sides symmetric.

Pure stdlib (`re`), no embeddings — cheap enough for the per-turn hot path.
"""

from __future__ import annotations

import re

from pmb.reference_data import extend_set as _extend_set

# Tokens too common to be evidence of anything. Multilingual stop-ish set +
# PMB-generic words + high-frequency English fillers + filesystem-path noise
# (users / appdata / local / …) that otherwise reads as a "strong" identifier
# token and produces coincidental matches on Windows paths in messages.
STOPWORDS: frozenset[str] = frozenset({
    # English fillers / function words
    "the", "and", "for", "with", "that", "this", "from", "not", "but", "use",
    "using", "used", "via", "per", "you", "your", "our", "its", "are", "was",
    "were", "has", "have", "had", "will", "would", "should", "must", "never",
    "always", "into", "onto", "than", "then", "else", "when", "what", "which",
    "already", "both", "before", "after", "first", "last", "next", "different",
    "default", "defaults", "same", "other", "another", "every", "each", "some",
    "any", "all", "more", "most", "less", "only", "just", "also", "still",
    "here", "there", "where", "while", "until", "they", "them", "their",
    "make", "made", "need", "want", "like", "such", "very", "much", "many",
    # (Russian/Ukrainian stop-ish words — incl. "pmb" spelled in Cyrillic —
    # live in the packs under text_match_stopwords and merge in below, keeping
    # this module Cyrillic-free.)
    # PMB-generic — appear in almost every lesson, no discriminating power
    "pmb", "lesson", "rule", "agent", "code", "file", "test", "run", "tests",
    "thing", "things", "stuff", "case", "cases", "time", "times", "work",
    # Filesystem-path noise (Windows + generic) — coincidental on any path
    "users", "appdata", "local", "roaming", "program", "programdata", "temp",
    "tmp", "windows", "documents", "desktop", "onedrive", "home", "path",
    "folder", "directory", "drive",
})

# Per-deployment extension: reference.yaml `stopwords`, then active language
# packs (extend-only — both no-ops without the respective files).
STOPWORDS = _extend_set("stopwords", STOPWORDS)
from pmb import lang as _lang  # noqa: E402

STOPWORDS = _lang.merged_set("stopwords", STOPWORDS)
# RU/UK lesson-keyword stopwords relocated out of the inline set (L1) — a
# DEDICATED category so they never bleed into pamvr's `stopwords` consumer.
STOPWORDS = _lang.merged_set("text_match_stopwords", STOPWORDS)

# A "distinctive" token: a word char start (incl. unicode), then word chars /
# - . — so identifiers like record_batch / qwen2.5 / paraphrase-multilingual
# survive intact. Length and stopword filtering happen in distinctive_tokens.
# The Cyrillic-block range lives in the ru pack (cyrillic_script_range) so this
# module is Cyrillic-free; with an active ru pack it reproduces the U+0400-04FF
# range the inline class used to spell out.
_TOKEN = re.compile(
    r"[A-Za-z0-9_À-ɏ" + "".join(str(x) for x in _lang.merged_list("cyrillic_script_range"))
    + r"][\w\-.]{2,}", re.UNICODE)


def is_strong(tok: str) -> bool:
    """A 'strong' token is specific enough that a coincidental match is
    unlikely: a long word (>= 7 chars) or an identifier (contains _ . - or a
    digit, e.g. record_batch / qwen2.5 / paraphrase-multilingual). Stopwords
    are never strong, so path noise added to STOPWORDS can't sneak back in."""
    if tok in STOPWORDS:
        return False
    if len(tok) >= 7:
        return True
    if any(c in tok for c in ("_", ".", "-")):
        return True
    if any(c.isdigit() for c in tok):
        return True
    return False


def distinctive_tokens(text: str) -> set[str]:
    """Lower-cased, length->=4, non-stopword, non-pure-digit tokens — the set
    we measure overlap on.

    Identifiers stay intact (record_batch, qwen2.5). For HYPHENATED compounds
    we keep BOTH the whole token and its parts (lesson-surfacing -> also
    'surfacing'; cold-start -> also 'cold', 'start'). Hyphens join compound
    words, so without the split a query word like 'surfacing' would miss a
    'lesson-surfacing' lesson — the previous \\W+ tokenizer split these, and
    keeping hyphens for identifiers must not silently drop compound-word
    recall. Underscores (identifier-internal: record_batch) are NOT split."""
    out: set[str] = set()

    def _emit(t: str) -> None:
        t = t.strip("-._")
        if len(t) >= 4 and not t.isdigit() and t not in STOPWORDS:
            out.add(t)

    for m in _TOKEN.finditer((text or "").lower()):
        tok = m.group(0).strip("-._")
        if not tok:
            continue
        _emit(tok)
        if "-" in tok:
            for part in tok.split("-"):
                _emit(part)
    return out


def relevance(query: str, content: str) -> tuple[int, int]:
    """(distinctive overlap count, strong overlap count) between two texts."""
    q = distinctive_tokens(query)
    c = distinctive_tokens(content)
    overlap = q & c
    strong = sum(1 for t in overlap if is_strong(t))
    return len(overlap), strong


def is_relevant(query: str, content: str, *, min_overlap: int = 2) -> bool:
    """Symmetric relevance bar: the lesson relates to the query if they share
    >= `min_overlap` distinctive tokens OR >= 1 strong (identifier-grade)
    token. One common word is NOT enough — that's the low bar that floods
    surfacing with noise. An empty query (no distinctive tokens) is treated as
    'no opinion' → True, so callers that want recency-only still get results."""
    q = distinctive_tokens(query)
    if not q:
        return True
    c = distinctive_tokens(content)
    overlap = q & c
    if any(is_strong(t) for t in overlap):
        return True
    return len(overlap) >= min_overlap
