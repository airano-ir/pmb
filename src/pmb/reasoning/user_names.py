"""
User-name auto-detect — closes the "Where does <Name> live?" recall gap.

Background: in assistant memory, the user's own name lives in ONE fact
("my name is <Name>") and their location in ANOTHER ("I live in <City>").
A query like "Where does <Name> live?" can't be answered by either fact
alone — it needs a JOIN. PAMVR's strict-entity check killed the location
fact (no "<Name>") and the name fact (no "live").

Fix: at recall time, if the query proper noun matches a known user name
AND the candidate uses a first-person marker ("I", localized equivalents),
accept it. The user-fact "I live in <City>" answers "Where does <Name>
live?" when we know <Name> = the user.

Cheap implementation:
  - At Engine init / refresh, scan facts for name-declaration patterns
    ("my name is X" and the localized equivalents). Cache the set of names.
  - PAMVR's entity check, when penalising a missed proper noun, checks
    if (a) noun is in user_names AND (b) candidate has first-person marker.
    If both true → match instead of penalty.

The name-declaration regexes and first-person markers for RU/UK live in the
built-in ru.yaml / uk.yaml packs (L1); English is inline above.

Cost: one regex pass over all active facts at refresh. ~10ms on 1000
events. Set lookup at query time = O(1).
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from pmb import lang as _lang

# Patterns that declare a name: extract group "name". The English patterns are
# inline; the RU/UK ones live in the built-in ru.yaml / uk.yaml packs under
# `name_statement_patterns` (L1 — keeps this module Cyrillic-free). Order is
# irrelevant; all patterns are tried.
_NAME_PATTERNS = [
    re.compile(r"\bmy\s+name\s+is\s+(?P<name>[A-Z][\w]+)", re.IGNORECASE),
    re.compile(r"\bi\s+am\s+(?P<name>[A-Z][a-z]{2,})\b"),
    re.compile(r"\bI'?m\s+(?P<name>[A-Z][a-z]{2,})\b"),
    re.compile(r"\buser'?s?\s+name\s+is\s+(?P<name>[A-Z][\w]+)", re.IGNORECASE),
] + _lang.compile_patterns("name_statement_patterns")


# Self-reference markers in content — when candidate has these, it's a
# first-person fact about the user. EN inline; RU/UK reflexives from the packs.
SELF_MARKERS = {
    "i", "i'm", "im", "ive", "my", "i've", "myself",
} | _lang.merged_set("self_markers", set())


# Explicit reflexives for the first-person regex; EN inline + RU/UK from packs
# (`self_re_markers`). Whole-word alternation, so order does not matter.
_SELF_ALTS = ["i", "i'm", "i've", "my", "myself"] + sorted(
    _lang.merged_set("self_re_markers", set()))
_SELF_RE = re.compile(r"\b(?:" + "|".join(_SELF_ALTS) + r")\b", re.IGNORECASE)


def detect_user_names(events: list[str]) -> set[str]:
    """Run name patterns over a list of fact contents. Returns lowercased
    names extracted from name-declaration facts."""
    out: set[str] = set()
    for text in events:
        if not text:
            continue
        for pat in _NAME_PATTERNS:
            for m in pat.finditer(text):
                name = m.group("name")
                if name and len(name) >= 3:
                    out.add(name.lower())
    return out


def looks_like_name_statement(content: str) -> bool:
    """True if CONTENT declares a user name ("My name is X" or a localized
    equivalent).
    Cheap regex check used by the write path to mark the user-name cache dirty
    so a freshly-recorded name takes effect on the very NEXT recall instead of
    waiting for the periodic refresh."""
    if not content:
        return False
    return any(p.search(content) for p in _NAME_PATTERNS)


def mine_user_names_from_db(db_path: Path, limit: int = 1000) -> set[str]:
    """Scan recent active fact events for user-name declarations.
    Cheap (~10ms / 1000 events). Called from Engine on warmup / refresh."""
    if not db_path.exists():
        return set()
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                "SELECT content FROM events "
                "WHERE archived_at IS NULL "
                "AND event_type IN ('fact', 'qa') "
                "ORDER BY ulid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        contents = [r[0] or "" for r in rows]
        return detect_user_names(contents)
    except Exception:
        return set()


def has_self_marker(text: str) -> bool:
    """True if the text contains a first-person reference. Used by
    PAMVR's entity check to rescue candidates that miss the proper noun
    but ARE about the user."""
    if not text:
        return False
    return bool(_SELF_RE.search(text))
