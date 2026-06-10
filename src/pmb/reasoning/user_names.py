"""
User-name auto-detect — closes the "Where does Алексей live?" recall gap.

Background: in assistant memory, the user's own name lives in ONE fact
("Меня зовут Алексей") and their location in ANOTHER ("Я живу в Киеве").
A query like "Where does Алексей live?" can't be answered by either fact
alone — it needs a JOIN. PAMVR's strict-entity check killed the location
fact (no "Алексей") and the name fact (no "live").

Fix: at recall time, if the query proper noun matches a known user name
AND the candidate uses first-person ("Я", "I", "мене", …), accept it.
The user-fact "Я живу в Киеве" answers "Where does Алексей live?" when
we know Алексей = the user.

Cheap implementation:
  - At Engine init / refresh, scan facts for "Меня зовут X" / "My name is X"
    / "Я Х" / "Мене звати X" patterns. Cache the set of names.
  - PAMVR's entity check, when penalising a missed proper noun, checks
    if (a) noun is in user_names AND (b) candidate has first-person marker.
    If both true → match instead of penalty.

Cost: one regex pass over all active facts at refresh. ~10ms on 1000
events. Set lookup at query time = O(1).
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path


# Patterns that declare a name: extract group "name"
_NAME_PATTERNS = [
    re.compile(r"\bменя\s+зовут\s+(?P<name>[А-ЯЁA-Z][\w]+)", re.IGNORECASE),
    re.compile(r"\bмене\s+звати\s+(?P<name>[А-ЯЁІЇЄҐA-Z][\w]+)", re.IGNORECASE),
    re.compile(r"\bmy\s+name\s+is\s+(?P<name>[A-Z][\w]+)", re.IGNORECASE),
    re.compile(r"\bi\s+am\s+(?P<name>[A-Z][a-z]{2,})\b"),
    re.compile(r"\bI'?m\s+(?P<name>[A-Z][a-z]{2,})\b"),
    re.compile(r"\buser'?s?\s+name\s+is\s+(?P<name>[A-Z][\w]+)", re.IGNORECASE),
    # "Я Алексей" — bare assertion
    re.compile(r"^Я\s+(?P<name>[А-ЯЁ][а-яё]{2,})\b", re.IGNORECASE | re.MULTILINE),
]


# Self-reference markers in content — when candidate has these, it's a
# first-person fact about the user.
SELF_MARKERS = {
    "я", "меня", "мне", "мной", "обо", "i", "i'm", "im", "ive", "my",
    "мене", "мені", "мною", "i've", "myself",
}


# Extra explicit RU/UK reflexives common in first-person fact phrasing
_SELF_RE = re.compile(
    r"\b(?:я|меня|мне|мной|i|i'm|i've|my|мене|мені|мною|myself)\b",
    re.IGNORECASE,
)


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
    """True if CONTENT declares a user name ("My name is X", "Меня зовут X").
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
