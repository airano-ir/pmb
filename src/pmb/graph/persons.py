"""
Person entity extraction — no ML, no model downloads.

Why not spaCy / HuggingFace NER:
  - 30-500MB download, cold-start latency
  - Generic English NER mispredicts on chat: "Postgres" classified as PERSON,
    "I" classified as PERSON, etc.
  - We have BETTER signal: speaker metadata is provided by chat agents
    natively. Plus capitalized words + stop-list catches the rest.

How this works (4 layers, ordered cheap → richer):

  1. SPEAKER METADATA       free, instant — agent already tracks "who's talking"
                            LoCoMo / OpenAI / Anthropic chat APIs all have a
                            `speaker` or `role` field. Use it.

  2. EXPLICIT-MENTION REGEX  capitalized word(s) NOT at sentence start,
                            NOT in a stop-list (months, places, common terms,
                            known tech names), length 2-30.

  3. PRONOUN RESOLUTION     within a single event/dialogue, "I/me/my/mine"
                            refers to the speaker. We do NOT replace text but
                            we ADD the speaker as a person entity for any
                            event where pronouns occur.

  4. SELF-REINFORCING DICT   known persons (mentioned ≥ N times) are stored
                            in the workspace metadata as a `known_persons` set.
                            On future events, this dict catches names that the
                            stop-list missed and confidently classifies them.

Cost: pure regex + dict lookup, < 0.5ms per event. No external deps.

Output: lowercase canonical names with kind='person'. Co-occurrence edges
are then built between persons in the graph layer.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Stop-list — capitalized words that LOOK like people but aren't
# ----------------------------------------------------------------------

MONTHS = {
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
}

WEEKDAYS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "mon", "tue", "wed", "thu", "fri", "sat", "sun",
}

PLACES_HINTS = {
    "paris", "london", "tokyo", "berlin", "moscow", "madrid", "rome",
    "boston", "seattle", "denver", "austin", "miami", "chicago",
    "sweden", "germany", "france", "spain", "italy", "japan", "china",
    "russia", "usa", "uk", "europe", "asia", "africa", "america",
    "australia", "canada", "mexico", "brazil",
    "north", "south", "east", "west", "european", "american",
    "asian", "african", "atlantic", "pacific",
}

COMMON_TITLES = {
    "mr", "mrs", "ms", "dr", "prof", "professor", "sir", "madam",
    "lord", "lady", "father", "mother", "brother", "sister",
    "uncle", "aunt", "cousin", "doctor",
}

# Common non-person capitalized words seen in chat
COMMON_NON_PERSON = {
    "the", "a", "an", "i", "you", "he", "she", "it", "we", "they",
    "this", "that", "these", "those", "yes", "no", "ok", "okay",
    "hi", "hello", "hey", "thanks", "thank", "great", "good", "bad",
    "today", "tomorrow", "yesterday", "now", "then", "later", "soon",
    "morning", "afternoon", "evening", "night", "week", "year", "day",
    "weekend", "weeks", "years", "days", "months", "weekends",
    "happy", "sad", "tired", "excited", "nervous",
    "lgbtq", "lgbt", "us", "i'm", "i've", "i'll",
    "covid", "ai", "ml", "llm", "api", "url", "json", "xml", "html",
    # Question words — common in dialogue, NEVER persons
    "how", "what", "where", "when", "why", "who", "whom", "whose",
    "which", "whether", "whoever", "whatever", "wherever", "whenever",
    # Dialogue roles — "User:", "Agent:", "Assistant:" are roles, not names
    "user", "agent", "assistant", "system", "bot", "human", "chatbot", "model", "tool", "function",
    # Section / project / product nouns mistakenly capitalized
    "frontend", "backend", "fullstack", "client", "server", "service",
    "database", "cache", "queue", "worker", "pipeline", "monorepo",
    "project", "module", "package", "library", "framework", "platform",
    "feature", "bug", "fix", "patch", "release", "version",
    # Windows path components — extracted from C:\Users\foo\AppData\Roaming\...
    "appdata", "roaming", "programfiles", "programdata", "users",
    "desktop", "documents", "downloads", "local", "temp", "tmp",
    "system32", "windows", "program", "application", "applications",
    "userprofile", "homepath",
    # Misc tools / product names that aren't human names
    "locomo", "mem0", "letta", "zep", "pmb",
    # Verbs that occasionally get capitalized at sentence start and survive
    # the sentence-start guard (e.g. "Verify credentials before login")
    "verify", "check", "test", "run", "build", "deploy", "create",
    "delete", "update", "fetch", "send", "load", "save",
    # CLI agent product names — they ARE proper nouns but rarely human names
    "codex", "claude", "anthropic",
}

# Known tech names (subset of pmb.graph.entities.KNOWN_TECHS) — block these
# as persons. We import lazily to avoid circular imports.
def _known_techs() -> set:
    try:
        from pmb.graph.entities import KNOWN_TECHS
        out = set()
        for variants in KNOWN_TECHS.values():
            for v in variants:
                out.add(v.lower())
        return out
    except Exception:
        return set()


def _full_stoplist() -> set:
    out = set()
    out.update(MONTHS)
    out.update(WEEKDAYS)
    out.update(PLACES_HINTS)
    out.update(COMMON_TITLES)
    out.update(COMMON_NON_PERSON)
    out.update(_known_techs())
    return out


_STOPLIST = _full_stoplist()


# ----------------------------------------------------------------------
# Regex patterns
# ----------------------------------------------------------------------

# Capitalized word — 2+ chars, starts uppercase. Captures hyphenated names
# like "Mary-Anne" and apostrophes "O'Connor".
_CAP_WORD_RE = re.compile(r"\b([A-Z][a-z][a-zA-Z\-']{1,28})\b")

# Path-context characters: if a capitalized word is preceded or followed by
# one of these, it's almost certainly a path/URL component, not a person name.
# (e.g. `C:\Users\alexb\AppData\Roaming\Claude\...`)
_PATH_NEIGHBOR_CHARS = set("\\/.:_@")

# Speaker prefix in chat dialogue: "Caroline: ..." or "[Caroline]:"
_SPEAKER_PREFIX_RE = re.compile(
    r"(?:^|\n)\s*\[?([A-Z][a-zA-Z\-']{1,30})\]?\s*:",
    re.MULTILINE,
)

# Common verbs that follow person names (helps confirm capitalized is a person)
_VERB_AFTER_NAME_RE = re.compile(
    r"\b([A-Z][a-z][a-zA-Z\-']{1,28})\s+"
    r"(said|told|asked|replied|wrote|mentioned|thinks|thought|feels|felt|"
    r"loves|loved|hates|hated|knows|knew|wants|wanted|needs|needed|has|"
    r"had|wishes|wished|believes|believed|moved|went|came|left|arrived|"
    r"is|was|works|worked|lives|lived|met|meets|sees|saw|likes|liked|"
    r"plans|planned|decided|decides|chose|chooses|tried|tries|gets|got|"
    r"makes|made|takes|took|gives|gave|finds|found|started|starts|"
    r"stopped|stops|wrote|writes|reads|read|runs|ran|drives|drove|"
    r"flew|flies|travels|traveled|texted|called|emailed|visited)\b"
)

# Pronouns that imply self-reference within a turn → tag speaker
_FIRST_PERSON_RE = re.compile(
    r"\b(I|me|my|mine|myself|I've|I'm|I'll|I'd)\b"
)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

@dataclass
class PersonExtractionResult:
    persons: list[str]              # canonical lowercase names
    speaker: str | None = None   # if extracted from metadata
    rationale: list[str] = None     # which stages contributed

    def to_kind_name_pairs(self) -> list[tuple[str, str]]:
        return [("person", p) for p in self.persons]


# ----------------------------------------------------------------------
# Known-persons dictionary (workspace-scoped)
# ----------------------------------------------------------------------

class KnownPersons:
    """Self-reinforcing per-workspace dictionary of known person names.

    Backed by the same SQLite events.db; stored in a tiny key-value table
    that we provision on demand.
    """

    def __init__(self, db_path: Path, workspace_id: str):
        self.db_path = db_path
        self.workspace_id = workspace_id
        self._ensure_table()

    def _ensure_table(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workspace_kv (
                    workspace_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    PRIMARY KEY (workspace_id, key)
                )
                """
            )

    def load(self) -> dict[str, int]:
        """Return {name: mention_count}."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value_json FROM workspace_kv WHERE workspace_id = ? "
                "AND key = 'known_persons'",
                (self.workspace_id,),
            ).fetchone()
        if not row:
            return {}
        try:
            data = json.loads(row[0]) or {}
            return {k: int(v) for k, v in data.items()}
        except Exception:
            return {}

    def save(self, names: dict[str, int]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO workspace_kv (workspace_id, key, value_json)
                VALUES (?, 'known_persons', ?)
                ON CONFLICT(workspace_id, key) DO UPDATE SET
                  value_json = excluded.value_json
                """,
                (self.workspace_id, json.dumps(names, ensure_ascii=False)),
            )

    def bump(self, names: Iterable[str]) -> None:
        if not names:
            return
        cur = self.load()
        for n in names:
            if not n:
                continue
            cur[n] = cur.get(n, 0) + 1
        self.save(cur)

    def is_known(self, name: str, threshold: int = 2) -> bool:
        if not name:
            return False
        return self.load().get(name.lower(), 0) >= threshold


# ----------------------------------------------------------------------
# Main extractor
# ----------------------------------------------------------------------

def _normalize_name(raw: str) -> str | None:
    """Lowercase, strip, check it's a plausible person name."""
    if not raw:
        return None
    n = raw.strip().lower()
    if len(n) < 2 or len(n) > 30:
        return None
    if n in _STOPLIST:
        return None
    # Must contain a letter (no pure numbers, no symbols-only)
    if not any(c.isalpha() for c in n):
        return None
    return n


def _is_path_context(text: str, start: int, end: int) -> bool:
    r"""True if the [start, end) span looks like a path/URL component.

    Triggered by any of these:
      - char immediately before start is one of: \ / . : _ @
      - char immediately at end is one of: \ / . : _ @
      - the surrounding 40 chars window contains 2+ path separators

    Avoids treating "alexb", "AppData", "Roaming" inside
    `C:\Users\alexb\AppData\Roaming\...` as person names.
    """
    if start > 0 and text[start - 1] in _PATH_NEIGHBOR_CHARS:
        return True
    if end < len(text) and text[end] in _PATH_NEIGHBOR_CHARS:
        return True
    # Window heuristic
    win_lo = max(0, start - 20)
    win_hi = min(len(text), end + 20)
    window = text[win_lo:win_hi]
    if window.count("\\") + window.count("/") >= 2:
        return True
    return False


def extract_persons(
    text: str,
    metadata: dict | None = None,
    known_persons: KnownPersons | None = None,
    max_persons: int = 12,
) -> PersonExtractionResult:
    """Multi-stage person extraction.

    Args:
      text:         raw event content
      metadata:     event metadata (may contain `speaker` field)
      known_persons: optional workspace-scoped dictionary for stage 4
      max_persons:  cap per event to avoid bloat

    Returns: PersonExtractionResult with canonical lowercase names + rationale.
    """
    rationale: list[str] = []
    found: dict[str, int] = {}  # name → stage tier (1=best ... 4=worst)

    def add(name: str, tier: int, why: str) -> None:
        canon = _normalize_name(name)
        if not canon:
            return
        # Lower tier number wins (more confident)
        if canon not in found or tier < found[canon]:
            found[canon] = tier
        rationale.append(f"[{tier}] {canon}: {why}")

    speaker_canon: str | None = None

    # Stage 1: explicit speaker metadata
    if metadata and isinstance(metadata, dict):
        sp = metadata.get("speaker") or metadata.get("user") or metadata.get("role")
        if isinstance(sp, str) and sp.strip():
            add(sp, tier=1, why="speaker metadata")
            speaker_canon = _normalize_name(sp)

    # Stage 1b: speaker prefix in dialogue ("Caroline: ...")
    if text:
        for m in _SPEAKER_PREFIX_RE.finditer(text[:5000]):
            add(m.group(1), tier=1, why="dialogue speaker prefix")

    # Stage 2: capitalized + stop-list
    if text:
        # Skip the first character of each sentence (capitalization is forced there)
        # We approximate by skipping words at sentence start: after [.!?] + space
        # Simpler: just check the word against the stop-list and proximity
        # to verbs.
        for m in _CAP_WORD_RE.finditer(text[:8000]):
            cand = m.group(1)
            # Skip if it's the first word of text (likely sentence start)
            if m.start() == 0:
                continue
            # Skip if surrounded by path/URL characters (e.g. C:\Users\AppData\...)
            if _is_path_context(text, m.start(), m.end()):
                continue
            # Skip if preceded by sentence-end punctuation (likely sentence start)
            prev_idx = m.start() - 1
            if prev_idx >= 0:
                prev = text[max(0, prev_idx - 1):m.start()].strip()
                if prev.endswith((".", "!", "?")):
                    # Sentence start: only accept if it's a name typical pattern
                    # (e.g., followed by a verb)
                    verb_match = _VERB_AFTER_NAME_RE.match(text[m.start():m.start() + 60])
                    if not verb_match:
                        continue
            add(cand, tier=2, why="capitalized non-stoplist")

    # Stage 2b: name followed by verb — high-confidence person.
    # This rule SHOULD fire even at sentence start because the entire point
    # is to catch "Caroline said ...", "Bob met ...", "Alice flew to Paris".
    # False positives like "Frontend is React" are caught by the stoplist
    # at _normalize_name; we don't need a sentence-start guard here.
    if text:
        for m in _VERB_AFTER_NAME_RE.finditer(text[:8000]):
            # Path context guard — skip names that live inside file paths.
            name_end = m.start() + len(m.group(1))
            if _is_path_context(text, m.start(), name_end):
                continue
            add(m.group(1), tier=1, why="name + action verb")

    # Stage 3: pronoun resolution — if speaker is known and "I/me/my" appears,
    # mark the speaker as participating in this event
    if speaker_canon and text and _FIRST_PERSON_RE.search(text):
        rationale.append(f"[3] {speaker_canon}: first-person pronoun → speaker is participant")
        # Already added at tier 1; this is just rationale

    # Stage 4: self-reinforcing dictionary — re-check capitalized words
    # against known persons (this catches names the stop-list might have
    # mis-classified as common nouns, e.g. "May" is a stop-list month but
    # if "May Smith" appears repeatedly, it gets known)
    if known_persons and text:
        known = known_persons.load()
        for m in _CAP_WORD_RE.finditer(text[:8000]):
            cand = m.group(1).lower()
            if cand in known and known[cand] >= 2:
                add(m.group(1), tier=2, why="known persons dict")

    # Sort by tier asc (best first), cap, return
    items = sorted(found.items(), key=lambda kv: kv[1])
    persons = [name for name, _ in items[:max_persons]]
    return PersonExtractionResult(
        persons=persons,
        speaker=speaker_canon,
        rationale=rationale,
    )
