"""Canonicalization of personal-attribute (keyed-fact) names + detection of
current-state statements.

A keyed fact is stored under a `subject::attribute` key (e.g. ``user::city``).
Without canonicalization, synonymous attribute labels — ``city`` /
``current_city`` / ``current_city_2026`` / ``lives_in`` / ``город`` — produce
INDEPENDENT keys that then compete in recall instead of one superseding the
next. That is exactly how a stale ``user::city = Warsaw`` kept out-ranking a
live ``user lives in Tampa`` fact.

This module is the single source of truth for:
  * ``canonicalize_attribute`` — collapse synonymous labels to one attribute.
  * ``keyed_fact_key`` — build the canonical ``subject::attribute`` key used
    by every keyed-fact read/write path.
  * ``detect_current_state`` — conservatively recognise a plain fact that
    states a CURRENT, mutable personal attribute ("I now live in Tampa") so
    it can update the matching keyed fact instead of just piling up.

The alias map is deliberately broad and extensible (location, work, contact,
status, …) — it is NOT city-specific. Unknown attributes pass through
normalized, so brand-new attributes still get a stable key; they simply have
no alias group yet. Add groups freely — that's the only step needed.
"""
from __future__ import annotations

import re
from typing import Optional

# ── canonical attribute → set of synonym labels ───────────────────────────
# Labels are matched after normalize_label() (lowercased, non-alnum collapsed
# to '_'), so write both 'current city' and 'current_city' as you like.
_ALIAS_GROUPS: dict[str, set[str]] = {
    "city": {
        "city", "current_city", "city_now", "current_city_2026", "town",
        "lives_in", "live_in", "living_in", "residence", "resides_in",
        "location", "current_location", "based_in", "home_city", "hometown",
        "город", "город_проживания", "живет", "живёт", "проживает",
        "место_жительства", "местожительства", "проживание",
    },
    "country": {
        "country", "current_country", "nation", "страна",
    },
    "employer": {
        "employer", "company", "current_employer", "current_company",
        "works_at", "work_at", "workplace", "works_for", "employed_at",
        "работодатель", "компания", "место_работы",
    },
    "job_title": {
        "job_title", "title", "role", "position", "occupation", "profession",
        "current_role", "current_title", "current_position",
        "должность", "профессия", "роль",
    },
    "email": {
        "email", "email_address", "e_mail", "mail", "contact_email", "почта",
        "электронная_почта",
    },
    "phone": {
        "phone", "phone_number", "mobile", "cell", "telephone", "tel",
        "телефон", "номер_телефона",
    },
    "current_project": {
        "current_project", "active_project", "working_on", "current_focus",
        "текущий_проект",
    },
    "timezone": {
        "timezone", "time_zone", "tz", "часовой_пояс",
    },
    "relationship_status": {
        "relationship_status", "marital_status", "семейное_положение",
    },
}

# reverse lookup alias → canonical, built once
_ALIAS_TO_CANON: dict[str, str] = {}
for _canon, _aliases in _ALIAS_GROUPS.items():
    _ALIAS_TO_CANON[_canon] = _canon
    for _a in _aliases:
        _ALIAS_TO_CANON[_a] = _canon


def normalize_label(label: str) -> str:
    """Lowercase, trim, and collapse any run of non-alphanumeric characters
    (spaces, hyphens, dots, …) to a single underscore.
    'Current City ' → 'current_city'."""
    s = (label or "").strip().lower()
    s = re.sub(r"[^0-9a-zа-яё]+", "_", s)
    return s.strip("_")


def canonicalize_attribute(attribute: str) -> str:
    """Map an attribute label to its canonical form. Unknown labels pass
    through normalized so new attributes still get a stable key."""
    return _ALIAS_TO_CANON.get(normalize_label(attribute), normalize_label(attribute))


def keyed_fact_key(subject: str, attribute: str) -> str:
    """Canonical ``subject::attribute`` key used by all keyed-fact storage.
    Subject keeps its historical normalization (strip+lower); only the
    attribute is canonicalized, since that's where the synonym collisions are.
    """
    return f"{(subject or '').strip().lower()}::{canonicalize_attribute(attribute)}"


# ── current-state detection (conservative, regex, no LLM) ──────────────────
#
# Each entry: (compiled_pattern, canonical_attribute). The pattern MUST imply
# a present, personal, mutable state — we require an explicit subject cue
# (I / user / my) AND, for location/work, a present-tense verb. Capture group
# 1 is the raw value. Order matters: first match wins.
_VALUE_TAIL = r"(.+?)"
_END = (
    r"(?:[.!?,;]"
    r"|\s+(?:now|currently|today|recently|yesterday|last\s+\w+|since\b"
    r"|in\s+\d|for\s+\d|as of\b|on an?\b|with\b).*"
    r"|$)"
)

# Every verb-based pattern requires an EXPLICIT present-state marker
# (now / currently / сейчас / теперь / moved-to), so a generic biography fact
# like "I live in Paris" is NOT auto-promoted — only deliberate "this is my
# CURRENT X" statements are. "my current X is …" / "moved to …" are inherently
# present-state and need no extra marker.
_CURRENT_STATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # English — location
    (re.compile(
        r"\b(?:i|user)\s+(?:currently|now)\s+(?:live|lives|reside|resides|am\s+living|is\s+living)\s+in\s+"
        + _VALUE_TAIL + _END, re.IGNORECASE), "city"),
    (re.compile(
        r"\b(?:i|user)\s+(?:live|lives|reside|resides)\s+in\s+" + _VALUE_TAIL
        + r"\s+(?:now|currently)\b", re.IGNORECASE), "city"),
    (re.compile(
        r"\b(?:my|user'?s)\s+(?:current\s+)?(?:city|location|residence)\s+(?:is|:)\s*"
        + _VALUE_TAIL + _END, re.IGNORECASE), "city"),
    (re.compile(
        r"\b(?:i|user)\s+(?:just\s+)?moved\s+to\s+" + _VALUE_TAIL + _END,
        re.IGNORECASE), "city"),
    # English — employer / role
    (re.compile(
        r"\b(?:i|user)\s+(?:currently|now)\s+(?:work|works|am\s+working|is\s+working)\s+(?:at|for)\s+"
        + _VALUE_TAIL + _END, re.IGNORECASE), "employer"),
    (re.compile(
        r"\b(?:my|user'?s)\s+(?:current\s+)?(?:employer|company)\s+(?:is|:)\s*"
        + _VALUE_TAIL + _END, re.IGNORECASE), "employer"),
    (re.compile(
        r"\b(?:my|user'?s)\s+(?:current\s+)?(?:job\s*title|role|position)\s+(?:is|:)\s*"
        + _VALUE_TAIL + _END, re.IGNORECASE), "job_title"),
    # Russian — location
    (re.compile(
        r"(?:сейчас|теперь)\s+жив[уёе]\w*\s+в\s+" + _VALUE_TAIL + _END,
        re.IGNORECASE), "city"),
    (re.compile(
        r"\b(?:переехал|переехала|переехали)\s+в\s+" + _VALUE_TAIL + _END,
        re.IGNORECASE), "city"),
    # Russian — work
    (re.compile(
        r"(?:сейчас|теперь)\s+работа\w+\s+в\s+" + _VALUE_TAIL + _END,
        re.IGNORECASE), "employer"),
]

# Stop-words that, if they're the whole captured value, mean we didn't really
# capture a value (avoid keying garbage).
_BAD_VALUES = {"", "the", "a", "an", "here", "there", "now", "это", "тут", "там"}

# Negation / meta-instruction guard. A statement like "Do NOT state that the
# user currently lives in Warsaw" or "treat ... as stale" must NEVER be read as
# a current-state assertion (it would re-promote the exact wrong value). Found
# by a live run against the real corpus — synthetic tests missed it.
_NEGATION_RE = re.compile(
    r"\b(?:do not|don'?t|does not|doesn'?t|did not|didn'?t|never|no longer|"
    r"not currently|isn'?t|aren'?t|wasn'?t|stop (?:saying|stating)|"
    r"treat\b[^.]*\bstale|не\s+жив|больше не|уже не)\b",
    re.IGNORECASE,
)


def _clean_value(raw: str) -> str:
    v = (raw or "").strip().strip("\"'`")
    # drop a trailing connector word the tail regex may have left
    v = re.sub(r"\s+(?:now|currently|today)\.?$", "", v, flags=re.IGNORECASE)
    # drop a leading article ("the United States" -> "United States")
    v = re.sub(r"^(?:the|a|an)\s+", "", v, flags=re.IGNORECASE)
    return v.strip(" .,:;!?").strip()


def detect_current_state(content: str) -> Optional[tuple[str, str]]:
    """If CONTENT plainly states a current, mutable personal attribute,
    return (canonical_attribute, value); else None.

    Deliberately conservative — only fires on explicit first-/second-person
    present-state phrasing, so arbitrary recorded text (project notes,
    reflections, code) does not get mis-promoted into keyed facts.
    """
    if not content or len(content) > 400:
        return None
    if _NEGATION_RE.search(content):
        return None  # negated / meta-instruction — not a current-state assertion
    for pat, attr in _CURRENT_STATE_PATTERNS:
        m = pat.search(content)
        if not m:
            continue
        value = _clean_value(m.group(1))
        if value.lower() in _BAD_VALUES or len(value) < 2:
            continue
        return attr, value
    return None
