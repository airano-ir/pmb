"""
Temporal extraction and bi-temporal scoring.

Why this exists:
  LoCoMo cat-4 (temporal) is a known weak spot for vector-only retrieval.
  Mem0 reports F1=48.93 on temporal vs F1=72+ on open-domain. Zep/Graphiti
  argue their bi-temporal edges close this gap.

What we do:
  - Extract `event_time` from event content (LLM-free regex): dates,
    months, "yesterday", "last week", "in December", etc. Stored as
    timestamp metadata.
  - At recall, if the QUERY has temporal markers, boost candidate events
    whose event_time is close to the query's anchor.

Cheap. No LLM. Implementable as a small re-rank pass.

References:
  - Zep/Graphiti (arXiv 2501.13956): bi-temporal edges, event-time + ingest-time
  - mem0-g vs mem0 ablation: temporal benefit from graph-side time signal
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime

# Month names → 1..12
MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}


# ISO-like date: 2025-12-05, 2025/12/05
_ISO_DATE_RE = re.compile(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b")
# Month + day: 'december 5' / 'Dec 5' / 'december 5th'
_MONTH_DAY_RE = re.compile(
    r"\b(" + "|".join(MONTHS.keys()) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)
# Day + month: '5 December' / '5th of December'
_DAY_MONTH_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?\s+(" + "|".join(MONTHS.keys()) + r")\b",
    re.IGNORECASE,
)
# Year only: '2024' or 'in 2024'
_YEAR_RE = re.compile(r"\b(?:in\s+)?(20\d{2})\b", re.IGNORECASE)
# Bare month with no day: 'in December', 'during October', 'December meeting'.
# We anchor to mid-month (day 15) when no day is provided.
_BARE_MONTH_RE = re.compile(
    r"\b(?:in|during|from|by|since|until|the)\s+(" + "|".join(MONTHS.keys()) + r")\b"
    r"|\b(" + "|".join(MONTHS.keys()) + r")\b(?=\s+(?:meeting|trip|event|happened|"
    r"month|of))",
    re.IGNORECASE,
)
# Relative day-counters - translated against a reference 'now'
_RELATIVE_RE = re.compile(
    r"\b(yesterday|today|tomorrow|"
    r"last\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"next\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"\d+\s+(?:days?|weeks?|months?|years?)\s+ago)\b",
    re.IGNORECASE,
)


# Words in a query that signal temporal intent
_TEMPORAL_QUERY_RE = re.compile(
    r"\b(when|date|day|month|year|after|before|during|in\s+\d{4}|"
    + r"|".join(MONTHS.keys())
    + r"|yesterday|today|last|next|ago|since|until|earlier|previously"
    + r")\b",
    re.IGNORECASE,
)


def is_temporal_query(query: str) -> bool:
    """Cheap detection: does this query reference time?"""
    if not query:
        return False
    return bool(_TEMPORAL_QUERY_RE.search(query))


def extract_event_time(
    text: str, reference_now: float | None = None,
) -> float | None:
    """Try to find an explicit date in `text` and return its UTC timestamp.

    Returns None if no clear date.
    `reference_now` is used when resolving relative expressions (yesterday,
    "last week"). Defaults to time.time() at call.
    """
    if not text:
        return None
    import time
    if reference_now is None:
        reference_now = time.time()

    # 1. ISO-like: 2025-12-05
    m = _ISO_DATE_RE.search(text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return _to_ts(y, mo, d)
        except ValueError:
            pass

    # 2. Month + day
    m = _MONTH_DAY_RE.search(text)
    if m:
        mo = MONTHS.get(m.group(1).lower())
        d = int(m.group(2))
        if mo and 1 <= d <= 31:
            year = _infer_year(reference_now, mo)
            try:
                return _to_ts(year, mo, d)
            except ValueError:
                pass

    # 3. Day + month
    m = _DAY_MONTH_RE.search(text)
    if m:
        d = int(m.group(1))
        mo = MONTHS.get(m.group(2).lower())
        if mo and 1 <= d <= 31:
            year = _infer_year(reference_now, mo)
            try:
                return _to_ts(year, mo, d)
            except ValueError:
                pass

    # 4. Year only
    m = _YEAR_RE.search(text)
    if m:
        try:
            return _to_ts(int(m.group(1)), 1, 1)
        except ValueError:
            pass

    # 4b. Bare month (no day). Anchor to mid-month (day 15) of inferred year.
    m = _BARE_MONTH_RE.search(text)
    if m:
        month_str = (m.group(1) or m.group(2) or "").lower()
        mo = MONTHS.get(month_str)
        if mo:
            year = _infer_year(reference_now, mo)
            try:
                return _to_ts(year, mo, 15)
            except ValueError:
                pass

    # 5. Relative
    m = _RELATIVE_RE.search(text)
    if m:
        return _resolve_relative(m.group(1).lower(), reference_now)

    return None


def temporal_proximity_boost(
    event_time: float | None,
    query_anchor_time: float | None,
    half_life_days: float = 14.0,
) -> float:
    """Returns a [0, 1] proximity boost. 1 = same day, decays exponentially.
    If either timestamp is missing, returns 0 (no boost)."""
    if event_time is None or query_anchor_time is None:
        return 0.0
    gap_sec = abs(event_time - query_anchor_time)
    half_life_sec = max(1.0, half_life_days * 86400.0)
    return float(math.exp(-gap_sec * math.log(2) / half_life_sec))


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------

def _to_ts(year: int, month: int, day: int) -> float:
    return datetime(year, month, day, tzinfo=UTC).timestamp()


def _infer_year(reference_now: float, month: int) -> int:
    """Pick the year that puts the date <= 6 months from reference (in the
    past for ambiguous month-day strings - most user references go
    backward not forward)."""
    now = datetime.fromtimestamp(reference_now, tz=UTC)
    candidate_this_year = datetime(now.year, month, 1, tzinfo=UTC)
    # If the month is later this year than now, it likely refers to last
    # year (memory references are backward-looking).
    if candidate_this_year > now + _delta_days(180):
        return now.year - 1
    return now.year


def _delta_days(days: int):
    from datetime import timedelta
    return timedelta(days=days)


def _resolve_relative(phrase: str, reference_now: float) -> float | None:
    from datetime import timedelta
    now = datetime.fromtimestamp(reference_now, tz=UTC)
    p = phrase.lower()
    if p == "yesterday":
        return (now - timedelta(days=1)).timestamp()
    if p == "today":
        return now.timestamp()
    if p == "tomorrow":
        return (now + timedelta(days=1)).timestamp()
    if p.startswith("last week"):
        return (now - timedelta(days=7)).timestamp()
    if p.startswith("last month"):
        return (now - timedelta(days=30)).timestamp()
    if p.startswith("last year"):
        return (now - timedelta(days=365)).timestamp()
    if p.startswith("next week"):
        return (now + timedelta(days=7)).timestamp()
    if p.startswith("next month"):
        return (now + timedelta(days=30)).timestamp()
    if p.startswith("next year"):
        return (now + timedelta(days=365)).timestamp()
    # "N days/weeks/months/years ago"
    m = re.match(r"(\d+)\s+(days?|weeks?|months?|years?)\s+ago", p)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("day"):
            delta = timedelta(days=n)
        elif unit.startswith("week"):
            delta = timedelta(days=7 * n)
        elif unit.startswith("month"):
            delta = timedelta(days=30 * n)
        elif unit.startswith("year"):
            delta = timedelta(days=365 * n)
        else:
            return None
        return (now - delta).timestamp()
    return None
