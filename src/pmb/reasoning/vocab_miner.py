"""
Auto VOCAB_BRIDGES — mine domain-specific vocabulary bridges from workspace events.

PAMVR's hand-curated VOCAB_BRIDGES (typing↔mypy, database↔Postgres, …) work
great on coding-agent queries but slightly regress on personal-conversation
domains because the lexicon is dev-flavoured. This module mines a per-
workspace bridge table from the user's own event history so PAMVR adapts to
whatever domain the user actually writes about — coding, personal, research,
business, …

Approach (no ML, no external deps):

  1. Tokenise every event's content into lowercase content-words (stop-words
     removed, len > 2).
  2. Walk a sliding window of ±W tokens. For every pair (a, b) inside the
     window, increment co-occurrence count.
  3. Compute pointwise mutual information (PMI):

         PMI(a, b) = log( P(a, b) / (P(a) · P(b)) )

     Pairs with high PMI co-occur much more than chance — that's exactly
     the relationship VOCAB_BRIDGES models ("typing" and "mypy" appear in
     the same event much more than their individual rates would predict).
  4. Filter:
        - count(a, b) >= MIN_COUNT  (default 3 — kill noise)
        - PMI >= MIN_PMI            (default 2.0 — keep ~top 5%)
        - len(a), len(b) > 2 and both alphabetic
        - drop entirely-numeric tokens and unicode-mixed garbage
  5. Group by first token: {"typing": ["mypy", "type_hints"], ...}

The result is merged with PAMVR's hand-curated VOCAB_BRIDGES at lookup time
(workspace-specific keys win; hand-curated stays as fallback).

Cache: written to `~/.pmb/workspaces/<id>/vocab_bridges.json`. Refreshed
when the workspace gains >= REFRESH_THRESHOLD events since last mine.

Cost: ~50 ms per 1000 events. Triggered async — never blocks recall.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional


# Stop-words shared with PAMVR (kept inline to avoid circular import).
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "to",
    "for", "and", "or", "with", "we", "i", "you", "do", "does", "did",
    "what", "who", "where", "when", "why", "how", "by", "at", "from", "as",
    "be", "have", "has", "had", "this", "that", "these", "those", "it",
    "our", "my", "your", "their", "his", "her", "about", "any", "some",
    "all", "more", "than", "but", "not", "now", "before", "previously",
    "going", "use", "using", "can", "will", "would", "could", "should",
    "may", "might", "just", "also", "very", "really", "much", "many",
    "such", "into", "out", "up", "down", "over", "under", "again",
    "then", "there", "here", "only", "own", "same", "so", "if", "no",
}


# Token regex: a Unicode letter start, then word chars (letters/digits/_) —
# keeps "type_hints" intact. Identical to the old Latin+Cyrillic class on EN/RU;
# also keeps other scripts' letters. Deliberately NO hyphens in the class:
# including them silently broke compound-word matching (caught by a live run).
_TOKEN_RE = re.compile(r"[^\W\d_][\w]{2,}", re.UNICODE)

# Default parameters — chosen empirically so a 200-event workspace yields
# ~50-150 bridges, low noise.
DEFAULT_WINDOW = 6           # ±6 tokens around each position
DEFAULT_MIN_COUNT = 3        # pair must appear ≥3 times
DEFAULT_MIN_PMI = 2.0        # PMI threshold
DEFAULT_MAX_BRIDGES_PER_KEY = 8
DEFAULT_REFRESH_THRESHOLD = 50  # refresh after 50 new events


def _tokens(text: str) -> list[str]:
    """Lowercase tokens, stop-words removed, alpha-numeric only."""
    if not text:
        return []
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        t = m.group(0).lower()
        if t in _STOP:
            continue
        # Drop pure-numeric and tokens with too few letters
        n_letters = sum(1 for c in t if c.isalpha())
        if n_letters < 3:
            continue
        out.append(t)
    return out


def mine_pmi_bridges(
    event_contents: Iterable[str],
    window: int = DEFAULT_WINDOW,
    min_count: int = DEFAULT_MIN_COUNT,
    min_pmi: float = DEFAULT_MIN_PMI,
    max_bridges_per_key: int = DEFAULT_MAX_BRIDGES_PER_KEY,
) -> dict[str, list[str]]:
    """Mine vocab bridges from a stream of event contents.

    Returns a dict {key_term: [bridge_term_1, bridge_term_2, ...]} sorted
    by PMI descending.
    """
    tok_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    total_tokens = 0

    for content in event_contents:
        toks = _tokens(content)
        if len(toks) < 2:
            continue
        # Update unigram counts
        for t in toks:
            tok_counts[t] += 1
            total_tokens += 1
        # Update window co-occurrence (unordered pairs)
        n = len(toks)
        for i in range(n):
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            seen: set[str] = set()
            for j in range(lo, hi):
                if j == i:
                    continue
                a, b = toks[i], toks[j]
                if a == b:
                    continue
                # Unordered key
                key = (a, b) if a < b else (b, a)
                if key not in seen:
                    pair_counts[key] += 1
                    seen.add(key)

    if total_tokens < 50 or not pair_counts:
        return {}

    # Compute PMI and filter
    # P(a) = count(a) / total ;  P(a,b) = pair_count / total_pairs
    total_pairs = sum(pair_counts.values())
    if total_pairs == 0:
        return {}

    scored_bridges: dict[str, list[tuple[float, str]]] = defaultdict(list)

    for (a, b), c_ab in pair_counts.items():
        if c_ab < min_count:
            continue
        c_a, c_b = tok_counts[a], tok_counts[b]
        if c_a < min_count or c_b < min_count:
            continue
        p_ab = c_ab / total_pairs
        p_a = c_a / total_tokens
        p_b = c_b / total_tokens
        denom = p_a * p_b
        if denom <= 0:
            continue
        pmi = math.log(p_ab / denom)
        if pmi < min_pmi:
            continue
        # Add both directions so we can look up by either term
        scored_bridges[a].append((pmi, b))
        scored_bridges[b].append((pmi, a))

    # Keep top-K bridges per key, sorted by PMI desc
    bridges: dict[str, list[str]] = {}
    for key, items in scored_bridges.items():
        items.sort(key=lambda x: -x[0])
        bridges[key] = [t for _, t in items[:max_bridges_per_key]]

    return bridges


def load_cached(path: Path) -> Optional[dict]:
    """Load cached bridges from disk. Returns None if missing/corrupt."""
    try:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        if not isinstance(blob, dict):
            return None
        if "bridges" not in blob:
            return None
        return blob
    except Exception:
        return None


def save_cached(
    path: Path,
    bridges: dict[str, list[str]],
    event_count_at_mine: int,
    params: Optional[dict] = None,
) -> None:
    """Persist bridges + provenance metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "bridges": bridges,
        "event_count_at_mine": event_count_at_mine,
        "n_keys": len(bridges),
        "n_bridges": sum(len(v) for v in bridges.values()),
        "params": params or {},
    }
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def should_refresh(
    cached_event_count: int,
    current_event_count: int,
    threshold: int = DEFAULT_REFRESH_THRESHOLD,
) -> bool:
    """True if enough new events have landed to warrant a re-mine."""
    if current_event_count <= 0:
        return False
    if cached_event_count <= 0:
        return current_event_count >= 5  # mine ASAP on empty cache
    return (current_event_count - cached_event_count) >= threshold


def fetch_event_contents(db_path: Path, limit: int = 5000) -> list[str]:
    """Read event content from SQLite. Newest first, capped at `limit`."""
    if not db_path.exists():
        return []
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                "SELECT content FROM events "
                "WHERE archived_at IS NULL "
                "ORDER BY ulid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [r[0] or "" for r in rows]
    except Exception:
        return []


def mine_workspace(
    db_path: Path,
    cache_path: Path,
    force: bool = False,
    window: int = DEFAULT_WINDOW,
    min_count: int = DEFAULT_MIN_COUNT,
    min_pmi: float = DEFAULT_MIN_PMI,
    max_bridges_per_key: int = DEFAULT_MAX_BRIDGES_PER_KEY,
    refresh_threshold: int = DEFAULT_REFRESH_THRESHOLD,
) -> dict[str, list[str]]:
    """High-level: load cache → check freshness → re-mine if needed.

    Returns the active bridges dict (possibly cached, possibly freshly mined).
    Safe to call on every Engine() init — fast when cache is fresh.
    """
    contents = fetch_event_contents(db_path)
    n = len(contents)

    cached = None if force else load_cached(cache_path)
    if cached and not should_refresh(
        cached.get("event_count_at_mine", 0), n, refresh_threshold
    ):
        return cached["bridges"]

    bridges = mine_pmi_bridges(
        contents,
        window=window,
        min_count=min_count,
        min_pmi=min_pmi,
        max_bridges_per_key=max_bridges_per_key,
    )
    save_cached(
        cache_path,
        bridges,
        event_count_at_mine=n,
        params={
            "window": window,
            "min_count": min_count,
            "min_pmi": min_pmi,
            "max_bridges_per_key": max_bridges_per_key,
        },
    )
    return bridges


def merge_bridges(
    base: dict[str, list[str]],
    extra: dict[str, list[str]],
    max_per_key: int = DEFAULT_MAX_BRIDGES_PER_KEY,
) -> dict[str, list[str]]:
    """Merge two bridge dicts. Extra (workspace-mined) extends base
    (hand-curated). Duplicates removed, order preserved (base first).
    """
    out: dict[str, list[str]] = {}
    keys = set(base.keys()) | set(extra.keys())
    for k in keys:
        seen: set[str] = set()
        merged: list[str] = []
        for src in (base.get(k, []), extra.get(k, [])):
            for v in src:
                if v == k or v in seen:
                    continue
                merged.append(v)
                seen.add(v)
                if len(merged) >= max_per_key:
                    break
            if len(merged) >= max_per_key:
                break
        if merged:
            out[k] = merged
    return out
