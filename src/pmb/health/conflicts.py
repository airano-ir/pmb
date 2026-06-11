"""
Conflict Detector — detects contradictions between facts from different times.

Idea: one key (for example `database`) can have different values at different
times. This is either evolution (Postgres 16 → Postgres 17, which is fine) or
an error (2 live facts contradict each other).

Detection strategies (without LLM calls):
1. **Same-key heuristics**: we look for "fact" events whose content has the
   pattern "X = Y" or "X is Y" with the same X but a different Y.
2. **Semantic clusters**: for qa events — high similarity between the
   questions (close by embedding) but low similarity between the answers.

In Phase 3 — a simple implementation via regex for facts. Semantic clusters
are a TODO for the next iteration.

Output: a list of FactConflict with suggested_resolution:
- "supersede": newer replaces older — automatic resolution
- "concurrent": both may be true (different branches)
- "needs_review": requires manual review
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmb.core.engine import Engine
    from pmb.core.events import Event


# Conservative patterns — only catch identifier-style keys, not free text.
# We deliberately dropped the prior generic "X is Y" pattern: it matched
# natural language like "the bug is fixed" and produced false conflicts.

_IDENT = r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*"

KEY_VALUE_PATTERNS = [
    # foo = value  (assignment, requires =, not just :)
    re.compile(rf"(?<![\w.])({_IDENT})\s*=\s*(.+)"),
    # project.foo = value, user.foo = value, also project.foo: value (config-style)
    re.compile(rf"(?<![\w.])((?:project|user)\.{_IDENT})\s*[=:]\s*(.+)", re.IGNORECASE),
    # "<ident> uses <value>" — single ident only
    re.compile(rf"(?<![\w.])({_IDENT})\s+uses\s+(.+)", re.IGNORECASE),
]

# Keys we refuse to treat as configuration — common natural-language words
# that would otherwise generate false-positive conflicts.
_KEY_BLOCKLIST = {
    "the", "a", "an", "this", "that", "it", "we", "i", "you", "they",
    "is", "was", "are", "be", "been", "being",
    "what", "where", "who", "when", "how", "why", "which",
    "and", "or", "but", "if", "then", "so", "because",
    "code", "function", "class", "method", "var", "let", "const",
    "true", "false", "null", "none", "yes", "no",
    "todo", "fixme", "note", "warning",
}

_MAX_VALUE_LEN = 80


@dataclass
class FactConflict:
    """A conflict between two facts."""

    key: str
    older_ulid: str
    older_value: str
    older_timestamp: float
    newer_ulid: str
    newer_value: str
    newer_timestamp: float
    suggested_resolution: str  # "supersede" | "concurrent" | "needs_review"
    confidence: float  # 0..1, how confident we are that this is a conflict

    def to_dict(self) -> dict:
        return asdict(self)


def extract_key_value(content: str) -> tuple[str, str] | None:
    """
    Conservatively extract a (key, value) pair.

    Returns the first plausible pair, or None. Keys must be identifier-like
    and not in the natural-language blocklist. Values are trimmed at the
    first newline / sentence end and length-capped.
    """
    for pattern in KEY_VALUE_PATTERNS:
        m = pattern.search(content)
        if not m:
            continue
        key = m.group(1).strip().lower()
        value = m.group(2).strip()
        # Strip at first newline or sentence end
        value = value.split("\n")[0]
        # Cut at first ". " or "; " — keep abbreviations like "v1.2" intact
        for sep in (". ", "; ", " — ", " - "):
            idx = value.find(sep)
            if idx > 0:
                value = value[:idx]
        value = value.strip().rstrip(".,;:!?")
        if len(value) > _MAX_VALUE_LEN:
            value = value[:_MAX_VALUE_LEN].rstrip() + "…"

        # Filter
        if not (2 <= len(key) <= 40 and 2 <= len(value) <= _MAX_VALUE_LEN + 1):
            continue
        # Block common natural-language words from being treated as keys
        head = key.split(".")[0]
        if head in _KEY_BLOCKLIST:
            continue
        return (key, value)
    return None


def _values_seem_different(v1: str, v2: str) -> bool:
    """Rough check: values are not identical and neither is a subset of the other."""
    v1n = v1.lower().strip().rstrip(".,;:!?")
    v2n = v2.lower().strip().rstrip(".,;:!?")
    if v1n == v2n:
        return False
    # If one is a substring of the other, it's evolution (e.g. "Postgres" vs "Postgres 17")
    if v1n in v2n or v2n in v1n:
        return False
    return True


class ConflictDetector:
    """Detects conflicts among facts in workspace."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def detect(self, max_age_days: float = 365.0) -> list[FactConflict]:
        """
        Find conflicts among active fact events.

        Returns a list of FactConflict, sorted by newer_timestamp DESC.
        """
        workspace_id = self.engine.workspace.id
        events = self.engine.events.list_active(
            workspace_id, limit=10000, event_type="fact",
        )
        # We also inspect qa events for key=value patterns
        qa_events = self.engine.events.list_active(
            workspace_id, limit=10000, event_type="qa",
        )
        all_candidates: list[Event] = events + qa_events

        cutoff = time.time() - max_age_days * 86400.0
        all_candidates = [e for e in all_candidates if e.timestamp >= cutoff]

        # Group by key
        key_to_events: dict[str, list[tuple[Event, str]]] = {}
        for ev in all_candidates:
            kv = extract_key_value(ev.content)
            if not kv:
                continue
            key, value = kv
            key_to_events.setdefault(key, []).append((ev, value))

        conflicts: list[FactConflict] = []
        for key, items in key_to_events.items():
            if len(items) < 2:
                continue
            # Sort by timestamp asc
            items.sort(key=lambda x: x[0].timestamp)
            # Compare adjacent pairs (older vs newer)
            for i in range(len(items) - 1):
                older_ev, older_val = items[i]
                newer_ev, newer_val = items[i + 1]
                if not _values_seem_different(older_val, newer_val):
                    continue

                # Same session? Then concurrent vs supersede
                same_session = (
                    older_ev.source_session_id == newer_ev.source_session_id
                    and older_ev.source_session_id is not None
                )
                # > 7 days apart? — more likely supersede
                age_diff_days = (newer_ev.timestamp - older_ev.timestamp) / 86400.0

                if age_diff_days > 7:
                    resolution = "supersede"
                    confidence = 0.7
                elif same_session:
                    resolution = "needs_review"
                    confidence = 0.5
                else:
                    resolution = "concurrent"
                    confidence = 0.4

                conflicts.append(FactConflict(
                    key=key,
                    older_ulid=older_ev.ulid,
                    older_value=older_val,
                    older_timestamp=older_ev.timestamp,
                    newer_ulid=newer_ev.ulid,
                    newer_value=newer_val,
                    newer_timestamp=newer_ev.timestamp,
                    suggested_resolution=resolution,
                    confidence=confidence,
                ))

        conflicts.sort(key=lambda c: -c.newer_timestamp)
        return conflicts

    def auto_resolve(self, dry_run: bool = True, merge_via_llm: bool = False) -> dict:
        """
        Resolve 'supersede' conflicts.

        Two modes:
          - default (merge_via_llm=False): just archive the older value.
            Fast, deterministic, no API calls.
          - LLM-merge (merge_via_llm=True): also create a merged fact that
            preserves both contexts ("X was Y until DATE, now Z") and
            archives both originals. This is the human-memory reconsolidation
            analogy: instead of throwing the old version away we synthesize
            an updated one with provenance.

        In dry_run we never write — we just describe what we would do.
        """
        conflicts = self.detect()
        actions = []
        archived = 0
        merged_created = 0

        llm = None
        if merge_via_llm and not dry_run:
            try:
                from pmb.health.consolidate import resolve_llm_client
                # Read backend from engine config if available, fallback to auto
                backend = "auto"
                try:
                    backend = self.engine.config.get("consolidate.backend")
                except Exception:
                    pass
                llm = resolve_llm_client(backend=backend)
            except Exception:
                # Fall back to non-merge mode rather than failing
                llm = None
                merge_via_llm = False

        def _is_protected(ulid: str) -> bool:
            """R12: a pinned (importance>=0.99) or lesson fact must NEVER be
            auto-archived by conflict resolution — every other archiver in the
            codebase protects those, this one didn't."""
            try:
                import json as _j
                import sqlite3 as _sql
                with _sql.connect(str(self.engine.workspace.db_path)) as _c:
                    row = _c.execute(
                        "SELECT importance, metadata_json FROM events WHERE ulid=?",
                        (ulid,)).fetchone()
                if not row:
                    return False
                if float(row[0] or 0.0) >= 0.99:
                    return True
                md = _j.loads(row[1] or "{}")
                return md.get("kind") == "lesson" or md.get("source") == "lesson"
            except Exception:
                return False

        for c in conflicts:
            if c.suggested_resolution != "supersede" or c.confidence < 0.6:
                continue
            # R12: skip the conflict entirely if archiving would touch a
            # protected fact (pinned / lesson). The older value is what gets
            # archived in supersede; in merge mode BOTH originals are archived.
            if _is_protected(c.older_ulid) or (
                    merge_via_llm and _is_protected(c.newer_ulid)):
                actions.append({"key": c.key, "older_ulid": c.older_ulid,
                                "newer_ulid": c.newer_ulid,
                                "mode": "skipped_protected"})
                continue

            action = {
                "key": c.key,
                "older_ulid": c.older_ulid,
                "older_value": c.older_value,
                "newer_ulid": c.newer_ulid,
                "newer_value": c.newer_value,
                "mode": "merge" if merge_via_llm else "archive_older",
            }

            if dry_run:
                actions.append(action)
                continue

            if merge_via_llm and llm is not None:
                # Ask the LLM to write one fact that carries both contexts.
                merged_text = _ask_llm_to_merge(c, llm)
                if merged_text:
                    new_ulid = self.engine.record_fact(
                        fact=merged_text,
                        importance=0.85,
                        metadata={
                            "merged_from": [c.older_ulid, c.newer_ulid],
                            "merged_at": time.time(),
                            "key": c.key,
                        },
                    )
                    # Archive both originals — merged fact now carries truth
                    self.engine.events.archive(c.older_ulid)
                    self.engine.events.archive(c.newer_ulid)
                    archived += 2
                    merged_created += 1
                    action["new_ulid"] = new_ulid
                    action["merged_text"] = merged_text
                    actions.append(action)
                    continue
                # Fall through to plain archive if LLM merge failed
                action["mode"] = "archive_older_fallback"

            # Default supersede: archive older, keep newer
            self.engine.events.archive(c.older_ulid)
            archived += 1
            actions.append(action)

        return {
            "n_conflicts": len(conflicts),
            "n_supersede_candidates": sum(
                1 for c in conflicts
                if c.suggested_resolution == "supersede" and c.confidence >= 0.6
            ),
            "n_archived": archived,
            "n_merged": merged_created,
            "dry_run": dry_run,
            "actions": actions,
        }


_MERGE_PROMPT = """\
Two facts about the same key contradict each other. The newer fact is
the current truth; the older is history. Write ONE sentence that
preserves both — current state first, then history — using natural
language.

Examples:
  older: "database = MySQL"
  newer: "database = Postgres 17"
  output: "Database is currently Postgres 17 (was MySQL until the recent migration)."

  older: "auth = JWT only"
  newer: "auth = JWT + 2FA via TOTP"
  output: "Auth uses JWT with 2FA TOTP added on top (was JWT-only before)."

Now write a single-sentence merged fact for the following — JSON only,
no prose or code fences:

{"merged": "<your one sentence>"}

Key:   %(key)s
Older: %(older)s
Newer: %(newer)s
"""


def _ask_llm_to_merge(c, llm) -> str:
    """Ask the LLM to produce a merged fact preserving both contexts."""
    prompt = _MERGE_PROMPT % {
        "key": c.key,
        "older": c.older_value,
        "newer": c.newer_value,
    }
    try:
        out = llm.consolidate([prompt])
    except Exception:
        return ""
    # The shared consolidate path returns {summary, confidence, ...}. Try
    # to find a JSON {"merged": "..."} object anywhere in the text.
    for key in ("summary", "reasoning"):
        raw = out.get(key) if isinstance(out, dict) else None
        if not raw:
            continue
        a = raw.find("{")
        b = raw.rfind("}")
        if a < 0 or b <= a:
            continue
        try:
            obj = json.loads(raw[a : b + 1])
            merged = obj.get("merged", "").strip()
            if merged:
                return merged
        except Exception:
            continue
    return ""
