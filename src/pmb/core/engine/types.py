"""Shared dataclasses and small helpers for the engine package."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from pmb.reference_data import override_dict as _override_dict


@dataclass
class RecallResult:
    """A recall result - an event plus ranking signals."""

    ulid: str
    event_type: str
    content: str
    metadata: dict
    timestamp: float
    score: float
    bm25_score: float
    vec_score: float
    importance: float
    recency_score: float
    # R3: absolute vector similarity 1/(1+dist) in [0,1] - un-normalized, so it
    # carries real meaning across queries (unlike the min-maxed score/vector).
    raw_vec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ulid": self.ulid,
            "event_type": self.event_type,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "date": self.resolved_date,
            "score": self.score,
            "signals": {
                "bm25": self.bm25_score,
                "vector": self.vec_score,
                "raw_cosine": self.raw_vec,
                "importance": self.importance,
                "recency": self.recency_score,
            },
        }

    @property
    def resolved_date(self) -> str | None:
        """Human-readable date this result REFERS to.

        Prefers the parsed `event_time` (the date *inside* the content), then a
        session-date string (chat-history imports), then the creation
        timestamp. Lets an agent answer "when ...?" without epoch math, and
        anchors relative-date reasoning to the EVENT, not to "today".
        Output-only - never affects ranking or latency.
        """
        meta = self.metadata if isinstance(self.metadata, dict) else {}
        et = meta.get("event_time")
        if isinstance(et, (int, float)):
            try:
                return time.strftime("%Y-%m-%d", time.gmtime(float(et)))
            except Exception:
                pass
        sd = meta.get("session_dt") or meta.get("session_date_time")
        if isinstance(sd, str) and sd.strip():
            return sd.strip()
        try:
            return time.strftime("%Y-%m-%d", time.gmtime(float(self.timestamp)))
        except Exception:
            return None


@dataclass
class RecallPack:
    """Structured response from recall - formatted for an LLM."""

    query: str
    workspace_name: str
    workspace_id: str
    results: list[RecallResult]
    n_total_in_workspace: int
    elapsed_ms: float
    # Optional escalation diagnostics, set by recall_smart / recall_deep:
    #   {"stages": [...], "stopped": "confidence_met"|"deadline_hit"|...,
    #    "elapsed_ms": ..., "deadline_ms": ..., "confidence": ...}
    # Lets a caller see what ran and why it stopped, so it doesn't fan out
    # redundant recalls after a low-confidence / timed-out result (#10).
    escalation: dict | None = None

    def to_dict(self) -> dict:
        d = {
            "query": self.query,
            "workspace": {"id": self.workspace_id, "name": self.workspace_name},
            "n_results": len(self.results),
            "n_total_in_workspace": self.n_total_in_workspace,
            "elapsed_ms": self.elapsed_ms,
            "confidence": self.confidence,
            "results": [r.to_dict() for r in self.results],
        }
        if self.escalation is not None:
            d["escalation"] = self.escalation
        return d

    @property
    def confidence(self) -> float:
        """Improvement G: confidence in this recall.

        Combines top-1 score with the gap to top-2 (larger gap = more
        confident the top hit is the right one). Returns 0..1.

        Used by escalation logic and by callers who want to decide whether
        to surface results or ask for clarification."""
        if not self.results:
            return 0.0
        # X3: calibration math lives in one named, tested function.
        from pmb.reasoning.scoring import calibrated_confidence
        top1 = float(self.results[0].score)
        top2 = float(self.results[1].score) if len(self.results) > 1 else None
        return calibrated_confidence(top1, top2)

    def to_text(self, max_results: int = 5) -> str:
        """Text representation for injection into a prompt."""
        if not self.results:
            return f"[Memory] No relevant memories found in workspace '{self.workspace_name}'."

        lines = [f"[Memory recall from '{self.workspace_name}']"]
        for r in self.results[:max_results]:
            ts = r.resolved_date or time.strftime("%Y-%m-%d", time.gmtime(r.timestamp))
            content_preview = r.content[:300] + "..." if len(r.content) > 300 else r.content
            lines.append(f"\n- [{ts}] [{r.event_type}] (score {r.score:.2f}):")
            lines.append(content_preview)
        return "\n".join(lines)


_MULTIHOP_RE = re.compile(
    r"\b(after|before|because|due to|caused|led to|then|next|earlier|"
    r"previously|why did|what happened (?:after|when)|since|until|"
    r"following|preceding|subsequently|as a result|consequence|"
    r"prior to|in response|reaction|triggered|prompted)\b",
    re.IGNORECASE,
)


def _looks_multihop(query: str) -> bool:
    """Cheap detection of multi-hop / temporal / causal query patterns."""
    if not query:
        return False
    return bool(_MULTIHOP_RE.search(query))


def _collapse_reflections(
    scored: list,
    event_store,
    workspace_id: str,
) -> list:
    """Collapse reflection events onto their source events.

    `scored` is a list of (SearchHit, Event, score, recency). For any
    reflection in the list with `metadata.source_ulid`:
      - if source_ulid is already a candidate: add reflection's score to
        the source's score and drop the reflection from the list
      - if source_ulid is not in candidates: load the source from store
        and REPLACE the reflection's entry with the source (keeping the
        reflection's score because it earned that ranking)
      - if source can't be loaded: keep the reflection as a fallback

    This is the key fix for benchmarks that score on source dia_ids:
    reflections served their bridge purpose during scoring, but the
    answer surfaced to the agent should be the original source event.
    """
    if not scored:
        return scored
    # Quick exit if there are no reflections
    has_refl = any(getattr(ev, "event_type", None) == "reflection" for _, ev, _, _ in scored)
    if not has_refl:
        return scored

    by_ulid = {ev.ulid: i for i, (_, ev, _, _) in enumerate(scored)}
    out: list = []
    drop_indices: set[int] = set()
    score_boost: dict[str, float] = {}
    add_back: list = []  # entries to add (source replaces reflection)

    for i, (h, ev, score, recency) in enumerate(scored):
        if ev.event_type != "reflection":
            continue
        src_ulid = (ev.metadata or {}).get("source_ulid") if ev.metadata else None
        if not src_ulid:
            continue
        if src_ulid in by_ulid:
            # Source already a candidate - transfer score
            score_boost[src_ulid] = score_boost.get(src_ulid, 0.0) + score * 0.5
            drop_indices.add(i)
        else:
            # Source not yet a candidate - fetch it, replace reflection
            src_ev = event_store.get_by_ulid(src_ulid)
            if src_ev is None or src_ev.archived_at is not None:
                continue  # keep reflection as fallback
            # Build a fresh SearchHit for the source carrying the reflection's score
            from pmb.core.search import SearchHit as _SH

            new_h = _SH(
                ulid=src_ev.ulid,
                score=score,
                bm25_score=h.bm25_score,
                vec_score=h.vec_score,
                importance=src_ev.importance,
                recency_score=h.recency_score,
            )
            add_back.append((new_h, src_ev, score, recency))
            drop_indices.add(i)

    # Build the rebuilt list
    for i, item in enumerate(scored):
        if i in drop_indices:
            continue
        h, ev, score, recency = item
        if ev.ulid in score_boost:
            score = score + score_boost[ev.ulid]
        out.append((h, ev, score, recency))
    out.extend(add_back)
    return out


# No-op context manager used when the embed-queue lock hasn't been created yet
class _DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_DUMMY_LOCK = _DummyLock()

# Improvement S: cross-kind entity dedup.
#
# When several extractors run over the same text, the same surface form often
# lands in multiple kinds:
#   - "alice"  → concept (length≥4 lowercase token)   AND   person
#   - "authmanager" → concept   AND   class   AND   function   AND   person
#   - "asyncpg"    → concept   AND   import
# We keep only the most specific kind. Order (highest priority first):
#   tech > file > class > function > import > person > theme > concept
#
# Code-AST kinds (class/function/import) outrank `person` because the regex
# person extractor will happily flag "AuthManager" as a capitalized name -
# but AST proves it's a code symbol. `person` still beats `concept` so
# "Alice" → person, not concept.
_KIND_PRIORITY: dict[str, int] = {
    "tech": 0,
    "file": 1,
    "class": 2,
    "function": 3,
    "import": 4,
    "person": 5,
    "theme": 6,
    "concept": 7,
}
# Per-deployment override: reference.yaml `kind_priority` adds/overrides ranks.
_KIND_PRIORITY = _override_dict("kind_priority", _KIND_PRIORITY)


def _truncate_marker(s: str, limit: int) -> str:
    if not isinstance(s, str) or len(s) <= limit:
        return s
    return s[:limit].rstrip() + "… [truncated by PMB]"


def _cap_batch_content(items: list[dict], max_content: int) -> list[dict]:
    """Improvement Z: cap content fields per item to avoid embedding-runaway
    on huge agent inputs (e.g. dumping full web-search results). Truncates
    `content`, `main`, `title`, and each `subfacts[i]` to `max_content` chars
    with a clear marker so downstream stays predictable.
    """
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            out.append(item)
            continue
        capped = dict(item)
        for k in ("content", "main", "title", "fact", "summary"):
            if k in capped and isinstance(capped[k], str):
                capped[k] = _truncate_marker(capped[k], max_content)
        if isinstance(capped.get("subfacts"), list):
            capped["subfacts"] = [
                _truncate_marker(s, max_content) if isinstance(s, str) else s
                for s in capped["subfacts"]
            ]
        out.append(capped)
    return out


def _dedupe_named_entities(named: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Collapse same-name entries to their highest-priority kind."""
    best: dict[str, tuple[int, str]] = {}  # name → (priority, kind)
    for kind, name in named:
        if not name:
            continue
        prio = _KIND_PRIORITY.get(kind, 99)
        cur = best.get(name)
        if cur is None or prio < cur[0]:
            best[name] = (prio, kind)
    # Stable order: same as original input
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for kind, name in named:
        if name in seen:
            continue
        seen.add(name)
        chosen_kind = best[name][1]
        out.append((chosen_kind, name))
    return out
