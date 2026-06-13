"""
Causation graph - typed edges between events.

Where this fits in PMB v2:
  Reflective Memory (phase 1) makes individual events richer.
  Causation makes the *connections between* events explicit.

What edges we extract:
  causes          - event A directly led to event B (decision → outcome)
  influenced      - event A shaped event B (preference → behaviour)
  enabled         - event A made event B possible (capability → action)
  blocked         - event A prevented something
  references      - event B explicitly mentions event A's subject
  temporal-next   - event B happened shortly after A in same context (free, no LLM)
  contradicts     - event B states something inconsistent with A

How extraction works:
  Two strategies, complementary:

  1. RULE-BASED (cheap, runs at write-time):
     - temporal-next: any two events within N minutes by same speaker/session
     - references: regex for explicit pronouns / quoted text from earlier event

  2. LLM-BASED (background, batched in sleep):
     - For each new event, fetch ~5 recent semantically-near events
     - Ask LLM: 'For each candidate, output one edge_type or "none"'
     - Persist non-'none' edges with confidence ≥ 0.5

How recall uses them:
  When the query router detects a multi-hop pattern (after/before/because/
  due-to), recall walks event_edges from initial hits forward AND backward,
  pulling 1-2 hops of connected events into the candidate pool.

Storage:
  Table `event_edges` (schema v4):
    source_ulid, target_ulid, edge_type, confidence, rationale, created_at

  Directional. (A causes B) ≠ (B causes A). For undirected edges (like
  'references') we store the version we extract; if mutual, both rows exist.

Cost:
  Rule edges: ~0ms (one SQL).
  LLM edges:  ~1 LLM call per new event (in idle), batched.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmb.core.engine import Engine
    from pmb.core.events import Event


log = logging.getLogger(__name__)


EDGE_TYPES = {
    "causes",
    "influenced",
    "enabled",
    "blocked",
    "references",
    "temporal-next",
    "contradicts",
}


CAUSATION_PROMPT = """\
You are analysing a memory system. The user just logged a new event.
For each candidate older event listed below, decide if there is a causal
or referential relationship FROM the older event TO the new one.

Output VALID JSON: a list of objects, one per candidate, in the same order:
  [
    {{"edge_type": "causes" | "influenced" | "enabled" | "blocked" |
                  "references" | "contradicts" | "none",
      "confidence": 0.0-1.0,
      "rationale": "one short sentence"}}
  ]

Rules:
- "none" if there is no real relationship.
- "causes": the older event DIRECTLY led to the new one.
- "influenced": the older event SHAPED the new one (softer).
- "enabled": the older event MADE POSSIBLE the new one.
- "blocked": the older event RULED OUT something the new one would have done.
- "references": the new event explicitly refers back to the older.
- "contradicts": the new event says something inconsistent with the older.
- Confidence below 0.4 → use "none".
- Output array length MUST equal number of candidates.

New event:
\"\"\"
{new_event}
\"\"\"

Candidates (in chronological order, oldest first):
{candidates}

JSON array:"""


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

@dataclass
class CausationEdge:
    source_ulid: str
    target_ulid: str
    edge_type: str
    confidence: float
    rationale: str = ""

    def is_valid(self) -> bool:
        return (
            self.edge_type in EDGE_TYPES
            and self.confidence >= 0.0
            and self.source_ulid != self.target_ulid
        )


# ----------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------

def upsert_edge(db_path, edge: CausationEdge) -> None:
    """Insert or bump confidence if already present."""
    if not edge.is_valid():
        return
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO event_edges
                (source_ulid, target_ulid, edge_type, confidence, rationale, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_ulid, target_ulid, edge_type) DO UPDATE SET
                confidence = MAX(event_edges.confidence, excluded.confidence),
                rationale  = COALESCE(NULLIF(excluded.rationale, ''), event_edges.rationale)
            """,
            (
                edge.source_ulid, edge.target_ulid, edge.edge_type,
                float(edge.confidence), edge.rationale or "", time.time(),
            ),
        )


def edges_from(db_path, source_ulid: str, min_confidence: float = 0.4) -> list[CausationEdge]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM event_edges WHERE source_ulid = ? AND confidence >= ?",
            (source_ulid, min_confidence),
        ).fetchall()
    return [
        CausationEdge(
            source_ulid=r["source_ulid"], target_ulid=r["target_ulid"],
            edge_type=r["edge_type"], confidence=float(r["confidence"]),
            rationale=r["rationale"] or "",
        )
        for r in rows
    ]


def edges_to(db_path, target_ulid: str, min_confidence: float = 0.4) -> list[CausationEdge]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM event_edges WHERE target_ulid = ? AND confidence >= ?",
            (target_ulid, min_confidence),
        ).fetchall()
    return [
        CausationEdge(
            source_ulid=r["source_ulid"], target_ulid=r["target_ulid"],
            edge_type=r["edge_type"], confidence=float(r["confidence"]),
            rationale=r["rationale"] or "",
        )
        for r in rows
    ]


def walk_edges(
    db_path,
    seed_ulids: Iterable[str],
    direction: str = "both",  # 'forward', 'backward', 'both'
    edge_types: set[str] | None = None,
    min_confidence: float = 0.4,
    hops: int = 1,
    max_results: int = 50,
) -> set[str]:
    """Walk the causation graph from seed events, return reached ulids.

    Used by multi-hop recall: given top hits from hybrid search, expand
    along typed edges to surface bridge events.
    """
    seen: set[str] = set(seed_ulids)
    frontier: set[str] = set(seed_ulids)
    out: set[str] = set()
    for _ in range(max(1, hops)):
        next_frontier: set[str] = set()
        if not frontier:
            break
        type_filter = ""
        params: list = []
        if edge_types:
            qmarks = ",".join("?" * len(edge_types))
            type_filter = f" AND edge_type IN ({qmarks})"
            params.extend(edge_types)
        ph = ",".join("?" * len(frontier))
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            if direction in ("forward", "both"):
                rows = conn.execute(
                    f"SELECT target_ulid AS u FROM event_edges "
                    f"WHERE source_ulid IN ({ph}) AND confidence >= ?{type_filter}",
                    (*frontier, min_confidence, *params),
                ).fetchall()
                for r in rows:
                    if r["u"] not in seen:
                        next_frontier.add(r["u"])
            if direction in ("backward", "both"):
                rows = conn.execute(
                    f"SELECT source_ulid AS u FROM event_edges "
                    f"WHERE target_ulid IN ({ph}) AND confidence >= ?{type_filter}",
                    (*frontier, min_confidence, *params),
                ).fetchall()
                for r in rows:
                    if r["u"] not in seen:
                        next_frontier.add(r["u"])
        out.update(next_frontier)
        seen.update(next_frontier)
        frontier = next_frontier
        if len(out) >= max_results:
            break
    return out


# ----------------------------------------------------------------------
# Cheap rule-based edges (no LLM)
# ----------------------------------------------------------------------

def add_temporal_next_edge(engine: Engine, event: Event) -> CausationEdge | None:
    """Link `event` to the immediately-previous event in the same workspace
    if their gap is short enough to suggest continuity. Free, no LLM."""
    GAP_THRESHOLD_SEC = 600.0  # 10 minutes default
    with sqlite3.connect(engine.workspace.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT ulid, timestamp FROM events
            WHERE workspace_id = ? AND ulid != ? AND archived_at IS NULL
              AND event_type != 'reflection'
              AND timestamp < ?
            ORDER BY timestamp DESC LIMIT 1
            """,
            (engine.workspace.id, event.ulid, event.timestamp),
        ).fetchone()
    if not row:
        return None
    gap = event.timestamp - row["timestamp"]
    if gap > GAP_THRESHOLD_SEC:
        return None
    edge = CausationEdge(
        source_ulid=row["ulid"],
        target_ulid=event.ulid,
        edge_type="temporal-next",
        confidence=1.0 - min(0.9, gap / GAP_THRESHOLD_SEC),
        rationale=f"~{int(gap)}s gap",
    )
    upsert_edge(engine.workspace.db_path, edge)
    return edge


# ----------------------------------------------------------------------
# LLM-based extraction (background)
# ----------------------------------------------------------------------

class CausationExtractor:
    """Run an LLM to type the edges between a new event and recent candidates."""

    def __init__(self, llm_client, max_candidates: int = 6):
        self.llm = llm_client
        self.max_candidates = max_candidates

    def extract(
        self, new_event: Event, candidates: list[Event],
    ) -> list[CausationEdge]:
        if not candidates:
            return []
        candidates = candidates[: self.max_candidates]
        cand_text = "\n\n".join(
            f"[{i+1}] (ts={int(c.timestamp)}) {(c.content or '')[:300]}"
            for i, c in enumerate(candidates)
        )
        prompt = CAUSATION_PROMPT.format(
            new_event=(new_event.content or "")[:600],
            candidates=cand_text,
        )
        try:
            text = self._call_llm(prompt)
        except Exception as e:
            log.warning("CausationExtractor LLM call failed: %s", e)
            return []
        try:
            arr = self._parse_array(text)
        except Exception as e:
            log.warning("CausationExtractor non-JSON output: %s | %s", e, text[:200])
            return []

        out: list[CausationEdge] = []
        for i, item in enumerate(arr[: len(candidates)]):
            if not isinstance(item, dict):
                continue
            etype = str(item.get("edge_type", "none")).strip().lower()
            if etype == "none" or etype not in EDGE_TYPES:
                continue
            conf = float(item.get("confidence", 0.0))
            if conf < 0.4:
                continue
            out.append(CausationEdge(
                source_ulid=candidates[i].ulid,
                target_ulid=new_event.ulid,
                edge_type=etype,
                confidence=conf,
                rationale=str(item.get("rationale", ""))[:200],
            ))
        return out

    def _call_llm(self, prompt: str) -> str:
        if hasattr(self.llm, "complete"):
            return self.llm.complete(prompt) or ""
        if callable(self.llm):
            return self.llm(prompt) or ""
        raise RuntimeError("CausationExtractor: llm must have .complete() or be callable")

    def _parse_array(self, text: str) -> list:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rsplit("```", 1)[0].strip()
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("no JSON array in output")
        return json.loads(text[start : end + 1])
