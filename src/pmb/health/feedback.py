"""
Recall feedback log — real-user signal source.

The point of this module: replace the closed-loop self-test as the primary
source of "which recalls actually matter" data. Self-test stays as fallback
when there's no user feedback yet.

Per-workspace `recall_feedback.jsonl` lines:
  {timestamp, ulid, verdict, query?, expected_ulid?, session_id?}

Verdicts:
  useful      — the returned event was what the agent/user needed
  wrong       — returned but not what was wanted (with optional expected_ulid)
  irrelevant  — should not have been retrieved at all

Aggregation (`summary`) returns:
  total, useful, wrong, irrelevant, useful_rate, n_unique_queries,
  most_useful_events (top 10), most_wrong_events (top 10).

These numbers are the metric to watch over 2 weeks of dogfooding.
A self-test acc@5 of 92% with useful_rate < 60% means the synthetic
metric is lying.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmb.core.engine import Engine


VALID_VERDICTS = {"useful", "wrong", "irrelevant"}


@dataclass
class FeedbackEntry:
    timestamp: float
    ulid: str
    verdict: str
    query: str | None = None
    expected_ulid: str | None = None
    session_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _feedback_path(engine: Engine) -> Path:
    return engine.workspace.storage_dir / "recall_feedback.jsonl"


def record_feedback(
    engine: Engine,
    ulid: str,
    verdict: str,
    query: str | None = None,
    expected_ulid: str | None = None,
) -> dict:
    """
    Record one feedback line and apply lightweight reinforcement.

    Side effects on importance:
    - useful      → small positive boost on `ulid`
    - wrong       → mild negative on `ulid`; if expected_ulid given, boost expected
    - irrelevant  → mild negative on `ulid`

    Pinned events (importance >= 0.99) are not touched in either direction.
    """
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"verdict must be one of {sorted(VALID_VERDICTS)}, got {verdict!r}")

    # Validate ULIDs against storage — silently accepting typos pollutes the
    # feedback log permanently. Only validate within this workspace.
    target = engine.events.get_by_ulid(ulid)
    if target is None or target.workspace_id != engine.workspace.id:
        raise LookupError(
            f"ulid {ulid!r} not found in workspace {engine.workspace.name!r}"
        )
    if expected_ulid:
        expected_ev = engine.events.get_by_ulid(expected_ulid)
        if expected_ev is None or expected_ev.workspace_id != engine.workspace.id:
            raise LookupError(
                f"expected_ulid {expected_ulid!r} not found in workspace"
            )

    # F3: a 'useful' verdict labels the surfaced result's channel-weight sample,
    # closing the X2 learning loop (best-effort, gated, never auto-applied).
    if verdict == "useful":
        try:
            if engine.config.get("recall.weight_learning"):
                from pmb.reasoning.weight_learning import note_recall_useful
                note_recall_useful(engine, ulid)
        except Exception:
            pass

    sess = engine.session_tracker.current(auto_create=False)
    entry = FeedbackEntry(
        timestamp=time.time(),
        ulid=ulid,
        verdict=verdict,
        query=query,
        expected_ulid=expected_ulid,
        session_id=sess.id if sess else None,
    )

    path = _feedback_path(engine)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    # Lightweight reinforcement — explicit user signal beats synthetic self-test
    if target.importance < 0.99:
        if verdict == "useful":
            new_imp = min(1.0, target.importance + 0.08 * (1.0 - target.importance))
            engine.events.update_importance(ulid, new_imp)
        elif verdict in ("wrong", "irrelevant"):
            new_imp = max(0.0, target.importance - 0.05)
            engine.events.update_importance(ulid, new_imp)

    if expected_ulid and expected_ev is not None and expected_ev.importance < 0.99:
        new_imp = min(1.0, expected_ev.importance + 0.15 * (1.0 - expected_ev.importance))
        engine.events.update_importance(expected_ulid, new_imp)

    return {
        "ulid": ulid,
        "verdict": verdict,
        "recorded_at": entry.timestamp,
        "expected_boosted": expected_ulid is not None,
    }


def history(engine: Engine, limit: int | None = None) -> list[FeedbackEntry]:
    """Read all feedback entries (or last N)."""
    path = _feedback_path(engine)
    if not path.exists():
        return []
    out: list[FeedbackEntry] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                out.append(FeedbackEntry(**data))
            except Exception:
                continue
    if limit is not None:
        out = out[-limit:]
    return out


def summary(engine: Engine) -> dict:
    """Aggregate feedback into a single dict."""
    entries = history(engine)
    if not entries:
        return {
            "total": 0,
            "useful": 0,
            "wrong": 0,
            "irrelevant": 0,
            "useful_rate": None,
            "n_unique_queries": 0,
            "most_useful_events": [],
            "most_wrong_events": [],
            "verdict": "no_data",
        }

    useful = sum(1 for e in entries if e.verdict == "useful")
    wrong = sum(1 for e in entries if e.verdict == "wrong")
    irrelevant = sum(1 for e in entries if e.verdict == "irrelevant")
    n_judged = useful + wrong + irrelevant
    rate = useful / n_judged if n_judged else None

    useful_counter: Counter = Counter()
    wrong_counter: Counter = Counter()
    for e in entries:
        if e.verdict == "useful":
            useful_counter[e.ulid] += 1
        elif e.verdict in ("wrong", "irrelevant"):
            wrong_counter[e.ulid] += 1

    queries = {e.query for e in entries if e.query}

    if rate is None:
        verdict = "no_data"
    elif rate >= 0.7:
        verdict = "healthy"
    elif rate >= 0.4:
        verdict = "mixed"
    else:
        verdict = "poor"

    return {
        "total": len(entries),
        "useful": useful,
        "wrong": wrong,
        "irrelevant": irrelevant,
        "useful_rate": rate,
        "n_unique_queries": len(queries),
        "most_useful_events": useful_counter.most_common(10),
        "most_wrong_events": wrong_counter.most_common(10),
        "verdict": verdict,
    }


def expected_ulid_boost_history(engine: Engine) -> dict[str, int]:
    """Count how many times each ulid was named as expected_ulid in 'wrong' feedback."""
    counts: dict[str, int] = {}
    for e in history(engine):
        if e.verdict == "wrong" and e.expected_ulid:
            counts[e.expected_ulid] = counts.get(e.expected_ulid, 0) + 1
    return counts
