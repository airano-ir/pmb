"""
Spaced-repetition rehearsal.

Cognitive analogy: a memory you haven't accessed in a while becomes harder
to retrieve. The brain combats this by re-activating high-value memories
periodically (the so-called "testing effect"). Without rehearsal, even
important facts drift below the noise floor and eventually archive out.

What this module does:
  - Walks all active events (or a sample) in importance ≥ threshold.
  - Skips events that have been accessed recently (within `min_idle_days`).
  - For each eligible event, simulates an internal recall using the event's
    own content as the query. The recall pipeline already includes the
    reinforcement step (`apply_recall_updates` bumps importance + touches),
    so the rehearsal naturally pushes the memory back up the tier ladder.

Rate-limited: we cap the number of rehearsed events per run to keep
runtime predictable. Designed to be triggered from `pmb rehearse` CLI
or weekly via cron.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmb.core.engine import Engine


@dataclass
class RehearseResult:
    n_candidates: int
    n_rehearsed: int
    n_failed: int
    elapsed_seconds: float
    rehearsed_ulids: list[str]

    def to_dict(self) -> dict:
        return {
            "n_candidates": self.n_candidates,
            "n_rehearsed": self.n_rehearsed,
            "n_failed": self.n_failed,
            "elapsed_seconds": self.elapsed_seconds,
            "rehearsed_ulids": self.rehearsed_ulids,
        }


def _query_from_event(text: str, max_tokens: int = 7) -> str:
    """Cheap query synthesis from event content — same approach as self-test
    so a rehearsal really exercises the recall pipeline."""
    import re
    tokens = [t for t in re.findall(r"\w+", text.lower()) if len(t) >= 3]
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_tokens:
            break
    return " ".join(out)


def rehearse(
    engine: Engine,
    importance_threshold: float = 0.5,
    min_idle_days: float = 7.0,
    max_rehearse: int = 20,
    seed: int | None = None,
) -> RehearseResult:
    """Run a rehearsal pass.

    Args:
      importance_threshold: rehearse memories at or above this importance
      min_idle_days: skip recently-accessed events
      max_rehearse: hard cap on rehearsals per call (keeps runtime bounded)
      seed: for deterministic sampling in tests
    """
    rng = random.Random(seed if seed is not None else int(time.time()))
    t0 = time.time()
    now = time.time()

    workspace_id = engine.workspace.id
    active = engine.events.list_active(workspace_id, limit=10000)

    # Eligibility
    idle_cutoff = now - min_idle_days * 86400.0
    eligible = [
        e for e in active
        if e.importance >= importance_threshold
        and e.importance < 0.99  # already-pinned needs no rehearsal
        and e.last_accessed <= idle_cutoff
        and e.event_type in ("qa", "fact", "git")
    ]

    if not eligible:
        return RehearseResult(
            n_candidates=0, n_rehearsed=0, n_failed=0,
            elapsed_seconds=time.time() - t0, rehearsed_ulids=[],
        )

    # Prioritise by importance: most-important-first, with a little jitter
    eligible.sort(key=lambda e: (-e.importance, rng.random()))
    sampled = eligible[:max_rehearse]

    n_ok = 0
    n_fail = 0
    rehearsed_ulids: list[str] = []
    for ev in sampled:
        query = _query_from_event(ev.content)
        if not query:
            continue
        try:
            pack = engine.recall(query, top_k=10, rerank=False)
        except Exception:
            n_fail += 1
            continue
        # Was the event actually surfaced? If yes, recall pipeline already
        # touched + bumped it via apply_recall_updates.
        found = any(r.ulid == ev.ulid for r in pack.results)
        if found:
            n_ok += 1
            rehearsed_ulids.append(ev.ulid)
        else:
            # Even on miss, give a small explicit importance nudge — the
            # memory is important enough that we don't want it to vanish
            # because our query synthesis was weak.
            new_imp = min(1.0, ev.importance + 0.02)
            engine.events.update_importance(ev.ulid, new_imp)
            n_ok += 1
            rehearsed_ulids.append(ev.ulid)

    return RehearseResult(
        n_candidates=len(eligible),
        n_rehearsed=n_ok,
        n_failed=n_fail,
        elapsed_seconds=time.time() - t0,
        rehearsed_ulids=rehearsed_ulids,
    )
