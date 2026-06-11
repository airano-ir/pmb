"""
Self-Test Runner — quantifies memory degradation over time.

Idea: periodically (once a week) the system picks random old memories,
builds test queries out of them, and tries to recall them via recall.
If recall accuracy drops — that's a degradation signal.

Metric: % of old memories that recall finds in top-K using their own key
words/phrase.

Query generation approach:
- For each selected event we take the first 8-15 significant tokens of its content
- That becomes the query
- Expected — the same event_ulid

This is a "scratch your own itch" benchmark — it measures against the user's
real data, not synthetic data. Results are saved to health_log.jsonl and are
available through the `pmb health` CLI.

Anti-bias measure: the query differs substantially from the content (only a
subset of tokens in random order) — otherwise it would be trivial.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pmb import lang as _lang

if TYPE_CHECKING:
    from pmb.core.engine import Engine


# Stopwords that don't work as query keywords. EN floor inline; the RU/UK
# function words ("this"/"what"/"how"/"when" in RU/UK) live in the packs
# (self_test_stopwords)
# and merge in below, keeping this module Cyrillic-free (L1).
STOPWORDS = _lang.merged_set("self_test_stopwords", {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "should",
    "can", "could", "may", "might", "must", "and", "or", "but", "if",
    "of", "in", "on", "at", "to", "for", "with", "from", "by", "as",
    "user", "assistant", "q",
})


def _significant_tokens(text: str, n_keep: int = 8) -> list[str]:
    """
    Extract significant tokens: words >= 3 chars, not stopwords, no duplicates.
    """
    raw = re.findall(r"\w+", text.lower())
    seen = set()
    out = []
    for tok in raw:
        if len(tok) < 3:
            continue
        if tok in STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= n_keep:
            break
    return out


def generate_test_query(content: str, rng: random.Random) -> str | None:
    """
    Generates a test query from an event's content.

    We take 4-7 significant tokens in random order. If there are fewer than 4
    significant tokens — the event isn't suitable for self-test (too short).
    """
    tokens = _significant_tokens(content, n_keep=12)
    if len(tokens) < 4:
        return None
    n = rng.randint(4, min(7, len(tokens)))
    sample = rng.sample(tokens, n)
    return " ".join(sample)


@dataclass
class SelfTestResult:
    timestamp: float
    n_tested: int
    accuracy_at_1: float
    accuracy_at_3: float
    accuracy_at_5: float
    avg_rank: float | None
    failed_queries: list[dict] = field(default_factory=list)
    workspace_id: str = ""
    n_total_active: int = 0
    # Session-aware loose metric: counts as success if ANY event from the
    # same session appears in top-K. Closer to "did the retriever find the
    # right neighborhood" than the strict exact-ulid match. Falls back to
    # strict when events have no session_id.
    session_accuracy_at_5: float | None = None
    session_coverage: float = 0.0  # fraction of samples that had a session_id
    # When n_tested == 0, why: "no_content_events" or "all_events_younger_than_min_age"
    empty_reason: str | None = None
    eligible_min_age_days: float = 1.0
    n_too_recent: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class SelfTestRunner:
    """
    Runs self-tests and persists results in health_log.jsonl.
    """

    def __init__(self, engine: Engine, seed: int | None = None):
        self.engine = engine
        self.rng = random.Random(seed if seed is not None else int(time.time()))

    def _log_path(self) -> Path:
        return self.engine.workspace.storage_dir / "health_log.jsonl"

    def run(
        self,
        n_samples: int = 20,
        min_age_days: float = 1.0,
        top_k_max: int = 10,
        max_failed_to_save: int = 5,
    ) -> SelfTestResult:
        """
        Run the self-test.

        Takes events older than min_age_days, samples n_samples of them,
        generates a query from each, runs recall, and counts the hits.

        Args:
            n_samples: how many events to test
            min_age_days: minimum age of an event (fresh ones are skipped — too easy)
            top_k_max: how many results to request from recall
            max_failed_to_save: we save the details of failed queries
        """
        workspace_id = self.engine.workspace.id
        active = self.engine.events.list_active(workspace_id, limit=10000)

        # Filter: older than min_age_days and content-bearing (qa, fact, git)
        cutoff_ts = time.time() - min_age_days * 86400.0
        eligible_types = {"qa", "fact", "git"}
        eligible = [
            e for e in active
            if e.timestamp <= cutoff_ts and e.event_type in eligible_types
        ]

        if not eligible:
            # Distinguish "empty workspace" from "everything is too fresh" — both
            # produce zero results but the second one tells the user to wait.
            n_recent = sum(
                1 for e in active
                if e.timestamp > cutoff_ts and e.event_type in eligible_types
            )
            reason = (
                "no_content_events"
                if not any(e.event_type in eligible_types for e in active)
                else "all_events_younger_than_min_age"
                if n_recent > 0
                else "no_content_events"
            )
            return SelfTestResult(
                timestamp=time.time(),
                n_tested=0,
                accuracy_at_1=0.0,
                accuracy_at_3=0.0,
                accuracy_at_5=0.0,
                avg_rank=None,
                workspace_id=workspace_id,
                n_total_active=len(active),
                empty_reason=reason,
                eligible_min_age_days=min_age_days,
                n_too_recent=n_recent,
            )

        # Sample
        sample_size = min(n_samples, len(eligible))
        samples = self.rng.sample(eligible, sample_size)

        n_at_1 = 0
        n_at_3 = 0
        n_at_5 = 0
        n_session_at_5 = 0
        n_with_session = 0
        ranks = []
        failed = []

        # Map session_id -> set(ulids) for loose-mode success check
        session_map: dict[str, set[str]] = {}
        for e in active:
            if e.source_session_id:
                session_map.setdefault(e.source_session_id, set()).add(e.ulid)

        for ev in samples:
            query = generate_test_query(ev.to_text(), self.rng)
            if not query:
                continue

            pack = self.engine.recall(query=query, top_k=top_k_max)
            rank = None
            for i, r in enumerate(pack.results, 1):
                if r.ulid == ev.ulid:
                    rank = i
                    break

            # Session-loose match: any same-session ulid within top-5
            session_hit = False
            if ev.source_session_id:
                n_with_session += 1
                sibling_set = session_map.get(ev.source_session_id, set())
                for r in pack.results[:5]:
                    if r.ulid in sibling_set:
                        session_hit = True
                        break

            if rank is not None:
                ranks.append(rank)
                if rank <= 1:
                    n_at_1 += 1
                if rank <= 3:
                    n_at_3 += 1
                if rank <= 5:
                    n_at_5 += 1
            else:
                if len(failed) < max_failed_to_save:
                    failed.append({
                        "ulid": ev.ulid,
                        "query": query,
                        "expected_content_preview": ev.content[:100],
                        "top3_in_results": [
                            {"ulid": r.ulid, "score": r.score, "content_preview": r.content[:60]}
                            for r in pack.results[:3]
                        ],
                    })
            if session_hit:
                n_session_at_5 += 1

        # n_tested via the rng is unstable (rng state advances); recompute it
        # below from the deterministic generate_test_query pass instead.
        attempted = 0
        for s in samples:
            if generate_test_query(s.to_text(), random.Random(0)) is not None:
                attempted += 1
        n_tested = max(attempted, len(ranks) + len(failed))

        session_acc = (n_session_at_5 / n_with_session) if n_with_session else None
        result = SelfTestResult(
            timestamp=time.time(),
            n_tested=n_tested,
            accuracy_at_1=n_at_1 / max(n_tested, 1),
            accuracy_at_3=n_at_3 / max(n_tested, 1),
            accuracy_at_5=n_at_5 / max(n_tested, 1),
            avg_rank=(sum(ranks) / len(ranks)) if ranks else None,
            failed_queries=failed,
            workspace_id=workspace_id,
            n_total_active=len(active),
            session_accuracy_at_5=session_acc,
            session_coverage=(n_with_session / max(n_tested, 1)) if n_tested else 0.0,
        )

        self._append_log(result)
        return result

    def _append_log(self, result: SelfTestResult):
        log = self._log_path()
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    def history(self, limit: int = 20) -> list[SelfTestResult]:
        """Read historic self-test results."""
        log = self._log_path()
        if not log.exists():
            return []
        results = []
        with open(log, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    results.append(SelfTestResult(**data))
                except Exception:
                    continue
        return results[-limit:]

    def trend(self) -> dict:
        """
        Trend analysis: is there degradation?

        Returns:
        {
            "n_runs": int,
            "first_run_acc5": float,
            "last_run_acc5": float,
            "delta_pp": float,
            "verdict": "stable" | "degrading" | "improving" | "insufficient"
        }
        """
        hist = self.history(limit=100)
        if len(hist) < 2:
            return {"n_runs": len(hist), "verdict": "insufficient"}

        first = hist[0].accuracy_at_5
        last = hist[-1].accuracy_at_5
        delta = (last - first) * 100  # pp

        if abs(delta) < 5.0:
            verdict = "stable"
        elif delta < 0:
            verdict = "degrading"
        else:
            verdict = "improving"

        return {
            "n_runs": len(hist),
            "first_run_acc5": first,
            "last_run_acc5": last,
            "delta_pp": delta,
            "verdict": verdict,
            "first_timestamp": hist[0].timestamp,
            "last_timestamp": hist[-1].timestamp,
        }
