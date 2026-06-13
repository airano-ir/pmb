"""Wall-clock + call-count budget for offline LLM passes.

Per-call timeouts bound a SINGLE call; they do nothing about a loop that makes
40 sequential calls of 15 s each (10 minutes). Every batched/offline LLM loop
- keyed-state suggestions, the declutter judge, consolidation clusters -
constructs one budget at the top and checks ``allow()`` before EVERY call so
the WHOLE pass is bounded.

Never used on a synchronous hot path; these passes are offline by design.
"""
from __future__ import annotations

import time


class LLMBudget:
    """A spend-as-you-go budget. ``allow()`` is the gate, ``spend()`` is called
    after each LLM call, ``report()`` is merged into the pass result so the CLI
    can show what was actually spent (and whether the cap stopped it early)."""

    def __init__(self, max_calls: int = 40, max_wall_s: float = 120.0):
        self.max_calls = max(1, int(max_calls))
        self.max_wall_s = max(0.0, float(max_wall_s))
        self._t0 = time.monotonic()
        self.calls = 0

    def allow(self) -> bool:
        """True while there is budget for one more call."""
        return (self.calls < self.max_calls
                and (time.monotonic() - self._t0) < self.max_wall_s)

    def spend(self) -> None:
        self.calls += 1

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._t0

    def report(self) -> dict:
        return {
            "llm_calls": self.calls,
            "elapsed_s": round(self.elapsed_s, 1),
            "max_calls": self.max_calls,
            "max_wall_s": self.max_wall_s,
        }


def budget_from_config(config, default_calls: int = 40,
                       default_wall_s: float = 120.0) -> LLMBudget:
    """Build an LLMBudget from the additive config keys, tolerant of a missing
    config object (tests, bare callers)."""
    calls = default_calls
    wall = default_wall_s
    try:
        calls = int(config.get("llm.offline_max_calls") or default_calls)
        wall = float(config.get("llm.offline_budget_s") or default_wall_s)
    except Exception:
        pass
    return LLMBudget(max_calls=calls, max_wall_s=wall)
