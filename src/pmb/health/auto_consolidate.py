"""
Auto-trigger for consolidation - sleep on a schedule, not by hand.

Cognitive analogy: the brain consolidates during sleep without you asking.
PMB should be similar - `pmb consolidate` shouldn't have to be a manual
ritual. This module tracks how many events have arrived since the last
consolidation and how much wall-clock time has passed; if either
threshold is crossed, the next idle moment triggers a sleep pass.

State persisted to `<workspace>/consolidation_state.yaml`:
  - last_consolidation_at: unix timestamp
  - n_events_at_last:      total event count at that moment
  - n_runs:                lifetime counter

Trigger logic (OR):
  - (current_total_events - n_events_at_last) >= auto_min_new_events
  - now - last_consolidation_at >= auto_min_days * 86400

Off by default. Even when on, can be disabled per call (`force=False`).
The trigger does NOT consolidate inline - callers pass in their LLM
client and decide whether to run synchronously. We just decide *when*.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from pmb.core.engine import Engine


@dataclass
class TriggerState:
    last_consolidation_at: float = 0.0
    n_events_at_last: int = 0
    n_runs: int = 0


def _state_path(engine: Engine) -> Path:
    return engine.workspace.storage_dir / "consolidation_state.yaml"


def load_state(engine: Engine) -> TriggerState:
    p = _state_path(engine)
    if not p.exists():
        return TriggerState()
    try:
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return TriggerState(
            last_consolidation_at=float(data.get("last_consolidation_at", 0.0)),
            n_events_at_last=int(data.get("n_events_at_last", 0)),
            n_runs=int(data.get("n_runs", 0)),
        )
    except Exception:
        return TriggerState()


def save_state(engine: Engine, state: TriggerState) -> None:
    p = _state_path(engine)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "last_consolidation_at": state.last_consolidation_at,
            "n_events_at_last": state.n_events_at_last,
            "n_runs": state.n_runs,
        }, f)


def should_trigger(engine: Engine) -> dict:
    """Inspect state vs thresholds, return decision dict.

    Returns:
      {
        "should_run": bool,
        "reason": str,
        "new_events_since_last": int,
        "days_since_last": float,
        "thresholds": {...},
      }
    """
    cfg = engine.config
    if not cfg.get("consolidate.auto_trigger"):
        return {"should_run": False, "reason": "auto_trigger_disabled"}

    state = load_state(engine)
    n_events_now = engine.events.count(engine.workspace.id, include_archived=True)
    new_events = max(0, n_events_now - state.n_events_at_last)
    days_since = (time.time() - state.last_consolidation_at) / 86400.0 if state.last_consolidation_at else float("inf")

    min_events = cfg.get("consolidate.auto_min_new_events")
    min_days = cfg.get("consolidate.auto_min_days")

    reason_parts = []
    if new_events >= min_events:
        reason_parts.append(f"{new_events} new events ≥ {min_events}")
    if days_since >= min_days:
        if days_since == float("inf"):
            reason_parts.append("never consolidated before")
        else:
            reason_parts.append(f"{days_since:.1f}d since last ≥ {min_days}d")

    return {
        "should_run": bool(reason_parts),
        "reason": " AND ".join(reason_parts) if reason_parts else "thresholds_not_met",
        "new_events_since_last": new_events,
        "days_since_last": days_since,
        "thresholds": {
            "min_new_events": min_events,
            "min_days": min_days,
        },
    }


def mark_consolidation_done(engine: Engine) -> None:
    """Reset the trigger state - call after a consolidation run completes."""
    state = load_state(engine)
    state.last_consolidation_at = time.time()
    state.n_events_at_last = engine.events.count(
        engine.workspace.id, include_archived=True,
    )
    state.n_runs += 1
    save_state(engine, state)
