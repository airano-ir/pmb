"""M1 — daemon self-maintenance tick.

Nothing ran decay / conflict-scan / declutter automatically: junk accumulated
unless the user hand-installed cron. The daemon already owns a process
lifecycle, so it's the natural place to tend the store. Once per
``daemon.maintenance_interval_h`` of uptime, and ONLY when the daemon has been
idle for ``daemon.maintenance_idle_min`` minutes (so it never competes with a
live request), it runs — each step independently guarded + errlogged so one
failure can't abort the rest:

  * archive_cold      — archive cold / low-value / old fact+activity rows.
                        Archive-only (reversible, ``archived_reason=decay_cold``);
                        NEVER lessons / goals, and decisions survive because they
                        carry high importance and live in the semantic tier (R7),
                        above ``decay.archive_cold_max_importance``.
  * detect_conflicts  — REPORT only (surfaced to ``pmb doctor``); never resolved.
  * declutter         — DRY-RUN only (report); never auto-applied.

Everything is archive-only or report-only: the tick NEVER hard-deletes, never
resolves a conflict, and never declutters for real without the user. Gated by
``daemon.maintenance`` (default on); the archive step additionally respects
``daemon.maintenance_archive`` (default on) so a cautious operator can keep the
reports while disabling auto-archive.
"""
from __future__ import annotations

import time
from typing import Any


def should_run_maintenance(
    now: float,
    last_tick_ts: float,
    interval_s: float,
    last_request_ts: float,
    idle_min_s: float,
) -> bool:
    """Pure scheduler predicate (injected clock → unit-testable). Run iff enough
    uptime has elapsed since the previous tick AND the daemon is idle right now."""
    if now - last_tick_ts < interval_s:
        return False
    if now - last_request_ts < idle_min_s:
        return False
    return True


def run_maintenance_tick(engine: Any, *, archive: bool = True,
                         now: float | None = None) -> dict:
    """Run one maintenance pass. Returns a summary dict
    ``{ran_at, ok, steps: {name: {...}}}``. Pure of scheduling — the caller
    decides WHEN; this decides WHAT, all archive-only / report-only."""
    summary: dict = {"ran_at": now if now is not None else time.time(),
                     "ok": True, "steps": {}}

    def _step(name: str, fn) -> None:
        try:
            summary["steps"][name] = fn()
        except Exception as e:  # one bad step must not abort the rest
            summary["steps"][name] = {"error": str(e)[:200]}
            summary["ok"] = False
            try:
                from pmb.core.errlog import log_error
                log_error(engine.workspace.db_path, f"maintenance.{name}", e)
            except Exception:
                pass

    # 1. archive cold/low-value/old rows (archive-only, reversible).
    if archive:
        _step("archive_cold",
              lambda: {"archived": int(engine.archive_cold(dry_run=False).get("n", 0))})
    else:
        summary["steps"]["archive_cold"] = {"skipped": "maintenance_archive=off"}

    # 2. conflict scan — REPORT only.
    _step("conflicts", lambda: {"found": len(engine.detect_conflicts())})

    # 3. declutter — DRY-RUN only (never apply).
    def _declutter_dryrun() -> dict:
        from pmb.maintenance.declutter import declutter
        r = declutter(engine, apply=False)
        return {"would_archive": int(r.get("n", 0))}
    _step("declutter_dryrun", _declutter_dryrun)

    return summary
