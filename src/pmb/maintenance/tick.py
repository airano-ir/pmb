"""M1 - daemon self-maintenance tick.

Nothing ran decay / conflict-scan / declutter automatically: junk accumulated
unless the user hand-installed cron. The daemon already owns a process
lifecycle, so it's the natural place to tend the store. Once per
``daemon.maintenance_interval_h`` of uptime, and ONLY when the daemon has been
idle for ``daemon.maintenance_idle_min`` minutes (so it never competes with a
live request), it runs - each step independently guarded + errlogged so one
failure can't abort the rest:

  * apply_daily_decay - tier-aware forgetting curve (importance decays by tier
                        half-life; lessons / decisions floored, pinned skipped).
                        Runs BEFORE archive_cold so freshly down-weighted rows
                        leave recall the same tick. Gated by
                        ``daemon.maintenance_decay`` (default on).
  * archive_cold      - archive cold / low-value / old fact+activity rows.
                        Archive-only (reversible, ``archived_reason=decay_cold``);
                        NEVER lessons / goals, and decisions survive because they
                        carry high importance and live in the semantic tier (R7),
                        above ``decay.archive_cold_max_importance``.
  * detect_conflicts  - REPORT only (surfaced to ``pmb doctor``); never resolved.
  * declutter         - DRY-RUN only (report); never auto-applied.

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
                         decay_days: float = 1.0,
                         now: float | None = None) -> dict:
    """Run one maintenance pass. Returns a summary dict
    ``{ran_at, ok, steps: {name: {...}}}``. Pure of scheduling - the caller
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

    # 1. decay importance (the forgetting curve) FIRST, so freshly
    # down-weighted rows drop under archive_cold's importance threshold and
    # leave recall this same tick. apply_daily_decay floors lessons / decisions
    # and skips pinned, so nothing durable evaporates.
    try:
        decay_on = bool(engine.config.get("daemon.maintenance_decay"))
    except Exception:
        decay_on = True
    if decay_on:
        def _decay() -> dict:
            r = engine.apply_daily_decay(days_since=max(0.0, decay_days))
            return {"decayed": int(r.get("n_decayed", 0)),
                    "archived": int(r.get("n_archived", 0))}
        _step("decay", _decay)
    else:
        summary["steps"]["decay"] = {"skipped": "maintenance_decay=off"}

    # 2. archive cold/low-value/old rows (archive-only, reversible).
    if archive:
        _step("archive_cold",
              lambda: {"archived": int(engine.archive_cold(dry_run=False).get("n", 0))})
    else:
        summary["steps"]["archive_cold"] = {"skipped": "maintenance_archive=off"}

    # 2. conflict scan - REPORT only.
    _step("conflicts", lambda: {"found": len(engine.detect_conflicts())})

    # 3. declutter - DRY-RUN only (never apply).
    def _declutter_dryrun() -> dict:
        from pmb.maintenance.declutter import declutter
        r = declutter(engine, apply=False)
        return {"would_archive": int(r.get("n", 0))}
    _step("declutter_dryrun", _declutter_dryrun)

    # 4. ALD (Phase D) - distil the anchor-fire log into $PMB_HOME/lang/auto.yaml
    # so the COLD lexical path learns this machine's languages from its own
    # traffic. Writes only when n-grams clear support/precision; safe + local.
    def _distill() -> dict:
        try:
            if not engine.config.get("lang.anchor_log"):
                return {"skipped": "lang.anchor_log=off"}
        except Exception:
            pass
        from pmb.maintenance.distill import distill_lexicon
        return distill_lexicon(engine)
    _step("distill", _distill)

    # 5. E1 - refresh corpus-derived stopwords (document frequency) for this
    # workspace and apply them to the live tokenizer. Gated; additive.
    def _corpus_stopwords() -> dict:
        try:
            if not engine.config.get("lang.corpus_stopwords"):
                return {"skipped": "lang.corpus_stopwords=off"}
        except Exception:
            pass
        from pmb.maintenance.corpus_stopwords import refresh_corpus_stopwords
        return refresh_corpus_stopwords(engine)
    _step("corpus_stopwords", _corpus_stopwords)

    # 6. F3 - propose channel weights from collected feedback samples (X2 loop).
    # Suggestion only; `pmb doctor` surfaces it, never auto-applied.
    def _weight_learning() -> dict:
        try:
            if not engine.config.get("recall.weight_learning"):
                return {"skipped": "recall.weight_learning=off"}
        except Exception:
            pass
        from pmb.reasoning.weight_learning import propose_from_samples
        return propose_from_samples(engine)
    _step("weight_learning", _weight_learning)

    return summary
