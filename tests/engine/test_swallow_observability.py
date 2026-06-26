"""Pillar-2: high-value silent swallows in the write path now leave an
error_log breadcrumb (via errlog.log_error) instead of vanishing - so a
SYSTEMIC enrichment failure shows up in `pmb doctor` instead of silently
degrading recall. The write itself stays non-blocking.
"""
from __future__ import annotations

from pmb.core.errlog import recent_errors


def test_write_enrichment_failure_is_observable(isolated_engine):
    # Force a best-effort enrichment step (event_time parsing) to blow up on
    # every write. The write must STILL succeed (non-blocking), but the failure
    # must no longer be silent: it lands in error_log under a stable component.
    def boom(*a, **k):
        raise RuntimeError("enrichment exploded")

    isolated_engine._attach_event_time = boom

    ulid = isolated_engine.record_fact("a durable fact about pillars", importance=0.7)
    assert ulid                                   # primary write was NOT blocked

    comps = {e["component"] for e in recent_errors(isolated_engine.workspace.db_path)}
    assert "attach_event_time" in comps           # the swallow is now observable


def test_successful_write_leaves_no_breadcrumb(isolated_engine):
    # The flip side: on the happy path the seam costs nothing and writes no
    # noise - log_error is only reached when an enrichment actually throws.
    isolated_engine.record_fact("an uneventful fact", importance=0.7)
    comps = {e["component"] for e in recent_errors(isolated_engine.workspace.db_path)}
    assert "attach_event_time" not in comps
