"""M1: daemon self-maintenance tick.

Two guarantees:
  * the scheduler predicate fires only after enough uptime AND while idle
    (so housekeeping never competes with a live request);
  * the tick archives genuine cold junk but a DECISION survives — the R7
    regression: decisions are high-importance / semantic-tier, above
    decay.archive_cold_max_importance, so the tick must never evaporate them.
"""
from __future__ import annotations

import sqlite3
import time

from pmb.core.engine import Engine
from pmb.maintenance.tick import run_maintenance_tick, should_run_maintenance


def test_should_run_maintenance_gates():
    # not enough uptime since the last tick → no
    assert not should_run_maintenance(
        now=1000.0, last_tick_ts=999.0, interval_s=3600.0,
        last_request_ts=0.0, idle_min_s=300.0)
    # enough uptime, but a request arrived recently (not idle) → no
    assert not should_run_maintenance(
        now=5000.0, last_tick_ts=0.0, interval_s=3600.0,
        last_request_ts=4900.0, idle_min_s=300.0)
    # enough uptime AND idle long enough → yes
    assert should_run_maintenance(
        now=5000.0, last_tick_ts=0.0, interval_s=3600.0,
        last_request_ts=4000.0, idle_min_s=300.0)


def test_tick_archives_junk_but_keeps_decisions(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
                 config_overrides={"recall.cache_size": 0})
    decision = eng.record_fact("we chose Postgres over MySQL for strong typing",
                               importance=0.8, metadata={"kind": "decision"})
    junk = eng.record_fact("zzz tmp scratch placeholder note",
                           importance=0.1, metadata={"kind": "fact"})
    # Backdate the junk so it is cold AND old AND never recalled — the only
    # shape archive_cold touches. The decision stays fresh & high-importance.
    old = time.time() - 999 * 86400.0
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        c.execute("UPDATE events SET timestamp=?, access_count=0 WHERE ulid=?",
                  (old, junk))

    summary = run_maintenance_tick(eng, archive=True, now=time.time())

    # ran all three steps, report-shaped
    assert set(summary["steps"]) >= {"archive_cold", "conflicts", "declutter_dryrun"}
    assert summary["steps"]["archive_cold"].get("archived", 0) >= 1

    active = eng.events.get_many([decision, junk], workspace_id=eng.workspace.id,
                                 only_active=True)
    assert decision in active, "R7: a decision must survive the maintenance tick"
    assert junk not in active, "cold/old/low-value junk should be archived"


def test_tick_report_only_when_archive_off(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
                 config_overrides={"recall.cache_size": 0})
    junk = eng.record_fact("zzz tmp scratch placeholder", importance=0.1,
                           metadata={"kind": "fact"})
    old = time.time() - 999 * 86400.0
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        c.execute("UPDATE events SET timestamp=?, access_count=0 WHERE ulid=?",
                  (old, junk))
    summary = run_maintenance_tick(eng, archive=False, now=time.time())
    assert "skipped" in summary["steps"]["archive_cold"]
    # archive disabled → junk stays active, but declutter still REPORTS it
    active = eng.events.get_many([junk], workspace_id=eng.workspace.id,
                                 only_active=True)
    assert junk in active
