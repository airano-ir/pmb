"""R7: decisions are durable memory, not working memory.

The documented agent pattern records a decision as
{"type":"activity","kind":"decision"}. Routed to the working tier it decays
with a ~2-day half-life and auto-archives within a week. R7 lands kind=decision
in the SEMANTIC tier (half-life ~346d) so the "why" survives; other activity
kinds stay working memory.

Uses the shared conftest fixtures (tmp_pmb_home / tmp_workspace_dir).
"""
from __future__ import annotations

import sqlite3

from pmb.core.engine import Engine


def _engine(ws, home):
    return Engine(cwd=ws, pmb_home=home, config_overrides={"recall.cache_size": 0})


def _tier(eng, ulid) -> str:
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        r = c.execute("SELECT tier FROM events WHERE ulid=?", (ulid,)).fetchone()
    return r[0] if r else ""


def _is_active(eng, ulid) -> bool:
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        r = c.execute("SELECT archived_at FROM events WHERE ulid=?", (ulid,)).fetchone()
    return bool(r) and r[0] is None


def _backdate(eng, ulid, days):
    import time
    ts = time.time() - days * 86400.0
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        c.execute("UPDATE events SET timestamp=?, last_accessed=? WHERE ulid=?",
                  (ts, ts, ulid))
        c.commit()


def test_decision_lands_in_semantic_tier(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    u = eng.record_activity("Chose Postgres 17 over Mongo for JSONB", kind="decision")
    assert _tier(eng, u) == "semantic"


def test_action_lands_in_working_tier(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    u = eng.record_activity("ran the test suite", kind="completed")
    assert _tier(eng, u) == "working"


def test_decision_survives_decay_action_does_not(tmp_pmb_home, tmp_workspace_dir):
    from pmb.signals.decay import apply_decay
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    dec = eng.record_activity("Set the Postgres pool to 20 after load test", kind="decision")
    act = eng.record_activity("opened the config file", kind="completed")
    # age both well past the working-tier 3-day archive floor
    _backdate(eng, dec, 120)
    _backdate(eng, act, 120)
    # simulate 120 days of decay in one compounded pass
    apply_decay(eng, days_since_last_decay=120)
    assert _is_active(eng, dec), "a decision must survive 120 days (semantic tier)"
    assert not _is_active(eng, act), "a stale working-tier action should archive"
