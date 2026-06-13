"""B4: durable write outbox — record_batch_async survives a crash between
accept and write, and B5: the error_log breadcrumb table.

The old fire-and-forget path ran the ENTIRE record_batch inside a daemon
thread, so a process exit between accept and write lost the items silently.
The outbox persists the batch synchronously first, then a drainer replays it;
recover_outbox() replays leftovers from a crashed previous process.
"""
from __future__ import annotations

import sqlite3

from pmb.core.engine import Engine


def _engine(ws, home, **over):
    cfg = {"recall.cache_size": 0}
    cfg.update(over)
    return Engine(cwd=ws, pmb_home=home, config_overrides=cfg)


def _fact_count(eng, needle: str) -> int:
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        rows = c.execute(
            "SELECT COUNT(*) FROM events WHERE event_type='fact' "
            "AND content LIKE ?", (f"%{needle}%",)).fetchone()
    return int(rows[0] or 0)


def _outbox_rows(eng, status=None):
    q = "SELECT id, status, attempts, last_error FROM write_outbox"
    args = ()
    if status:
        q += " WHERE status=?"
        args = (status,)
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(q, args).fetchall()]


# ── happy path: durable enqueue → drainer writes → wait drains ──────────────

def test_async_write_lands_via_outbox(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    res = eng.record_batch_async([{"type": "fact", "content": "outbox marker ALPHA"}])
    assert res["ok"] and "outbox_id" in res
    assert eng.wait_for_writes(timeout=30)
    assert _fact_count(eng, "outbox marker ALPHA") == 1
    # the row is marked done, not lingering pending
    assert _outbox_rows(eng, status="pending") == []


# ── crash recovery: enqueued but never drained → recover_outbox replays ─────

def test_recover_outbox_replays_leftovers(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    # Simulate a crash: the durable row is written, but the drainer never runs.
    monkeypatch.setattr(eng, "_ensure_outbox_drainer", lambda: None)
    res = eng.record_batch_async([{"type": "fact", "content": "crash marker BETA"}])
    assert "outbox_id" in res
    assert _fact_count(eng, "crash marker BETA") == 0          # not written yet
    assert len(_outbox_rows(eng, status="pending")) == 1

    # A fresh process opens the same workspace and recovers.
    eng2 = _engine(tmp_workspace_dir, tmp_pmb_home)
    n = eng2.recover_outbox()
    assert n >= 1
    assert eng2.wait_for_writes(timeout=30)
    assert _fact_count(eng2, "crash marker BETA") == 1


# ── a permanently failing batch goes to 'failed' after retries, logs error ──

def test_failed_batch_marked_failed_and_logged(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)

    def _boom(items):
        raise RuntimeError("simulated write failure")
    monkeypatch.setattr(eng, "record_batch", _boom)

    eng.record_batch_async([{"type": "fact", "content": "doomed GAMMA"}])
    # wait_for_writes returns once the row reaches a terminal state (failed)
    assert eng.wait_for_writes(timeout=60)
    failed = _outbox_rows(eng, status="failed")
    assert len(failed) == 1
    assert failed[0]["attempts"] >= 5
    assert "simulated write failure" in (failed[0]["last_error"] or "")
    # the failure left an error_log breadcrumb (B5)
    from pmb.core.errlog import error_counts
    assert error_counts(eng.workspace.db_path).get("write_outbox", 0) >= 1


# ── legacy path (write.outbox off) still records ────────────────────────────

def test_legacy_path_still_writes(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home, **{"write.outbox": False})
    res = eng.record_batch_async([{"type": "fact", "content": "legacy marker DELTA"}])
    assert res["ok"] and "outbox_id" not in res
    assert eng.wait_for_writes(timeout=30)
    assert _fact_count(eng, "legacy marker DELTA") == 1
    # nothing was written to the outbox table
    assert _outbox_rows(eng) == []


# ── B5: error_log module is write-safe and queryable ────────────────────────

def test_errlog_records_and_reads(tmp_pmb_home, tmp_workspace_dir):
    from pmb.core.errlog import error_counts, log_error, recent_errors
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    try:
        raise ValueError("boom in component X")
    except ValueError as e:
        log_error(eng.workspace.db_path, "unit_test", e, note="row 42")
    rows = recent_errors(eng.workspace.db_path)
    assert any(r["component"] == "unit_test" and "boom in component X" in r["message"]
               for r in rows)
    assert error_counts(eng.workspace.db_path).get("unit_test", 0) >= 1


def test_errlog_never_raises_on_bad_path(tmp_pmb_home):
    from pmb.core.errlog import log_error
    # An unwritable / nonexistent directory must not raise out of the logger.
    log_error("/nonexistent_dir_zzz/none.db", "x", RuntimeError("y"))
