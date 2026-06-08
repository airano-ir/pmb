"""Phase 2 / issue #11: perf tracking records recall_smart stage diagnostics,
client-timeout flag, and outcome — with an additive, idempotent schema
migration. A client-timed-out call is NOT counted as a clean success."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmb.mcp import perf


def test_schema_migration_idempotent(tmp_path):
    db = tmp_path / "perf.sqlite"
    perf._ensure_schema(db)
    perf._schema_ready_paths.discard(str(db))  # force a second migration pass
    perf._ensure_schema(db)                     # must not raise on existing cols
    with sqlite3.connect(db) as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(mcp_calls)")}
    for col in ("stages_json", "backend", "cache_hit", "client_timeout", "outcome"):
        assert col in cols


def test_record_call_persists_new_fields(tmp_path):
    db = tmp_path / "perf.sqlite"
    perf.record_call(
        db, "ws", "recall_smart", 12.5, success=True,
        stages_json='{"stages": ["local"], "stopped": "confidence_met"}',
        cache_hit=False, client_timeout=False, outcome="ok",
    )
    stats = perf.get_perf_stats(db)
    assert stats["total_calls"] == 1
    rec = stats["recent"][0]
    assert rec["outcome"] == "ok"
    assert rec["client_timeout"] is False
    assert rec["stages"]["stopped"] == "confidence_met"


def test_wrapper_captures_escalation_and_client_timeout(tmp_path, monkeypatch):
    db = tmp_path / "perf.sqlite"
    # tiny client timeout → every call is flagged as "client already gave up"
    monkeypatch.setenv("PMB_MCP_CLIENT_TIMEOUT_MS", "0.001")
    timed = perf.make_timing_wrapper(db, "ws")

    @timed
    def recall_smart(q):
        return {
            "results": [],
            "escalation": {"stages": ["local", "local_rerank"],
                           "stopped": "deadline_hit"},
        }

    recall_smart("where do I live")
    stats = perf.get_perf_stats(db)
    rec = stats["recent"][0]
    assert rec["stages"]["stopped"] == "deadline_hit"
    assert rec["stages"]["stages"] == ["local", "local_rerank"]
    assert rec["client_timeout"] is True
    assert rec["outcome"] == "client_timeout"
    assert stats["client_timeout_count"] == 1


def test_old_call_signature_still_works(tmp_path):
    """Back-compat: callers that don't pass the new fields still record."""
    db = tmp_path / "perf.sqlite"
    perf.record_call(db, "ws", "recall", 3.0)  # no new kwargs
    stats = perf.get_perf_stats(db)
    assert stats["total_calls"] == 1
    assert stats["recent"][0]["stages"] is None
    assert stats["recent"][0]["outcome"] is None
