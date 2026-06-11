"""Tests for LLM-based conflict merging (reconsolidation)."""
from __future__ import annotations

import json
import time

from pmb.core.engine import Engine
from pmb.health.conflicts import ConflictDetector, _ask_llm_to_merge


class _StubLLM:
    """Returns a merged fact in the shape _ask_llm_to_merge expects."""

    def __init__(self, merged_text: str = "Database is Postgres now (was MySQL)."):
        self.merged_text = merged_text
        self.calls = 0

    def consolidate(self, texts):
        self.calls += 1
        payload = json.dumps({"merged": self.merged_text})
        return {
            "consolidate": False,
            "summary": payload,
            "confidence": 1.0,
            "reasoning": "",
        }


def test_ask_llm_to_merge_extracts_json():
    class _C:
        key = "database"
        older_value = "MySQL"
        newer_value = "Postgres 17"
    out = _ask_llm_to_merge(_C(), _StubLLM("Now Postgres 17 (was MySQL)."))
    assert "Postgres" in out
    assert "MySQL" in out


def test_default_auto_resolve_only_archives(tmp_pmb_home, tmp_workspace_dir):
    """Without merge_via_llm flag, behaviour is unchanged: archive older."""
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    older = eng.record_fact("database = MySQL")
    # Backdate older event by 30 days so detector marks supersede
    import sqlite3
    with sqlite3.connect(eng.workspace.db_path) as conn:
        conn.execute("UPDATE events SET timestamp = ? WHERE ulid = ?",
                     (time.time() - 30 * 86400, older))
    newer = eng.record_fact("database = Postgres 17")

    result = ConflictDetector(eng).auto_resolve(dry_run=False, merge_via_llm=False)
    assert result["n_archived"] >= 1
    assert result["n_merged"] == 0
    # older archived, newer still active
    assert eng.events.get_by_ulid(older).archived_at is not None
    assert eng.events.get_by_ulid(newer).archived_at is None


def test_merge_via_llm_creates_merged_fact(
    tmp_pmb_home, tmp_workspace_dir, monkeypatch,
):
    """With merge_via_llm=True, the LLM produces a merged fact and both
    originals get archived. Mimics human-memory reconsolidation."""

    stub = _StubLLM("Database is Postgres 17 now (was MySQL before).")
    # Make resolve_llm_client return our stub regardless of backend
    monkeypatch.setattr(
        "pmb.health.consolidate.resolve_llm_client",
        lambda backend="auto", model=None: stub,
    )

    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    older = eng.record_fact("database = MySQL")
    import sqlite3
    with sqlite3.connect(eng.workspace.db_path) as conn:
        conn.execute("UPDATE events SET timestamp = ? WHERE ulid = ?",
                     (time.time() - 30 * 86400, older))
    newer = eng.record_fact("database = Postgres 17")

    result = ConflictDetector(eng).auto_resolve(dry_run=False, merge_via_llm=True)

    assert result["n_merged"] == 1
    assert result["n_archived"] >= 2  # both originals archived

    # Both originals archived
    assert eng.events.get_by_ulid(older).archived_at is not None
    assert eng.events.get_by_ulid(newer).archived_at is not None

    # A new merged fact exists carrying both contexts
    merged_action = next(a for a in result["actions"] if a.get("mode") == "merge")
    new_ulid = merged_action["new_ulid"]
    new_ev = eng.events.get_by_ulid(new_ulid)
    assert new_ev is not None
    assert "Postgres" in new_ev.content
    assert "MySQL" in new_ev.content
    assert new_ev.importance == 0.85
    assert older in new_ev.metadata["merged_from"]
    assert newer in new_ev.metadata["merged_from"]


def test_merge_falls_back_when_llm_fails(
    tmp_pmb_home, tmp_workspace_dir, monkeypatch,
):
    """If the LLM call raises, fall back to plain archive (no crash)."""
    class _BadLLM:
        def consolidate(self, texts):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "pmb.health.consolidate.resolve_llm_client",
        lambda backend="auto", model=None: _BadLLM(),
    )

    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    older = eng.record_fact("framework = Vue")
    import sqlite3
    with sqlite3.connect(eng.workspace.db_path) as conn:
        conn.execute("UPDATE events SET timestamp = ? WHERE ulid = ?",
                     (time.time() - 30 * 86400, older))
    newer = eng.record_fact("framework = React")

    result = ConflictDetector(eng).auto_resolve(dry_run=False, merge_via_llm=True)
    # No merged fact, but older still archived
    assert result["n_merged"] == 0
    assert eng.events.get_by_ulid(older).archived_at is not None


def test_dry_run_makes_no_writes(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    older = eng.record_fact("lang = Python")
    import sqlite3
    with sqlite3.connect(eng.workspace.db_path) as conn:
        conn.execute("UPDATE events SET timestamp = ? WHERE ulid = ?",
                     (time.time() - 30 * 86400, older))
    newer = eng.record_fact("lang = Go")

    result = ConflictDetector(eng).auto_resolve(dry_run=True, merge_via_llm=True)
    assert result["n_archived"] == 0
    assert result["n_merged"] == 0
    assert eng.events.get_by_ulid(older).archived_at is None
    assert eng.events.get_by_ulid(newer).archived_at is None
