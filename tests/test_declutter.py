"""T7: `pmb declutter` — heuristic junk sweep + optional bounded-LLM judge.

Archive-only, dry-run by default. Heuristics never need the LLM; the LLM judge
is capped, timeout-clamped, and behind the recall circuit breaker.
"""
from __future__ import annotations

import json
import sqlite3
import time

import pytest

from pmb.core.engine import Engine
from pmb.core.events import Event
from pmb.maintenance.declutter import declutter

NOW = time.time()


@pytest.fixture(autouse=True)
def _reset_breaker():
    from pmb.core import circuit_breaker
    circuit_breaker.reset()
    yield
    circuit_breaker.reset()






@pytest.fixture
def eng(tmp_workspace_dir, tmp_pmb_home):
    return Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
                  config_overrides={"recall.cache_size": 0})


def _seed(eng, content, *, etype="fact", imp=0.2, ac=0, ts=NOW - 100, meta=None):
    ev = Event(workspace_id=eng.workspace.id, event_type=etype, content=content,
               metadata=meta or {}, importance=imp, access_count=ac, timestamp=ts)
    return eng.events.append(ev).ulid


def _is_active(eng, ulid):
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        row = c.execute("SELECT archived_at FROM events WHERE ulid=?",
                        (ulid,)).fetchone()
    return row is not None and row[0] is None


def _meta(eng, ulid):
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        row = c.execute("SELECT metadata_json FROM events WHERE ulid=?",
                        (ulid,)).fetchone()
    return json.loads(row[0] or "{}") if row else {}


def _by_ulid(res):
    return {c["ulid"]: c["reason"] for c in res["candidates"]}


# ── heuristics ──────────────────────────────────────────────────────────────

def test_test_artifact_keyed_key(eng):
    u = _seed(eng, "user test_attr_39da34: final_value", imp=0.8,
              meta={"keyed_fact_key": "user::test_attr_39da34"})
    cand = _by_ulid(declutter(eng, apply=False))
    assert cand.get(u) == "test_artifact"


def test_empty_project_index_artifact_is_decluttered(eng):
    u = _seed(
        eng,
        "File: docs/empty.md (markdown, 8 lines)\n"
        "Symbols: (no symbols)\nImports: (no imports)",
        imp=0.55,
        meta={
            "source": "project",
            "project_name": "acme",
            "file_path": "docs/empty.md",
            "symbols": [],
            "imports": [],
        },
    )
    res = declutter(eng, apply=True)
    assert _by_ulid(res).get(u) == "project_index_empty"
    assert not _is_active(eng, u)


def test_real_tool_names_are_not_test_artifacts(eng):
    """Regression (caught on the real personal corpus): a fact mentioning the
    `asdf` version manager must NOT be flagged as a test artifact."""
    u = _seed(eng, "PMB ops: installed Mise (formerly rtx), replaces asdf as "
                   "the language version manager", imp=0.6)
    cand = _by_ulid(declutter(eng, apply=False))
    assert u not in cand


def test_near_empty_content(eng):
    u = _seed(eng, "ok")
    cand = _by_ulid(declutter(eng, apply=False))
    assert cand.get(u) == "near_empty"


def test_exact_duplicate_keeps_newest(eng):
    older = _seed(eng, "Postgres runs on port 5432", ts=NOW - 200)
    newer = _seed(eng, "Postgres runs on port 5432", ts=NOW - 50)
    cand = _by_ulid(declutter(eng, apply=False))
    assert cand.get(older) == "exact_duplicate"
    assert newer not in cand  # newest copy kept (equal importance → recency)


# ── A5: short non-stopword facts are review-only, never auto-archived ────────

@pytest.mark.parametrize("val", ["O+", "HIV+", "ADHD", "Tampa", "B-", "INTJ"])
def test_short_real_fact_is_review_only(eng, val):
    u = _seed(eng, val, imp=0.2)
    res = declutter(eng, apply=False)
    assert _by_ulid(res).get(u) == "short_review"
    # default --apply must NOT archive a short_review item
    res2 = declutter(eng, apply=True)
    assert _is_active(eng, u)
    assert res2["n_applied"] == 0


def test_short_fact_archived_only_with_aggressive(eng):
    u = _seed(eng, "Tampa", imp=0.2)
    res = declutter(eng, apply=True, aggressive=True)
    assert not _is_active(eng, u)
    assert res["n_applied"] == 1


def test_keyed_short_value_not_decluttered(eng):
    """A keyed value ("O+" under user::blood_type) is structure, not junk —
    it must never appear as a near_empty/short candidate."""
    eng.record_keyed_fact("user", "blood_type", "O+")
    cand = _by_ulid(declutter(eng, apply=False))
    # no candidate carries the O+ content under a near_empty/short reason
    assert all(r not in ("near_empty", "short_review") for r in cand.values())


# ── A6: duplicate keep-policy prefers the most valuable copy ─────────────────

def test_duplicate_keeps_most_valuable_not_newest(eng):
    older_important = _seed(eng, "we deploy on fly.io", imp=0.9, ts=NOW - 200)
    newer_cheap = _seed(eng, "we deploy on fly.io", imp=0.4, ts=NOW - 50)
    cand = _by_ulid(declutter(eng, apply=False))
    assert cand.get(newer_cheap) == "exact_duplicate"  # cheap newer archived
    assert older_important not in cand                  # valuable older kept


def test_duplicate_keeps_more_accessed_copy(eng):
    accessed = _seed(eng, "the cron runs at 03:00 UTC", imp=0.5, ac=7, ts=NOW - 200)
    fresh = _seed(eng, "the cron runs at 03:00 UTC", imp=0.5, ac=0, ts=NOW - 50)
    cand = _by_ulid(declutter(eng, apply=False))
    assert cand.get(fresh) == "exact_duplicate"
    assert accessed not in cand


def test_duplicate_archive_stamps_duplicate_of(eng):
    older = _seed(eng, "redis on 6379", imp=0.9, ts=NOW - 200)
    newer = _seed(eng, "redis on 6379", imp=0.4, ts=NOW - 50)
    declutter(eng, apply=True)
    assert _meta(eng, newer).get("duplicate_of") == older


def test_pinned_and_lessons_protected(eng):
    pinned = _seed(eng, "ok", imp=1.0)
    lesson = _seed(eng, "x", imp=0.1, meta={"kind": "lesson"})
    cand = _by_ulid(declutter(eng, apply=False))
    assert pinned not in cand
    assert lesson not in cand


def test_apply_archives_with_reason(eng):
    u = _seed(eng, "placeholder lorem ipsum text", imp=0.2)
    res = declutter(eng, apply=True)
    assert res["dry_run"] is False
    assert not _is_active(eng, u)
    assert _meta(eng, u).get("archived_reason") == "declutter"


def test_dry_run_changes_nothing(eng):
    u = _seed(eng, "ok")
    declutter(eng, apply=False)
    assert _is_active(eng, u)


# ── optional LLM judge ──────────────────────────────────────────────────────

class _FakeLLM:
    timeout = 120.0

    def complete(self, prompt, max_tokens=120):
        return '{"junk": true, "reason": "low-signal musing"}'


def test_llm_judge_flags_borderline(eng, monkeypatch):
    monkeypatch.setattr("pmb.health.consolidate.resolve_llm_client",
                        lambda **k: _FakeLLM())
    # low-value, NOT a heuristic match → only the LLM can flag it
    u = _seed(eng, "the afternoon felt vaguely pleasant, perhaps", imp=0.1)
    res = declutter(eng, apply=False, use_llm=True)
    assert res["llm_used"] is True
    cand = _by_ulid(res)
    assert u in cand and cand[u].startswith("llm:")


def test_breaker_open_skips_llm_but_keeps_heuristics(eng, monkeypatch):
    from pmb.core import circuit_breaker
    monkeypatch.setattr(circuit_breaker, "is_open", lambda backend: True)

    def _boom(**k):
        raise AssertionError("resolve_llm_client called while breaker open")
    monkeypatch.setattr("pmb.health.consolidate.resolve_llm_client", _boom)

    junk = _seed(eng, "ok")  # heuristic near_empty
    borderline = _seed(eng, "some idle low value musing about nothing here", imp=0.1)
    cand = _by_ulid(declutter(eng, apply=False, use_llm=True))
    assert junk in cand          # heuristics still run
    assert borderline not in cand  # LLM judge skipped
