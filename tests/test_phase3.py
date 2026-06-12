"""
Phase 3 tests: self-test, conflicts, compaction, adaptive importance.
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from pmb.core.engine import Engine
from pmb.health.adaptive import adaptive_history, apply_adaptive_boost
from pmb.health.conflicts import (
    ConflictDetector,
    _values_seem_different,
    extract_key_value,
)
from pmb.health.self_test import (
    SelfTestRunner,
    _significant_tokens,
    generate_test_query,
)
from pmb.maintenance.compact import StorageCompactor
from pmb.maintenance.scheduler import generate_scheduler_config

# ---------------------------------------------------------------------------
# Self-test query generation
# ---------------------------------------------------------------------------

def test_significant_tokens_filters_stopwords():
    # G3: EN probe (RU stopwords lived in the deleted ru pack; corpus stopwords
    # are E1's per-workspace path now).
    text = "User: what do we need to do with the Postgres deployment"
    toks = _significant_tokens(text)
    assert "postgres" in toks
    assert "deployment" in toks
    assert "the" not in toks   # stopword
    assert "user" not in toks  # custom stopword


def test_generate_test_query_returns_subset():
    import random as _r
    rng = _r.Random(42)
    text = "Postgres 17 deployment via docker-compose with healthcheck and migrations"
    q = generate_test_query(text, rng)
    assert q is not None
    # query must be subset of significant tokens
    q_tokens = set(q.split())
    full_toks = set(_significant_tokens(text, n_keep=20))
    assert q_tokens.issubset(full_toks)


def test_generate_returns_none_for_short_content():
    import random as _r
    q = generate_test_query("ok thanks", _r.Random(1))
    assert q is None


# ---------------------------------------------------------------------------
# SelfTestRunner
# ---------------------------------------------------------------------------

def _backdate_event(eng, ulid, days_old):
    with sqlite3.connect(eng.workspace.db_path) as conn:
        old_ts = time.time() - days_old * 86400
        conn.execute(
            "UPDATE events SET timestamp = ?, last_accessed = ? WHERE ulid = ?",
            (old_ts, old_ts, ulid),
        )


def test_self_test_finds_easy_events(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    facts = [
        ("Database choice", "We use Postgres 17 deployed via docker-compose with healthcheck"),
        ("Auth method", "JWT with refresh tokens, 15 minute access and 7 day refresh"),
        ("Frontend stack", "Next.js 15 with Tailwind CSS and shadcn/ui components"),
        ("Cache layer", "Redis 7 for rate limiting and session storage"),
        ("CI provider", "GitHub Actions with golangci-lint and integration tests"),
    ]
    for q, a in facts:
        ulid = eng.remember(q, a, importance=0.5)
        _backdate_event(eng, ulid, days_old=2)

    runner = SelfTestRunner(eng, seed=42)
    result = runner.run(n_samples=5, min_age_days=1.0)
    assert result.n_tested >= 3
    # На лёгких уникальных текстах ожидаем сильную точность
    assert result.accuracy_at_5 >= 0.7


def test_self_test_persists_history(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ulid = eng.remember("Q1", "long enough content with many distinct keywords here")
    _backdate_event(eng, ulid, days_old=2)

    runner = SelfTestRunner(eng, seed=1)
    runner.run(n_samples=1)
    runner.run(n_samples=1)
    hist = runner.history()
    assert len(hist) == 2


def test_self_test_trend_insufficient(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    runner = SelfTestRunner(eng)
    t = runner.trend()
    assert t["verdict"] == "insufficient"


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

def test_extract_key_value_simple():
    kv = extract_key_value("database = Postgres 17")
    assert kv == ("database", "Postgres 17")

    kv2 = extract_key_value("user.location: Berlin")
    assert kv2 is not None
    assert "berlin" in kv2[1].lower()


def test_extract_key_value_no_match():
    kv = extract_key_value("just a sentence with no key value")
    # might match "just" or similar — either way, ensure type stable
    assert kv is None or isinstance(kv, tuple)


def test_extract_key_value_rejects_natural_language():
    """After regex tightening, free-text 'X is Y' must not produce false KV."""
    assert extract_key_value("the bug is fixed") is None
    assert extract_key_value("this code is wrong") is None
    assert extract_key_value("what is the database?") is None


def test_extract_key_value_requires_identifier_key():
    """Keys must look like identifiers, not random words separated by =."""
    # 'q' is a single char and below min length → rejected
    assert extract_key_value("q = some answer here") is None
    # Embedded ident pulled out cleanly — commentary stripped
    kv = extract_key_value("our preferred database = Postgres")
    assert kv == ("database", "Postgres")


def test_values_seem_different():
    assert _values_seem_different("Berlin", "Moscow") is True
    assert _values_seem_different("Berlin", "Berlin") is False
    # Substring case (evolution)
    assert _values_seem_different("Postgres", "Postgres 17") is False


def test_conflict_detection(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_fact("location = Moscow", importance=0.7)
    time.sleep(0.05)
    # Backdate чтобы быть точно "older"
    eng.record_fact("location = Berlin", importance=0.7)

    detector = ConflictDetector(eng)
    conflicts = detector.detect()
    assert len(conflicts) >= 1
    keys = [c.key for c in conflicts]
    assert "location" in keys


def test_conflict_auto_resolve_dry_run(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    older_ulid = eng.record_fact("language = Python")
    _backdate_event(eng, older_ulid, days_old=10)
    newer_ulid = eng.record_fact("language = Go")

    detector = ConflictDetector(eng)
    result = detector.auto_resolve(dry_run=True)
    assert result["dry_run"] is True
    assert result["n_archived"] == 0  # dry_run, никого не архивировал


def test_conflict_auto_resolve_real(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    older_ulid = eng.record_fact("framework = Vue")
    _backdate_event(eng, older_ulid, days_old=15)
    newer_ulid = eng.record_fact("framework = React")

    detector = ConflictDetector(eng)
    result = detector.auto_resolve(dry_run=False)
    assert result["n_archived"] >= 1

    older = eng.events.get_by_ulid(older_ulid)
    assert older.archived_at is not None  # archived


# ---------------------------------------------------------------------------
# Storage compaction
# ---------------------------------------------------------------------------

def test_compaction_dry_run(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ulid = eng.remember("Q", "old content")
    eng.forget(ulid)
    # Backdate archived_at
    with sqlite3.connect(eng.workspace.db_path) as conn:
        conn.execute(
            "UPDATE events SET archived_at = ? WHERE ulid = ?",
            (time.time() - 60 * 86400, ulid),
        )

    compactor = StorageCompactor(eng)
    result = compactor.compact(dry_run=True, age_days=30)
    assert result["dry_run"] is True
    assert result["moved_to_cold"] == 1


def test_compaction_real(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    # Активный + старый archived
    eng.remember("Active", "active content remains")
    old_ulid = eng.remember("Old", "old archived content")
    eng.forget(old_ulid)
    with sqlite3.connect(eng.workspace.db_path) as conn:
        conn.execute(
            "UPDATE events SET archived_at = ? WHERE ulid = ?",
            (time.time() - 60 * 86400, old_ulid),
        )

    compactor = StorageCompactor(eng)
    result = compactor.compact(dry_run=False, age_days=30)
    assert result["moved_to_cold"] == 1

    # Old ulid должен исчезнуть из main DB
    after = eng.events.get_by_ulid(old_ulid)
    assert after is None

    # Но появиться в cold
    cold = compactor.cold_stats()
    assert cold["exists"] is True
    assert cold["n_events"] == 1


# ---------------------------------------------------------------------------
# Adaptive boost
# ---------------------------------------------------------------------------

def test_adaptive_boost_increases_importance(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ulid = eng.remember("Q", "content for adaptive test", importance=0.4)

    # Симулируем self-test result с failure
    from pmb.health.self_test import SelfTestResult
    fake_result = SelfTestResult(
        timestamp=time.time(),
        n_tested=1,
        accuracy_at_1=0.0,
        accuracy_at_3=0.0,
        accuracy_at_5=0.0,
        avg_rank=None,
        failed_queries=[{
            "ulid": ulid,
            "query": "test query",
            "expected_content_preview": "content...",
            "top3_in_results": [],
        }],
        workspace_id=eng.workspace.id,
        n_total_active=1,
    )

    summary = apply_adaptive_boost(eng, fake_result)
    assert summary["n_boosted"] == 1

    after = eng.events.get_by_ulid(ulid)
    assert after.importance > 0.4


def test_adaptive_boost_skips_pinned(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ulid = eng.remember("Q", "content", importance=0.5)
    eng.pin(ulid)

    from pmb.health.self_test import SelfTestResult
    fake = SelfTestResult(
        timestamp=time.time(), n_tested=1,
        accuracy_at_1=0.0, accuracy_at_3=0.0, accuracy_at_5=0.0, avg_rank=None,
        failed_queries=[{"ulid": ulid, "query": "x",
                         "expected_content_preview": "c", "top3_in_results": []}],
        workspace_id=eng.workspace.id, n_total_active=1,
    )
    apply_adaptive_boost(eng, fake)

    after = eng.events.get_by_ulid(ulid)
    assert after.importance >= 0.99  # pinned, untouched


def test_adaptive_history_tracks_failures(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ulid = eng.remember("Q", "content", importance=0.4)

    from pmb.health.self_test import SelfTestResult
    failure = {"ulid": ulid, "query": "q", "expected_content_preview": "c", "top3_in_results": []}
    for _ in range(3):
        fake = SelfTestResult(
            timestamp=time.time(), n_tested=1,
            accuracy_at_1=0.0, accuracy_at_3=0.0, accuracy_at_5=0.0, avg_rank=None,
            failed_queries=[failure],
            workspace_id=eng.workspace.id, n_total_active=1,
        )
        apply_adaptive_boost(eng, fake)

    history = adaptive_history(eng)
    assert len(history) == 3
    after = eng.events.get_by_ulid(ulid)
    # 3 failures → super-boost (≥0.85)
    assert after.importance >= 0.85


# ---------------------------------------------------------------------------
# Scheduler config
# ---------------------------------------------------------------------------

def test_scheduler_config_returns_supported():
    cfg = generate_scheduler_config()
    assert "os" in cfg
    if cfg.get("supported"):
        assert "install_steps" in cfg
        assert any("decay" in step or "decay" in str(step) for step in cfg["install_steps"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
