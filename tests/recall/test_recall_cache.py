"""Tests for the recall LRU cache layer."""
from __future__ import annotations

import time

from pmb.core.engine import Engine
from pmb.core.recall_cache import RecallCache, make_recall_cache_key

# ----------------------------------------------------------------------
# Pure unit tests of RecallCache
# ----------------------------------------------------------------------


def test_disabled_when_max_size_zero():
    c = RecallCache(max_size=0)
    c.put("k", "v")
    assert c.get("k") is None


def test_get_miss_returns_none():
    c = RecallCache(max_size=8)
    assert c.get("k") is None
    assert c.stats()["misses"] == 1


def test_put_get_roundtrip():
    c = RecallCache(max_size=8)
    c.put("k", "v")
    assert c.get("k") == "v"
    assert c.stats()["hits"] == 1


def test_lru_evicts_oldest():
    c = RecallCache(max_size=2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)  # evicts "a"
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("c") == 3


def test_touching_promotes_recency():
    c = RecallCache(max_size=2)
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")        # touches a → makes b oldest
    c.put("c", 3)     # evicts b, NOT a
    assert c.get("a") == 1
    assert c.get("b") is None


def test_ttl_expires_entry():
    c = RecallCache(max_size=8, ttl_seconds=0.01)
    c.put("k", "v")
    time.sleep(0.05)
    assert c.get("k") is None


def test_bump_generation_invalidates_all():
    c = RecallCache(max_size=8)
    c.put("a", 1)
    c.put("b", 2)
    c.bump_generation()
    assert c.get("a") is None
    assert c.get("b") is None


def test_cache_key_is_stable():
    k1 = make_recall_cache_key("hello", 5, 30.0, 0.15, False, 25)
    k2 = make_recall_cache_key("hello", 5, 30.0, 0.15, False, 25)
    assert k1 == k2


def test_cache_key_changes_on_params():
    base = make_recall_cache_key("q", 5, 30.0, 0.15, False, 25)
    assert base != make_recall_cache_key("q", 5, 30.0, 0.15, True, 25)
    assert base != make_recall_cache_key("q", 10, 30.0, 0.15, False, 25)
    assert base != make_recall_cache_key("q", 5, 30.0, 0.30, False, 25)


# ----------------------------------------------------------------------
# Integration with Engine
# ----------------------------------------------------------------------


def test_repeat_recall_hits_cache(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.remember("Q1", "Postgres 17 setup")
    eng.remember("Q2", "Redis cache")
    # 1st recall — miss
    p1 = eng.recall("postgres", top_k=3)
    # 2nd recall — hit
    p2 = eng.recall("postgres", top_k=3)
    stats = eng.recall_cache.stats()
    assert stats["hits"] >= 1
    # Same number of results (cached pack returned)
    assert len(p1.results) == len(p2.results)


def test_write_invalidates_cache(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.remember("Q1", "alpha content")
    eng.recall("alpha", top_k=3)
    hits_before = eng.recall_cache.stats()["hits"]
    eng.remember("Q2", "beta content")  # write → bump generation
    eng.recall("alpha", top_k=3)
    hits_after = eng.recall_cache.stats()["hits"]
    # The second alpha recall should NOT have been a cache hit (was invalidated)
    assert hits_after == hits_before
    misses = eng.recall_cache.stats()["misses"]
    assert misses >= 2


def test_cache_disabled_via_config(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={"recall.cache_size": 0},
    )
    eng.remember("Q", "content")
    eng.recall("content", top_k=3)
    eng.recall("content", top_k=3)
    # Disabled → no hits, no entries
    stats = eng.recall_cache.stats()
    assert stats["hits"] == 0
    assert stats["size"] == 0
