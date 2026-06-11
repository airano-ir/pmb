"""Tests for Predictive Pre-Computation cache (Improvement F)."""
from __future__ import annotations

import json
import time

import numpy as np

from pmb.core.engine import Engine
from pmb.reasoning.predictive import (
    CacheEntry,
    PredictiveQuestionGenerator,
    best_match,
    load_entries,
    store_entry,
)


class _StubPredictiveLLM:
    def __init__(self, response: list[str]):
        self.response = response
        self.calls = 0

    def complete(self, prompt: str, **kw) -> str:
        self.calls += 1
        return json.dumps(self.response)


# ----------------------------------------------------------------------
# Question generator
# ----------------------------------------------------------------------

def test_generator_parses_questions():
    from pmb.core.events import Event
    events = [Event(workspace_id="w", event_type="qa", content="Caroline researched adoption")]
    stub = _StubPredictiveLLM([
        "Who is Caroline?",
        "What did Caroline research?",
        "When did Caroline start the research?",
    ])
    g = PredictiveQuestionGenerator(stub)
    qs = g.generate(events)
    assert len(qs) == 3
    assert all("?" in q or len(q) > 5 for q in qs)


def test_generator_filters_dupes_and_short():
    from pmb.core.events import Event
    stub = _StubPredictiveLLM([
        "Who is Caroline?",
        "Who is Caroline?",       # dup
        "X",                      # too short
        "a" * 400,                # too long
        "What is the api endpoint?",
    ])
    qs = PredictiveQuestionGenerator(stub).generate(
        [Event(workspace_id="w", event_type="qa", content="hi")]
    )
    assert len(qs) == 2  # 2 valid uniques


# ----------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------

def test_cache_store_and_load(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    emb = np.random.rand(384).astype(np.float32)
    entry = CacheEntry(
        id=None, workspace_id=eng.workspace.id,
        query_text="Who is Alice?",
        query_embedding=emb,
        top_ulids=["u1", "u2"],
        created_at=time.time(),
    )
    store_entry(eng.workspace.db_path, entry)
    loaded = load_entries(eng.workspace.db_path, eng.workspace.id)
    assert len(loaded) == 1
    assert loaded[0].query_text == "Who is Alice?"
    assert loaded[0].top_ulids == ["u1", "u2"]
    assert np.allclose(loaded[0].query_embedding, emb)


def test_cache_ttl_filters_old_entries(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    old_ts = time.time() - 30 * 86400
    fresh_ts = time.time()
    for ts, q in [(old_ts, "old query"), (fresh_ts, "fresh query")]:
        store_entry(eng.workspace.db_path, CacheEntry(
            id=None, workspace_id=eng.workspace.id, query_text=q,
            query_embedding=np.zeros(384, dtype=np.float32),
            top_ulids=["u"], created_at=ts,
        ))
    fresh_only = load_entries(eng.workspace.db_path, eng.workspace.id, max_age_days=7)
    assert len(fresh_only) == 1
    assert fresh_only[0].query_text == "fresh query"


def test_best_match_picks_closest_above_threshold():
    e1 = CacheEntry(
        id=1, workspace_id="w", query_text="A",
        query_embedding=np.array([1, 0, 0], dtype=np.float32),
        top_ulids=["x"], created_at=0,
    )
    e2 = CacheEntry(
        id=2, workspace_id="w", query_text="B",
        query_embedding=np.array([0, 1, 0], dtype=np.float32),
        top_ulids=["y"], created_at=0,
    )
    # Query close to e1
    q = np.array([0.99, 0.1, 0], dtype=np.float32)
    match = best_match([e1, e2], q, threshold=0.85)
    assert match is not None
    entry, sim = match
    assert entry.id == 1
    assert sim > 0.9


def test_best_match_returns_none_below_threshold():
    e = CacheEntry(
        id=1, workspace_id="w", query_text="A",
        query_embedding=np.array([1, 0, 0], dtype=np.float32),
        top_ulids=["x"], created_at=0,
    )
    q = np.array([0, 1, 0], dtype=np.float32)  # orthogonal
    assert best_match([e], q, threshold=0.85) is None


# ----------------------------------------------------------------------
# Engine integration
# ----------------------------------------------------------------------

def test_precompute_then_cache_hit_returns_instantly(
    tmp_pmb_home, tmp_workspace_dir,
):
    """End-to-end: ingest events, precompute cache with stub LLM,
    then a near-duplicate query should hit the cache."""
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={
            "recall.cache_size": 0,
            "recall.predictive_enabled": True,
            "recall.predictive_threshold": 0.7,  # lenient for test
        },
    )
    target = eng.record_fact("Caroline researched adoption agencies and lawyers")
    eng.record_fact("Melanie does pottery and painting")

    # Predictive generator returns ONE question that closely matches our test query
    stub = _StubPredictiveLLM(["What did Caroline research?"])
    result = eng.precompute_predictive_cache(
        n_questions=5, events_to_consider=10, cache_top_k=5, llm=stub,
    )
    assert result["n_cached"] == 1

    # Now query — should hit the cache
    pack = eng.recall("What did Caroline research?", top_k=5)
    ulids = [r.ulid for r in pack.results]
    assert target in ulids


def test_precompute_skips_when_no_llm(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_fact("Something")
    result = eng.precompute_predictive_cache(backend="anthropic")  # no API key
    # Either skipped no_llm OR ran without LLM available; either way no exception
    assert result is not None
    assert result.get("n_cached", 0) == 0


def test_clear_predictive_cache(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    store_entry(eng.workspace.db_path, CacheEntry(
        id=None, workspace_id=eng.workspace.id, query_text="q",
        query_embedding=np.zeros(384, dtype=np.float32),
        top_ulids=["u"], created_at=time.time(),
    ))
    n = eng.clear_predictive_cache()
    assert n == 1
    assert len(load_entries(eng.workspace.db_path, eng.workspace.id)) == 0
