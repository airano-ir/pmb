"""Tests for Adaptive Query Decomposition (PMB v2.2)."""
from __future__ import annotations

import json

from pmb.core.engine import Engine
from pmb.reasoning.decompose import (
    QueryDecomposer,
    reciprocal_rank_fuse,
)

# ----------------------------------------------------------------------
# Stub LLM
# ----------------------------------------------------------------------

class _StubDecomposeLLM:
    def __init__(self, responses: list[list[str]]):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, prompt: str, **kw) -> str:
        out = self.responses[self.calls]
        self.calls += 1
        return json.dumps(out)


# ----------------------------------------------------------------------
# RRF
# ----------------------------------------------------------------------

def test_rrf_fuses_rankings():
    r1 = ["a", "b", "c"]
    r2 = ["b", "c", "d"]
    fused = reciprocal_rank_fuse([r1, r2])
    # 'b' is rank 2 in r1 and rank 1 in r2 → should beat 'a' (rank 1 only in r1)
    ulids = [u for u, _ in fused]
    assert ulids[0] == "b"


def test_rrf_single_ranking_preserves_order():
    r = ["a", "b", "c"]
    fused = reciprocal_rank_fuse([r])
    ulids = [u for u, _ in fused]
    assert ulids == r


# ----------------------------------------------------------------------
# Decomposer
# ----------------------------------------------------------------------

def test_decomposer_parses_llm_array(tmp_workspace_dir):
    stub = _StubDecomposeLLM([[
        "When did Alice meet Bob?",
        "What did Alice do after Dec 5?",
    ]])
    d = QueryDecomposer(stub, cache_dir=tmp_workspace_dir)
    out = d.decompose("What did Alice do after meeting Bob?")
    assert len(out.sub_queries) == 2
    assert "Alice" in out.sub_queries[0]


def test_decomposer_caches_to_disk(tmp_workspace_dir):
    stub = _StubDecomposeLLM([
        ["sub1", "sub2"],
        # second call shouldn't happen — cache hit
    ])
    d = QueryDecomposer(stub, cache_dir=tmp_workspace_dir)
    out1 = d.decompose("Test query")
    assert stub.calls == 1
    # Fresh decomposer instance should see same cache
    d2 = QueryDecomposer(stub, cache_dir=tmp_workspace_dir)
    out2 = d2.decompose("Test query")
    assert out2.from_cache
    assert out2.sub_queries == out1.sub_queries
    assert stub.calls == 1  # no extra call


def test_decomposer_fallback_on_llm_error(tmp_workspace_dir):
    class _Bad:
        def complete(self, prompt, **kw):
            raise RuntimeError("LLM down")
    d = QueryDecomposer(_Bad(), cache_dir=tmp_workspace_dir)
    out = d.decompose("multi-hop question")
    # Falls back to single query
    assert out.sub_queries == ["multi-hop question"]


# ----------------------------------------------------------------------
# Engine integration: smoke that adaptive_decompose actually fires
# ----------------------------------------------------------------------

def test_engine_adaptive_decomposition_fuses_results(
    tmp_pmb_home, tmp_workspace_dir,
):
    """When adaptive_decompose is on AND query looks multi-hop, sub-queries
    run + RRF merges. This is end-to-end: stub LLM splits, real recall
    runs each sub-query, RRF picks the union of best results."""
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={
            "recall.cache_size": 0,
            "recall.adaptive_decompose": True,
            "recall.spreading_activation": False,
        },
    )
    a = eng.record_fact("Alice flew to Paris on January 8")
    b = eng.record_fact("Bob met Alice for coffee on December 5")
    eng.record_fact("Random unrelated text about lunch")

    # Stub the LLM client resolution path with our decomposer LLM
    # by monkey-patching the import inside _recall_with_decomposition
    import pmb.health.consolidate as _conso
    original_resolver = _conso.resolve_llm_client
    _stub = _StubDecomposeLLM([[
        "When did Alice meet Bob?",
        "What did Alice do after that?",
    ]])
    _conso.resolve_llm_client = lambda **kw: _stub

    try:
        pack = eng.recall(
            "what did Alice do after meeting Bob?", top_k=5,
        )
        ulids = [r.ulid for r in pack.results]
        # Both source events should surface in the fused top-K
        assert a in ulids and b in ulids, (
            f"Expected both Alice events surfaced via RRF; got {ulids}"
        )
    finally:
        _conso.resolve_llm_client = original_resolver


def test_engine_decompose_off_by_default(tmp_pmb_home, tmp_workspace_dir):
    """Default config has adaptive_decompose=False — no LLM call should
    happen even for multi-hop queries."""
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_fact("Some fact")
    # Should not raise even though no LLM is configured
    pack = eng.recall("what happened after the meeting?", top_k=3)
    assert pack is not None
