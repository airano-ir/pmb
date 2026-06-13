"""Tests for the causation graph layer (PMB v2 Phase 2).

The claim: typed event-to-event edges (caused, influenced, references,
temporal-next) let recall walk to bridge events that hybrid+BM25 miss.
"""
from __future__ import annotations

import json

from pmb.core.engine import Engine, _looks_multihop
from pmb.reasoning.causation import (
    CausationEdge,
    CausationExtractor,
    edges_from,
    edges_to,
    upsert_edge,
    walk_edges,
)

# ----------------------------------------------------------------------
# Multi-hop detector
# ----------------------------------------------------------------------

def test_looks_multihop_catches_temporal_patterns():
    assert _looks_multihop("what happened after the meeting?")
    assert _looks_multihop("why did Alice leave?")
    assert _looks_multihop("Bob caused the outage")
    assert _looks_multihop("things that led to the migration")


def test_looks_multihop_ignores_simple_lookups():
    assert not _looks_multihop("postgres version")
    assert not _looks_multihop("api endpoint")
    assert not _looks_multihop("Alice")


# ----------------------------------------------------------------------
# Edge storage + walk
# ----------------------------------------------------------------------

def test_upsert_and_query_edges(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    a = eng.record_fact("FactA")
    b = eng.record_fact("FactB")
    c = eng.record_fact("FactC")

    upsert_edge(eng.workspace.db_path,
                CausationEdge(a, b, "causes", 0.9, "A led to B"))
    upsert_edge(eng.workspace.db_path,
                CausationEdge(b, c, "influenced", 0.7, "B shaped C"))

    # Filter to the edge types we care about (record_fact also creates
    # automatic 'temporal-next' edges between back-to-back writes)
    causes_from_a = [
        e for e in edges_from(eng.workspace.db_path, a) if e.edge_type == "causes"
    ]
    assert len(causes_from_a) == 1
    assert causes_from_a[0].target_ulid == b

    influenced_into_c = [
        e for e in edges_to(eng.workspace.db_path, c) if e.edge_type == "influenced"
    ]
    assert len(influenced_into_c) == 1
    assert influenced_into_c[0].source_ulid == b


def test_walk_edges_chain(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    a = eng.record_fact("A")
    b = eng.record_fact("B")
    c = eng.record_fact("C")
    d = eng.record_fact("D (unrelated)")

    db = eng.workspace.db_path
    upsert_edge(db, CausationEdge(a, b, "causes", 0.9))
    upsert_edge(db, CausationEdge(b, c, "causes", 0.9))

    # 1-hop forward from A → {B}
    out1 = walk_edges(db, [a], direction="forward", hops=1)
    assert out1 == {b}

    # 2-hop forward from A → {B, C}
    out2 = walk_edges(db, [a], direction="forward", hops=2)
    assert out2 == {b, c}

    # Backward from C with hops=2 → {A, B}
    out3 = walk_edges(db, [c], direction="backward", hops=2)
    assert out3 == {a, b}

    # D should never appear
    assert d not in (out1 | out2 | out3)


def test_temporal_next_edge_autoplaced(tmp_pmb_home, tmp_workspace_dir):
    """Recording two events back-to-back should create a temporal-next edge."""
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    a = eng.record_fact("first thing")
    b = eng.record_fact("second thing")
    fwd = edges_from(eng.workspace.db_path, a)
    assert any(e.edge_type == "temporal-next" and e.target_ulid == b for e in fwd), (
        f"expected temporal-next from {a} to {b}; got {[(e.target_ulid, e.edge_type) for e in fwd]}"
    )


# ----------------------------------------------------------------------
# LLM causation extractor (stub)
# ----------------------------------------------------------------------

class _StubCausationLLM:
    def __init__(self, responses_per_call: list[list[dict]]):
        self.responses = responses_per_call
        self.call_idx = 0

    def complete(self, prompt: str, **kw) -> str:
        out = self.responses[self.call_idx]
        self.call_idx += 1
        return json.dumps(out)


def test_causation_extractor_parses_llm_output(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    a_ul = eng.record_fact("Alice complained that Postgres was slow")
    b_ul = eng.record_fact("Switched the project to use Redis cache layer")

    a = eng.events.get_by_ulid(a_ul)
    b = eng.events.get_by_ulid(b_ul)

    llm = _StubCausationLLM([[
        {"edge_type": "causes", "confidence": 0.85, "rationale": "complaint led to cache"},
    ]])
    extractor = CausationExtractor(llm)
    edges = extractor.extract(b, [a])
    assert len(edges) == 1
    assert edges[0].edge_type == "causes"
    assert edges[0].source_ulid == a_ul
    assert edges[0].target_ulid == b_ul
    assert edges[0].confidence >= 0.8


def test_causation_walk_in_recall_surfaces_bridge(
    tmp_pmb_home, tmp_workspace_dir,
):
    """The killer claim: query 'what happened after Alice's complaint' surfaces
    event B (cache switch) via causation walk even though B doesn't mention
    'Alice' or 'complaint'."""
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={
            "recall.cache_size": 0,
            "recall.spreading_activation": False,
            "recall.causation_walk": True,
            "recall.causation_boost": 0.4,  # strong for test clarity
        },
    )
    a = eng.record_fact("Alice complained that Postgres was slow")
    # Decoy events with same word frequency but no causation link
    eng.record_fact("Postgres uses port 5432 by default")
    eng.record_fact("MySQL is sometimes faster for reads")
    b = eng.record_fact("We added a Redis cache layer to the API")

    # Manually install the causation edge (in production this comes from LLM)
    upsert_edge(eng.workspace.db_path,
                CausationEdge(a, b, "causes", 0.9, "complaint → cache"))

    # Multi-hop query — note no overlap with B's content
    pack = eng.recall("what happened after Alice complained", top_k=5)
    ulids = [r.ulid for r in pack.results]
    assert b in ulids, (
        f"causation walk should surface {b} (the bridge); got {ulids}"
    )
