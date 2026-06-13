"""Tests for Adaptive Layer Routing (Improvement E)."""
from __future__ import annotations

from pmb.core.engine import Engine
from pmb.reasoning.router import QueryRouter

# ----------------------------------------------------------------------
# Pure router classification
# ----------------------------------------------------------------------

def test_router_classifies_direct_lookup():
    r = QueryRouter()
    intent = r.classify("What is the API endpoint?")
    assert "direct" in intent.types
    # Direct lookups boost atomic facts
    assert intent.weights.facts_boost > 1.0
    # And reduce arc/causation noise
    assert intent.weights.arc_boost_mul < 1.0


def test_router_classifies_temporal():
    r = QueryRouter()
    intent = r.classify("When did Alice meet Bob?")
    assert "temporal" in intent.types
    assert intent.weights.temporal_boost_mul >= 2.0


def test_router_classifies_multi_hop():
    r = QueryRouter()
    intent = r.classify("What happened after the migration?")
    assert "multi_hop" in intent.types
    assert intent.weights.causation_boost_mul >= 2.0
    assert intent.weights.ppr_weight_mul >= 1.5


def test_router_classifies_narrative():
    r = QueryRouter()
    intent = r.classify("Tell me about the Postgres adoption")
    assert "narrative" in intent.types
    # Narrative downweights atomic facts, upweights raw/arcs
    assert intent.weights.facts_boost < 1.0
    assert intent.weights.arc_boost_mul >= 2.0


def test_router_classifies_inferential():
    r = QueryRouter()
    intent = r.classify("Why did Caroline switch careers?")
    # Note: "why" triggers inferential, but it ALSO matches "_MULTIHOP_RE"
    # because 'why' is one of the multi-hop indicators, so both types may
    # appear.
    assert "inferential" in intent.types
    assert intent.weights.reflections_boost >= 1.5


def test_router_multi_intent_composes():
    """A query like 'when did X happen after Y?' should detect both
    temporal AND multi-hop, composing weights."""
    r = QueryRouter()
    intent = r.classify("when did the migration happen after the meeting?")
    assert "temporal" in intent.types
    assert "multi_hop" in intent.types
    assert intent.weights.temporal_boost_mul >= 2.0
    assert intent.weights.causation_boost_mul >= 2.0


def test_router_empty_query_defaults_to_direct():
    r = QueryRouter()
    intent = r.classify("")
    assert "direct" in intent.types


# ----------------------------------------------------------------------
# Engine integration — narrative query routes to raw events, not facts
# ----------------------------------------------------------------------

def test_routing_boosts_facts_for_direct_lookup(
    tmp_pmb_home, tmp_workspace_dir,
):
    """For 'what is X?' a fact_atom should rank above raw text with same
    relevance, thanks to facts_boost = 1.5."""
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={
            "recall.cache_size": 0,
            "recall.adaptive_routing": True,
            "recall.spreading_activation": False,
        },
    )
    # Raw fact-event
    eng.record_fact("Postgres runs on port 5433 for the api service")

    # Manually inject a fact_atom-typed event covering the same info
    from pmb.core.events import TIER_SEMANTIC, Event
    fact_atom = Event(
        event_type="fact_atom",
        content="Fact: Postgres runs on port 5433",
        metadata={"source_ulid": "x", "kind": "atomic_fact"},
        workspace_id=eng.workspace.id,
        importance=0.7, tier=TIER_SEMANTIC,
    )
    eng.events.append(fact_atom)
    eng.search.add(fact_atom.ulid, fact_atom.to_text())

    pack = eng.recall("What port does Postgres use?", top_k=5)
    # Atomic fact should appear (boost helped)
    assert any(r.event_type == "fact_atom" for r in pack.results)


def test_routing_downweights_facts_for_narrative(
    tmp_pmb_home, tmp_workspace_dir,
):
    """For 'tell me about X' narrative query, atomic facts get penalty
    (facts_boost = 0.6); raw events should dominate."""
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={
            "recall.cache_size": 0,
            "recall.adaptive_routing": True,
            "recall.spreading_activation": False,
            "recall.arc_expansion": False,  # isolate the boost effect
        },
    )
    raw = eng.record_fact("Postgres adoption journey: we evaluated, chose, migrated")
    from pmb.core.events import TIER_SEMANTIC, Event
    for i in range(3):
        fa = Event(
            event_type="fact_atom",
            content=f"Fact: Postgres detail number {i}",
            metadata={"source_ulid": raw, "kind": "atomic_fact"},
            workspace_id=eng.workspace.id,
            importance=0.7, tier=TIER_SEMANTIC,
        )
        eng.events.append(fa)
        eng.search.add(fa.ulid, fa.to_text())

    pack = eng.recall("Tell me about Postgres adoption", top_k=5)
    ulids = [r.ulid for r in pack.results]
    # Raw event should appear in top results
    assert raw in ulids, f"raw event should surface for narrative; got {ulids}"


def test_routing_can_be_disabled(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={"recall.adaptive_routing": False},
    )
    eng.record_fact("Something")
    pack = eng.recall("Tell me about something", top_k=3)
    assert pack is not None
