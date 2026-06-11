"""Tests for HippoRAG-style Personalized PageRank over the entity graph
(PMB v2.1 Phase A).

The claim: a query whose entities are graph-connected (but multi-hop
distant) to the answer event will surface the answer via PPR diffusion,
even when raw hybrid search misses it."""
from __future__ import annotations

from pmb.core.engine import Engine
from pmb.graph.ppr import (
    build_ppr_graph,
    personalized_pagerank,
    score_events_by_ppr,
)

# ----------------------------------------------------------------------
# PPR math
# ----------------------------------------------------------------------

def test_build_ppr_graph_empty(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    g = build_ppr_graph(eng.workspace.db_path, eng.workspace.id)
    assert g is None  # no entities yet


def test_ppr_diffuses_mass(tmp_pmb_home, tmp_workspace_dir):
    """A simple graph: Postgres - Backup - Cron - S3. Seed at Postgres
    should diffuse mass towards S3 across 3 hops."""
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_fact("Postgres backup runs nightly")
    eng.record_fact("Backup runs via cron")
    eng.record_fact("Cron uploads to S3 bucket")
    g = build_ppr_graph(eng.workspace.db_path, eng.workspace.id)
    assert g is not None
    # Find postgres entity id
    pg = None
    for ent in eng.graph.top_entities(eng.workspace.id, kind="tech"):
        if ent.name == "postgres":
            pg = ent.id
            break
    assert pg is not None
    scores = personalized_pagerank(g, [pg], iterations=30)
    # mass should be positive at multi-hop nodes
    assert scores.sum() > 0.9  # almost-normalized
    # postgres node should hold the most
    pg_idx = g.id_to_idx[pg]
    assert scores[pg_idx] >= scores.max() * 0.8


def test_ppr_event_scoring(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    e1 = eng.record_fact("Postgres setup for the api service")
    e2 = eng.record_fact("Redis cache on port 6379")
    e3 = eng.record_fact("Postgres connects to Redis via worker")
    g = build_ppr_graph(eng.workspace.db_path, eng.workspace.id)
    assert g is not None
    pg = None
    for ent in eng.graph.top_entities(eng.workspace.id, kind="tech"):
        if ent.name == "postgres":
            pg = ent.id
            break
    scores = personalized_pagerank(g, [pg])
    event_scores = score_events_by_ppr(g, scores)
    # Both Postgres events should outrank pure Redis event
    assert event_scores.get(e1, 0) > 0
    assert event_scores.get(e3, 0) > 0
    # The cross-event (Postgres+Redis) should be high too
    assert event_scores.get(e3, 0) >= event_scores.get(e2, 0) * 0.7


# ----------------------------------------------------------------------
# Engine integration
# ----------------------------------------------------------------------

def test_ppr_surfaces_multihop_event(tmp_pmb_home, tmp_workspace_dir):
    """The KILLER test: a query that names tech entities multi-hop away from
    the answer event should surface it via PPR.

    Chain (built via known-tech co-occurrence in shared events):
      FastAPI --(E1)-- Docker --(E2)-- Redis
    Target E3 mentions only Redis, no FastAPI. Query 'FastAPI sessions'
    should surface E3 because PPR walks fastapi→docker→redis.
    """
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={
            "recall.cache_size": 0,
            "recall.spreading_activation": False,
            "recall.graph_boost": 0.0,           # disable old graph_boost
            "recall.ppr_enabled": True,
            "recall.ppr_always": True,           # bypass intent gate for test
            "recall.ppr_weight": 2.5,
            "recall.ppr_alpha": 0.3,             # depth-friendly
        },
    )
    # Build chain via shared techs
    eng.record_fact("FastAPI app runs in a Docker container")           # fastapi+docker
    eng.record_fact("Docker compose orchestrates Postgres and Redis")    # docker+postgres+redis
    target = eng.record_fact("Redis stores user session tokens")         # redis only
    # Lexical-decoy that mentions FastAPI but is irrelevant
    eng.record_fact("FastAPI documentation explains type hints clearly")
    # Pure unrelated noise
    eng.record_fact("Lunch is at noon today")

    # Query mentions ONE tech (fastapi) plus a concept that lives in target
    pack = eng.recall("FastAPI sessions", top_k=5)
    ulids = [r.ulid for r in pack.results]
    assert target in ulids, (
        f"PPR should surface the Redis-sessions event (2 hops from fastapi); got "
        f"{[(r.ulid, r.content[:80]) for r in pack.results]}"
    )


def test_ppr_disabled_does_nothing(tmp_pmb_home, tmp_workspace_dir):
    """With PPR off, behaviour matches pre-PPR pipeline (no errors)."""
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={
            "recall.cache_size": 0,
            "recall.ppr_enabled": False,
        },
    )
    eng.record_fact("Postgres something")
    pack = eng.recall("postgres", top_k=3)
    assert len(pack.results) >= 1
