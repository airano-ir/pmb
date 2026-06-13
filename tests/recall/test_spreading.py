"""Tests for graph-based spreading activation (priming)."""
from __future__ import annotations

from pmb.core.engine import Engine
from pmb.graph.spreading import apply_spreading_activation


def test_spreading_boosts_graph_neighbours(tmp_pmb_home, tmp_workspace_dir):
    """Two facts that share entities should prime each other."""
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={"recall.spreading_activation": False},  # disable auto so test controls it
    )
    u_hit = eng.record_fact("Use Postgres 17 with WAL replication")
    u_neighbour = eng.record_fact("pgbouncer pool sits in front of Postgres 17")
    u_unrelated = eng.record_fact("React with Tailwind for the frontend")

    imp_before_n = eng.events.get_by_ulid(u_neighbour).importance
    imp_before_u = eng.events.get_by_ulid(u_unrelated).importance

    hit = eng.events.get_by_ulid(u_hit)
    result = apply_spreading_activation(
        eng, hit_events=[hit], boost=0.2, half_life_hours=24.0,
    )

    after_n = eng.events.get_by_ulid(u_neighbour).importance
    after_u = eng.events.get_by_ulid(u_unrelated).importance

    assert result["primed"] >= 1
    assert after_n > imp_before_n, (
        f"neighbour importance must rise: {imp_before_n:.3f} → {after_n:.3f}"
    )
    # Unrelated shouldn't move much (or not at all if no shared entities)
    assert abs(after_u - imp_before_u) < 0.01


def test_spreading_skips_pinned(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={"recall.spreading_activation": False},
    )
    u_hit = eng.record_fact("Use Postgres 17")
    u_neighbour = eng.record_fact("Postgres tuning shared_buffers 4GB")
    eng.pin(u_neighbour)

    hit = eng.events.get_by_ulid(u_hit)
    apply_spreading_activation(eng, hit_events=[hit], boost=0.5)
    # Pinned stays at 1.0
    assert eng.events.get_by_ulid(u_neighbour).importance >= 0.99


def test_spreading_noop_when_no_entities(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    result = apply_spreading_activation(eng, hit_events=[], boost=0.1)
    assert result["primed"] == 0
    assert result["skipped"] == "no_hits"


def test_recall_triggers_spreading_when_enabled(tmp_pmb_home, tmp_workspace_dir):
    """End-to-end: enable spreading in config and a recall must lift neighbours."""
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={"recall.spreading_activation": True,
                          "recall.spreading_boost": 0.2,
                          "recall.spreading_half_life_hours": 24.0,
                          "recall.cache_size": 0},   # disable cache so writes happen
    )
    u_postgres = eng.record_fact("Use Postgres 17 in production")
    u_neighbour = eng.record_fact(
        "Postgres connection pool via pgbouncer with 50 connections"
    )
    eng.record_fact("Tailwind for styling on the frontend side")

    before = eng.events.get_by_ulid(u_neighbour).importance
    eng.recall("postgres database", top_k=3)
    after = eng.events.get_by_ulid(u_neighbour).importance
    assert after > before, (
        f"neighbour should be primed: {before:.3f} → {after:.3f}"
    )


def test_spreading_respects_max_primed(tmp_pmb_home, tmp_workspace_dir):
    """Even with many candidates, we cap the boost set to max_primed."""
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={"recall.spreading_activation": False},
    )
    # Create 10 events all mentioning "postgres" → all share the postgres entity
    u_hit = eng.record_fact("First fact about postgres")
    for i in range(10):
        eng.record_fact(f"postgres-related fact number {i}")

    hit = eng.events.get_by_ulid(u_hit)
    result = apply_spreading_activation(
        eng, hit_events=[hit], boost=0.1, max_primed=3,
    )
    assert result["primed"] <= 3
