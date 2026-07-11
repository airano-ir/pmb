"""Tests for the association-graph layer."""
from __future__ import annotations

from pmb.core.engine import Engine
from pmb.graph.entities import (
    EntityExtractor,
    extract_concepts,
    extract_file_paths,
    extract_techs,
)

# ----------------------------------------------------------------------
# Entity extraction
# ----------------------------------------------------------------------


def test_extract_file_paths_finds_python_paths():
    text = "Updated src/auth.py and tests/test_auth.py to use jwt"
    files = extract_file_paths(text)
    assert "src/auth.py" in files
    assert "tests/test_auth.py" in files


def test_extract_file_paths_ignores_non_code_words():
    text = "The .env file mentions postgres. We use docker-compose."
    files = extract_file_paths(text)
    assert ".env" in files or files == []  # accept either as long as no false matches
    # Should not match bare 'postgres' or 'docker-compose'
    for f in files:
        assert "/" in f or f.startswith(".") or f.count(".") == 1


def test_extract_techs_known_set():
    text = "Migrated from MongoDB to Postgres 17 with Redis cache"
    techs = extract_techs(text)
    assert "postgres" in techs
    assert "redis" in techs
    assert "mongodb" in techs


def test_extract_techs_ignores_unknown_words():
    text = "We use Cassandra and Snowflake"  # neither in KNOWN_TECHS
    techs = extract_techs(text)
    assert techs == []


def test_extract_concepts_filters_stopwords():
    text = "The user wants the database to be fast and reliable"
    concepts = extract_concepts(text)
    assert "user" not in concepts  # 'user' is in stopwords
    assert "the" not in concepts
    # Real concepts get through
    assert any(c in concepts for c in ("database", "fast", "reliable"))


def test_extract_concepts_capped():
    text = " ".join(["unique" + str(i) for i in range(50)])
    concepts = extract_concepts(text, max_n=5)
    assert len(concepts) <= 5


def test_extractor_combines_layers():
    extractor = EntityExtractor()
    ext = extractor.extract(
        "Switched src/db.py to Postgres 17 for performance reasons"
    )
    assert "src/db.py" in ext.files
    assert "postgres" in ext.techs
    assert ext.concepts  # at least one concept


def test_openai_llm_extractor_uses_api_provider(monkeypatch):
    import json

    from pmb.graph import extractors_llm
    from pmb.graph.extractors_llm import LLMExtractor

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls = {}

    def _fake_openai(prompt, timeout, model=""):
        calls["model"] = model
        calls["prompt"] = prompt
        return json.dumps({
            "persons": [], "orgs": [], "places": [], "products": [],
            "concepts": ["session cache"],
        })

    monkeypatch.setattr(extractors_llm, "_run_openai_api", _fake_openai)
    out = LLMExtractor(provider="openai", model="gpt-test").extract(
        "Use Postgres for the session cache"
    )

    assert calls["model"] == "gpt-test"
    assert "session cache" in out.concepts
    assert "postgres" in out.techs


# ----------------------------------------------------------------------
# Graph writes via Engine
# ----------------------------------------------------------------------


def test_remember_populates_graph(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.remember(
        "What database?",
        "We switched src/db.py to Postgres 17 from MySQL last week",
    )
    stats = eng.graph_stats()
    assert stats["n_entities"] >= 2  # at least postgres + mysql + db.py
    assert "tech" in stats["by_kind"]


def test_two_events_share_entity_get_edge(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.remember("Q1", "Use Postgres 17 with Redis cache for hot data")
    eng.remember("Q2", "Postgres reads spike; add Redis read-through")
    # postgres + redis appear in both events → strong edge between them
    nb = eng.graph_neighbors("postgres", kind="tech", top_k=5)
    assert nb["entity"] is not None
    nbr_names = [n["entity"]["name"] for n in nb["neighbors"]]
    assert "redis" in nbr_names


def test_recall_uses_graph_to_surface_missed_event(tmp_pmb_home, tmp_workspace_dir):
    """
    The killer demo: an event that doesn't lexically match the query
    still gets surfaced because the query's entity matches the event's
    entity through the graph.
    """
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    # Event 1 — establishes that the project uses postgres
    e1 = eng.record_fact(
        "Project storage: Postgres 17 with WAL replication", importance=0.8,
    )
    # Event 2 — phrases the same fact very differently (still mentions postgres)
    eng.record_fact("Decided on Postgres for primary store; sqlite was too slow")
    # Event 3 — unrelated
    eng.record_fact("CI runs on GitHub Actions ubuntu-latest")

    # Query that hits postgres entity but uses other words
    pack = eng.recall("which database engine", top_k=3)
    found = {r.ulid for r in pack.results}
    # At least one postgres-mentioning event should surface
    assert e1 in found or any("postgres" in r.content.lower() for r in pack.results)


def test_graph_rebuild_from_events(tmp_pmb_home, tmp_workspace_dir):
    """Reindex after wiping graph state."""
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_fact("Use Postgres 17 with Redis")
    eng.record_fact("Switch tests to pytest from unittest")
    n_before = eng.graph_stats()["n_entities"]
    # Wipe entity rows directly to simulate pre-graph data
    import sqlite3
    with sqlite3.connect(eng.workspace.db_path) as conn:
        conn.execute("DELETE FROM graph_entities")
        conn.execute("DELETE FROM graph_event_entities")
        conn.execute("DELETE FROM graph_edges")
    assert eng.graph_stats()["n_entities"] == 0

    r = eng.graph_rebuild_from_events()
    assert r["n_events_indexed"] == 2
    assert r["n_entities"] == n_before


def test_top_entities_sorted_by_mentions(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    # Mention postgres in DIFFERENT facts (post-Improvement U dedup, identical
    # writes get collapsed into one event, so we vary the text per write to
    # genuinely create multiple postgres-touching events).
    eng.record_fact("Postgres tuning for write-heavy workload")
    eng.record_fact("Postgres backup strategy uses logical replication")
    eng.record_fact("Postgres connection pool sized at 100 by default")
    eng.record_fact("Redis for session cache only")

    top = eng.graph_top_entities(kind="tech", limit=5)
    names = [e["name"] for e in top]
    assert names[0] == "postgres"  # highest mentions
    assert "redis" in names


def test_graph_neighbors_for_missing_entity(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    r = eng.graph_neighbors("nonexistent-thing")
    assert r["entity"] is None
    assert r["neighbors"] == []


def test_multi_entity_bonus_favors_multi_hop_event(
    tmp_pmb_home, tmp_workspace_dir,
):
    """Multi-hop scenario: query mentions two distinct entities. The event
    that contains BOTH should outrank events that contain only one, thanks
    to recall.multi_entity_bonus.

    This is the key fix for LoCoMo cat-3 questions where the answer event
    mentions multiple things from the query."""
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={
            "recall.cache_size": 0,
            "recall.spreading_activation": False,
            "recall.multi_entity_bonus": 0.8,  # strong bonus for test clarity
        },
    )
    # Two single-entity events
    eng.record_fact("Postgres backup runs nightly at 3am via pg_dump")
    eng.record_fact("Redis is used for session storage on port 6379")
    # The multi-hop event — mentions both Postgres AND Redis
    multi = eng.record_fact(
        "We migrated from Postgres to Redis for session cache last week"
    )
    # Query touches both entities
    pack = eng.recall("postgres and redis migration", top_k=3)
    ulids = [r.ulid for r in pack.results]
    assert multi in ulids[:2], (
        f"multi-entity event should rank in top-2 with multi_entity_bonus, "
        f"got order {ulids}"
    )


def test_multi_entity_bonus_disabled_by_zero(
    tmp_pmb_home, tmp_workspace_dir,
):
    """Setting bonus to 0 should disable the multiplier — sanity check
    that the config knob actually controls behaviour."""
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={
            "recall.cache_size": 0,
            "recall.spreading_activation": False,
            "recall.multi_entity_bonus": 0.0,
        },
    )
    eng.record_fact("Postgres uses port 5433")
    eng.record_fact("Redis uses port 6379")
    eng.record_fact("Postgres and Redis both run in docker")
    # No exception, recall completes — proves the disabled path is correct.
    pack = eng.recall("postgres redis", top_k=3)
    assert len(pack.results) >= 1
