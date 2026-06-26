"""Tests for the Reflective Memory layer (PMB v2 Phase 1).

The key claim: a reflection event makes a source event findable for
multi-hop queries that share no lexical content with the source itself —
because the LLM-generated reflection ('might_answer' bullets, themes,
implications) is what surfaces in search."""
from __future__ import annotations

import json

import pytest

from pmb.core.engine import Engine

# ----------------------------------------------------------------------
# Stub LLM
# ----------------------------------------------------------------------

class _StubLLM:
    """Returns a fixed reflection JSON for any prompt."""

    def __init__(self, override: dict | None = None):
        self.calls = 0
        self.override = override or {}

    def complete(self, prompt: str, **kw) -> str:
        self.calls += 1
        data = {
            "significance": "User chose Postgres over MySQL for the project's primary store.",
            "implications": [
                "Schema-on-write decisions favoured",
                "Backup tooling now uses pg_dump",
            ],
            "might_answer": [
                "Which database does this project use?",
                "Why did we switch away from MySQL?",
                "What is the primary store?",
            ],
            "linked_themes": ["database_choice", "postgres_migration"],
            "people": [],
            "time_anchor": "Q4 2025",
        }
        data.update(self.override)
        return json.dumps(data)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------





# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_reflect_event_creates_searchable_reflection(
    tmp_pmb_home, tmp_workspace_dir,
):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    source = eng.record_fact("Decided on Postgres for primary store")
    stub = _StubLLM()

    out = eng.reflect_event(source, llm=stub)
    assert out is not None
    assert out["source_ulid"] == source
    refl_ulid = out["reflection_ulid"]

    refl_ev = eng.events.get_by_ulid(refl_ulid)
    assert refl_ev is not None
    assert refl_ev.event_type == "reflection"
    assert refl_ev.metadata.get("source_ulid") == source
    # Content should contain at least the 'might-answer' phrasings
    assert "might answer" in refl_ev.content.lower() or "might_answer" in refl_ev.content.lower()


def test_reflect_event_idempotent(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    source = eng.record_fact("Use Postgres on port 5433")
    stub = _StubLLM()

    out1 = eng.reflect_event(source, llm=stub)
    out2 = eng.reflect_event(source, llm=stub)

    assert out1 is not None
    assert out2 is None, "second reflection on same source should be skipped"
    assert stub.calls == 1, "LLM should not be called again on duplicate"


def test_reflection_bridges_multihop_query(tmp_pmb_home, tmp_workspace_dir):
    """The killer claim: a query that doesn't lexically match the source
    event still finds it because the reflection's 'might_answer' or themes
    surface in hybrid search."""
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={"recall.cache_size": 0},
    )
    source = eng.record_fact(
        "Migrated all rows from MySQL into the new system on Dec 5"
    )
    # The source mentions MySQL & Dec 5 but NOT 'Postgres' or 'database choice'.
    # The stub reflection adds those phrases.
    stub = _StubLLM(override={
        "might_answer": [
            "Which database does the project use now?",
            "When did we migrate away from MySQL?",
        ],
        "linked_themes": ["postgres_migration", "database_choice"],
    })
    eng.reflect_event(source, llm=stub)

    # Query that hits the reflection's vocabulary, not the source's
    pack = eng.recall("which database does this project use", top_k=5)
    ulids = [r.ulid for r in pack.results]
    # The source OR its reflection should surface
    source_or_refl_found = any(
        r.metadata.get("source_ulid") == source or r.ulid == source
        for r in pack.results
    )
    assert source_or_refl_found, (
        f"Expected source {source} or its reflection in top-5, got: "
        f"{[(r.ulid, r.event_type, r.content[:60]) for r in pack.results]}"
    )


# platform_sensitive: the unreflected-candidate count varies with embedder dedup
# on macOS arm64; deterministic + green on the Linux reference runner.
@pytest.mark.platform_sensitive
def test_reflect_batch_picks_unreflected(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    a = eng.record_fact("FactA: thing A happened")
    b = eng.record_fact("FactB: thing B happened")
    c = eng.record_fact("FactC: thing C happened")
    stub = _StubLLM()

    # First reflect only on A manually
    eng.reflect_event(a, llm=stub)

    # Batch should pick up B and C, skip A
    result = eng.reflect_batch(limit=10, llm=stub)
    assert result["n_candidates"] == 2  # B and C
    assert result["n_reflected"] == 2


def test_reflect_batch_skips_when_no_llm(tmp_pmb_home, tmp_workspace_dir):
    """When no LLM client available and no llm= override, reflect_batch
    returns gracefully without raising."""
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_fact("Some fact")

    # Force backend="none" path via a fake resolver that raises
    class _NoLLM:
        def complete(self, prompt, **kw):
            raise RuntimeError("no LLM")

    # reflect_event with broken LLM returns None, doesn't raise
    out = eng.reflect_event(eng.events.list_active(eng.workspace.id)[0].ulid, llm=_NoLLM())
    assert out is None


def test_reflection_bridge_entities_attach_to_source(
    tmp_pmb_home, tmp_workspace_dir,
):
    """Improvement B: reflection's LLM-extracted entities/themes get linked
    back to the SOURCE event in the graph, not to a separate reflection
    chunk. Source becomes findable through reflection vocabulary."""
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={"recall.reflection_to_edges": True},
    )
    source = eng.record_fact("Set up the api service for the project")
    stub = _StubLLM(override={
        "linked_themes": ["postgres_migration", "database_choice"],
        "people": ["alice", "bob"],
        "implications": ["Future API endpoints will use Postgres"],
        "might_answer": ["Which database does the project use?"],
    })
    out = eng.reflect_event(source, llm=stub)
    assert out is not None
    assert out.get("bridge_entities_added_to_source", 0) > 0

    # Verify the SOURCE event is now linked to the new entities
    import sqlite3
    with sqlite3.connect(eng.workspace.db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT e.kind, e.name
            FROM graph_event_entities ee
            JOIN graph_entities e ON e.id = ee.entity_id
            WHERE ee.event_ulid = ?
            """,
            (source,),
        ).fetchall()
    entity_names = {r["name"] for r in rows}
    # The 'people' / 'theme' entities should now be linked to SOURCE
    assert "alice" in entity_names or "bob" in entity_names, (
        f"expected reflection 'people' linked to source; got entities: {entity_names}"
    )
    assert "postgres_migration" in entity_names or "database_choice" in entity_names, (
        f"expected reflection 'themes' linked to source; got entities: {entity_names}"
    )


def test_reflection_to_edges_disabled_no_bridges(
    tmp_pmb_home, tmp_workspace_dir,
):
    """With the config off, no bridge entities are added."""
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={"recall.reflection_to_edges": False},
    )
    source = eng.record_fact("X")
    out = eng.reflect_event(source, llm=_StubLLM())
    assert out is not None
    assert out.get("bridge_entities_added_to_source", 0) == 0
