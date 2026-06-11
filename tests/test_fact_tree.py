"""Tests for hierarchical fact trees (Improvement P)."""
from __future__ import annotations

from pmb.core.engine import Engine


def test_record_fact_tree_creates_main_and_subfacts(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    result = eng.record_fact_tree(
        "On May 23, 2026, user fell down stairs and broke arm",
        subfacts=[
            "Time of fall: 18:52",
            "Recommended: visit ER for X-ray",
            "First aid: ice 15-20min, don't drive",
        ],
        importance=0.9,
    )
    assert result["n_subfacts"] == 3
    assert result["main_ulid"]
    assert len(result["subfact_ulids"]) == 3

    # Main event has has_subfacts=True
    main = eng.events.get_by_ulid(result["main_ulid"])
    assert main.metadata.get("has_subfacts") is True
    assert main.importance == 0.9

    # Each subfact has parent_ulid + lower importance
    for sub_ulid in result["subfact_ulids"]:
        sub = eng.events.get_by_ulid(sub_ulid)
        assert sub.metadata.get("parent_ulid") == result["main_ulid"]
        assert sub.metadata.get("is_subfact") is True
        assert sub.importance < main.importance  # 0.85x


def test_get_subfacts_returns_linked_facts(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    result = eng.record_fact_tree(
        "User decided to migrate to Postgres",
        subfacts=["Reason: JSONB", "Target: end Q2", "Convert auth schema first"],
    )
    subs = eng.get_subfacts(result["main_ulid"])
    assert len(subs) == 3
    assert "Reason: JSONB" in [s["content"] for s in subs]


def test_get_parent_fact(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    result = eng.record_fact_tree(
        "Main event",
        subfacts=["sub a", "sub b"],
    )
    sub_ulid = result["subfact_ulids"][0]
    parent = eng.get_parent_fact(sub_ulid)
    assert parent is not None
    assert parent["ulid"] == result["main_ulid"]
    assert "Main event" in parent["content"]


def test_subfacts_searchable_independently(tmp_pmb_home, tmp_workspace_dir):
    """Each subfact lives in the index — recall should find them."""
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={"recall.cache_size": 0},
    )
    eng.record_fact_tree(
        "User broke arm on May 23",
        subfacts=[
            "First aid: ice 15-20 minutes",
            "Warning signs: numb fingers, bleeding, deformation",
        ],
        importance=0.9,
    )
    # Query that should hit the warning-signs subfact
    pack = eng.recall("когда вызывать 911 numbness", top_k=3)
    contents = [r.content for r in pack.results]
    assert any("Warning signs" in c for c in contents)


def test_no_subfacts_main_only(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    result = eng.record_fact_tree(
        "Just a single fact",
        subfacts=[],
    )
    assert result["n_subfacts"] == 0
    main = eng.events.get_by_ulid(result["main_ulid"])
    assert main.metadata.get("has_subfacts") is False


def test_empty_subfacts_skipped(tmp_pmb_home, tmp_workspace_dir):
    """Empty strings in subfacts list are skipped."""
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    result = eng.record_fact_tree(
        "Main",
        subfacts=["valid", "", "  ", "also valid"],
    )
    assert result["n_subfacts"] == 2


def test_subfact_has_causation_edge_to_parent(tmp_pmb_home, tmp_workspace_dir):
    """Improvement P creates an event_edge 'references' between main and each subfact."""
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    result = eng.record_fact_tree("Main", subfacts=["sub1", "sub2"])
    import sqlite3
    with sqlite3.connect(eng.workspace.db_path) as conn:
        rows = conn.execute(
            "SELECT source_ulid, target_ulid, edge_type FROM event_edges "
            "WHERE source_ulid = ? AND edge_type = 'references'",
            (result["main_ulid"],),
        ).fetchall()
    targets = {r[1] for r in rows}
    assert set(result["subfact_ulids"]) <= targets
