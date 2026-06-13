"""Project-index records stay useful without polluting memory or the graph."""
from __future__ import annotations

import sqlite3

from pmb.core.engine import Engine
from pmb.core.events import Event
from pmb.ingest.project import index_project


def _engine(ws, home):
    return Engine(cwd=ws, pmb_home=home, config_overrides={"recall.cache_size": 0})


def test_index_project_skips_empty_structure_and_root_is_idempotent(
    tmp_pmb_home, tmp_workspace_dir,
):
    (tmp_workspace_dir / "useful.py").write_text(
        "import sqlite3\n\ndef load():\n    return sqlite3.connect(':memory:')\n",
        encoding="utf-8",
    )
    (tmp_workspace_dir / "empty.md").write_text("# Notes\nNothing structural.\n",
                                                 encoding="utf-8")
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)

    first = index_project(eng, tmp_workspace_dir)
    assert eng.wait_for_writes(timeout=30)
    second = index_project(eng, tmp_workspace_dir)
    assert eng.wait_for_writes(timeout=30)

    assert first["n_indexed"] == 1
    assert first["n_low_signal_skipped"] == 1
    assert second["n_low_signal_skipped"] == 1
    with sqlite3.connect(eng.workspace.db_path) as conn:
        roots = conn.execute(
            "SELECT COUNT(*) FROM events WHERE workspace_id=? "
            "AND json_extract(metadata_json, '$.project_root')=1 "
            "AND archived_at IS NULL",
            (eng.workspace.id,),
        ).fetchone()[0]
        empty_rows = conn.execute(
            "SELECT COUNT(*) FROM events WHERE workspace_id=? "
            "AND json_extract(metadata_json, '$.file_path')='empty.md' "
            "AND archived_at IS NULL",
            (eng.workspace.id,),
        ).fetchone()[0]
    assert roots == 1
    assert empty_rows == 0


def test_project_index_uses_structured_graph_entities(
    tmp_pmb_home, tmp_workspace_dir,
):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.record_fact(
        "File: tests/test_auth.py (python, 20 lines)\n"
        "Symbols: test_login\nImports: pytest",
        metadata={
            "source": "project",
            "project_name": "PMB",
            "file_path": "tests/test_auth.py",
            "file_path_posix": "tests/test_auth.py",
            "language": "python",
            "symbols": ["test_login"],
            "imports": ["pytest"],
        },
    )
    with sqlite3.connect(eng.workspace.db_path) as conn:
        rows = conn.execute(
            "SELECT kind, lower(name) FROM graph_entities WHERE workspace_id=?",
            (eng.workspace.id,),
        ).fetchall()
    pairs = set(rows)
    assert ("project", "pmb") in pairs
    assert ("file", "tests/test_auth.py") in pairs
    assert ("import", "pytest") in pairs
    assert not any(kind == "person" for kind, _ in pairs)
    assert not any(name in {"file", "symbols", "imports"} for _, name in pairs)


def test_project_overview_prefers_exact_entity_and_hides_file_artifacts(
    tmp_pmb_home, tmp_workspace_dir,
):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    root = eng.record_fact(
        "Project: PMB\nPath: C:/work/pmb",
        metadata={
            "source": "project",
            "project_name": "PMB",
            "project_path": "C:/work/pmb",
            "project_root": True,
        },
    )
    eng.record_fact(
        "File: src/pmb/core.py (python, 20 lines)\n"
        "Symbols: Engine\nImports: sqlite3",
        metadata={
            "source": "project",
            "project_name": "PMB",
            "file_path": "src/pmb/core.py",
            "language": "python",
            "symbols": ["Engine"],
            "imports": ["sqlite3"],
        },
    )
    goal_done = eng.record_goal("Ship PMB v0.8", status="done")
    goal_open = eng.record_goal("Document PMB install", status="pending")

    pmb_entity = eng.graph.find_entities_by_name(
        eng.workspace.id, ["PMB"], kinds=("project",)
    )[0]
    eng.graph.link_event(root, [pmb_entity.id])
    eng.graph.link_event(goal_done, [pmb_entity.id])
    eng.graph.link_event(goal_open, [pmb_entity.id])
    for _ in range(30):
        eng.graph.upsert_entity(eng.workspace.id, "concept", "PMB")
    for _ in range(20):
        eng.graph.upsert_entity(eng.workspace.id, "concept", "tmp_pmb_home")

    overview = eng.project_overview("PMB")
    assert overview["entity"]["name"].lower() == "pmb"
    assert all("File:" not in fact["content"] for fact in overview["key_facts"])
    assert {g["ulid"] for g in overview["open_goals"]} == {goal_open}
    assert {g["ulid"] for g in overview["completed_goals"]} == {goal_done}


def test_project_overview_legacy_graph_uses_exact_project_metadata(
    tmp_pmb_home, tmp_workspace_dir,
):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    root = eng.events.append(
        Event(
            workspace_id=eng.workspace.id,
            event_type="fact",
            content="Project: PMB",
            metadata={"source": "project", "project_name": "PMB", "project_root": True},
        )
    )
    for _ in range(30):
        eng.graph.upsert_entity(eng.workspace.id, "concept", "tmp_pmb_home")
    noisy = eng.graph.find_entities_by_name(eng.workspace.id, ["tmp_pmb_home"])[0]
    eng.graph.link_event(root.ulid, [noisy.id])

    overview = eng.project_overview("PMB")
    assert overview["entity"]["name"] == "PMB"
    assert overview["entity"]["kind"] == "project"
    assert overview["entity"]["id"] is None
