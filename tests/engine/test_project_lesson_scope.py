"""Project routing and lesson scoping in multi-project memory workspaces."""
from __future__ import annotations

from pmb.core.engine import Engine


def _engine(ws, home):
    eng = Engine(
        cwd=ws,
        pmb_home=home,
        config_overrides={"recall.cache_size": 0, "dedup.enable": False},
    )
    eng.workspace.name = "PMB"
    return eng


def _project_entity(eng, name: str, mentions: int) -> None:
    for _ in range(mentions):
        eng.graph.upsert_entity(eng.workspace.id, "person", name)


def _lesson(eng, content: str) -> None:
    eng.record_fact(content, metadata={"kind": "lesson", "source": "lesson"})


def test_current_repo_wins_over_more_popular_project_in_same_message(
    tmp_pmb_home, tmp_workspace_dir,
):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    _project_entity(eng, "LoadGuard", 30)
    _project_entity(eng, "PMB", 3)

    detected = eng.detect_project_in_text(
        "PMB surfaced irrelevant LoadGuard lessons; improve PMB precision."
    )

    assert detected is not None
    assert detected["name"].lower() == "pmb"


def test_foreign_project_still_detects_when_current_repo_is_not_named(
    tmp_pmb_home, tmp_workspace_dir,
):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    _project_entity(eng, "LoadGuard", 30)
    _project_entity(eng, "PMB", 3)

    detected = eng.detect_project_in_text("Fix the pricing view in LoadGuard.")

    assert detected is not None
    assert detected["name"].lower() == "loadguard"


def test_project_scoped_lessons_drop_explicit_foreign_projects(
    tmp_pmb_home, tmp_workspace_dir,
):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    for name in ("PMB", "LoadGuard", "ApplyPilot", "HackerNoon"):
        _project_entity(eng, name, 3)

    _lesson(eng, "PMB lesson surfacing must prefer project-specific rules.")
    _lesson(eng, "LoadGuard lesson surfacing must verify live Relay fields.")
    _lesson(eng, "Adversarial verification for LoadGuard must use primary sources.")
    _lesson(eng, "ApplyPilot lesson surfacing must keep the browser visible.")
    _lesson(eng, "HackerNoon lesson surfacing requires rendered HTML paste.")
    _lesson(eng, "General lesson surfacing should preserve exact identifier matches.")

    hits = eng.find_lessons(
        query="improve PMB lesson surfacing precision",
        project="PMB",
        limit=10,
    )
    contents = " || ".join(x["content"] for x in hits)

    assert "PMB lesson surfacing" in contents
    assert "General lesson surfacing" in contents
    assert "LoadGuard lesson surfacing" not in contents
    assert "Adversarial verification for LoadGuard" not in contents
    assert "ApplyPilot lesson surfacing" not in contents
    assert "HackerNoon lesson surfacing" not in contents


def test_record_batch_persists_structured_lesson_project(
    tmp_pmb_home, tmp_workspace_dir,
):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    result = eng.record_batch([
        {
            "type": "lesson",
            "content": "Always run the scoped PMB regression gate.",
            "project": "PMB",
        },
    ])
    assert result["n_ok"] == 1

    pmb_hits = eng.find_lessons("scoped regression gate", project="PMB")
    other_hits = eng.find_lessons("scoped regression gate", project="LoadGuard")

    assert any("scoped PMB regression" in x["content"] for x in pmb_hits)
    assert not any("scoped PMB regression" in x["content"] for x in other_hits)
