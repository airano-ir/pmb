"""Project observer — ambient for MCP-only agents by watching git.

Cursor / VS Code / Zed / Gemini give no hooks, so we observe the project's
git working tree instead of the agent. These tests pin the change detection
(new/modified files become Edit actions, deletions ignored, snapshot diffing
avoids re-reporting unchanged files) on a real temp git repo.
"""

from __future__ import annotations

import subprocess

import pytest

from pmb.hooks.project_observer import (
    is_git_repo,
    snapshot_changes,
)


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True, text=True)


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "proj"
    r.mkdir()
    try:
        _git(r, "init")
        _git(r, "config", "user.email", "t@t.t")
        _git(r, "config", "user.name", "t")
    except Exception:
        pytest.skip("git not available")
    (r / "base.py").write_text("x = 1\n", encoding="utf-8")
    _git(r, "add", "-A")
    _git(r, "commit", "-m", "init")
    return r


def test_is_git_repo(repo, tmp_path):
    assert is_git_repo(repo) is True
    assert is_git_repo(tmp_path / "not-a-repo") is False


def test_detects_new_and_modified_files(repo):
    (repo / "auth.py").write_text("auth", encoding="utf-8")
    (repo / "base.py").write_text("x = 2\n", encoding="utf-8")  # modify
    actions, state = snapshot_changes(repo, {})
    targets = {a["target"] for a in actions}
    assert "auth.py" in targets
    assert "base.py" in targets
    assert all(a["tool"] == "Edit" for a in actions)


def test_snapshot_diffing_no_repeat(repo):
    (repo / "a.py").write_text("a", encoding="utf-8")
    actions1, state1 = snapshot_changes(repo, {})
    assert any(a["target"] == "a.py" for a in actions1)
    # second scan with no new changes → nothing new
    actions2, state2 = snapshot_changes(repo, state1)
    assert actions2 == []
    # now touch a NEW file → only that shows up
    (repo / "b.py").write_text("b", encoding="utf-8")
    actions3, state3 = snapshot_changes(repo, state2)
    assert {a["target"] for a in actions3} == {"b.py"}


def test_modified_again_re_reported(repo):
    import time
    (repo / "c.py").write_text("v1", encoding="utf-8")
    _, state = snapshot_changes(repo, {})
    time.sleep(0.02)
    (repo / "c.py").write_text("v2 changed", encoding="utf-8")  # mtime bumps
    actions, _ = snapshot_changes(repo, state)
    assert any(a["target"] == "c.py" for a in actions)


def test_deletions_ignored(repo):
    (repo / "base.py").unlink()  # delete a tracked file
    actions, _ = snapshot_changes(repo, {})
    assert all(a["target"] != "base.py" for a in actions)


def test_full_observer_to_autowrite(tmp_path, monkeypatch, repo):
    """End-to-end: observe project changes → record → autowrite synthesizes."""
    monkeypatch.setenv("PMB_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PMB_WORKSPACE", "obs_e2e")
    from pmb.core.engine import Engine
    from pmb.core.ambient_log import insert_agent_action
    from pmb.hooks import run_autowrite

    eng = Engine()
    (repo / "auth.py").write_text("a", encoding="utf-8")
    (repo / "utils.py").write_text("u", encoding="utf-8")
    (repo / "db.py").write_text("d", encoding="utf-8")
    actions, _ = snapshot_changes(repo, {})
    for a in actions:
        insert_agent_action(eng.workspace.db_path, eng.workspace.id,
                            tool=a["tool"], target=a["target"], status="ok")
    res = run_autowrite(eng, window_minutes=60, min_actions=2,
                        synthesizer="template", apply=True)
    assert res.wrote is True
    assert "file(s)" in res.summary
