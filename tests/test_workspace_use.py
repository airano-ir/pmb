"""T4: persisted default workspace + `pmb workspace use / current / --clear`.

Locks the resolution order (env > project config > saved default > git/cwd)
and the CLI surface. The new "saved default" step must be invisible to setups
that never call `use` (absent file == old behaviour).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmb.core.workspace import (
    Workspace,
    detect_workspace,
    read_default_workspace,
    set_default_workspace,
)


def _make_ws(pmb_home: Path, ws_id: str, name: str = "WS") -> Workspace:
    ws = Workspace(id=ws_id, name=name, root=Path("/proj"),
                   pmb_home=pmb_home, source="explicit")
    ws.ensure_dirs()
    ws.save_meta()
    return ws


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("PMB_WORKSPACE", raising=False)
    monkeypatch.delenv("PMB_HOME", raising=False)


# ── resolution order ────────────────────────────────────────────────────────

def test_absent_default_is_old_behaviour(tmp_path, clean_env):
    """No saved default → resolve from cwd as before (source cwd/git), NOT
    'default'."""
    home = tmp_path / "home"
    cwd = tmp_path / "proj"
    cwd.mkdir()
    ws = detect_workspace(cwd=cwd, pmb_home=home)
    assert ws.source in ("cwd", "git")


def test_saved_default_beats_cwd_fallback(tmp_path, clean_env):
    home = tmp_path / "home"
    _make_ws(home, "personal", "Personal")
    set_default_workspace(home, "personal")
    cwd = tmp_path / "random"
    cwd.mkdir()
    ws = detect_workspace(cwd=cwd, pmb_home=home)
    assert ws.id == "personal"
    assert ws.source == "default"


def test_env_beats_saved_default(tmp_path, clean_env, monkeypatch):
    home = tmp_path / "home"
    _make_ws(home, "personal", "Personal")
    set_default_workspace(home, "personal")
    monkeypatch.setenv("PMB_WORKSPACE", "envws")
    cwd = tmp_path / "random"
    cwd.mkdir()
    ws = detect_workspace(cwd=cwd, pmb_home=home)
    assert ws.id == "envws"
    assert ws.source == "env"


def test_project_config_beats_saved_default(tmp_path, clean_env):
    home = tmp_path / "home"
    _make_ws(home, "personal", "Personal")
    set_default_workspace(home, "personal")
    cwd = tmp_path / "proj"
    (cwd / ".pmb").mkdir(parents=True)
    (cwd / ".pmb" / "workspace.yaml").write_text(
        "id: projws\nname: ProjWS\n", encoding="utf-8")
    ws = detect_workspace(cwd=cwd, pmb_home=home)
    assert ws.id == "projws"
    assert ws.source == "config"


def test_default_pointing_at_missing_workspace_is_ignored(tmp_path, clean_env):
    """A saved default whose storage no longer exists must NOT phantom-create
    it — fall through to normal detection."""
    home = tmp_path / "home"
    set_default_workspace(home, "ghost")  # never created
    cwd = tmp_path / "random"
    cwd.mkdir()
    ws = detect_workspace(cwd=cwd, pmb_home=home)
    assert ws.id != "ghost"
    assert ws.source in ("cwd", "git")


def test_set_and_clear_roundtrip(tmp_path):
    home = tmp_path / "home"
    assert read_default_workspace(home) is None
    set_default_workspace(home, "personal")
    assert read_default_workspace(home) == "personal"
    set_default_workspace(home, None)
    assert read_default_workspace(home) is None


# ── CLI ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def cli(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from pmb.cli.main import app
    home = tmp_path / "home"
    monkeypatch.setenv("PMB_HOME", str(home))
    monkeypatch.delenv("PMB_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)
    _make_ws(home, "personal", "Personal")
    return CliRunner(), app, home


def test_cli_use_sets_default(cli):
    runner, app, home = cli
    r = runner.invoke(app, ["workspace", "use", "personal"])
    assert r.exit_code == 0, r.output
    assert read_default_workspace(home) == "personal"


def test_cli_use_unknown_name_exits_1(cli):
    runner, app, home = cli
    r = runner.invoke(app, ["workspace", "use", "doesnotexist"])
    assert r.exit_code == 1
    assert read_default_workspace(home) is None


def test_cli_use_clear(cli):
    runner, app, home = cli
    set_default_workspace(home, "personal")
    r = runner.invoke(app, ["workspace", "use", "--clear"])
    assert r.exit_code == 0, r.output
    assert read_default_workspace(home) is None


def test_cli_current_reports_saved_default(cli):
    runner, app, home = cli
    runner.invoke(app, ["workspace", "use", "personal"])
    r = runner.invoke(app, ["workspace", "current"])
    assert r.exit_code == 0, r.output
    assert "personal" in r.output.lower()
