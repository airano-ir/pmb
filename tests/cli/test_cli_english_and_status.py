"""T3: the user-facing CLI is English-only and bare `pmb` shows a status
dashboard (instead of dumping --help).

  * every command/group `--help` text is free of Cyrillic;
  * bare `pmb` (no subcommand) renders the status panel and exits 0;
  * `pmb --help` still lists commands (panel did not replace help).
"""
from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from pmb.cli.main import app

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point PMB at a throwaway home + cwd so the status panel opens a fresh,
    empty workspace instead of the developer's real memory."""
    monkeypatch.setenv("PMB_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("PMB_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_help_tree_has_no_cyrillic():
    """Walk the whole Click command tree and assert no `--help` leaks Russian."""
    from click.testing import CliRunner as ClickRunner
    from typer.main import get_command

    root = get_command(app)
    runner = ClickRunner()
    bad: list[str] = []

    def walk(cmd, path):
        out = runner.invoke(root, path + ["--help"]).output or ""
        if _CYRILLIC.search(out):
            bad.append(" ".join(path) or "<root>")
        for name, sub in sorted(getattr(cmd, "commands", {}).items()):
            walk(sub, path + [name])

    walk(root, [])
    assert not bad, f"Cyrillic found in --help for: {bad}"


def test_root_help_still_lists_commands():
    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0, r.output
    # A few representative commands must appear in the group help.
    for cmd in ("recall", "remember", "workspace", "stats"):
        assert cmd in r.output, r.output


def test_bare_pmb_renders_status_panel(isolated_home):
    r = CliRunner().invoke(app, [])
    assert r.exit_code == 0, r.output
    # The panel title carries the ✦ pmb wordmark + version.
    assert "✦" in r.output and "pmb" in r.output
    # The panel labels the active workspace + offers the switch hint.
    assert "workspace" in r.output.lower()
    assert "pmb --help" in r.output
