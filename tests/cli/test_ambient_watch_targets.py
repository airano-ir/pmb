from __future__ import annotations

import os

from pmb.cli.commands.ambient import (
    _load_ambient_watch_paths,
    _save_ambient_watch_paths,
    _split_ambient_watch_paths,
)


def test_split_ambient_watch_paths_uses_platform_separator():
    raw = os.pathsep.join(["app", " api ", "", "docs"])
    assert _split_ambient_watch_paths(raw) == ["app", "api", "docs"]


def test_save_and_load_ambient_watch_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("PMB_HOME", str(tmp_path))
    paths = [str(tmp_path / "app"), str(tmp_path / "api")]

    _save_ambient_watch_paths(paths)

    assert _load_ambient_watch_paths() == paths
