"""`_untar_workspace` must refuse malicious tar members (CodeQL tar-slip).

It rejects path-traversal and link/device members BEFORE writing anything, on
every Python version, and the no-'data'-filter fallback extracts the validated
members by hand (never tar.extract on a tainted path).
"""
from __future__ import annotations

import io
import tarfile

import pytest

from pmb.core.encryption import _untar_workspace


def _targz(members: list[dict]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for spec in members:
            ti = tarfile.TarInfo(spec["name"])
            if spec.get("link") is not None:
                ti.type = tarfile.SYMTYPE
                ti.linkname = spec["link"]
                tar.addfile(ti)
            else:
                payload = spec.get("data", b"x")
                ti.size = len(payload)
                tar.addfile(ti, io.BytesIO(payload))
    return buf.getvalue()


def test_untar_extracts_safe_members(tmp_path):
    data = _targz([{"name": "a.txt", "data": b"hello"},
                   {"name": "sub/b.txt", "data": b"world"}])
    names = _untar_workspace(data, tmp_path)
    assert set(names) == {"a.txt", "sub/b.txt"}
    assert (tmp_path / "a.txt").read_bytes() == b"hello"
    assert (tmp_path / "sub" / "b.txt").read_bytes() == b"world"


def test_untar_rejects_path_traversal(tmp_path):
    data = _targz([{"name": "../escape.txt", "data": b"pwn"}])
    with pytest.raises(ValueError):
        _untar_workspace(data, tmp_path / "dest")
    assert not (tmp_path / "escape.txt").exists()


def test_untar_rejects_symlink_member(tmp_path):
    data = _targz([{"name": "link", "link": "/etc/passwd"}])
    with pytest.raises(ValueError):
        _untar_workspace(data, tmp_path / "dest")


def test_untar_manual_fallback_when_no_data_filter(tmp_path, monkeypatch):
    """Force the old-runtime path (extractall without the 'data' filter raises
    TypeError) and confirm the by-hand extraction writes the validated members."""
    orig = tarfile.TarFile.extractall

    def no_filter(self, *args, **kwargs):
        if "filter" in kwargs:
            raise TypeError("filter unsupported on this runtime")
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(tarfile.TarFile, "extractall", no_filter)
    data = _targz([{"name": "a.txt", "data": b"hi"},
                   {"name": "d/e.txt", "data": b"yo"}])
    _untar_workspace(data, tmp_path)
    assert (tmp_path / "a.txt").read_bytes() == b"hi"
    assert (tmp_path / "d" / "e.txt").read_bytes() == b"yo"
