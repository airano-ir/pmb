"""Regression: the hook reads stdin as UTF-8, not the platform locale.

Claude Code pipes the user message as UTF-8. On Windows, sys.stdin.read()
defaults to cp1251 and mangles Cyrillic into mojibake, which makes the
intent regexes miss — auto-recall silently produces nothing for any
non-ASCII message. _read_stdin_utf8 must decode UTF-8 explicitly.
"""

from __future__ import annotations

import io

from pmb.cli.main import _read_stdin_utf8


class _FakeStdin:
    """Mimics sys.stdin with a .buffer that yields raw bytes."""
    def __init__(self, raw: bytes):
        self.buffer = io.BytesIO(raw)

    def read(self):  # text fallback path
        return self.buffer.getvalue().decode("utf-8", errors="replace")


def test_reads_utf8_cyrillic(monkeypatch):
    msg = "почему мы выбрали playwright для e2e"
    monkeypatch.setattr("sys.stdin", _FakeStdin(msg.encode("utf-8")))
    assert _read_stdin_utf8() == msg


def test_reads_utf8_mixed_scripts(monkeypatch):
    msg = "виправ баг у LoadGuard — чому ми обрали osrm?"
    monkeypatch.setattr("sys.stdin", _FakeStdin(msg.encode("utf-8")))
    assert _read_stdin_utf8() == msg


def test_bad_bytes_do_not_crash(monkeypatch):
    # Invalid UTF-8 must not raise — errors='replace'.
    monkeypatch.setattr("sys.stdin", _FakeStdin(b"\xff\xfe bad \x80 bytes"))
    out = _read_stdin_utf8()
    assert isinstance(out, str)
    assert "bad" in out


def test_no_buffer_falls_back_to_read(monkeypatch):
    # pytest-style StringIO stdin (no .buffer) must still work.
    monkeypatch.setattr("sys.stdin", io.StringIO("plain text message"))
    assert _read_stdin_utf8() == "plain text message"
