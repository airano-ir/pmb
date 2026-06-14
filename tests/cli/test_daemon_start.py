"""`pmb daemon start` hardening: a busy port must NOT silently kill the daemon.

  * `_free_port` skips an occupied port and returns the next bindable one
    (so a streamable-http MCP server already on 8765 no longer collides);
  * `_daemon_log_path` resolves under the PMB home, so the spawned daemon's
    stdout/stderr land in a diagnosable log instead of DEVNULL.
"""
from __future__ import annotations

import socket

from pmb.cli.commands.daemon import _daemon_log_path, _free_port


def test_free_port_skips_a_busy_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))           # OS picks a free port
    busy = s.getsockname()[1]
    s.listen(1)
    try:
        got = _free_port("127.0.0.1", busy)
        assert got != busy, "should have skipped the occupied port"
        assert busy < got < busy + 20
    finally:
        s.close()


def test_free_port_returns_a_truly_free_port_unchanged():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    free = s.getsockname()[1]
    s.close()                          # release it so it's bindable
    assert _free_port("127.0.0.1", free) == free


def test_daemon_log_path_under_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PMB_HOME", str(tmp_path))
    monkeypatch.delenv("PMB_WORKSPACE", raising=False)
    p = _daemon_log_path()
    assert p.name == "daemon.log"
    assert str(tmp_path) in str(p)
