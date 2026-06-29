from __future__ import annotations

import socket
from unittest.mock import patch

from typer.testing import CliRunner

from pmb.cli.commands.capture import _dashboard_port_or_next
from pmb.cli.main import app


def test_dashboard_port_or_next_skips_busy_default_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    busy = s.getsockname()[1]
    s.listen(1)
    try:
        picked = _dashboard_port_or_next("127.0.0.1", busy, tries=20)
        assert picked != busy
        assert busy < picked < busy + 20
    finally:
        s.close()


def test_dashboard_port_or_next_keeps_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    free = s.getsockname()[1]
    s.close()
    assert _dashboard_port_or_next("127.0.0.1", free, tries=20) == free


def test_dashboard_command_auto_skips_busy_default_port():
    runner = CliRunner()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    busy = s.getsockname()[1]
    s.listen(1)
    called: list[tuple[str, int]] = []

    def fake_run_dashboard(_engine, host="127.0.0.1", port=8765):
        called.append((host, port))

    try:
        with patch("pmb.cli.commands.capture.Engine", lambda: object()), \
             patch("pmb.cli.commands.capture._DASHBOARD_DEFAULT_PORT", busy), \
             patch("pmb.dashboard.server.run_dashboard", fake_run_dashboard):
            result = runner.invoke(app, ["dashboard"])
    finally:
        s.close()

    assert result.exit_code == 0
    assert f"Port {busy} is unavailable" in result.output
    assert called and called[0][1] != busy


def test_dashboard_command_prints_friendly_bind_error_for_explicit_port():
    runner = CliRunner()

    def raising_run_dashboard(_engine, host="127.0.0.1", port=8765):
        raise PermissionError(10013, "forbidden")

    with patch("pmb.cli.commands.capture.Engine", lambda: object()), \
         patch("pmb.dashboard.server.run_dashboard", raising_run_dashboard):
        result = runner.invoke(app, ["dashboard", "--port", "18888"])

    assert result.exit_code == 2
    assert "Dashboard could not bind" in result.output
    assert "pmb dashboard --port 18888" in result.output
