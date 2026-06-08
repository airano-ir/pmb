"""Phase 2 / issue #6: running-server registry + `pmb mcp status` + HTTP
singleton guard. No real server is started — we test the registry primitives
and the CLI surface directly."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmb.mcp import registry as R

DEAD_PID = 999_999_999  # astronomically unlikely to be a live process


@pytest.fixture
def pmb_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PMB_HOME", str(tmp_path / "home"))
    return tmp_path


def test_register_and_list_self(pmb_home):
    entry = R.register_server(transport="stdio", workspace="personal")
    assert entry["pid"] == os.getpid()
    servers = R.list_servers()
    assert len(servers) == 1
    assert servers[0]["pid"] == os.getpid()
    assert servers[0]["alive"] is True
    assert servers[0]["transport"] == "stdio"


def test_dead_pid_is_pruned(pmb_home):
    R.register_server(transport="stdio", workspace="x", pid=DEAD_PID)
    R.register_server(transport="stdio", workspace="me")  # self, alive
    servers = R.list_servers(prune=True)
    pids = {s["pid"] for s in servers}
    assert DEAD_PID not in pids
    assert os.getpid() in pids


def test_find_live_http_matches_host_port(pmb_home):
    R.register_server(transport="streamable-http", host="127.0.0.1", port=18999)
    hit = R.find_live_http("127.0.0.1", 18999)
    assert hit is not None
    assert hit["pid"] == os.getpid()
    # wrong port / wrong host → no match
    assert R.find_live_http("127.0.0.1", 19000) is None
    assert R.find_live_http("0.0.0.0", 18999) is None


def test_find_live_http_ignores_dead_and_stdio(pmb_home):
    R.register_server(transport="streamable-http", host="127.0.0.1",
                      port=18000, pid=DEAD_PID)           # dead http
    R.register_server(transport="stdio", workspace="me")  # alive but stdio
    assert R.find_live_http("127.0.0.1", 18000) is None


def test_unregister(pmb_home):
    R.register_server(transport="stdio", workspace="me")
    assert len(R.list_servers()) == 1
    R.unregister_server(os.getpid())
    assert R.list_servers() == []


# ── CLI: pmb mcp status ────────────────────────────────────────────────────

def test_mcp_status_empty(pmb_home):
    from typer.testing import CliRunner
    from pmb.cli.main import app
    r = CliRunner().invoke(app, ["mcp", "status"])
    assert r.exit_code == 0, r.output
    assert "No PMB MCP servers" in r.output


def test_mcp_status_lists_registered(pmb_home):
    from typer.testing import CliRunner
    from pmb.cli.main import app
    R.register_server(transport="streamable-http", host="127.0.0.1",
                      port=8765, path="/mcp", workspace="personal")
    r = CliRunner().invoke(app, ["mcp", "status"])
    assert r.exit_code == 0, r.output
    assert "127.0.0.1:8765" in r.output
    assert str(os.getpid()) in r.output
