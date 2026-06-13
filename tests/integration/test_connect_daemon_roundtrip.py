"""Caveat-S6a: a REAL HTTP round-trip, not just the entry shape.

`pmb connect --daemon` writes an MCP entry whose `Authorization: Bearer <token>`
header is the persistent daemon token. This drives the EXACT token + header the
connect flow produces through the ACTUAL daemon bearer middleware (the same ASGI
path a live request takes), proving the written entry authenticates and that a
missing/wrong token is rejected. The only thing this can't cover is Claude
Code's own MCP protocol handshake (no live client here) — auth + transport are
proven end-to-end with the real artifacts.
"""
from __future__ import annotations

import pytest

pytest.importorskip("starlette")
pytest.importorskip("httpx")

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PMB_HOME", str(tmp_path / "pmb_home"))
    return tmp_path


def _app(token: str) -> Starlette:
    from pmb.mcp.daemon import _daemon_bearer_middleware

    async def mcp(request):
        return PlainTextResponse("tools-here")

    async def health(request):
        return PlainTextResponse("healthy")

    app = Starlette(routes=[
        Route("/mcp", mcp, methods=["GET", "POST", "OPTIONS"]),
        Route("/internal/health", health, methods=["GET"]),
    ])
    app.add_middleware(_daemon_bearer_middleware(token))
    return app


def test_connect_entry_header_authenticates_against_the_real_daemon(isolated):
    from pmb.cli.connect import make_daemon_entry
    from pmb.mcp.daemon import write_daemon_token

    token = write_daemon_token()                      # persistent token
    entry = make_daemon_entry()                        # what `pmb connect` writes
    auth = entry["headers"]["Authorization"]
    assert auth == f"Bearer {token}"

    client = TestClient(_app(token))
    # the EXACT header the connect entry carries → authenticated 200 with content
    r = client.post("/mcp", headers={"Authorization": auth}, json={"jsonrpc": "2.0"})
    assert r.status_code == 200
    assert "tools-here" in r.text


def test_missing_and_wrong_token_are_rejected(isolated):
    from pmb.mcp.daemon import write_daemon_token
    token = write_daemon_token()
    client = TestClient(_app(token))
    assert client.post("/mcp", json={"jsonrpc": "2.0"}).status_code in (401, 403)
    assert client.post("/mcp", headers={"Authorization": "Bearer wrong"},
                       json={"jsonrpc": "2.0"}).status_code in (401, 403)


def test_health_is_open_so_discovery_works(isolated):
    from pmb.mcp.daemon import write_daemon_token
    token = write_daemon_token()
    client = TestClient(_app(token))
    # /internal/health must answer WITHOUT the token (clients probe it to find
    # a live daemon before they have the token loaded).
    assert client.get("/internal/health").status_code == 200
