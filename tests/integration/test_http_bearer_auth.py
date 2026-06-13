"""E2e for the streamable-http transport's bearer-token auth.

Rather than boot a full uvicorn server (slow, flaky to tear down on
Windows), we mount the real `_build_bearer_middleware` middleware on a tiny
Starlette app and drive it through Starlette's TestClient — the same ASGI
path a live request takes. This pins the auth contract:

  • no token / wrong token              → 401
  • correct token (constant-time match) → passes through (200)
  • CORS preflight (OPTIONS) + health    → pass through unauthenticated
  • empty token config                  → middleware is None (auth disabled)
"""

from __future__ import annotations

import pytest

starlette = pytest.importorskip("starlette")
pytest.importorskip("httpx")

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from pmb.mcp.server import _build_bearer_middleware

TOKEN = "s3cr3t-bearer-token-xyz"


def _app(token: str) -> Starlette:
    async def ok(request):
        return PlainTextResponse("ok")

    async def health(request):
        return PlainTextResponse("healthy")

    app = Starlette(routes=[
        Route("/mcp", ok, methods=["GET", "POST", "OPTIONS"]),
        Route("/healthz", health, methods=["GET"]),
        Route("/", health, methods=["GET"]),
    ])
    mw = _build_bearer_middleware(token)
    if mw is not None:
        app.add_middleware(mw)
    return app


def test_no_token_rejected():
    client = TestClient(_app(TOKEN))
    r = client.post("/mcp", json={"jsonrpc": "2.0"})
    assert r.status_code == 401
    assert r.json().get("error") == "unauthorized"


def test_wrong_token_rejected():
    client = TestClient(_app(TOKEN))
    r = client.post("/mcp", headers={"Authorization": "Bearer nope"},
                    json={"jsonrpc": "2.0"})
    assert r.status_code == 401


def test_correct_token_passes():
    client = TestClient(_app(TOKEN))
    r = client.post("/mcp", headers={"Authorization": f"Bearer {TOKEN}"},
                    json={"jsonrpc": "2.0"})
    assert r.status_code == 200
    assert r.text == "ok"


def test_malformed_header_rejected():
    client = TestClient(_app(TOKEN))
    # right token value but missing the "Bearer " scheme prefix
    r = client.post("/mcp", headers={"Authorization": TOKEN},
                    json={"jsonrpc": "2.0"})
    assert r.status_code == 401


def test_cors_preflight_passes_without_auth():
    client = TestClient(_app(TOKEN))
    r = client.options("/mcp")
    assert r.status_code < 400  # preflight allowed through


def test_health_endpoints_pass_without_auth():
    client = TestClient(_app(TOKEN))
    assert client.get("/healthz").status_code == 200
    assert client.get("/").status_code == 200


def test_empty_token_disables_auth():
    # No token configured → middleware factory returns None → no auth gate.
    assert _build_bearer_middleware("") is None
    client = TestClient(_app(""))
    r = client.post("/mcp", json={"jsonrpc": "2.0"})
    assert r.status_code == 200  # unauthenticated but allowed (network-ACL mode)


def test_case_insensitive_header_name():
    # HTTP header names are case-insensitive; the check must still work.
    client = TestClient(_app(TOKEN))
    r = client.post("/mcp", headers={"AUTHORIZATION": f"Bearer {TOKEN}"},
                    json={"jsonrpc": "2.0"})
    assert r.status_code == 200
