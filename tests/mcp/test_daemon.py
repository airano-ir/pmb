"""B1/B2/B3: the persistent memory daemon.

- B2: the internal /internal/health + /internal/hook/prepare-context routes
  answer against the SAME warm engine (tested in-process via TestClient, no
  heavy model load needed — lesson surfacing is lexical).
- B1/B3: the registry knows a daemon (find_live_daemon) and the hook client
  (_try_daemon_prepare) talks to it, honoring the version handshake and
  degrading to None when absent / mismatched.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import pmb

# ── B2: internal routes answer against the warm engine (in-process) ─────────

def _build_daemon_app(cwd, seed_lesson: str | None = None):
    from starlette.testclient import TestClient

    from pmb.mcp.daemon import _register_internal_routes
    from pmb.mcp.server import build_server

    server = build_server(cwd=cwd, prewarm=False)  # no 20s model load in tests
    engine = server._pmb_engine
    if seed_lesson:
        engine.record_batch([{"type": "lesson", "content": seed_lesson}])
        engine.wait_for_writes(timeout=30)
    _register_internal_routes(server, engine)
    app = server.http_app(path="/mcp")
    return TestClient(app), engine


def test_internal_health_reports_version_and_workspace(tmp_pmb_home, tmp_workspace_dir):
    client, engine = _build_daemon_app(tmp_workspace_dir)
    with client:
        r = client.get("/internal/health")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["version"] == pmb.__version__
        assert body["workspace"] == engine.workspace.id


def test_internal_prepare_context_surfaces_a_lesson(tmp_pmb_home, tmp_workspace_dir):
    client, _ = _build_daemon_app(
        tmp_workspace_dir,
        seed_lesson="Always use pnpm in this repo, never npm.")
    with client:
        r = client.post("/internal/hook/prepare-context",
                        json={"message": "should I run npm or pnpm to install deps here?"})
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "daemon"
        assert body["version"] == pmb.__version__
        # lexical lesson surfacing needs no embedding model → deterministic
        assert "pnpm" in body["context"].lower()


def test_internal_prepare_context_empty_message(tmp_pmb_home, tmp_workspace_dir):
    client, _ = _build_daemon_app(tmp_workspace_dir)
    with client:
        r = client.post("/internal/hook/prepare-context", json={"message": ""})
        assert r.status_code == 200
        assert r.json()["context"] == ""


def test_internal_recall_returns_serialized_pack(tmp_pmb_home, tmp_workspace_dir):
    """The /internal/recall route lets `pmb recall` reuse the warm engine: it
    returns a serialized pack (results + the bm25/vector signal breakdown the
    CLI renders)."""
    client, engine = _build_daemon_app(tmp_workspace_dir)
    engine.record_batch([
        {"type": "fact", "content": "zz canonical deploy region is eu-central-7"},
    ])
    engine.wait_for_writes(timeout=30)
    with client:
        r = client.post("/internal/recall",
                        json={"query": "canonical deploy region", "top_k": 5})
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "daemon"
        assert body["version"] == pmb.__version__
        hit = [x for x in body.get("results", []) if "eu-central-7" in x["content"]]
        assert hit, "seeded fact should be recalled by the daemon"
        assert "signals" in hit[0] and "bm25" in hit[0]["signals"]


# ── B1: registry tracks a daemon; find_live_daemon matches kind+home ────────

def test_find_live_daemon_matches_kind(tmp_pmb_home):
    from pmb.mcp.registry import find_live_daemon, register_server, unregister_server
    # a plain mcp server must NOT be returned as a daemon
    register_server(transport="streamable-http", kind="mcp",
                    host="127.0.0.1", port=9111)
    assert find_live_daemon() is None
    # a daemon for THIS pmb_home is found
    entry = register_server(transport="streamable-http", kind="daemon",
                            host="127.0.0.1", port=9112)
    got = find_live_daemon()
    assert got is not None and got["port"] == 9112
    unregister_server(entry["pid"])


# ── B3: the hook client talks to a stub daemon + honors the version gate ────

class _StubHandler(BaseHTTPRequestHandler):
    version_to_send = pmb.__version__

    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        _ = self.rfile.read(n)
        body = json.dumps({
            "context": "STUB CONTEXT FROM DAEMON",
            "version": self.version_to_send,
            "source": "daemon",
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def stub_daemon(tmp_pmb_home):
    """Run a stub HTTP daemon + register it so find_live_daemon() resolves it."""
    from pmb.mcp.daemon import write_daemon_token
    from pmb.mcp.registry import register_server, unregister_server

    write_daemon_token()
    srv = HTTPServer(("127.0.0.1", 0), _StubHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    entry = register_server(transport="streamable-http", kind="daemon",
                            host="127.0.0.1", port=port)
    try:
        yield srv
    finally:
        unregister_server(entry["pid"])
        srv.shutdown()


def test_hook_client_uses_daemon(stub_daemon, tmp_pmb_home):
    from pmb.cli.commands.ambient import _try_daemon_prepare
    _StubHandler.version_to_send = pmb.__version__
    out = _try_daemon_prepare("where do I live?", 4000, timeout=2.0)
    assert out == "STUB CONTEXT FROM DAEMON"


def test_hook_client_rejects_version_mismatch(stub_daemon, tmp_pmb_home):
    from pmb.cli.commands.ambient import _try_daemon_prepare
    _StubHandler.version_to_send = "0.0.0-other"
    out = _try_daemon_prepare("where do I live?", 4000, timeout=2.0)
    assert out is None  # mismatched version → treat as absent, go cold
    _StubHandler.version_to_send = pmb.__version__


def test_hook_client_none_when_no_daemon(tmp_pmb_home):
    from pmb.cli.commands.ambient import _try_daemon_prepare
    # no daemon registered → immediate None (cold path)
    assert _try_daemon_prepare("anything", 4000, timeout=0.3) is None
