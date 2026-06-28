"""Tests for the web dashboard server (Improvement I)."""
from __future__ import annotations

import socket
import threading

import pytest
from _http import get_json as _get_json
from _http import get_text, wait_ready
from _http import post_json as _post_json

from pmb.core.engine import Engine
from pmb.dashboard.server import make_handler


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def running_dashboard(tmp_pmb_home, tmp_workspace_dir):
    """Start a dashboard server in a background thread; yield (engine, port)."""
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_fact("Caroline researched adoption agencies")
    eng.record_fact("Melanie does pottery")

    from http.server import ThreadingHTTPServer
    handler = make_handler(eng)
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # Block until the server answers (readiness), not a fixed sleep - a cold
    # per-test engine on a loaded CI runner can be slow to first respond.
    wait_ready(port)
    yield (eng, port)
    server.shutdown()
    server.server_close()


# ----------------------------------------------------------------------
# Static + API. The HTTP helpers (_get_json / _post_json) and the readiness
# wait live in tests/_http.py - resilient to CI timing flakes, shared so no
# test re-rolls a one-off urlopen(timeout=5).
# ----------------------------------------------------------------------

def test_dashboard_serves_html(running_dashboard):
    _, port = running_dashboard
    body = get_text(port, "/")
    # The dashboard's HTML page identifies itself with the PMB brand and is
    # an HTML document; the rest of the markup (tabs, graph, panels) is
    # asserted via the JSON APIs in the tests below.
    assert "PMB" in body
    assert "<html" in body


def test_static_route_rejects_path_traversal(running_dashboard):
    """`/static/../server.py` points at a real file just outside the static
    root. Without the confinement guard the server would read and leak it; with
    it, the request must 404. Sent over a raw socket so the client can't
    normalise the `..` away (CodeQL path-traversal)."""
    _, port = running_dashboard
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    sock.sendall(
        b"GET /static/../server.py HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\nConnection: close\r\n\r\n")
    resp = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        resp += chunk
    sock.close()
    status_line = resp.split(b"\r\n", 1)[0]
    assert b"404" in status_line, status_line
    assert b"make_handler" not in resp, "static traversal leaked source code"


def test_dashboard_api_stats(running_dashboard):
    eng, port = running_dashboard
    data = _get_json(port, "/api/stats")
    assert "workspace" in data
    assert "graph" in data


def test_dashboard_api_events(running_dashboard):
    eng, port = running_dashboard
    events = _get_json(port, "/api/events?limit=10")
    assert isinstance(events, list)
    assert len(events) >= 2
    assert "ulid" in events[0]
    assert "content" in events[0]


def test_dashboard_api_entities(running_dashboard):
    eng, port = running_dashboard
    ents = _get_json(port, "/api/entities?limit=20")
    assert isinstance(ents, list)


def test_dashboard_api_recall(running_dashboard):
    eng, port = running_dashboard
    # Pre-warm the engine in this process so the HTTP recall call below doesn't
    # eat the model-load latency (~10-30 s cold) and trip the socket timeout.
    # The dashboard server shares the same Engine instance, so a warmup here
    # makes its in-server recall hot too.
    try:
        eng.warmup()
    except AttributeError:
        eng.recall("warmup", top_k=1)
    res = _post_json(port, "/api/recall",
                     {"query": "adoption agencies", "top_k": 3}, timeout=60)
    assert "results" in res
    assert "elapsed_ms" in res


def test_dashboard_event_detail(running_dashboard):
    eng, port = running_dashboard
    events = _get_json(port, "/api/events?limit=1")
    ulid = events[0]["ulid"]
    detail = _get_json(port, f"/api/event/{ulid}")
    assert "entities" in detail
    assert detail["ulid"] == ulid


def test_dashboard_api_lessons(running_dashboard):
    """GET /api/lessons returns the self-improvement-loop aggregates the
    Lessons tab renders: total_surfaces / followed / ignored / unknown +
    per_lesson rows. Drives a real surface→follow→ignore cycle first."""
    eng, port = running_dashboard
    from pmb.hooks import run_auto_context

    eng.record_fact("This repo uses pnpm, never npm",
                    metadata={"kind": "lesson", "source": "lesson"})
    eng.record_fact("Pin numpy below 2.x for lancedb compatibility",
                    metadata={"kind": "lesson", "source": "lesson"})

    r1 = run_auto_context(eng, "do we have a rule about pnpm and npm")
    r2 = run_auto_context(eng, "do we have a rule about numpy and lancedb")
    eng.mark_lesson_followed(r1.lessons[0]["surface_id"], followed=True, note="ok")
    eng.mark_lesson_followed(r2.lessons[0]["surface_id"], followed=False, note="legacy")

    data = _get_json(port, "/api/lessons?days=1")
    assert data["total_surfaces"] >= 2
    assert data["followed"] >= 1
    assert data["ignored"] >= 1
    assert isinstance(data["per_lesson"], list) and data["per_lesson"]
    # every row carries the fields the frontend badge logic needs
    row = data["per_lesson"][0]
    for k in ("lesson_ulid", "surfaces", "followed", "ignored", "content"):
        assert k in row


def test_dashboard_api_adherence(running_dashboard):
    """GET /api/adherence must NOT crash on a fresh workspace (no mcp_calls
    table yet) and must still report lesson metrics + a per-day series."""
    eng, port = running_dashboard
    from pmb.hooks import run_auto_context

    eng.record_fact("Dashboard SVG overlay uses requestAnimationFrame",
                    metadata={"kind": "lesson", "source": "lesson"})
    res = run_auto_context(eng, "do we have a rule about dashboard svg overlay")
    eng.mark_lesson_followed(res.lessons[0]["surface_id"], followed=True, note="x")

    data = _get_json(port, "/api/adherence?days=1")
    assert "error" not in data, f"adherence endpoint errored: {data.get('error')}"
    # lesson metrics survive even though mcp_calls is absent
    assert data.get("lesson_surfaces", 0) >= 1
    assert data.get("lesson_followed", 0) >= 1
    assert "series" in data and isinstance(data["series"], list)


def test_dashboard_404_on_unknown_route(running_dashboard):
    _, port = running_dashboard
    import urllib.error
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get_json(port, "/nonsense")
    assert exc.value.code == 404
