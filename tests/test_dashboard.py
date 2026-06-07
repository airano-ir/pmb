"""Tests for the web dashboard server (Improvement I)."""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmb.core.engine import Engine
from pmb.dashboard.server import make_handler, run_dashboard


@pytest.fixture
def tmp_pmb_home():
    import gc, shutil, time as _t
    tmp = tempfile.mkdtemp()
    home = Path(tmp) / "pmb_home"
    os.environ["PMB_HOME"] = str(home)
    try:
        yield home
    finally:
        os.environ.pop("PMB_HOME", None)
        gc.collect()
        for _ in range(3):
            try:
                shutil.rmtree(tmp, ignore_errors=False)
                break
            except (OSError, PermissionError):
                _t.sleep(0.2)
                gc.collect()
        else:
            shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def tmp_workspace_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


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
    # Give it a moment
    time.sleep(0.1)
    yield (eng, port)
    server.shutdown()
    server.server_close()


# ----------------------------------------------------------------------
# Static + API
# ----------------------------------------------------------------------

def _get_json(port, path: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(port, path: str, payload, timeout: float = 5):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_dashboard_serves_html(running_dashboard):
    _, port = running_dashboard
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp:
        body = resp.read().decode("utf-8")
    # The dashboard's HTML page identifies itself with the PMB brand and is
    # an HTML document; the rest of the markup (tabs, graph, panels) is
    # asserted via the JSON APIs in the tests below.
    assert "PMB" in body
    assert "<html" in body


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

    r1 = run_auto_context(eng, "какие правила про pnpm и npm")
    r2 = run_auto_context(eng, "какие правила про numpy lancedb")
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
    res = run_auto_context(eng, "какие правила про dashboard svg overlay")
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
