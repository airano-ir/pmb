"""`pmb delete` / `pmb restore` CLI flow + the dashboard POST /api/delete route.

Both use a fake engine so they're deterministic and model-free. The dashboard
test binds port 0 (OS picks a free one) so it never collides with the daemon /
dashboard default of 8765.
"""
from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from typer.testing import CliRunner

from pmb.cli.main import app

runner = CliRunner()


# ── CLI: pmb delete / pmb restore ────────────────────────────────────────────

class _Ev:
    def __init__(self, ulid):
        self.ulid = ulid
        self.content = "a memory about the billing migration"
        self.event_type = "fact"


class _Events:
    def get_by_ulid(self, u):
        return _Ev(u) if u.startswith("ok") else None


class _FakeEngine:
    last = None

    def __init__(self, *a, **k):
        self.events = _Events()
        self.deleted = []
        self.restored = []
        _FakeEngine.last = self

    def delete_event(self, ulid, hard=False):
        self.deleted.append((ulid, hard))
        return {"ulid": ulid, "mode": "hard" if hard else "soft", "ok": True}

    def unforget(self, ulid):
        self.restored.append(ulid)


def _patch(monkeypatch):
    monkeypatch.setattr("pmb.cli.commands.capture.Engine", _FakeEngine)


def test_delete_soft_default_with_confirm(monkeypatch):
    _patch(monkeypatch)
    r = runner.invoke(app, ["delete", "ok1"], input="y\n")
    assert r.exit_code == 0, r.output
    assert _FakeEngine.last.deleted == [("ok1", False)]
    assert "Archived" in r.output


def test_delete_cancel_changes_nothing(monkeypatch):
    _patch(monkeypatch)
    r = runner.invoke(app, ["delete", "ok1"], input="n\n")
    assert r.exit_code == 0
    assert _FakeEngine.last.deleted == []
    assert "Cancelled" in r.output


def test_delete_hard_yes_skips_prompt_and_purges_each(monkeypatch):
    _patch(monkeypatch)
    r = runner.invoke(app, ["delete", "ok1", "ok2", "--hard", "--yes"])
    assert r.exit_code == 0, r.output
    assert _FakeEngine.last.deleted == [("ok1", True), ("ok2", True)]
    assert "permanently" in r.output.lower()


def test_delete_missing_ulid_exits_nonzero(monkeypatch):
    _patch(monkeypatch)
    r = runner.invoke(app, ["delete", "nope"])  # get_by_ulid -> None
    assert r.exit_code == 1
    assert "Nothing to delete" in r.output


def test_restore_calls_unforget(monkeypatch):
    _patch(monkeypatch)
    r = runner.invoke(app, ["restore", "ok1"])
    assert r.exit_code == 0, r.output
    assert _FakeEngine.last.restored == ["ok1"]
    assert "Restored" in r.output


# ── Dashboard: POST /api/delete/<ulid> ───────────────────────────────────────

class _DashFakeEngine:
    def __init__(self):
        self.calls = []

    def delete_event(self, ulid, hard=False):
        self.calls.append((ulid, hard))
        return {"ulid": ulid, "mode": "hard" if hard else "soft", "ok": True}


def _post(port, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_dashboard_api_delete_routes_soft_and_hard():
    from pmb.dashboard.server import make_handler
    eng = _DashFakeEngine()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(eng))  # 0 = free port
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        soft = _post(port, "/api/delete/ABC", {"hard": False})
        assert soft["mode"] == "soft" and soft["ok"] is True
        hard = _post(port, "/api/delete/ABC", {"hard": True})
        assert hard["mode"] == "hard"
        assert eng.calls == [("ABC", False), ("ABC", True)]
    finally:
        srv.shutdown()
        srv.server_close()
