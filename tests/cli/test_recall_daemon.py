"""CLI `pmb recall` routes through the warm daemon when available (bug 7)."""
from __future__ import annotations

import json

from pmb.cli.commands.capture import _pack_from_daemon, _try_daemon_recall


def test_pack_from_daemon_maps_all_signal_fields():
    out = {
        "workspace_name": "myws",
        "n_total_in_workspace": 42,
        "elapsed_ms": 12.5,
        "results": [{
            "ulid": "u1", "event_type": "fact", "content": "hello",
            "metadata": {"kind": "lesson"}, "timestamp": 1000.0, "score": 1.23,
            "signals": {"bm25": 0.9, "vector": 0.8, "raw_cosine": 0.5,
                        "importance": 0.7, "recency": 0.6},
        }],
    }
    pack = _pack_from_daemon("q", out)
    assert pack.workspace_name == "myws"
    assert pack.n_total_in_workspace == 42
    assert pack.elapsed_ms == 12.5
    assert len(pack.results) == 1
    r = pack.results[0]
    assert r.ulid == "u1" and r.content == "hello"
    assert r.score == 1.23
    assert r.bm25_score == 0.9 and r.vec_score == 0.8
    assert r.importance == 0.7 and r.recency_score == 0.6 and r.raw_vec == 0.5
    assert r.metadata == {"kind": "lesson"}


def test_try_daemon_recall_returns_none_without_daemon(monkeypatch):
    """No live daemon -> None, so the CLI falls back to a cold local engine."""
    monkeypatch.setattr(
        "pmb.mcp.registry.find_live_daemon", lambda: None, raising=False)
    assert _try_daemon_recall("anything", 5) is None


def test_try_daemon_recall_rejects_version_mismatch(monkeypatch):
    """A live but stale-version daemon is rejected so the CLI falls back."""
    monkeypatch.setattr(
        "pmb.mcp.registry.find_live_daemon",
        lambda: {"host": "127.0.0.1", "port": 65500}, raising=False)
    monkeypatch.setattr(
        "pmb.mcp.daemon.read_daemon_token", lambda: "tok", raising=False)
    monkeypatch.setattr(
        "pmb.core.workspace.detect_workspace",
        lambda: type("W", (), {"id": "ws"})(), raising=False)

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"version": "0.0.0-stale", "results": []}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
    assert _try_daemon_recall("q", 5) is None
