"""Tests for the self-contained memory-graph visualization (pmb viz)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmb.dashboard.viz import build_memory_html


class _Ws:
    id = "w"
    name = "demo"


class _Graph:
    def viz_graph(self, wsid, limit=300, max_edges=4000):
        return {
            "nodes": [
                {"id": 1, "kind": "person", "name": "Alex", "mentions": 5, "last_seen": 1.0},
                {"id": 2, "kind": "project", "name": "PMB <x>", "mentions": 3, "last_seen": 2.0},
            ],
            "edges": [{"a": 1, "b": 2, "w": 4}],
        }

    def stats(self, wsid):
        return {"n_entities": 2, "n_edges": 1, "by_kind": {"person": 1, "project": 1}}


class _Eng:
    def __init__(self):
        self.workspace = _Ws()
        self.graph = _Graph()


def _db_json(html: str) -> dict:
    line = next(l for l in html.splitlines() if l.startswith("const DB = "))
    return json.loads(line[len("const DB = "):].rstrip(";"))


def test_viz_html_structure():
    html = build_memory_html(_Eng(), limit=10)
    assert "<canvas" in html
    assert "/*__DATA__*/null" not in html          # data placeholder replaced
    assert "__TITLE__" not in html                 # title placeholder replaced
    # a '<' inside an entity name must be escaped so it can't break out of <script>
    assert "PMB <x>" not in html
    assert "\\u003c" in html


def test_viz_embedded_json_is_valid():
    data = _db_json(build_memory_html(_Eng(), limit=10))
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1
    assert data["stats"]["n_entities"] == 2
    # escaped name round-trips correctly through JSON (< -> <)
    assert any(n["name"] == "PMB <x>" for n in data["nodes"])


def test_viz_empty_workspace_has_empty_state():
    class _G(_Graph):
        def viz_graph(self, *a, **k):
            return {"nodes": [], "edges": []}

        def stats(self, *a, **k):
            return {"n_entities": 0, "n_edges": 0, "by_kind": {}}

    e = _Eng()
    e.graph = _G()
    html = build_memory_html(e)
    assert "<canvas" in html
    assert "No entities yet" in html


def test_viz_integration_real_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("PMB_HOME", str(tmp_path / "h"))
    monkeypatch.setenv("PMB_WORKSPACE", "vizt")
    from pmb.core.engine import Engine

    eng = Engine()
    eng.record_fact("Alex works on the PMB project with Postgres and Redis")
    eng.record_fact("PMB uses LanceDB and SQLite for storage")

    html = build_memory_html(eng, limit=50)
    assert "<canvas" in html
    data = _db_json(html)
    # graph indexing is synchronous, so at least some entities should exist
    assert isinstance(data["nodes"], list)
    assert data["stats"]["n_entities"] >= 1
