"""D3: MCP recall-response compaction.

`_compact_recall` caps each result's content and drops null/empty top-level
fields, gated by config. 0/False are KEPT (they carry meaning). Never raises.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmb.mcp.tools import _compact_recall


class _Cfg:
    def __init__(self, d):
        self.d = d

    def get(self, k):
        return self.d.get(k)


class _Eng:
    def __init__(self, **cfg):
        self.config = _Cfg(cfg)


def _pack(content):
    return {
        "results": [{"content": content, "score": 0.0, "event_type": "fact"}],
        "lessons": [],            # empty list → KEPT (structural)
        "project_context": None,  # genuinely null → dropped
        "n": 1,
    }


def test_compaction_caps_content_and_drops_null():
    eng = _Eng(**{"mcp.compact_responses": True, "mcp.max_item_chars": 20})
    out = _compact_recall(_pack("x" * 100), eng)
    assert "project_context" not in out         # None dropped
    assert out["lessons"] == []                 # empty list KEPT (structural)
    assert out["n"] == 1
    body = out["results"][0]["content"]
    assert body.endswith("…") and len(body) <= 21
    assert out["results"][0]["score"] == 0.0   # 0.0 kept (meaningful)


def test_empty_results_key_is_never_dropped():
    """Regression: `results` is indexed directly by callers (rc["results"]),
    so an empty results list must survive compaction."""
    eng = _Eng(**{"mcp.compact_responses": True, "mcp.max_item_chars": 600})
    out = _compact_recall({"results": [], "lessons": [], "x": None}, eng)
    assert out["results"] == [] and out["lessons"] == []
    assert "x" not in out


def test_compaction_off_is_passthrough():
    eng = _Eng(**{"mcp.compact_responses": False, "mcp.max_item_chars": 20})
    pack = _pack("x" * 100)
    out = _compact_recall(pack, eng)
    assert out is pack                          # unchanged object
    assert "project_context" in out and out["results"][0]["content"] == "x" * 100


def test_cap_zero_keeps_full_content_but_still_drops_null():
    eng = _Eng(**{"mcp.compact_responses": True, "mcp.max_item_chars": 0})
    out = _compact_recall(_pack("y" * 100), eng)
    assert out["results"][0]["content"] == "y" * 100   # not capped
    assert "project_context" not in out                # None still dropped
    assert out["lessons"] == []                        # empty list kept


def test_short_content_untouched():
    eng = _Eng(**{"mcp.compact_responses": True, "mcp.max_item_chars": 600})
    out = _compact_recall(_pack("a short fact"), eng)
    assert out["results"][0]["content"] == "a short fact"


def test_never_raises_on_weird_input():
    eng = _Eng(**{"mcp.compact_responses": True, "mcp.max_item_chars": 20})
    assert _compact_recall("not a dict", eng) == "not a dict"
    assert _compact_recall({"results": "not-a-list", "x": None}, eng) == {"results": "not-a-list"}
