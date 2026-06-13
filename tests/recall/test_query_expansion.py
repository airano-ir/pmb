"""Tests for LLM-based graph query expansion."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from pmb.graph.expansion import _cache_path, expand_query


class _StubLLM:
    """Returns a fixed entity list — used to drive expansion tests."""

    def __init__(self, entities=None, raise_error=False):
        self._entities = entities or []
        self._raise = raise_error
        self.calls = 0

    def consolidate(self, texts):
        self.calls += 1
        if self._raise:
            raise RuntimeError("simulated llm failure")
        payload = json.dumps({"entities": self._entities})
        return {
            "consolidate": False,
            "summary": payload,
            "confidence": 1.0,
            "reasoning": "",
        }


@pytest.fixture
def tmp_storage():
    with tempfile.TemporaryDirectory() as t:
        yield Path(t)


def test_expand_returns_entities_from_llm(tmp_storage):
    llm = _StubLLM(["postgres", "redis"])
    out = expand_query("describe our backend stack", tmp_storage, llm=llm)
    assert out == ["postgres", "redis"]
    assert llm.calls == 1


def test_expand_caches_repeat_queries(tmp_storage):
    llm = _StubLLM(["argon2"])
    expand_query("how do we hash passwords", tmp_storage, llm=llm)
    expand_query("how do we hash passwords", tmp_storage, llm=llm)
    # Second call should be served from cache → LLM not called again
    assert llm.calls == 1


def test_expand_caches_empty_results(tmp_storage):
    """An empty result is still cached so we don't re-pay LLM."""
    llm = _StubLLM([])
    expand_query("vague vague vague", tmp_storage, llm=llm)
    expand_query("vague vague vague", tmp_storage, llm=llm)
    assert llm.calls == 1


def test_expand_without_llm_returns_empty(tmp_storage):
    assert expand_query("anything", tmp_storage, llm=None) == []


def test_expand_swallows_llm_errors(tmp_storage):
    llm = _StubLLM(raise_error=True)
    out = expand_query("anything", tmp_storage, llm=llm)
    assert out == []


def test_cache_file_is_jsonl(tmp_storage):
    llm = _StubLLM(["postgres"])
    expand_query("which db", tmp_storage, llm=llm)
    path = _cache_path(tmp_storage)
    assert path.exists()
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["query"] == "which db"
    assert rec["entities"] == ["postgres"]


def test_corrupt_cache_does_not_crash(tmp_storage):
    path = _cache_path(tmp_storage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json\n{still not}\n", encoding="utf-8")
    llm = _StubLLM(["redis"])
    out = expand_query("which cache", tmp_storage, llm=llm)
    # Falls through to live call, returns entity, appends new line
    assert out == ["redis"]
