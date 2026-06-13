"""Async LLM graph extraction — the write path must never block on an LLM.

PMB's rule: the write path does NO blocking LLM call. An LLM graph extractor
(`llm:claude` / `llm:ollama` / `llm:codex`) does a CLI round-trip of up to
graph.llm_timeout_s per event; running that inline made records hang. These
tests pin the fix:

  • regex backend            → inline, synchronous, graph ready immediately
  • LLM backend (async on)   → write returns instantly, graph fills later
  • LLM backend (async off)  → inline (deterministic mode for tests)
"""

from __future__ import annotations

import time

import pytest

from pmb.graph.entities import ExtractedEntities


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("PMB_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PMB_WORKSPACE", "async_graph")
    from pmb.core.engine import Engine
    return Engine()


class _SlowLLM:
    """Mimics an LLM extractor: a slow CLI round-trip that yields one concept."""
    backend_name = "llm:fake"

    def __init__(self, delay=1.0, concept="asyncconcept"):
        self.delay = delay
        self.concept = concept
        self.calls = 0

    def extract(self, text, files_hint=None):
        self.calls += 1
        time.sleep(self.delay)
        return ExtractedEntities(files=[], techs=[], concepts=[self.concept])


def test_regex_backend_stays_inline(engine):
    # Default backend is regex — graph indexing is synchronous, nothing queued.
    assert getattr(engine.entity_extractor, "backend_name", "regex") == "regex"
    engine.record_fact("Chose Postgres over Redis for durability",
                       metadata={"kind": "decision"})
    assert engine.graph_queue_pending() == 0  # done inline


def test_llm_backend_does_not_block_write(engine):
    engine.entity_extractor = _SlowLLM(delay=1.5)
    t0 = time.time()
    engine.record_fact("A fact that needs slow LLM graph extraction",
                       metadata={"kind": "decision"})
    elapsed = time.time() - t0
    # Write returned WITHOUT waiting 1.5s for the extractor.
    assert elapsed < 0.5, f"write blocked on LLM extractor: {elapsed:.2f}s"
    # The event is queued (or already in-flight) for the background worker.
    assert engine.graph_queue_pending() >= 1


def test_llm_graph_eventually_consistent(engine):
    engine.entity_extractor = _SlowLLM(delay=0.3, concept="eventualconcept")
    engine.record_fact("Fact mentioning eventualconcept for the graph",
                       metadata={"kind": "decision"})
    res = engine.wait_for_graph_queue(timeout_seconds=10)
    assert res["drained"] is True
    assert engine.graph_queue_pending() == 0
    names = [e.get("name") for e in engine.graph_top_entities(limit=30)]
    assert "eventualconcept" in names


def test_async_off_runs_inline(engine):
    # With async disabled, the LLM extractor runs inline (blocks the write).
    engine.config._overrides["graph.async_llm"] = False
    slow = _SlowLLM(delay=0.4, concept="inlineconcept")
    engine.entity_extractor = slow
    t0 = time.time()
    engine.record_fact("Inline concept inlineconcept synchronous",
                       metadata={"kind": "decision"})
    elapsed = time.time() - t0
    # Ran inline → took at least the extractor delay, nothing queued.
    assert elapsed >= 0.3
    assert engine.graph_queue_pending() == 0
    names = [e.get("name") for e in engine.graph_top_entities(limit=30)]
    assert "inlineconcept" in names


def test_batch_write_does_not_block_on_llm(engine):
    engine.entity_extractor = _SlowLLM(delay=1.0)
    items = [
        {"type": "fact", "content": f"Batch fact number {i} about systems"}
        for i in range(5)
    ]
    t0 = time.time()
    engine.record_batch(items=items)
    elapsed = time.time() - t0
    # 5 events × 1s each would be ≥5s inline; async must keep it fast.
    assert elapsed < 1.0, f"batch blocked on LLM: {elapsed:.2f}s"


def test_wait_for_graph_queue_on_idle_engine(engine):
    # Nothing queued → returns drained immediately.
    res = engine.wait_for_graph_queue(timeout_seconds=1)
    assert res["drained"] is True
    assert res["remaining"] == 0
