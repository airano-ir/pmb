"""Tests for atomic-fact extraction (Improvement D)."""
from __future__ import annotations

import json

from pmb.core.engine import Engine
from pmb.reasoning.facts import FactExtractor


class _StubFactLLM:
    def __init__(self, response: list[str]):
        self.response = response
        self.calls = 0

    def complete(self, prompt: str, **kw) -> str:
        self.calls += 1
        return json.dumps(self.response)


def test_extractor_parses_atomic_facts():
    from pmb.core.events import Event
    ev = Event(
        ulid="src1", workspace_id="ws", event_type="qa",
        content="Caroline: I researched adoption agencies last week and met with my mentor",
        metadata={"session_dt": "2025-05-15"},
    )
    stub = _StubFactLLM([
        "Caroline researched adoption agencies",
        "Caroline met with her mentor on 2025-05-15",
    ])
    ext = FactExtractor(stub)
    facts = ext.extract(ev)
    assert len(facts) == 2
    assert all(f.source_ulid == "src1" for f in facts)
    assert "adoption agencies" in facts[0].text


def test_extractor_handles_markdown_fences():
    from pmb.core.events import Event
    ev = Event(ulid="x", workspace_id="ws", event_type="qa", content="hi")
    class _LLM:
        def complete(self, p, **k):
            return '```json\n["a fact about something"]\n```'
    facts = FactExtractor(_LLM()).extract(ev)
    assert len(facts) == 1


def test_extractor_filters_junk():
    from pmb.core.events import Event
    ev = Event(ulid="x", workspace_id="ws", event_type="qa", content="content")
    stub = _StubFactLLM([
        "ok",         # too short
        "hi",         # too short
        "a" * 300,    # too long
        "Caroline researched adoption agencies",  # valid
        "Caroline researched adoption agencies",  # duplicate
    ])
    facts = FactExtractor(stub).extract(ev)
    assert len(facts) == 1


def test_engine_extract_facts_batch_creates_searchable_facts(
    tmp_pmb_home, tmp_workspace_dir,
):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    source = eng.record_fact(
        "Caroline researched adoption agencies and met with her mentor"
    )
    stub = _StubFactLLM([
        "Caroline researched adoption agencies",
        "Caroline met with her mentor",
    ])
    result = eng.extract_facts_batch(limit=10, llm=stub)
    assert result["n_facts_added"] == 2

    # Verify facts are searchable
    pack = eng.recall("what did Caroline research?", top_k=5)
    contents = [r.content for r in pack.results]
    assert any("adoption agencies" in c for c in contents), (
        f"expected fact 'adoption agencies' searchable; got {contents}"
    )


def test_engine_extract_facts_idempotent(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_fact("Postgres on port 5433")
    stub = _StubFactLLM(["Postgres uses port 5433"])
    eng.extract_facts_batch(limit=10, llm=stub)
    # Second call shouldn't process the same event again
    second = eng.extract_facts_batch(limit=10, llm=stub)
    assert second["n_candidates"] == 0
    assert stub.calls == 1


def test_engine_extract_facts_skips_when_no_llm(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    eng.record_fact("Something")
    # No llm passed and no real backend
    result = eng.extract_facts_batch(limit=10, backend="anthropic")
    # Either skipped no_llm or attempted but failed gracefully
    assert "n_facts_added" in result
    assert result["n_facts_added"] == 0
