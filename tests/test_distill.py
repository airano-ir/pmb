"""Tests for auto lesson distillation (zero-command memory growth).

Uses a FAKE llm client (.complete) so no real backend is needed. Verifies
parsing, dedup, and recording. Distillation is off the recall path, so these
are about the extraction wiring, not ranking.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmb.health.distill_lessons import _parse_items, distill_lessons


class FakeLLM:
    def __init__(self, response: str):
        self._r = response
        self.calls = 0

    def complete(self, prompt, max_tokens=600):
        self.calls += 1
        return self._r


# ----------------------------------------------------------------------
# _parse_items - defensive JSON extraction
# ----------------------------------------------------------------------

def test_parse_clean_json():
    txt = '[{"type":"lesson","content":"use pnpm not npm"},{"type":"failure","content":"numpy 2.x broke lancedb"}]'
    items = _parse_items(txt, 8)
    assert len(items) == 2
    assert items[0]["type"] == "lesson"
    assert items[1]["type"] == "failure"


def test_parse_json_with_surrounding_text():
    txt = 'Sure! Here are the lessons:\n[{"type":"lesson","content":"always run make fmt"}]\nHope that helps.'
    items = _parse_items(txt, 8)
    assert len(items) == 1
    assert "make fmt" in items[0]["content"]


def test_parse_bad_type_defaults_to_lesson():
    txt = '[{"type":"weird","content":"this is a reusable rule about builds"}]'
    items = _parse_items(txt, 8)
    assert items[0]["type"] == "lesson"


def test_parse_skips_too_short_and_caps():
    txt = '[{"type":"lesson","content":"ok"},{"type":"lesson","content":"a genuinely useful durable rule"}]'
    items = _parse_items(txt, 8)
    assert len(items) == 1  # "ok" dropped (too short)


def test_parse_garbage_returns_empty():
    assert _parse_items("not json at all", 8) == []
    assert _parse_items("", 8) == []


# ----------------------------------------------------------------------
# distill_lessons end-to-end with a fake LLM + real engine
# ----------------------------------------------------------------------

@pytest.fixture
def engine(monkeypatch):
    tmp = tempfile.mkdtemp()
    monkeypatch.setenv("PMB_HOME", tmp)
    monkeypatch.setenv("PMB_WORKSPACE", "distill_test")
    from pmb.core.engine import Engine
    eng = Engine()
    # seed some session-like events
    eng.record_batch([
        {"type": "activity", "kind": "edit", "content": "switched npm to pnpm after install errors"},
        {"type": "activity", "kind": "decision", "content": "user said: always use pnpm here"},
    ])
    return eng


def test_distill_records_lessons(engine):
    llm = FakeLLM('[{"type":"lesson","content":"this repo uses pnpm, never npm"}]')
    res = distill_lessons(engine, llm=llm)
    assert llm.calls == 1
    assert res["n_recorded"] == 1
    # the lesson is now stored and tagged
    evs = engine.events.list_active(engine.workspace.id, limit=50)
    lessons = [e for e in evs if (e.metadata or {}).get("kind") == "lesson"]
    assert any("pnpm" in e.content for e in lessons)
    assert lessons[0].metadata.get("distilled") is True


def test_distill_dedups_existing(engine):
    # pre-store the same lesson
    engine.record_batch([{"type": "lesson", "content": "this repo uses pnpm, never npm"}])
    llm = FakeLLM('[{"type":"lesson","content":"this repo uses pnpm, never npm"}]')
    res = distill_lessons(engine, llm=llm)
    assert res["n_recorded"] == 0  # duplicate not re-recorded
    assert res["fresh"] == []


def test_distill_dry_run_records_nothing(engine):
    llm = FakeLLM('[{"type":"failure","content":"bumping numpy to 2.x broke lancedb"}]')
    res = distill_lessons(engine, llm=llm, dry_run=True)
    assert res["n_recorded"] == 0
    assert len(res["candidates"]) == 1


def test_distill_empty_llm_output(engine):
    res = distill_lessons(engine, llm=FakeLLM("[]"))
    assert res["n_recorded"] == 0


def test_distill_no_events(monkeypatch):
    tmp = tempfile.mkdtemp()
    monkeypatch.setenv("PMB_HOME", tmp)
    monkeypatch.setenv("PMB_WORKSPACE", "empty_distill")
    from pmb.core.engine import Engine
    eng = Engine()
    res = distill_lessons(eng, llm=FakeLLM("[]"))
    assert res.get("skipped") == "no_events"
