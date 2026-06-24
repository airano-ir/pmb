"""Tests for auto-capture of exploration memos from a Claude Code transcript."""
from __future__ import annotations

import json

from pmb.hooks.exploration_capture import (
    extract_exploration_from_transcript,
    run_autocapture,
)


def _write(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for o in entries:
            f.write(json.dumps(o) + "\n")


def test_extract_basic(tmp_path):
    t = tmp_path / "t.jsonl"
    _write(t, [
        {"type": "user", "message": {"role": "user", "content": "How does auth work?"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}},
            {"type": "tool_use", "name": "Read", "input": {"file_path": "b.py"}},
            {"type": "tool_use", "name": "Grep", "input": {"path": "c.py", "pattern": "x"}},
        ]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": "..."}]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Auth is handled in a.py login()."},
        ]}},
    ])
    info = extract_exploration_from_transcript(str(t), min_files=3)
    assert info is not None
    assert info["intent"] == "How does auth work?"
    assert set(info["files"]) == {"a.py", "b.py", "c.py"}
    assert "a.py" in info["conclusion"]


def test_extract_below_threshold(tmp_path):
    t = tmp_path / "t.jsonl"
    _write(t, [
        {"type": "user", "message": {"content": "q"}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Read", "input": {"file_path": "only.py"}},
            {"type": "text", "text": "answer"},
        ]}},
    ])
    assert extract_exploration_from_transcript(str(t), min_files=3) is None


def test_extract_no_user_message(tmp_path):
    t = tmp_path / "t.jsonl"
    _write(t, [{"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}}])
    assert extract_exploration_from_transcript(str(t), min_files=1) is None


def test_run_autocapture_records(tmp_path):
    t = tmp_path / "t.jsonl"
    _write(t, [
        {"type": "user", "message": {"content": "how does X work"}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}},
            {"type": "tool_use", "name": "Read", "input": {"file_path": "b.py"}},
            {"type": "tool_use", "name": "Read", "input": {"file_path": "c.py"}},
            {"type": "text", "text": "X is implemented in a.py"},
        ]}},
    ])
    captured: dict = {}

    class _FakeEng:
        def record_batch_async(self, items):
            captured["items"] = items
            return {"ok": True}

    res = run_autocapture(_FakeEng(), str(t), min_files=3, apply=True)
    assert res["captured"] is True
    assert captured["items"][0]["metadata"]["source"] == "exploration-memo"


def test_run_autocapture_dry_run(tmp_path):
    t = tmp_path / "t.jsonl"
    _write(t, [
        {"type": "user", "message": {"content": "q"}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}},
            {"type": "tool_use", "name": "Read", "input": {"file_path": "b.py"}},
            {"type": "tool_use", "name": "Read", "input": {"file_path": "c.py"}},
            {"type": "text", "text": "ans"},
        ]}},
    ])
    res = run_autocapture(None, str(t), min_files=3, apply=False)
    assert res["captured"] is False
    assert res.get("dry_run") is True
    assert len(res["files"]) == 3


def test_run_autocapture_missing_transcript(tmp_path):
    res = run_autocapture(None, str(tmp_path / "nope.jsonl"), min_files=3)
    assert res["captured"] is False
