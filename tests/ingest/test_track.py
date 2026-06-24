"""Tests for git-tied semantic change tracking + module summaries.

We monkeypatch `record_batch_async` to capture the items PMB would write, so
these tests exercise the git parsing / summarisation / cursor logic without
loading the embedding + LanceDB write pipeline (fast, no flake).
"""
from __future__ import annotations

import subprocess

from pmb.ingest.track import (
    _read_cursor,
    _write_cursor,
    summarize_modules,
    track_changes,
)


class _StubLLM:
    """Returns a fixed summary; counts calls."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, max_tokens: int = 256, **kw) -> str:
        self.calls += 1
        return "Adds X to accomplish Y.\n- a.py: new helper"


def _git(cwd, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True,
                   capture_output=True, text=True)


def _make_repo(d) -> None:
    d.mkdir(parents=True, exist_ok=True)
    _git(d, "init")
    _git(d, "config", "user.email", "t@example.com")
    _git(d, "config", "user.name", "Test")
    (d / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    _git(d, "add", "-A")
    _git(d, "commit", "-m", "add a.py")


def test_track_changes_records_intent_and_advances_cursor(
    isolated_engine, tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    _make_repo(repo)

    captured: dict = {}
    monkeypatch.setattr(
        isolated_engine, "record_batch_async",
        lambda items: captured.update(items=items) or {"ok": True},
    )

    stub = _StubLLM()
    res = track_changes(isolated_engine, repo, llm=stub)

    assert res.get("n_commits", 0) == 1
    assert res["queued"] is True
    assert stub.calls == 1
    assert res["cursor"]

    item = captured["items"][0]
    # Stored as a fact so record_batch keeps our metadata (an activity's would
    # be replaced with {actor, activity_kind}).
    assert item["type"] == "fact"
    assert item["metadata"]["source"] == "git-change"
    assert item["metadata"]["commit_short"]
    assert "a.py" in item["metadata"]["files"]
    assert "Why:" in item["content"]

    # Idempotent: the cursor advanced, so a second run sees nothing new.
    res2 = track_changes(isolated_engine, repo, llm=stub)
    assert res2["n_commits"] == 0


def test_track_changes_since_override(isolated_engine, tmp_path, monkeypatch):
    repo = tmp_path / "repo2"
    _make_repo(repo)
    (repo / "b.py").write_text("x = 2\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "add b.py")

    monkeypatch.setattr(isolated_engine, "record_batch_async", lambda items: {"ok": True})
    # Only the most recent commit is after HEAD~1.
    res = track_changes(isolated_engine, repo, since="HEAD~1", llm=_StubLLM())
    assert res["n_commits"] == 1


def test_track_changes_not_a_repo(isolated_engine, tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    res = track_changes(isolated_engine, plain, llm=_StubLLM())
    assert "error" in res
    assert "not a git repository" in res["error"]


def test_summarize_modules_requires_index(isolated_engine, tmp_path):
    res = summarize_modules(isolated_engine, tmp_path, llm=_StubLLM())
    assert "error" in res
    assert "index project" in res["error"]


def test_summarize_modules_happy_and_idempotent(isolated_engine, monkeypatch):
    from pmb.ingest import track

    monkeypatch.setattr(track, "_indexed_files", lambda e, p: [
        {"file_path": "a.py", "content": "File: a.py\nSymbols: def f", "language": "python"},
    ])
    monkeypatch.setattr(track, "_files_with_summary", lambda e, p: set())

    captured: dict = {}
    monkeypatch.setattr(
        isolated_engine, "record_batch_async",
        lambda items: captured.update(items=items) or {"ok": True},
    )

    res = summarize_modules(isolated_engine, ".", llm=_StubLLM())
    assert res["n_summarized"] == 1
    md = captured["items"][0]["metadata"]
    assert md["source"] == "module-summary"
    assert md["file_path"] == "a.py"

    # Idempotent: once a file has a summary it is skipped.
    monkeypatch.setattr(track, "_files_with_summary", lambda e, p: {"a.py"})
    res2 = summarize_modules(isolated_engine, ".", llm=_StubLLM())
    assert res2["n_summarized"] == 0


def test_cursor_roundtrip(isolated_engine):
    assert _read_cursor(isolated_engine, "/repo/x") is None
    _write_cursor(isolated_engine, "/repo/x", "abc123")
    assert _read_cursor(isolated_engine, "/repo/x") == "abc123"
    # second repo key does not clobber the first
    _write_cursor(isolated_engine, "/repo/y", "def456")
    assert _read_cursor(isolated_engine, "/repo/x") == "abc123"
    assert _read_cursor(isolated_engine, "/repo/y") == "def456"
