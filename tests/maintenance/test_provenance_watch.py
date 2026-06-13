"""Tests for source attribution (provenance) + watch-folder scanning.

Pure functions - no Engine, no models. Fast, CI-friendly.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from pmb.ingest.watch import _split_paragraphs, load_state, save_state, scan_new_chunks
from pmb.provenance import describe_source, source_key

# ----------------------------------------------------------------------
# provenance.describe_source
# ----------------------------------------------------------------------

def test_describe_chatgpt_with_conversation():
    s = describe_source({"source": "chatgpt", "conversation": "Project planning", "role": "user"})
    assert s == "chatgpt · Project planning (user)"


def test_describe_markdown_file():
    assert describe_source({"source": "markdown", "file": "notes.md"}) == "markdown · notes.md"


def test_describe_cli_note():
    assert describe_source({"source": "cli-note"}) == "note (cli)"


def test_describe_lesson():
    assert describe_source({"source": "lesson", "kind": "lesson"}) == "lesson"
    assert source_key({"source": "lesson"}) == "lesson"


def test_describe_failure():
    assert describe_source({"source": "lesson", "kind": "failure"}) == "failure"


def test_describe_watch():
    assert describe_source({"source": "watch", "file": "journal.md"}) == "watch · journal.md"


def test_describe_mem0():
    assert describe_source({"source": "mem0"}) == "import:mem0"


def test_describe_agent_inferred():
    assert describe_source({"actor": "agent"}) == "agent"
    assert describe_source({"agent_id": "frontend-x"}) == "agent"


def test_describe_unknown():
    assert describe_source({}) == "-"
    assert describe_source(None) == "-"


def test_source_key_buckets():
    assert source_key({"source": "chatgpt", "conversation": "x"}) == "chatgpt"
    assert source_key({"actor": "agent"}) == "agent"
    assert source_key({}) == "unknown"


# ----------------------------------------------------------------------
# watch.scan_new_chunks
# ----------------------------------------------------------------------

def test_split_paragraphs():
    text = "first para line1\nfirst para line2\n\nsecond para\n\n\nthird"
    paras = _split_paragraphs(text)
    assert paras == ["first para line1\nfirst para line2", "second para", "third"]


def test_scan_new_then_incremental():
    with tempfile.TemporaryDirectory() as t:
        f = Path(t) / "journal.md"
        f.write_text(
            "We decided to use Postgres for the backend storage layer.\n\n"
            "Alice is the new tech lead and lives in Berlin now.\n",
            encoding="utf-8",
        )
        # first scan: both paragraphs are new
        items, seen = scan_new_chunks(f, set())
        assert len(items) == 2
        assert all(it["file"] == "journal.md" for it in items)

        # second scan with same seen: nothing new
        items2, seen2 = scan_new_chunks(f, seen)
        assert items2 == []

        # append a new paragraph: only it is ingested
        f.write_text(f.read_text(encoding="utf-8")
                     + "\n\nWe migrated the API from REST to gRPC last quarter.\n",
                     encoding="utf-8")
        items3, seen3 = scan_new_chunks(f, seen2)
        assert len(items3) == 1
        assert "gRPC" in items3[0]["content"]


def test_scan_skips_short_paragraphs():
    with tempfile.TemporaryDirectory() as t:
        f = Path(t) / "n.md"
        f.write_text("ok\n\nThis is a sufficiently long paragraph to be ingested.\n", encoding="utf-8")
        items, _ = scan_new_chunks(f, set())
        assert len(items) == 1
        assert "sufficiently long" in items[0]["content"]


def test_scan_directory_tree():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        (root / "a.md").write_text("My cat Whiskers is twelve years old this spring.", encoding="utf-8")
        sub = root / "sub"; sub.mkdir()
        (sub / "b.md").write_text("The deploy runs on GitHub Actions on every push to main.", encoding="utf-8")
        items, _ = scan_new_chunks(root, set())
        assert len(items) == 2
        files = {it["file"] for it in items}
        assert files == {"a.md", "b.md"}


def test_state_roundtrip():
    with tempfile.TemporaryDirectory() as t:
        sp = Path(t) / "watch_state.json"
        assert load_state(sp) == set()        # absent -> empty
        save_state(sp, {"a", "b", "c"})
        assert load_state(sp) == {"a", "b", "c"}


def test_state_load_corrupt_is_empty():
    with tempfile.TemporaryDirectory() as t:
        sp = Path(t) / "watch_state.json"
        sp.write_text("not json", encoding="utf-8")
        assert load_state(sp) == set()
