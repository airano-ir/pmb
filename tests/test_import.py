"""Tests for memory importers (chatgpt / claude / mem0 / markdown).

Pure parsing - no Engine, no embeddings. Uses synthetic exports that match
each tool's real schema.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from pmb.ingest import PARSERS, parse_source
from pmb.ingest.importers import parse_chatgpt, parse_claude, parse_markdown, parse_mem0


@pytest.fixture
def tmp():
    with tempfile.TemporaryDirectory() as t:
        yield Path(t)


# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------

def test_unknown_source_raises(tmp):
    with pytest.raises(ValueError):
        parse_source("notion", tmp)


def test_all_sources_registered():
    assert set(PARSERS) == {"chatgpt", "claude", "mem0", "markdown"}


# ----------------------------------------------------------------------
# ChatGPT
# ----------------------------------------------------------------------

def test_chatgpt_user_only(tmp):
    export = [{
        "title": "Project planning",
        "mapping": {
            "n1": {"message": {"author": {"role": "user"},
                               "create_time": 1700000000.0,
                               "content": {"content_type": "text",
                                           "parts": ["We decided to use Postgres for the backend"]}}},
            "n2": {"message": {"author": {"role": "assistant"},
                               "content": {"content_type": "text",
                                           "parts": ["Good choice, Postgres handles JSONB well."]}}},
            "n3": {"message": {"author": {"role": "system"},
                               "content": {"content_type": "text", "parts": ["You are helpful."]}}},
        },
    }]
    f = tmp / "conversations.json"
    f.write_text(json.dumps(export), encoding="utf-8")
    res = parse_chatgpt(f, roles={"user"})
    assert res.n_parsed == 1
    it = res.items[0]
    assert "Postgres" in it["content"]
    assert it["metadata"]["role"] == "user"
    assert it["metadata"]["conversation"] == "Project planning"
    assert it["metadata"]["original_ts"] == 1700000000.0


def test_chatgpt_both_roles(tmp):
    export = [{
        "title": "t",
        "mapping": {
            "n1": {"message": {"author": {"role": "user"},
                               "content": {"parts": ["I prefer dark mode in all my editors"]}}},
            "n2": {"message": {"author": {"role": "assistant"},
                               "content": {"parts": ["Noted, dark mode it is for everything."]}}},
        },
    }]
    f = tmp / "conversations.json"
    f.write_text(json.dumps(export), encoding="utf-8")
    res = parse_chatgpt(f, roles={"user", "assistant"})
    assert res.n_parsed == 2


def test_chatgpt_dir_path(tmp):
    (tmp / "conversations.json").write_text(json.dumps([
        {"title": "x", "mapping": {"n": {"message": {
            "author": {"role": "user"},
            "content": {"parts": ["My startup is building a Rust database engine"]}}}}}
    ]), encoding="utf-8")
    res = parse_chatgpt(tmp, roles={"user"})  # pass the directory
    assert res.n_parsed == 1


# ----------------------------------------------------------------------
# Claude
# ----------------------------------------------------------------------

def test_claude_chat_messages(tmp):
    export = [{
        "name": "Architecture review",
        "chat_messages": [
            {"sender": "human", "text": "I'm migrating the API from REST to gRPC next quarter",
             "created_at": "2026-04-01T10:00:00Z"},
            {"sender": "assistant", "text": "gRPC will help with the streaming endpoints."},
        ],
    }]
    f = tmp / "conversations.json"
    f.write_text(json.dumps(export), encoding="utf-8")
    res = parse_claude(f, roles={"user"})
    assert res.n_parsed == 1
    assert "gRPC" in res.items[0]["content"]
    assert res.items[0]["metadata"]["source"] == "claude"
    assert res.items[0]["metadata"]["original_ts"] == "2026-04-01T10:00:00Z"


def test_claude_content_block_schema(tmp):
    export = [{
        "name": "n",
        "chat_messages": [
            {"sender": "human", "content": [
                {"type": "text", "text": "We chose Tailwind over plain CSS for the dashboard"}]},
        ],
    }]
    f = tmp / "conversations.json"
    f.write_text(json.dumps(export), encoding="utf-8")
    res = parse_claude(f, roles={"user"})
    assert res.n_parsed == 1
    assert "Tailwind" in res.items[0]["content"]


# ----------------------------------------------------------------------
# mem0
# ----------------------------------------------------------------------

def test_mem0_list_of_dicts(tmp):
    f = tmp / "mem0.json"
    f.write_text(json.dumps([
        {"memory": "User is allergic to peanuts", "created_at": "2026-01-01"},
        {"text": "Prefers TypeScript over JavaScript"},
        {"junk": "no text field here"},
    ]), encoding="utf-8")
    res = parse_mem0(f, roles=set())
    assert res.n_parsed == 2  # two valid, one skipped
    assert res.skipped == 1
    assert all(i["importance"] == 0.7 for i in res.items)


def test_mem0_wrapped_in_results_key(tmp):
    f = tmp / "mem0.json"
    f.write_text(json.dumps({"results": [{"memory": "Lives in Berlin since 2024"}]}), encoding="utf-8")
    res = parse_mem0(f, roles=set())
    assert res.n_parsed == 1


# ----------------------------------------------------------------------
# Markdown
# ----------------------------------------------------------------------

def test_markdown_single_file_splits_on_headers(tmp):
    f = tmp / "notes.md"
    f.write_text(
        "# Project\nThe main goal is shipping v1 by June.\n\n"
        "## Stack\nWe use Go for the backend and Vite for the frontend.\n\n"
        "## People\nAlice leads frontend, Bob leads infra.\n",
        encoding="utf-8",
    )
    res = parse_markdown(f, roles=set())
    assert res.n_parsed >= 3  # 3 header sections
    assert any("Go for the backend" in i["content"] for i in res.items)
    assert all(i["metadata"]["file"] == "notes.md" for i in res.items)


def test_markdown_directory_tree(tmp):
    (tmp / "a.md").write_text("My cat's name is Whiskers and she is twelve years old", encoding="utf-8")
    sub = tmp / "sub"
    sub.mkdir()
    (sub / "b.md").write_text("The deploy pipeline runs on GitHub Actions every push", encoding="utf-8")
    res = parse_markdown(tmp, roles=set())
    assert res.n_parsed == 2
    files = {i["metadata"]["file"] for i in res.items}
    assert files == {"a.md", "b.md"}


def test_markdown_empty_dir_notes(tmp):
    res = parse_markdown(tmp, roles=set())
    assert res.n_parsed == 0
    assert res.notes  # explains no .md files


def test_short_content_skipped(tmp):
    f = tmp / "conversations.json"
    f.write_text(json.dumps([{"title": "t", "mapping": {
        "n": {"message": {"author": {"role": "user"}, "content": {"parts": ["ok"]}}},
    }}]), encoding="utf-8")
    res = parse_chatgpt(f, roles={"user"})
    assert res.n_parsed == 0  # "ok" is below the min-length floor
    assert res.skipped == 1


def test_missing_file_is_graceful(tmp):
    res = parse_source("chatgpt", tmp / "nope.json")
    assert res.n_parsed == 0
    assert res.notes  # explains the read failure, no crash
