"""End-to-end MCP tests via the fastmcp in-memory client.

Exercises the REAL MCP tool routing, schema validation and serialization
(no subprocess/stdio fragility), focused on the long-chat scenario the new
features target:

  - session_brief on a long session (decisions / done / lessons digest)
  - answer quality: recall surfaces the right memory to answer a question
  - lessons: a recorded lesson surfaces for a related task

The recall-based tests load the embedding model once (poll while the async
embed queue drains), so they're the slow part of this file.
"""
from __future__ import annotations

import asyncio

import pytest


@pytest.fixture
def mcp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PMB_HOME", str(tmp_path / "h"))
    monkeypatch.setenv("PMB_WORKSPACE", "mcpe2e")
    monkeypatch.setenv("PMB_TOOL_PROFILE", "default")


def _client():
    from fastmcp import Client

    from pmb.mcp.server import build_server
    return Client(build_server(cwd=None, workspace_id="mcpe2e"))


async def _call(client, name, args=None):
    res = await client.call_tool(name, args or {})
    return res.data


async def _poll(coro_fn, pred, tries=60, delay=0.3):
    """Await coro_fn() until pred(result) is true (handles the async write +
    embed queue). Returns the last result either way."""
    last = None
    for _ in range(tries):
        last = await coro_fn()
        if pred(last):
            return last
        await asyncio.sleep(delay)
    return last


# ----------------------------------------------------------------------
# Tool exposure (the profile-gating fix)
# ----------------------------------------------------------------------

async def test_mcp_exposes_new_tools(mcp_env):
    async with _client() as c:
        names = {t.name for t in await c.list_tools()}
    # the new tools must be reachable by the agent under the default profile
    assert "session_brief" in names
    assert "overview" in names
    assert "recall" in names
    assert "record_batch" in names


# ----------------------------------------------------------------------
# Long chat -> session_brief (model-free: reads events)
# ----------------------------------------------------------------------

async def test_mcp_long_chat_session_brief(mcp_env):
    # one batch with a session's worth of work (atomic async write - avoids a
    # race between several concurrent fire-and-forget batches)
    items = [
        {"type": "activity", "kind": "decision",
         "content": "Chose Postgres 17 over Mongo for JSONB"},
        {"type": "activity", "kind": "completed",
         "content": "Refactored auth.py and fixed the JWT 24h-expiry bug"},
        {"type": "lesson", "content": "this repo uses pnpm, never npm"},
        {"type": "activity", "kind": "decision",
         "content": "Set the Postgres connection pool to 20"},
        {"type": "goal", "title": "Ship v1 by June", "status": "in_progress"},
    ]
    async with _client() as c:
        await _call(c, "record_batch", {"items": items})
        # record_batch is fire-and-forget; poll until the writes land
        b = await _poll(
            lambda: _call(c, "session_brief", {}),
            lambda b: b and b.get("n_events", 0) >= 4,
            tries=80, delay=0.25,
        )

    assert b["n_events"] >= 4, f"only {b['n_events']} landed"
    assert any("Postgres" in d["content"] for d in b["decisions"])
    assert any("JWT" in d["content"] for d in b["done"])
    assert any("pnpm" in l["content"] for l in b["lessons"])


# ----------------------------------------------------------------------
# Answer quality + lessons (model: recall must surface the right memory)
# ----------------------------------------------------------------------

async def test_mcp_recall_answer_quality_and_lessons(mcp_env):
    async with _client() as c:
        await _call(c, "record_batch", {"items": [
            {"type": "fact", "content": "The API runs Postgres 17 on port 5432"},
            {"type": "fact", "content": "We use Redis for rate limiting"},
            {"type": "lesson", "content": "this repo uses pnpm, never npm"},
        ]})

        # ANSWER QUALITY: asking for the port must surface the 5432 fact
        rc = await _poll(
            lambda: _call(c, "recall", {"query": "what port does the database use", "top_k": 5}),
            lambda r: r and any("5432" in x["content"] for x in r.get("results", [])),
            tries=80, delay=0.3,
        )
        assert any("5432" in x["content"] for x in rc["results"]), \
            f"port fact not surfaced: {[x['content'][:40] for x in rc.get('results', [])]}"

        # LESSONS: a related coding task surfaces the lesson in the dedicated
        # top-level MCP field. The general `results` list may remain empty
        # while the semantic index warms; `lessons` is the stable contract.
        lr = await _call(c, "recall",
                         {"query": "package manager npm or pnpm install", "top_k": 5})
        assert any("pnpm" in x["content"] for x in lr.get("lessons", [])), \
            f"lesson not surfaced: {lr}"

        # OVERVIEW groups what we know about the topic
        ov = await _call(c, "overview", {"topic": "postgres", "max_events": 20})
        assert ov["n_memories"] >= 1
        assert not ov.get("empty")
