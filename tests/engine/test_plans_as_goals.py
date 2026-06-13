"""T9: future-intent ("plan") routing — plans become goals, not facts.

  * `looks_like_future_intent` distinguishes "next we'll do X" from settled
    facts / past decisions;
  * record_batch accepts {"type": "plan"} as a goal (kind=plan);
  * record_fact flags forward-looking facts with metadata.suggest_goal
    (a hint — never an auto-convert);
  * the MCP docstrings + connect template carry the routing rule (so it
    can't silently regress);
  * `pmb goals` lists and `pmb goals done` closes.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from pmb.core.engine import Engine
from pmb.reasoning.attributes import looks_like_future_intent

# ── pure-function ───────────────────────────────────────────────────────────

# G3: EN cold matrix; RU future-intent ("будем делать …") is the warm
# statement.future_intent anchor now (test_statement_anchors).
@pytest.mark.parametrize("text", [
    "next we'll wire the Groq backend",
    "next steps: migrate to Groq",
    "we will ship v1.0 next week",
    "plan: refactor auth, then the frontend",
    "let's add tests for the recall path",
])
def test_future_intent_positive(text):
    assert looks_like_future_intent(text) is True


@pytest.mark.parametrize("text", [
    "we decided to use Groq yesterday",
    "Project uses Postgres 17 on port 5432",
    "User prefers dark mode",
    "the migration happened last March",
    "Anna is my wife",
])
def test_future_intent_negative(text):
    assert looks_like_future_intent(text) is False


# ── engine fixtures ─────────────────────────────────────────────────────────





def _engine(ws, home, **over):
    cfg = {"recall.cache_size": 0}
    cfg.update(over)
    return Engine(cwd=ws, pmb_home=home, config_overrides=cfg)


def _row(eng, ulid):
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        c.row_factory = sqlite3.Row
        r = c.execute("SELECT event_type, metadata_json FROM events WHERE ulid=?",
                      (ulid,)).fetchone()
    meta = json.loads(r["metadata_json"] or "{}") if r else {}
    return (r["event_type"] if r else None), meta


# ── batch plan alias ────────────────────────────────────────────────────────

def test_batch_plan_alias_becomes_goal(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    res = eng.record_batch(items=[
        {"type": "plan", "title": "next we'll wire the Groq backend"},
    ])
    rec = res["results"][0] if "results" in res else res[0]
    assert rec["type"] == "plan"
    etype, meta = _row(eng, rec["ulid"])
    assert etype == "goal"
    assert meta.get("kind") == "plan"
    assert meta.get("goal_status") == "pending"


# ── record_fact suggest_goal flag ───────────────────────────────────────────

def test_record_fact_flags_future_intent(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    u = eng.record_fact("next we'll migrate the extractor to Groq")
    _etype, meta = _row(eng, u)
    assert meta.get("suggest_goal") is True


def test_record_fact_does_not_flag_settled_fact(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    u = eng.record_fact("We decided to use Groq for the extractor")
    _etype, meta = _row(eng, u)
    assert "suggest_goal" not in meta


# ── routing rule lives in the agent-facing text (regression guard) ──────────

def test_routing_rule_present_in_docstrings_and_template():
    src = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "src" / "pmb"
    tools = (src / "mcp" / "tools.py").read_text(encoding="utf-8")
    assert "FUTURE INTENT" in tools  # record_fact rule
    connect = (src / "cli" / "connect.py").read_text(encoding="utf-8")
    assert "FUTURE intent" in connect  # WRITE-triggers row


# ── CLI ─────────────────────────────────────────────────────────────────────

def test_cli_goals_list_and_done(tmp_pmb_home, monkeypatch):
    from typer.testing import CliRunner

    from pmb.cli.main import app
    # Pin the workspace via env (NOT chdir — chdir into a temp dir blocks its
    # cleanup on Windows) so the test's engine and the CLI share one workspace.
    monkeypatch.setenv("PMB_WORKSPACE", "goaltest")
    eng = Engine(pmb_home=tmp_pmb_home)
    ulid = eng.record_goal("Ship v0.6.0", status="in_progress")

    runner = CliRunner()
    r = runner.invoke(app, ["goals"])
    assert r.exit_code == 0, r.output
    assert "Ship v0.6.0" in r.output

    r2 = runner.invoke(app, ["goals", "done", ulid])
    assert r2.exit_code == 0, r2.output
    # now closed → absent from default (open-only) list
    r3 = runner.invoke(app, ["goals"])
    assert "Ship v0.6.0" not in r3.output
