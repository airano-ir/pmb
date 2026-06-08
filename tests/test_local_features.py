"""Tests for the local-use feature commands (timeline / insights / digest /
export / forget-topic / ttl / prune-expired / tags / reminders / snapshot).

All of these are CLI + display + write-layer only - they never call
engine.recall(), so they cannot affect retrieval quality or latency. These
tests verify the wiring + the EventStore.set_metadata / list_all helpers they
rely on. No embedding model is loaded: events are seeded directly into the
store so the suite stays fast and deterministic.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmb.cli.main import app  # noqa: E402
from pmb.cli._common import _parse_duration  # noqa: E402  (moved here in the split)
from pmb.core.events import Event  # noqa: E402

runner = CliRunner()

DAY = 86400.0


@pytest.fixture
def ws(monkeypatch):
    """A workspace seeded with a handful of events of every relevant kind.

    Returns (engine, ulids) where ulids maps a short label -> event ulid.
    Events are appended directly (no dedup / embedding) for determinism.
    """
    tmp = tempfile.mkdtemp()
    monkeypatch.setenv("PMB_HOME", tmp)
    monkeypatch.setenv("PMB_WORKSPACE", "local_feat_test")
    from pmb.core.engine import Engine
    eng = Engine()
    wid = eng.workspace.id
    now = time.time()
    ulids: dict[str, str] = {}

    def seed(label, content, *, etype="fact", meta=None, ts=None, importance=0.6):
        ev = Event(
            workspace_id=wid, event_type=etype, content=content,
            metadata=meta or {}, importance=importance,
            timestamp=ts if ts is not None else now,
        )
        ev = eng.events.append(ev)
        ulids[label] = ev.ulid
        return ev.ulid

    seed("pg", "decided to use Postgres for JSONB on project-x",
         meta={"source": "cli-note"})
    seed("pnpm", "this repo uses pnpm, never npm",
         etype="fact", meta={"source": "lesson", "kind": "lesson"}, importance=0.85)
    seed("old", "early architecture sketch", ts=now - 40 * DAY)
    seed("acme", "call with ACME about the integration",
         meta={"source": "note", "project": "acme"})
    # goals for reminders
    seed("overdue", "ship v1.0 release", etype="goal", importance=0.7,
         meta={"goal_status": "in_progress", "due_at": now - 5 * DAY})
    seed("soon", "write launch blog post", etype="goal", importance=0.7,
         meta={"goal_status": "pending", "due_at": now + 2 * DAY})
    seed("donegoal", "finished onboarding flow", etype="goal", importance=0.7,
         meta={"goal_status": "done", "due_at": now - 3 * DAY})
    return eng, ulids


# ----------------------------------------------------------------------
# _parse_duration - unit
# ----------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("30d", 30 * 86400),
    ("12h", 12 * 3600),
    ("2w", 2 * 604800),
    ("3mo", 3 * 2592000),
    ("1y", 31536000),
    ("45", 45 * 86400),   # bare integer = days
    ("10 d", 10 * 86400),  # whitespace tolerated
])
def test_parse_duration_ok(text, expected):
    assert _parse_duration(text) == float(expected)


@pytest.mark.parametrize("text", ["", "abc", "5x", "d10", "-3d", None])
def test_parse_duration_bad(text):
    assert _parse_duration(text) is None


# ----------------------------------------------------------------------
# EventStore.set_metadata / list_all - store level
# ----------------------------------------------------------------------

def test_set_metadata_roundtrip(ws):
    eng, ulids = ws
    eng.events.set_metadata(ulids["pg"], {"tags": ["work"], "source": "cli-note"})
    ev = eng.events.get_by_ulid(ulids["pg"])
    assert ev.metadata.get("tags") == ["work"]


def test_list_all_includes_archived(ws):
    eng, ulids = ws
    active_before = len(eng.events.list_active(eng.workspace.id, limit=10_000))
    eng.forget(ulids["old"])  # archive one
    active_after = len(eng.events.list_active(eng.workspace.id, limit=10_000))
    assert active_after == active_before - 1
    # list_all with include_archived sees it again
    all_n = len(eng.events.list_all(eng.workspace.id, include_archived=True))
    active_n = len(eng.events.list_all(eng.workspace.id, include_archived=False))
    assert all_n == active_after + 1
    assert active_n == active_after


# ----------------------------------------------------------------------
# Read-only commands
# ----------------------------------------------------------------------

def test_timeline_runs(ws):
    r = runner.invoke(app, ["timeline"])
    assert r.exit_code == 0, r.output
    assert "PMB timeline" in r.output


def test_timeline_days_filter(ws):
    # the 'old' event is 40d back; --days 7 should exclude it
    r = runner.invoke(app, ["timeline", "--days", "7"])
    assert r.exit_code == 0, r.output
    assert "early architecture sketch" not in r.output


def test_insights_runs(ws):
    r = runner.invoke(app, ["insights"])
    assert r.exit_code == 0, r.output
    assert "PMB insights" in r.output
    assert "Highlights" in r.output


def test_digest_today(ws):
    r = runner.invoke(app, ["digest", "today"])
    assert r.exit_code == 0, r.output
    # the 40d-old event must not show in 'today'
    assert "early architecture sketch" not in r.output


def test_digest_month_includes_recent(ws):
    r = runner.invoke(app, ["digest", "month"])
    assert r.exit_code == 0, r.output
    assert "PMB digest" in r.output


def test_digest_bad_period(ws):
    r = runner.invoke(app, ["digest", "decade"])
    assert r.exit_code == 2


# ----------------------------------------------------------------------
# export
# ----------------------------------------------------------------------

def _parse_json_stdout(text: str) -> dict:
    start, end = text.index("{"), text.rindex("}")
    return json.loads(text[start:end + 1])


def test_export_json(ws):
    r = runner.invoke(app, ["export", "--format", "json"])
    assert r.exit_code == 0, r.output
    payload = _parse_json_stdout(r.output)
    assert payload["n_events"] >= 7
    contents = [e["content"] for e in payload["events"]]
    assert any("Postgres" in c for c in contents)


def test_export_markdown_to_file(ws, tmp_path):
    out = tmp_path / "memory.md"
    r = runner.invoke(app, ["export", "--out", str(out)])
    assert r.exit_code == 0, r.output
    body = out.read_text(encoding="utf-8")
    assert "# PMB memory export" in body
    assert "pnpm" in body


# ----------------------------------------------------------------------
# tags / collections
# ----------------------------------------------------------------------

def test_tag_then_tagged_and_tags(ws):
    eng, ulids = ws
    r = runner.invoke(app, ["tag", ulids["pg"], "work", "db"])
    assert r.exit_code == 0, r.output

    r = runner.invoke(app, ["tagged", "work"])
    assert r.exit_code == 0, r.output
    assert "Postgres" in r.output

    r = runner.invoke(app, ["tags"])
    assert r.exit_code == 0, r.output
    assert "work" in r.output and "db" in r.output


def test_untag(ws):
    eng, ulids = ws
    runner.invoke(app, ["tag", ulids["pg"], "work", "temp"])
    r = runner.invoke(app, ["untag", ulids["pg"], "temp"])
    assert r.exit_code == 0, r.output
    ev = eng.events.get_by_ulid(ulids["pg"])
    assert ev.metadata.get("tags") == ["work"]


# ----------------------------------------------------------------------
# TTL / expiry
# ----------------------------------------------------------------------

def test_ttl_set_and_clear(ws):
    eng, ulids = ws
    r = runner.invoke(app, ["ttl", ulids["acme"], "30d"])
    assert r.exit_code == 0, r.output
    ev = eng.events.get_by_ulid(ulids["acme"])
    assert isinstance(ev.metadata.get("expires_at"), (int, float))

    r = runner.invoke(app, ["ttl", ulids["acme"], "clear"])
    assert r.exit_code == 0, r.output
    ev = eng.events.get_by_ulid(ulids["acme"])
    assert "expires_at" not in ev.metadata


def test_prune_expired_archives_past_ttl(ws):
    eng, ulids = ws
    # backdate an expiry into the past, directly
    meta = dict(eng.events.get_by_ulid(ulids["acme"]).metadata or {})
    meta["expires_at"] = time.time() - 10
    eng.events.set_metadata(ulids["acme"], meta)

    r = runner.invoke(app, ["prune-expired"])
    assert r.exit_code == 0, r.output
    assert "Archived" in r.output
    # acme should no longer be active
    active = {e.ulid for e in eng.events.list_active(eng.workspace.id, limit=10_000)}
    assert ulids["acme"] not in active


def test_prune_expired_dry_run_changes_nothing(ws):
    eng, ulids = ws
    meta = dict(eng.events.get_by_ulid(ulids["acme"]).metadata or {})
    meta["expires_at"] = time.time() - 10
    eng.events.set_metadata(ulids["acme"], meta)
    r = runner.invoke(app, ["prune-expired", "--dry-run"])
    assert r.exit_code == 0, r.output
    active = {e.ulid for e in eng.events.list_active(eng.workspace.id, limit=10_000)}
    assert ulids["acme"] in active  # still there


# ----------------------------------------------------------------------
# forget-topic
# ----------------------------------------------------------------------

def test_forget_topic_archives_matches(ws):
    eng, ulids = ws
    r = runner.invoke(app, ["forget-topic", "project-x", "--yes"])
    assert r.exit_code == 0, r.output
    assert "Archived" in r.output
    active = {e.ulid for e in eng.events.list_active(eng.workspace.id, limit=10_000)}
    assert ulids["pg"] not in active  # the project-x note is gone


def test_forget_topic_dry_run(ws):
    eng, ulids = ws
    r = runner.invoke(app, ["forget-topic", "project-x", "--dry-run"])
    assert r.exit_code == 0, r.output
    active = {e.ulid for e in eng.events.list_active(eng.workspace.id, limit=10_000)}
    assert ulids["pg"] in active  # untouched


def test_forget_topic_no_match(ws):
    r = runner.invoke(app, ["forget-topic", "nonexistent-zzz", "--yes"])
    assert r.exit_code == 0, r.output
    assert "No active memories match" in r.output


# ----------------------------------------------------------------------
# reminders
# ----------------------------------------------------------------------

def test_reminders_overdue_and_soon(ws):
    r = runner.invoke(app, ["reminders"])
    assert r.exit_code == 0, r.output
    assert "Overdue" in r.output
    assert "Due soon" in r.output
    # the DONE goal must not appear
    assert "onboarding" not in r.output


def test_reminders_done_goal_excluded(ws):
    # within=0 -> nothing "soon", but overdue still shows; done excluded
    r = runner.invoke(app, ["reminders", "--within", "0"])
    assert r.exit_code == 0, r.output
    assert "onboarding" not in r.output


# ----------------------------------------------------------------------
# snapshots
# ----------------------------------------------------------------------

def test_snapshot_create_and_list(ws):
    r = runner.invoke(app, ["snapshot", "create", "--note", "before-refactor"])
    assert r.exit_code == 0, r.output
    assert "Snapshot" in r.output

    r = runner.invoke(app, ["snapshot", "list"])
    assert r.exit_code == 0, r.output
    assert "before-refactor" in r.output
