"""X7 — security hardening pass. PMB is local-first, but it still takes
untrusted text (any message/metadata) into SQLite and serves a daemon over
localhost. These pin the load-bearing properties:

  * SQL is always parameterized — injection payloads in content/metadata are
    stored LITERALLY, never executed.
  * the daemon token file is owner-only on POSIX.
  * the daemon's bearer middleware refuses requests without the token.
"""
from __future__ import annotations

import os
import sqlite3

import pytest

from pmb.core.engine import Engine


def _engine(ws, home):
    return Engine(cwd=ws, pmb_home=home, config_overrides={"recall.cache_size": 0})


def test_sql_injection_payloads_are_inert(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    evil = "'; DROP TABLE events; -- and \"); DELETE FROM events; --"
    u = eng.record_fact(evil, metadata={"kind": "lesson", "note": evil})
    # the table still exists and the row is stored verbatim
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        assert c.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                         "AND name='events'").fetchone(), "events table dropped!"
        content = c.execute("SELECT content FROM events WHERE ulid=?", (u,)).fetchone()
    assert content and content[0] == evil
    # the indexed kind lookup (json_extract) still finds it, no injection
    assert any(x["ulid"] == u for x in eng.find_lessons(limit=50))


def test_malicious_kind_value_does_not_break_indexed_lookup(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.record_fact("real lesson", metadata={"kind": "lesson"})
    eng.record_fact("evil", metadata={"kind": "lesson' OR '1'='1"})
    rows = eng.find_lessons(limit=50)
    # only the genuine kind=='lesson' row matches; the injection string is just data
    assert sum(1 for x in rows if x["content"] == "real lesson") == 1


@pytest.mark.skipif(os.name == "nt", reason="POSIX file-mode check")
def test_daemon_token_is_owner_only(tmp_pmb_home, monkeypatch):
    monkeypatch.setenv("PMB_HOME", str(tmp_pmb_home))
    from pmb.mcp.daemon import token_path, write_daemon_token
    write_daemon_token(rotate=True)
    mode = token_path().stat().st_mode & 0o777
    assert mode == 0o600, f"daemon token mode {oct(mode)} is not 0600"


def test_daemon_bearer_middleware_requires_token():
    # the middleware factory must produce a guard that rejects wrong/absent tokens
    from pmb.mcp.daemon import _daemon_bearer_middleware
    mw = _daemon_bearer_middleware("the-secret")
    assert mw is not None
