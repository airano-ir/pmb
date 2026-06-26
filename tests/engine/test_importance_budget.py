"""R13: per-day high-importance budget stops importance spam.

importance_factor = 0.5 + 0.5·importance, so a fact stamped 0.95 gets a ~2×
ranking edge over a 0.5 fact. Agents habitually stamp 0.95 on everything. The
budget caps how many >0.9 fact writes land per rolling day; past it, importance
is clamped to 0.7 with a breadcrumb. Keyed/lesson facts are exempt; pinned
facts are exempt because pin() resets importance to 1.0.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from pmb.core.engine import Engine


def _engine(ws, home, budget):
    return Engine(cwd=ws, pmb_home=home,
                  config_overrides={"recall.cache_size": 0,
                                    "write.high_importance_daily_budget": budget})


def _row(eng, ulid):
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        c.row_factory = sqlite3.Row
        r = c.execute("SELECT importance, metadata_json FROM events WHERE ulid=?",
                      (ulid,)).fetchone()
    return float(r["importance"]), json.loads(r["metadata_json"] or "{}")


# platform_sensitive: the high-importance budget count's visibility races under
# macOS-CI load; deterministic + green on the Linux reference runner.
@pytest.mark.platform_sensitive
def test_high_importance_facts_clamped_past_budget(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home, budget=2)
    u1 = eng.record_fact("important fact one", importance=0.95)
    u2 = eng.record_fact("important fact two", importance=0.95)
    u3 = eng.record_fact("important fact three", importance=0.95)
    assert _row(eng, u1)[0] == 0.95          # within budget
    assert _row(eng, u2)[0] == 0.95
    imp3, meta3 = _row(eng, u3)
    assert imp3 == 0.7                        # over budget → clamped
    assert meta3.get("importance_budget_clamped") == 0.95


def test_normal_importance_untouched(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home, budget=1)
    eng.record_fact("hi-imp", importance=0.95)
    u = eng.record_fact("a routine fact", importance=0.6)   # below 0.9 → never clamped
    assert _row(eng, u)[0] == 0.6


def test_lessons_and_keyed_exempt(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home, budget=1)
    eng.record_fact("burn the budget", importance=0.95)
    les = eng.record_fact("a lesson rule", importance=0.95,
                          metadata={"kind": "lesson", "source": "lesson"})
    assert _row(eng, les)[0] == 0.95   # lessons are deliberately important


def test_budget_zero_disables(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home, budget=0)
    for i in range(5):
        u = eng.record_fact(f"fact {i}", importance=0.95)
    assert _row(eng, u)[0] == 0.95     # disabled → never clamps
