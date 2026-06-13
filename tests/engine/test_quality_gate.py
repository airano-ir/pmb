"""T8: write-time quality gate (config write.quality_gate, default ON in 0.6.0+).

OFF → record every input unweighted. ON → suspected junk is FLAGGED (never
rejected): importance capped at 0.2, metadata.quality_flag=suspect_junk, and
excluded from keyed-fact promotion. Good content is untouched. (The "off" tests
set the flag explicitly now that ON is the default.)
"""
from __future__ import annotations

import json
import sqlite3

from pmb.core.engine import Engine


def _engine(ws, home, **over):
    cfg = {"recall.cache_size": 0}
    cfg.update(over)
    return Engine(cwd=ws, pmb_home=home, config_overrides=cfg)


def _row(eng, ulid):
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        c.row_factory = sqlite3.Row
        r = c.execute("SELECT importance, metadata_json FROM events WHERE ulid=?",
                      (ulid,)).fetchone()
    return r["importance"], json.loads(r["metadata_json"] or "{}")


def _active_keyed(eng, subj, attr):
    return [h for h in eng.get_keyed_fact_history(subj, attr) if h["is_current"]]


# ── default OFF ─────────────────────────────────────────────────────────────

def test_gate_off_does_not_flag(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home,
                  **{"write.quality_gate": False})  # explicit off
    u = eng.record_fact("ok", importance=0.7)
    imp, meta = _row(eng, u)
    assert "quality_flag" not in meta
    assert abs(imp - 0.7) < 1e-6


# ── ON ──────────────────────────────────────────────────────────────────────

def test_gate_on_flags_and_caps_junk(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home,
                  **{"write.quality_gate": True})
    u = eng.record_fact("ok", importance=0.7)
    imp, meta = _row(eng, u)
    assert meta.get("quality_flag") == "suspect_junk"
    assert imp <= 0.2


def test_gate_on_leaves_good_content_untouched(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home,
                  **{"write.quality_gate": True})
    u = eng.record_fact("Postgres runs on port 5432 in production", importance=0.7)
    imp, meta = _row(eng, u)
    assert "quality_flag" not in meta
    assert abs(imp - 0.7) < 1e-6


def test_gate_on_excludes_flagged_from_keyed_promotion(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home,
                  **{"write.quality_gate": True})
    # current-state SHAPE but junk VALUE (matches the test-pattern heuristic)
    eng.record_fact("I now live in final_value")
    assert _active_keyed(eng, "user", "city") == []  # gate blocked promotion


def test_gate_off_would_promote_same_content(tmp_pmb_home, tmp_workspace_dir):
    """Contrast: with the gate OFF the junkish current-state IS promoted —
    showing the gate is what prevents it."""
    eng = _engine(tmp_workspace_dir, tmp_pmb_home,
                  **{"write.quality_gate": False})  # explicit off
    eng.record_fact("I now live in final_value")
    active = _active_keyed(eng, "user", "city")
    assert len(active) == 1 and active[0]["value"] == "final_value"
