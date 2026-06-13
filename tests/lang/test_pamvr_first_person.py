"""B3 — first-person flag precomputed at write time (metadata.fp), read by the
per-candidate PAMVR loop instead of re-derived. Hermetic (lexical, no model)."""
from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

from pmb.core.engine import Engine
from pmb.reasoning.pamvr import _first_person_flag


def _eng(ws, home):
    return Engine(cwd=ws, pmb_home=home, config_overrides={"recall.cache_size": 0})


def _meta(eng, ulid):
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        row = c.execute("SELECT metadata_json FROM events WHERE ulid = ?", (ulid,)).fetchone()
    return json.loads(row[0]) if row and row[0] else {}


def test_first_person_fact_gets_fp_flag(tmp_pmb_home, tmp_workspace_dir):
    eng = _eng(tmp_workspace_dir, tmp_pmb_home)
    u = eng.record_fact("I live in Tampa")
    assert _meta(eng, u).get("fp") == 1


def test_non_first_person_fact_has_no_fp(tmp_pmb_home, tmp_workspace_dir):
    eng = _eng(tmp_workspace_dir, tmp_pmb_home)
    u = eng.record_fact("the build passed on CI")
    assert "fp" not in _meta(eng, u)


def test_flag_reads_precomputed_metadata_over_content():
    # fp=1 in metadata wins even if the content text has no first-person marker.
    ev = SimpleNamespace(metadata={"fp": 1})
    assert _first_person_flag(ev, "no person marker here") is True
    ev0 = SimpleNamespace(metadata={"fp": 0})
    assert _first_person_flag(ev0, "I live here and I work there") is False


def test_flag_falls_back_to_lexical_without_metadata():
    ev = SimpleNamespace(metadata={})
    assert _first_person_flag(ev, "I live in Berlin") is True
    assert _first_person_flag(ev, "the cat is asleep") is False
