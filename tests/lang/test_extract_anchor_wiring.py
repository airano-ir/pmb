"""C2 wiring + F1 — warm anchor keyed-extraction on the write path, gated off by
default, recording extraction-confidence metadata. Loads the model once.
"""
from __future__ import annotations

import json
import sqlite3

from pmb.core.engine import Engine


def _warm(ws, home, **over):
    cfg = {"recall.cache_size": 0}
    cfg.update(over)
    eng = Engine(cwd=ws, pmb_home=home, config_overrides=cfg)
    eng._is_warm = True
    return eng


def _current_city(eng):
    hist = eng.get_keyed_fact_history("user", "city")
    cur = [h for h in hist if h.get("is_current")]
    return cur[0] if cur else None


def _meta_of(eng, ulid):
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        row = c.execute("SELECT metadata_json FROM events WHERE ulid = ?", (ulid,)).fetchone()
    return json.loads(row[0]) if row and row[0] else {}


def test_anchor_keyed_off_by_default(tmp_pmb_home, tmp_workspace_dir):
    eng = _warm(tmp_workspace_dir, tmp_pmb_home)
    # German self-statement the en/ru/uk regex tier does NOT cover.
    eng._anchor_promote_keyed("Ich wohne in Berlin", {"keyed_fact_subject": "user"}, 0.7)
    assert _current_city(eng) is None, "must not extract when extract.anchor_keyed is off"


def test_anchor_keyed_creates_fact_with_confidence(tmp_pmb_home, tmp_workspace_dir):
    eng = _warm(tmp_workspace_dir, tmp_pmb_home, **{"extract.anchor_keyed": True})
    eng._anchor_promote_keyed("Ich wohne in Berlin", {"keyed_fact_subject": "user"}, 0.7)
    cur = _current_city(eng)
    assert cur is not None and cur["value"] == "Berlin"
    meta = _meta_of(eng, cur["ulid"])
    assert meta.get("source") == "anchor_extract"
    ex = meta.get("extract") or {}
    assert ex.get("tier") == "anchor"
    assert isinstance(ex.get("margin"), (int, float)) and ex["margin"] > 0
    assert ex.get("model")


def test_anchor_keyed_cold_engine_noop(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
                 config_overrides={"recall.cache_size": 0,
                                   "extract.anchor_keyed": True})   # warm flag NOT set
    eng._anchor_promote_keyed("Ich wohne in Berlin", {"keyed_fact_subject": "user"}, 0.7)
    assert _current_city(eng) is None, "cold engine must never extract (no model load)"


# ── F2: write-time contradiction via the anti-hypothesis ──────────────────────


def test_anchor_negation_closes_current_value(tmp_pmb_home, tmp_workspace_dir):
    import time
    eng = _warm(tmp_workspace_dir, tmp_pmb_home, **{"extract.anchor_keyed": True})
    eng.record_keyed_fact("user", "city", "Tampa")
    assert _current_city(eng)["value"] == "Tampa"
    # The anti-hypothesis of the CURRENT value wins on this negation → close it.
    eng._anchor_close_on_negation("I no longer live in Tampa", "neg-ulid",
                                  time.time() + 5)
    assert _current_city(eng) is None, "F2 must close the negated keyed value"


def test_anchor_negation_keeps_unrelated_value(tmp_pmb_home, tmp_workspace_dir):
    import time
    eng = _warm(tmp_workspace_dir, tmp_pmb_home, **{"extract.anchor_keyed": True})
    eng.record_keyed_fact("user", "city", "Tampa")
    # An unrelated self-statement must NOT close the city.
    eng._anchor_close_on_negation("I really enjoy hiking on weekends", "u",
                                  time.time() + 5)
    cur = _current_city(eng)
    assert cur is not None and cur["value"] == "Tampa"


# ── B3: non-English first-person flag via the about_self anchor ───────────────


def test_b3_anchor_first_person_non_english(tmp_pmb_home, tmp_workspace_dir):
    eng = _warm(tmp_workspace_dir, tmp_pmb_home, **{"extract.anchor_keyed": True})
    # German self-statement: no English/RU lexical first-person marker, so the
    # flag must come from the warm statement.about_self anchor (B3).
    u = eng.record_fact("Ich wohne in München")
    assert _meta_of(eng, u).get("fp") == 1


# ── C2: cosine-merge of inflected keyed values ────────────────────────────────


def test_c2_cosine_merge_keeps_shorter_form(tmp_pmb_home, tmp_workspace_dir):
    from pmb.reasoning.extract_anchor import values_are_alias
    eng = _warm(tmp_workspace_dir, tmp_pmb_home, **{"extract.anchor_keyed": True})
    # sanity: the embedder treats the inflection as an alias
    assert values_are_alias(eng, "Киеве", "Киев")
    assert not values_are_alias(eng, "Киев", "Берлин")
    # nominative first, then an inflected restatement — must NOT churn into the
    # inflected form; the merge keeps the shorter/nominative value.
    eng.record_keyed_fact("user", "city", "Киев")
    eng.record_keyed_fact("user", "city", "Киеве")
    cur = _current_city(eng)
    assert cur is not None and cur["value"] == "Киев"
