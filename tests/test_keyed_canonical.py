"""Phase 1: keyed-fact attribute canonicalization, current-state promotion,
and the repair command.

The bug: synonymous attribute labels (city / current_city / current_city_2026
/ lives_in / город) created INDEPENDENT keys, so a stale `user::city = Warsaw`
out-ranked the live "lives in Tampa". These tests lock in:
  * one canonical key per attribute (general, not city-only);
  * a plain "I now live in X" fact upserts the keyed attribute;
  * `repair_keyed_facts` collapses pre-existing conflicts (archive-only).
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmb.core.engine import Engine
from pmb.core.events import Event
from pmb.reasoning.attributes import (
    canonicalize_attribute,
    detect_current_state,
    keyed_fact_key,
)


# ── pure-function tests (no engine, fast) ──────────────────────────────────

@pytest.mark.parametrize("label", [
    "city", "current_city", "current_city_2026", "lives_in", "current location",
    "residence", "город", "живёт",
])
def test_location_aliases_canonicalize_to_city(label):
    assert canonicalize_attribute(label) == "city"


def test_non_city_attributes_have_own_canon():
    assert canonicalize_attribute("works_at") == "employer"
    assert canonicalize_attribute("current company") == "employer"
    assert canonicalize_attribute("job title") == "job_title"
    assert canonicalize_attribute("phone number") == "phone"


def test_unknown_attribute_passes_through_normalized():
    assert canonicalize_attribute("Favourite Colour") == "favourite_colour"


def test_keyed_fact_key_canonicalizes_attribute_only():
    assert keyed_fact_key("user", "current_city_2026") == "user::city"
    assert keyed_fact_key("User", "city") == "user::city"


@pytest.mark.parametrize("text,attr,val", [
    ("I now live in Tampa", "city", "Tampa"),
    ("user now lives in Tampa, Florida", "city", "Tampa"),
    ("My current city is Tampa", "city", "Tampa"),
    ("I just moved to Tampa", "city", "Tampa"),
    ("I moved to Austin last week", "city", "Austin"),  # trailing time stripped
    ("I currently work at Anthropic", "employer", "Anthropic"),
    ("Сейчас живу в Тампе", "city", "Тампе"),
])
def test_detect_current_state_positive(text, attr, val):
    hit = detect_current_state(text)
    assert hit is not None, text
    assert hit[0] == attr
    assert hit[1] == val


@pytest.mark.parametrize("text", [
    "I live in Paris",                       # no present-state marker
    "The config file lives in src/config.py",  # not a person
    "We discussed where Alice lives",         # third party, no marker
    "Postgres listens on port 5433",
    # negation / meta-instruction — must NOT be read as a current state
    # (real-corpus false positive: would have re-promoted the stale value)
    "Do not state that the user currently lives in Warsaw.",
    "The user no longer lives in Warsaw.",
    "The user does not currently live in Warsaw.",
])
def test_detect_current_state_negative(text):
    assert detect_current_state(text) is None


# ── engine tests ───────────────────────────────────────────────────────────

@pytest.fixture
def tmp_pmb_home():
    import gc
    import shutil
    import time as _t
    tmp = tempfile.mkdtemp()
    home = Path(tmp) / "pmb_home"
    os.environ["PMB_HOME"] = str(home)
    try:
        yield home
    finally:
        os.environ.pop("PMB_HOME", None)
        gc.collect()
        for _ in range(3):
            try:
                shutil.rmtree(tmp, ignore_errors=False)
                break
            except (OSError, PermissionError):
                _t.sleep(0.2)
                gc.collect()
        else:
            shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def tmp_workspace_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def _engine(ws, home, **over):
    cfg = {"recall.cache_size": 0}
    cfg.update(over)
    return Engine(cwd=ws, pmb_home=home, config_overrides=cfg)


def _active_keyed(eng, key):
    hist = eng.get_keyed_fact_history(*key.split("::", 1))
    return [h for h in hist if h["is_current"]]


def test_synonym_attribute_supersedes_under_one_key(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.record_keyed_fact("user", "city", "Warsaw")
    # a DIFFERENT label for the same attribute must hit the SAME canonical key
    eng.record_keyed_fact("user", "current_city_2026", "Tampa")

    active = _active_keyed(eng, "user::city")
    assert len(active) == 1, [a["value"] for a in active]
    assert active[0]["value"] == "Tampa"
    # history still has Warsaw (archived, not deleted)
    hist = eng.get_keyed_fact_history("user", "city")
    assert any(h["value"] == "Warsaw" and not h["is_current"] for h in hist)


def test_record_fact_promotes_current_state(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.record_keyed_fact("user", "city", "Warsaw")
    # a plain user fact stating the CURRENT city should upsert the keyed value
    eng.record_fact("I now live in Tampa")

    active = _active_keyed(eng, "user::city")
    assert len(active) == 1
    assert active[0]["value"] == "Tampa"


def test_promotion_off_when_disabled(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home,
                  **{"keyed.auto_detect_current_state": False})
    eng.record_fact("I now live in Tampa")
    assert _active_keyed(eng, "user::city") == []


def test_internal_source_is_not_promoted(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    # a reflection / project-index style fact must never be auto-keyed
    eng.record_fact("I now live in Tampa", metadata={"source": "reflection"})
    assert _active_keyed(eng, "user::city") == []


def _seed_raw_keyed(eng, key, attr, value, ts):
    """Seed an ACTIVE keyed fact directly (bypassing canonicalization), to
    reproduce the pre-canonicalization conflicting state on disk."""
    ev = Event(
        workspace_id=eng.workspace.id, event_type="fact",
        content=f"user {attr}: {value}",
        metadata={
            "keyed_fact_key": key, "keyed_fact_subject": "user",
            "keyed_fact_attribute": attr, "keyed_fact_value": value,
        },
        importance=0.9, timestamp=ts,
    )
    return eng.events.append(ev).ulid


def test_repair_collapses_preexisting_conflicts(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    now = time.time()
    _seed_raw_keyed(eng, "user::city", "city", "Warsaw", now - 100)
    _seed_raw_keyed(eng, "user::current_city_2026", "current_city_2026",
                    "Tampa", now - 10)

    # dry-run: reports the plan, changes nothing
    dry = eng.repair_keyed_facts(dry_run=True)
    assert dry["dry_run"] is True
    assert dry["n_archived"] == 1
    assert len(_active_keyed(eng, "user::city")) >= 1  # untouched

    # apply: newest (Tampa) survives under canonical key, Warsaw archived
    res = eng.repair_keyed_facts(dry_run=False)
    assert res["n_archived"] == 1
    assert res["n_recanonicalized"] == 1
    active = _active_keyed(eng, "user::city")
    assert len(active) == 1
    assert active[0]["value"] == "Tampa"


def test_repair_noop_when_clean(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.record_keyed_fact("user", "city", "Tampa")
    res = eng.repair_keyed_facts(dry_run=False)
    assert res["groups"] == []
    assert res["n_archived"] == 0


def test_backfill_promotes_current_state_skipping_subfacts_and_lessons(
    tmp_pmb_home, tmp_workspace_dir
):
    """Regression for the real-corpus finding: a stale keyed value (Warsaw)
    must be superseded by the current-state PARENT fact (Tampa), while a
    later-timestamped subfact ("moved to the United States") and a negated
    lesson ("do not say Warsaw") are correctly ignored."""
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    now = time.time()
    eng.record_keyed_fact("user", "city", "Warsaw")  # stale keyed value

    def seed(content, ts, meta):
        ev = Event(workspace_id=eng.workspace.id, event_type="fact",
                   content=content, metadata=meta, importance=0.8, timestamp=ts)
        return eng.events.append(ev).ulid

    seed("As of today, the user currently lives in Tampa, Florida, USA.",
         now + 10, {"has_subfacts": True})
    seed("The user moved to the United States on an O-1 visa.",
         now + 11, {"is_subfact": True})              # newer, but a subfact
    seed("Do not state that the user currently lives in Warsaw; treat as stale.",
         now + 12, {"kind": "lesson"})                # newest, but a lesson

    bf = eng.backfill_keyed_from_facts(dry_run=False)
    assert ("city", "Tampa") in {(p["attribute"], p["new_value"]) for p in bf["promotions"]}
    active = _active_keyed(eng, "user::city")
    assert len(active) == 1
    assert active[0]["value"] == "Tampa"
