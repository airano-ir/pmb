"""T5: negation / "unknown" facts must be retired once a positive keyed value
exists.

Real-corpus bug (workspace 0019ea88): the fact "As of June 8 2026, the user
does not currently live in Warsaw; current city is unknown." stayed active
forever, asserting ignorance about a city that was, minutes later, known to be
Tampa. v0.5.0 only stopped PROMOTING such text; nothing retired the fact.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmb.core.engine import Engine
from pmb.core.events import Event
from pmb.reasoning.attributes import detect_negated_state


# ── pure-function tests ─────────────────────────────────────────────────────

@pytest.mark.parametrize("text,attr", [
    ("As of June 8 2026, the user does not currently live in Warsaw; "
     "current city is unknown.", "city"),
    ("I no longer live in Kyiv.", "city"),
    ("The user's current city is unknown.", "city"),
    ("The user no longer works at Stripe.", "employer"),
    ("My current employer is unknown.", "employer"),
    ("The user no longer lives in Warsaw.", "city"),
])
def test_detect_negated_state_positive(text, attr):
    assert detect_negated_state(text) == attr


@pytest.mark.parametrize("text", [
    "I don't like Postgres.",                       # negation but no attribute
    "The user doesn't work on weekends.",           # 'work on', not 'work at/for'
    "He no longer lives in Paris.",                 # third party — no subject cue
    "The user currently lives in Tampa.",           # positive, not negated
    "Do not log secrets to the console.",           # unrelated instruction
])
def test_detect_negated_state_negative(text):
    assert detect_negated_state(text) is None


# ── engine fixtures ─────────────────────────────────────────────────────────

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


def _seed_fact(eng, content, ts, meta=None, importance=0.8):
    ev = Event(workspace_id=eng.workspace.id, event_type="fact",
               content=content, metadata=meta or {}, importance=importance,
               timestamp=ts)
    return eng.events.append(ev).ulid


def _is_active(eng, ulid):
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        row = c.execute("SELECT archived_at FROM events WHERE ulid=?",
                        (ulid,)).fetchone()
    return row is not None and row[0] is None


def _meta(eng, ulid):
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        row = c.execute("SELECT metadata_json FROM events WHERE ulid=?",
                        (ulid,)).fetchone()
    return json.loads(row[0] or "{}") if row else {}


# ── write-time hook ─────────────────────────────────────────────────────────

def test_warsaw_unknown_negation_archived_on_positive_value(
    tmp_pmb_home, tmp_workspace_dir
):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    now = time.time()
    eng.record_keyed_fact("user", "city", "Warsaw")
    neg = _seed_fact(
        eng,
        "As of June 8 2026, the user does not currently live in Warsaw; "
        "current city is unknown.",
        now - 50,
    )
    assert _is_active(eng, neg)

    # The live value arrives → the stale "unknown" fact is retired.
    eng.record_keyed_fact("user", "city", "Tampa")
    assert not _is_active(eng, neg)
    m = _meta(eng, neg)
    assert m.get("superseded_reason") == "negation_obsoleted_by_value"


def test_lesson_with_matching_phrasing_is_not_archived(
    tmp_pmb_home, tmp_workspace_dir
):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    now = time.time()
    les = _seed_fact(eng, "The user no longer lives in Warsaw.", now - 50,
                     meta={"kind": "lesson"})
    eng.record_keyed_fact("user", "city", "Tampa")
    assert _is_active(eng, les)  # lessons are instructions, never auto-archived


def test_pinned_negation_is_not_archived(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    now = time.time()
    pinned = _seed_fact(eng, "The user no longer lives in Warsaw.", now - 50,
                        importance=1.0)
    eng.record_keyed_fact("user", "city", "Tampa")
    assert _is_active(eng, pinned)


def test_config_off_keeps_negation(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home,
                  **{"keyed.archive_obsolete_negations": False})
    now = time.time()
    neg = _seed_fact(eng, "The user no longer lives in Warsaw.", now - 50)
    eng.record_keyed_fact("user", "city", "Tampa")
    assert _is_active(eng, neg)


def test_unrelated_attribute_negation_survives(tmp_pmb_home, tmp_workspace_dir):
    """Setting user::city must NOT archive a negation about a DIFFERENT
    attribute (employer)."""
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    now = time.time()
    emp_neg = _seed_fact(eng, "The user no longer works at Stripe.", now - 50)
    eng.record_keyed_fact("user", "city", "Tampa")
    assert _is_active(eng, emp_neg)  # different attribute → untouched


# ── repair pass ─────────────────────────────────────────────────────────────

def test_repair_pass_archives_preexisting_negation(tmp_pmb_home, tmp_workspace_dir):
    # Hook OFF while seeding so the negation survives into a pre-existing state.
    eng = _engine(tmp_workspace_dir, tmp_pmb_home,
                  **{"keyed.archive_obsolete_negations": False})
    now = time.time()
    neg = _seed_fact(eng, "The user no longer lives in Warsaw; "
                          "current city is unknown.", now - 100)
    eng.record_keyed_fact("user", "city", "Tampa")  # newer positive value
    assert _is_active(eng, neg)  # hook was off

    # dry-run reports but changes nothing
    dry = eng.archive_negations_for_current_keys(dry_run=True)
    assert any(p["attribute"] == "city" for p in dry["plan"])
    assert _is_active(eng, neg)

    # apply archives it
    res = eng.archive_negations_for_current_keys(dry_run=False)
    assert res["n"] >= 1
    assert not _is_active(eng, neg)
    assert _meta(eng, neg).get("superseded_reason") == "negation_obsoleted_by_value"
