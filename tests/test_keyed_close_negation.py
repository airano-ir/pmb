"""E1: a user negation CLOSES the matching keyed value.

Task-5 archived stale NEGATIONS when a positive value arrived. The reverse was
unhandled: with user::city = Tampa, "I no longer live in Tampa" left Tampa
asserted as current forever. Now the negation closes the keyed value (archive +
valid_to/closed_by), so recall stops asserting it while keyed_fact_as_of keeps
it as history.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmb.core.engine import Engine


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


def _current_city(eng):
    return [h for h in eng.get_keyed_fact_history("user", "city")
            if h["is_current"]]


def test_negation_closes_current_keyed_value(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.record_keyed_fact("user", "city", "Tampa")
    assert _current_city(eng) and _current_city(eng)[0]["value"] == "Tampa"

    # time passes, then the user negates it
    time.sleep(0.01)
    eng.record_fact("I no longer live in Tampa.")

    # no current city value is asserted anymore
    assert _current_city(eng) == []
    # the keyed fact was closed (archived) with a reason, not deleted
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        c.row_factory = sqlite3.Row
        row = c.execute(
            "SELECT archived_at, metadata_json FROM events WHERE event_type='fact' "
            "AND metadata_json LIKE '%\"keyed_fact_value\": \"Tampa\"%'").fetchone()
    assert row["archived_at"] is not None
    import json
    assert json.loads(row["metadata_json"]).get("closed_reason") == "negated_by_user"


def test_as_of_still_sees_closed_value_as_history(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.record_keyed_fact("user", "city", "Tampa")
    t_mid = time.time()
    time.sleep(0.01)
    eng.record_fact("I no longer live in Tampa.")
    # as-of BEFORE the negation still returns Tampa (history preserved)
    asof = eng.keyed_fact_as_of("user", "city", t_mid)
    assert asof and asof["value"] == "Tampa"


def test_config_off_keeps_value(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home,
                  **{"keyed.close_on_negation": False})
    eng.record_keyed_fact("user", "city", "Tampa")
    time.sleep(0.01)
    eng.record_fact("I no longer live in Tampa.")
    assert _current_city(eng) and _current_city(eng)[0]["value"] == "Tampa"


def test_third_party_negation_does_not_close(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.record_keyed_fact("user", "city", "Tampa")
    time.sleep(0.01)
    # about Alice, not the user → must NOT close the user's city (relies on A1)
    eng.record_fact("I heard Alice no longer lives in Tampa.")
    assert _current_city(eng) and _current_city(eng)[0]["value"] == "Tampa"


def test_does_not_close_a_newer_value(tmp_pmb_home, tmp_workspace_dir):
    """A negation must not close a keyed value recorded AFTER it (only older)."""
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.record_fact("I no longer live in Tampa.")  # negation first
    time.sleep(0.01)
    eng.record_keyed_fact("user", "city", "Tampa")  # then a fresh positive value
    assert _current_city(eng) and _current_city(eng)[0]["value"] == "Tampa"
