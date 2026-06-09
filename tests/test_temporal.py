"""Tests for bi-temporal index (Improvement C)."""
from __future__ import annotations

import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmb.core.engine import Engine
from pmb.reasoning.temporal import (
    extract_event_time, is_temporal_query, temporal_proximity_boost,
)


@pytest.fixture
def tmp_pmb_home():
    import gc, shutil, time as _t
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


# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------

def test_extract_iso_date():
    t = extract_event_time("Migrated MySQL to Postgres on 2025-12-05")
    assert t is not None
    d = datetime.fromtimestamp(t, tz=timezone.utc)
    assert d.year == 2025 and d.month == 12 and d.day == 5


def test_extract_month_day():
    ref = datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp()
    t = extract_event_time("We met on December 5", reference_now=ref)
    assert t is not None
    d = datetime.fromtimestamp(t, tz=timezone.utc)
    # Dec 5 > Jun 1 + 6 months → infer year is last year (2024)
    assert d.month == 12 and d.day == 5
    assert d.year in (2024, 2025)  # tolerance


def test_extract_relative_yesterday():
    ref = datetime(2025, 6, 10, 12, 0, tzinfo=timezone.utc).timestamp()
    t = extract_event_time("Came home yesterday", reference_now=ref)
    assert t is not None
    d = datetime.fromtimestamp(t, tz=timezone.utc)
    assert d.year == 2025 and d.month == 6 and d.day == 9


def test_extract_n_days_ago():
    ref = datetime(2025, 6, 20, tzinfo=timezone.utc).timestamp()
    t = extract_event_time("Visited 5 days ago", reference_now=ref)
    assert t is not None
    d = datetime.fromtimestamp(t, tz=timezone.utc)
    assert d.day == 15  # 20 - 5


def test_extract_returns_none_for_no_date():
    assert extract_event_time("just a fact about Postgres") is None
    assert extract_event_time("") is None


# ----------------------------------------------------------------------
# Query classifier
# ----------------------------------------------------------------------

def test_is_temporal_query_positive():
    assert is_temporal_query("When did we migrate?")
    assert is_temporal_query("What happened in December")
    assert is_temporal_query("After the meeting last week")


def test_is_temporal_query_negative():
    assert not is_temporal_query("Postgres port")
    assert not is_temporal_query("What is the api endpoint")


# ----------------------------------------------------------------------
# Proximity
# ----------------------------------------------------------------------

def test_proximity_decays_exponentially():
    t1 = 1_700_000_000.0
    same_day = temporal_proximity_boost(t1, t1, half_life_days=14)
    a_week = temporal_proximity_boost(t1, t1 + 7 * 86400, half_life_days=14)
    a_month = temporal_proximity_boost(t1, t1 + 30 * 86400, half_life_days=14)
    assert same_day == pytest.approx(1.0, abs=1e-6)
    assert 0.6 < a_week < 0.8  # ~0.71
    assert a_month < 0.5


def test_proximity_zero_if_missing_time():
    assert temporal_proximity_boost(None, 1_700_000_000.0) == 0.0
    assert temporal_proximity_boost(1_700_000_000.0, None) == 0.0


# ----------------------------------------------------------------------
# Engine integration
# ----------------------------------------------------------------------

def test_event_time_stored_on_write(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    u = eng.record_fact("Trip to Paris on 2025-08-12 was great")
    ev = eng.events.get_by_ulid(u)
    assert ev.metadata.get("event_time") is not None
    parsed = datetime.fromtimestamp(ev.metadata["event_time"], tz=timezone.utc)
    assert parsed.year == 2025 and parsed.month == 8 and parsed.day == 12


def test_temporal_query_boosts_proximate_event(tmp_pmb_home, tmp_workspace_dir):
    """A temporal query ('when did the December meeting happen?') must rank the
    event whose time anchor is CLOSEST to the query's anchor highest."""
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={
            "recall.cache_size": 0,
            "recall.temporal_enabled": True,
            "recall.temporal_boost": 1.0,  # strong for test signal
            # The query parses to a December anchor and the events are months
            # away; with the default 14-day half-life the proximity boost is
            # ~0 for ALL of them, so ranking fell to vector luck (the flake).
            # A long half-life lets proximity actually differentiate near<far.
            "recall.temporal_half_life_days": 400.0,
            "recall.spreading_activation": False,
        },
    )
    # `near` and `far` are near-identical "Meeting with Alice …" facts — only
    # the DATE differs, so the temporal signal is isolated. `decoy` is undated.
    near = eng.record_fact("Meeting with Alice on 2025-12-05 about budgets")
    far = eng.record_fact("Meeting with Alice on 2024-01-15 about onboarding")
    decoy = eng.record_fact("General meeting topics that are not dated")

    # Drain the async embed queue so vectors are ready before we rank — else
    # recall races the queue (part of the pre-existing flake under load).
    eng.wait_for_embed_queue(timeout_seconds=60.0)

    pack = eng.recall("When did the December meeting happen?", top_k=3)
    ulids = [r.ulid for r in pack.results]
    assert near in ulids
    # Temporal proximity: the December-2025 meeting (closest to the query's
    # December anchor) must outrank the January-2024 one. They differ only by
    # date, so this isolates the temporal boost deterministically — unlike the
    # old near-vs-undated-decoy check, whose margin was pure vector luck.
    pos_near = ulids.index(near)
    pos_far = ulids.index(far) if far in ulids else 99
    assert pos_near < pos_far, (
        f"proximate (Dec 2025) should outrank distant (Jan 2024); got {ulids}"
    )


def test_temporal_disabled_means_no_event_time(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={"recall.temporal_enabled": False},
    )
    u = eng.record_fact("Trip on 2025-08-12")
    ev = eng.events.get_by_ulid(u)
    assert ev.metadata.get("event_time") is None
