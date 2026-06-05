"""Tests for the three answer-quality (J-score) levers:

  Lever 1 — resolved human date in recall output (RecallResult.resolved_date)
  Lever 2 — write-time atomic extraction patterns (relationship status, origin)
  Lever 3 — validity windows + as-of query (record_keyed_fact / keyed_fact_as_of)

Levers 1 & 2 are pure units (no engine/model). Lever 3 uses a real engine but
only the write path (no model load), so the suite stays fast and RAM-light.
None of these touch the recall ranking hot path.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmb.core.engine import RecallResult  # noqa: E402
from pmb.reasoning.fact_extract import extract_atomic_facts  # noqa: E402

# 1683417600 == 2023-05-07T00:00:00Z
MAY7_2023 = 1683417600.0


def _mk_result(metadata: dict, timestamp: float = MAY7_2023) -> RecallResult:
    return RecallResult(
        ulid="01ABC", event_type="fact", content="…", metadata=metadata,
        timestamp=timestamp, score=0.5, bm25_score=0.1, vec_score=0.2,
        importance=0.5, recency_score=0.3,
    )


# ----------------------------------------------------------------------
# Lever 1 — resolved date in recall output
# ----------------------------------------------------------------------

def test_resolved_date_prefers_event_time():
    r = _mk_result({"event_time": MAY7_2023}, timestamp=time.time())
    assert r.resolved_date == "2023-05-07"


def test_resolved_date_uses_session_dt_string():
    r = _mk_result({"session_dt": "1:56 pm on 7 May, 2023"})
    assert r.resolved_date == "1:56 pm on 7 May, 2023"


def test_resolved_date_falls_back_to_timestamp():
    r = _mk_result({}, timestamp=MAY7_2023)
    assert r.resolved_date == "2023-05-07"


def test_to_dict_includes_date_field():
    r = _mk_result({"event_time": MAY7_2023})
    d = r.to_dict()
    assert d["date"] == "2023-05-07"
    # ranking signals still present (output is additive, not a rewrite)
    assert "signals" in d and "score" in d


def test_resolved_date_event_time_beats_timestamp():
    # event_time (2023) must win over a 'now' creation timestamp (2025+)
    r = _mk_result({"event_time": MAY7_2023}, timestamp=1_750_000_000.0)
    assert r.resolved_date == "2023-05-07"


# ----------------------------------------------------------------------
# Lever 2 — atomic extraction patterns (relationship status, origin)
# ----------------------------------------------------------------------

def test_atomic_relationship_status_and_origin():
    text = ("I caught up with Caroline today at the cafe. "
            "Caroline is single and she moved from Sweden.")
    facts = extract_atomic_facts(text)
    kinds = {f.kind for f in facts}
    contents = " | ".join(f.content for f in facts)
    assert "relationship_status" in kinds, contents
    assert "origin" in kinds, contents
    assert any("single" in f.content.lower() for f in facts)
    assert any("Sweden" in f.content for f in facts)


def test_atomic_no_false_positive_on_lowercase_subject():
    # "the file is single-threaded" must NOT trigger relationship_status
    text = ("We refactored the worker pool this week. "
            "the config is single threaded by default now.")
    facts = extract_atomic_facts(text)
    assert "relationship_status" not in {f.kind for f in facts}


# ----------------------------------------------------------------------
# Lever 3 — validity windows + as-of query
# ----------------------------------------------------------------------

@pytest.fixture
def engine(monkeypatch):
    tmp = tempfile.mkdtemp()
    monkeypatch.setenv("PMB_HOME", tmp)
    monkeypatch.setenv("PMB_WORKSPACE", "asof_test")
    from pmb.core.engine import Engine
    return Engine()


def test_keyed_fact_stamps_validity_windows(engine):
    r1 = engine.record_keyed_fact("user", "lives in", "Kyiv")
    r2 = engine.record_keyed_fact("user", "lives in", "Warsaw")
    assert r2["superseded_ulids"] == [r1["new_ulid"]]

    old = engine.events.get_by_ulid(r1["new_ulid"])
    new = engine.events.get_by_ulid(r2["new_ulid"])
    # old value's window is closed; new value's window is open
    assert isinstance(old.metadata.get("valid_to"), (int, float))
    assert old.metadata.get("superseded_by") == r2["new_ulid"]
    assert isinstance(new.metadata.get("valid_from"), (int, float))
    assert new.metadata.get("valid_to") is None
    # window is non-empty (Warsaw written after Kyiv)
    assert old.metadata["valid_to"] > old.metadata["valid_from"]


def test_keyed_fact_as_of_returns_historical_value(engine):
    r1 = engine.record_keyed_fact("user", "lives in", "Kyiv")
    engine.record_keyed_fact("user", "lives in", "Warsaw")
    old = engine.events.get_by_ulid(r1["new_ulid"])
    vf, vt = old.metadata["valid_from"], old.metadata["valid_to"]

    # mid-window → the historical value (Kyiv)
    mid = (vf + vt) / 2.0
    assert engine.keyed_fact_as_of("user", "lives in", mid)["value"] == "Kyiv"
    # after supersession → current value (Warsaw)
    assert engine.keyed_fact_as_of("user", "lives in", vt + 100)["value"] == "Warsaw"
    # before anything was known → None
    assert engine.keyed_fact_as_of("user", "lives in", vf - 100) is None


def test_keyed_fact_as_of_unknown_key(engine):
    assert engine.keyed_fact_as_of("user", "favorite color", time.time()) is None
