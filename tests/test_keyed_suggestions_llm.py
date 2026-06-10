"""T11: offline LLM tier — keyed current-state suggestions.

The cheap regex (detect_current_state) handles obvious phrasing on the hot
path; for the rest, an OFFLINE bounded LLM (during consolidation) proposes
{attribute, value, negation, confidence}. confidence>=0.8 positives upsert via
the canonical keyed path; weaker/negation ones are tagged for review. Bounded,
breaker-protected, config-gated, never on recall. LLM is mocked here.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmb.core.engine import Engine
from pmb.core.events import Event


@pytest.fixture(autouse=True)
def _reset_breaker():
    from pmb.core import circuit_breaker
    circuit_breaker.reset()
    yield
    circuit_breaker.reset()


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


# A plain fact the cheap regex does NOT catch ("relocated home base" isn't in
# detect_current_state's patterns) → it becomes an LLM candidate.
_CAND = "The user relocated their home base to Berlin earlier this season."


def _seed(eng, content=_CAND):
    ev = Event(workspace_id=eng.workspace.id, event_type="fact",
               content=content, metadata={}, importance=0.6)
    return eng.events.append(ev).ulid


def _meta(eng, ulid):
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        row = c.execute("SELECT metadata_json FROM events WHERE ulid=?",
                        (ulid,)).fetchone()
    return json.loads(row[0] or "{}") if row else {}


def _active_keyed(eng, subj, attr):
    return [h for h in eng.get_keyed_fact_history(subj, attr) if h["is_current"]]


class _FakeLLM:
    timeout = 120.0

    def __init__(self, payload):
        self._payload = payload

    def complete(self, prompt, max_tokens=120):
        return self._payload


def _patch_llm(monkeypatch, payload):
    monkeypatch.setattr("pmb.health.consolidate.resolve_llm_client",
                        lambda **k: _FakeLLM(payload))


def test_high_confidence_positive_is_upserted(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    _seed(eng)
    _patch_llm(monkeypatch,
               '{"subject":"user","attribute":"city","value":"Berlin",'
               '"negation":false,"confidence":0.92}')
    res = eng.suggest_keyed_from_llm(dry_run=False)
    assert res["applied"] == 1
    active = _active_keyed(eng, "user", "city")
    assert len(active) == 1 and active[0]["value"] == "Berlin"


def test_low_confidence_is_tagged_not_applied(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    u = _seed(eng)
    _patch_llm(monkeypatch,
               '{"subject":"user","attribute":"city","value":"Berlin",'
               '"negation":false,"confidence":0.5}')
    res = eng.suggest_keyed_from_llm(dry_run=False)
    assert res["applied"] == 0 and res["tagged"] == 1
    assert _active_keyed(eng, "user", "city") == []
    assert _meta(eng, u).get("suggested_key", {}).get("attribute") == "city"


def test_negation_suggestion_is_tagged_not_applied(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    _seed(eng)
    _patch_llm(monkeypatch,
               '{"subject":"user","attribute":"city","value":"",'
               '"negation":true,"confidence":0.95}')
    res = eng.suggest_keyed_from_llm(dry_run=False)
    assert res["applied"] == 0
    assert _active_keyed(eng, "user", "city") == []


def test_dry_run_writes_nothing(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    _seed(eng)
    _patch_llm(monkeypatch,
               '{"subject":"user","attribute":"city","value":"Berlin",'
               '"negation":false,"confidence":0.92}')
    res = eng.suggest_keyed_from_llm(dry_run=True)
    assert res["applied"] == 0          # nothing written in a dry run
    assert res["would_apply"] == 1      # but it WOULD have applied
    assert _active_keyed(eng, "user", "city") == []  # nothing written


# ── A2: third-party facts must never become user state ──────────────────────

def test_third_party_fact_never_reaches_llm(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    """A fact with no first-person/user cue ('Alice relocated to Berlin') is
    filtered out BEFORE the LLM is called — gate 1."""
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    _seed(eng, content="Alice relocated to Berlin this spring.")

    def _boom(**k):
        raise AssertionError("LLM called on a third-party fact")
    monkeypatch.setattr("pmb.health.consolidate.resolve_llm_client", _boom)
    res = eng.suggest_keyed_from_llm(dry_run=False)
    assert res["applied"] == 0 and not res["suggestions"]


def test_llm_other_person_verdict_not_applied(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    """Even when a fact mentions the user ('my sister'), if the LLM says the
    attribute belongs to another person it is dropped — gate 2."""
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    _seed(eng, content="My sister Alice relocated her home base to Berlin.")
    _patch_llm(monkeypatch,
               '{"subject":"other_person","attribute":"city","value":"Berlin",'
               '"negation":false,"confidence":0.95}')
    res = eng.suggest_keyed_from_llm(dry_run=False)
    assert res["applied"] == 0 and not res["suggestions"]
    assert _active_keyed(eng, "user", "city") == []


def test_quality_flagged_fact_is_not_a_candidate(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    """A fact flagged suspect_junk by the write-time gate is never sent to the
    LLM — gate 3."""
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    ev = Event(workspace_id=eng.workspace.id, event_type="fact",
               content=_CAND, metadata={"quality_flag": "suspect_junk"},
               importance=0.2)
    eng.events.append(ev)

    def _boom(**k):
        raise AssertionError("LLM called on a quality-flagged fact")
    monkeypatch.setattr("pmb.health.consolidate.resolve_llm_client", _boom)
    res = eng.suggest_keyed_from_llm(dry_run=False)
    assert res["applied"] == 0 and not res["suggestions"]


# ── A3: the whole pass is wall-clock / call bounded ─────────────────────────

def test_offline_budget_caps_calls(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home,
                  **{"llm.offline_max_calls": 2})
    for i in range(6):
        _seed(eng, content=f"The user relocated their home base to City{i} lately.")
    calls = {"n": 0}

    class _CountLLM:
        timeout = 120.0

        def complete(self, prompt, max_tokens=120):
            calls["n"] += 1
            return '{"subject":"unknown","attribute":"","value":"","negation":false,"confidence":0.1}'

    monkeypatch.setattr("pmb.health.consolidate.resolve_llm_client",
                        lambda **k: _CountLLM())
    res = eng.suggest_keyed_from_llm(dry_run=True)
    assert calls["n"] == 2
    assert res.get("stopped") == "budget"
    assert res["budget"]["llm_calls"] == 2


def test_breaker_open_skips_quietly(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    from pmb.core import circuit_breaker
    monkeypatch.setattr(circuit_breaker, "is_open", lambda backend: True)

    def _boom(**k):
        raise AssertionError("resolve_llm_client called while breaker open")
    monkeypatch.setattr("pmb.health.consolidate.resolve_llm_client", _boom)

    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    _seed(eng)
    res = eng.suggest_keyed_from_llm(dry_run=False)
    assert res.get("skipped") == "breaker_open"


def test_config_off_skips(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home,
                  **{"consolidate.suggest_keyed": False})
    _seed(eng)

    def _boom(**k):
        raise AssertionError("resolve_llm_client called while disabled")
    monkeypatch.setattr("pmb.health.consolidate.resolve_llm_client", _boom)
    res = eng.suggest_keyed_from_llm(dry_run=False)
    assert res.get("skipped") == "disabled"
