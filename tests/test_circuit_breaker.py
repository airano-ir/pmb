"""Phase 2 / issue #12: backend circuit breaker — opens after N consecutive
failures, closes on success or cooldown, and recall_smart skips a tripped LLM
backend instead of paying its latency."""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmb.core import circuit_breaker as cb
from pmb.core.engine import Engine


# ── unit (no engine) ───────────────────────────────────────────────────────

def test_opens_after_threshold():
    cb.reset("svc")
    assert cb.is_open("svc") is False
    cb.record_failure("svc", threshold=2, cooldown_s=30)
    assert cb.is_open("svc") is False     # 1 failure — not yet
    cb.record_failure("svc", threshold=2, cooldown_s=30)
    assert cb.is_open("svc") is True      # 2 → open
    cb.reset("svc")


def test_success_closes():
    cb.reset("svc")
    cb.record_failure("svc", threshold=1, cooldown_s=30)
    assert cb.is_open("svc") is True
    cb.record_success("svc")
    assert cb.is_open("svc") is False
    cb.reset("svc")


def test_cooldown_expires():
    cb.reset("svc")
    cb.record_failure("svc", threshold=1, cooldown_s=0.05)
    assert cb.is_open("svc") is True
    time.sleep(0.12)
    assert cb.is_open("svc") is False
    cb.reset("svc")


def test_status_shape():
    cb.reset("llm")
    cb.record_failure("llm", threshold=1, cooldown_s=30, error="boom")
    st = cb.status()
    assert st["llm"]["open"] is True
    assert st["llm"]["total_failures"] == 1
    assert st["llm"]["last_error"] == "boom"
    cb.reset("llm")


# ── integration with recall_smart ───────────────────────────────────────────

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


def test_recall_smart_skips_llm_when_breaker_open(
    tmp_pmb_home, tmp_workspace_dir, monkeypatch
):
    cb.reset("llm")
    calls = {"n": 0}

    def _spy(*a, **k):
        calls["n"] += 1
        raise RuntimeError("LLM must not be resolved while breaker is open")

    monkeypatch.setattr("pmb.health.consolidate.resolve_llm_client", _spy)
    cb.record_failure("llm", threshold=1, cooldown_s=60)  # force open

    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
                 config_overrides={"recall.cache_size": 0,
                                   "recall.smart_allow_llm": True})
    eng.record_fact("an unrelated note about ships")
    pack = eng.recall_smart("totally unrelated quantum chromodynamics xyz",
                            top_k=3, confidence_threshold=0.99)
    assert calls["n"] == 0
    assert "llm_skipped_breaker_open" in pack.escalation["stages"]
    assert eng.breaker_status()["llm"]["open"] is True
    cb.reset("llm")
