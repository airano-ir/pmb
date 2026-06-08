"""Phase 0: recall_smart is deadline-bounded and never touches an LLM /
Claude CLI on the interactive (foreground) path by default.

Regression guard for the 120s foreground hang: a low-confidence interactive
query must NOT spawn LLM query-decomposition (which resolved the Claude CLI
subprocess with a 120s timeout). Deep LLM recall is opt-in only.
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


def test_fg_path_never_resolves_llm_by_default(
    tmp_pmb_home, tmp_workspace_dir, monkeypatch
):
    """Default config: even a low-confidence query must not call
    resolve_llm_client (no Claude CLI / Ollama on the interactive path)."""
    calls = {"n": 0}

    def _spy(*a, **k):
        calls["n"] += 1
        time.sleep(5.0)  # would blow the timing assertion below if ever reached
        raise RuntimeError("LLM must not be resolved on the fg path")

    monkeypatch.setattr("pmb.health.consolidate.resolve_llm_client", _spy)

    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.record_fact("Postgres uses port 5433 on the api service")
    eng.record_fact("Random unrelated note about lunch tacos")

    # Warm the embedding model OUTSIDE the timed region — its one-time cold
    # load (~15-20s) is unrelated to the foreground-LLM regression we test.
    eng.recall("warm up the embedding model", top_k=1)

    t0 = time.perf_counter()
    # high threshold forces escalation attempts
    pack = eng.recall_smart(
        "totally unrelated quantum chromodynamics",
        top_k=3, confidence_threshold=0.99,
    )
    elapsed = time.perf_counter() - t0

    assert calls["n"] == 0, "resolve_llm_client was called on the fg path"
    assert elapsed < 3.0, f"fg recall_smart took {elapsed:.2f}s — LLM likely ran"
    assert pack is not None and hasattr(pack, "results")
    assert pack.escalation is not None
    assert "llm_decompose" not in pack.escalation["stages"]


def test_escalation_diag_within_deadline(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home, **{"recall.smart_deadline_ms": 1500})
    eng.record_fact("Alice prefers tea over coffee")
    pack = eng.recall_smart(
        "what does nobody know", top_k=3, confidence_threshold=0.99,
    )
    assert pack.escalation is not None
    diag = pack.escalation
    assert diag["stages"][0] == "local"
    assert diag["elapsed_ms"] <= diag["deadline_ms"] + 1000  # comfortably under budget
    assert diag["stopped"] in (
        "confidence_met", "exhausted_escalations", "deadline_hit",
    )


def test_low_conf_no_reranker_returns_stage1(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    # No cross-encoder model is loaded in the test → stage 2 must skip, not block.
    assert getattr(eng.search, "reranker", None) is None
    eng.record_fact("the cat sat on the mat")
    pack = eng.recall_smart("xyzzy plugh", top_k=3, confidence_threshold=0.99)
    assert pack is not None
    assert "rerank_skipped_cold" in pack.escalation["stages"]
    assert "llm_decompose" not in pack.escalation["stages"]


def test_opt_in_llm_is_clamped_to_budget(
    tmp_pmb_home, tmp_workspace_dir, monkeypatch
):
    """With recall.smart_allow_llm=True the LLM stage runs, but the client's
    timeout is clamped to the remaining wall-clock budget (never 120s)."""

    class _FakeLLM:
        def __init__(self):
            self.timeout = 120.0

    fake = _FakeLLM()

    def _resolve(*a, **k):
        return fake

    class _Decomp:
        sub_queries = ["single"]  # len 1 → _recall_with_decomposition returns None fast

    class _FakeDecomposer:
        def __init__(self, llm, cache_dir=None):
            pass

        def decompose(self, q):
            return _Decomp()

    monkeypatch.setattr("pmb.health.consolidate.resolve_llm_client", _resolve)
    monkeypatch.setattr("pmb.reasoning.decompose.QueryDecomposer", _FakeDecomposer)

    eng = _engine(
        tmp_workspace_dir, tmp_pmb_home,
        **{"recall.smart_allow_llm": True, "recall.smart_deadline_ms": 8000},
    )
    eng.record_fact("an unrelated fact about ships")
    pack = eng.recall_smart(
        "what is the meaning of nothing here", top_k=3, confidence_threshold=0.99,
    )
    assert pack is not None
    assert fake.timeout <= 8.0, f"LLM timeout not clamped to budget: {fake.timeout}"
    assert "llm_decompose" in pack.escalation["stages"]
