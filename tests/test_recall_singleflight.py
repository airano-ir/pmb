"""D4: recall singleflight — concurrent identical top-level recalls run once.

Followers reuse the leader's result; on timeout or a leader error they compute
independently (no deadlock). Recursive (_skip_decompose) calls and the disabled
config bypass it.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
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


def _slow_impl(calls, delay=0.3):
    def impl(query, top_k=None, *a, **k):
        calls.append(query)
        time.sleep(delay)
        return f"RESULT::{query}::{len(calls)}"
    return impl


def _fire(eng, query, n, results):
    threads = []
    for i in range(n):
        t = threading.Thread(target=lambda i=i: results.__setitem__(i, eng.recall(query)))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)


def test_concurrent_identical_recalls_run_once(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    calls: list = []
    monkeypatch.setattr(eng, "_recall_impl", _slow_impl(calls))
    results: dict = {}
    _fire(eng, "where do I live", 5, results)
    assert len(calls) == 1, f"expected 1 underlying recall, got {len(calls)}"
    # every caller got the leader's result
    assert len(results) == 5 and len(set(results.values())) == 1


def test_disabled_runs_each_call(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home,
                  **{"recall.singleflight": False})
    calls: list = []
    monkeypatch.setattr(eng, "_recall_impl", _slow_impl(calls, delay=0.05))
    results: dict = {}
    _fire(eng, "where do I live", 4, results)
    assert len(calls) == 4  # no collapsing when disabled


def test_different_queries_not_collapsed(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    calls: list = []
    monkeypatch.setattr(eng, "_recall_impl", _slow_impl(calls, delay=0.1))
    r: dict = {}
    t1 = threading.Thread(target=lambda: r.__setitem__("a", eng.recall("query A")))
    t2 = threading.Thread(target=lambda: r.__setitem__("b", eng.recall("query B")))
    t1.start(); t2.start(); t1.join(5); t2.join(5)
    assert len(calls) == 2  # distinct queries each compute
    assert r["a"] != r["b"]


def test_skip_decompose_bypasses(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    calls: list = []
    monkeypatch.setattr(eng, "_recall_impl", _slow_impl(calls, delay=0.0))
    eng.recall("x", _skip_decompose=True)
    eng.recall("x", _skip_decompose=True)
    assert len(calls) == 2  # recursive sub-calls never collapse


def test_leader_error_lets_followers_recompute(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    """If the leader raises, followers must not hang or inherit the error —
    they fall back to their own compute."""
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    state = {"first": True}
    calls: list = []

    def impl(query, top_k=None, *a, **k):
        calls.append(query)
        time.sleep(0.1)
        if state["first"]:
            state["first"] = False
            raise RuntimeError("leader boom")
        return "RECOVERED"
    monkeypatch.setattr(eng, "_recall_impl", impl)

    errs: dict = {}
    oks: dict = {}

    def call(i):
        try:
            oks[i] = eng.recall("same")
        except Exception as e:  # noqa: BLE001
            errs[i] = str(e)
    ts = [threading.Thread(target=call, args=(i,)) for i in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    # at least one follower recomputed successfully; nobody hung
    assert "RECOVERED" in oks.values()
