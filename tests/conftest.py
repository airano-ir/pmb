"""Shared pytest setup for PMB.

Two responsibilities:

1. **sys.path fallback.** When the package is installed editable
   (`pip install -e .`) every test can simply `from pmb.X import Y` and
   nothing here matters. But if someone runs pytest from a clean check-out
   where `src/` is not on the path yet, individual tests have historically
   patched `sys.path` themselves; conftest centralises that fallback so
   new tests do not need the boilerplate.

2. **Common fixtures.** A `tmp_workspace` fixture that returns an isolated
   tmp directory pair (PMB_HOME, workspace cwd), useful for any test that
   needs to instantiate an Engine without colliding with `~/.pmb/`.

Existing tests that still do `sys.path.insert(...)` themselves keep
working unchanged - the operation is idempotent.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# sys.path fallback - safe even if pmb is already installed editable
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Friendly venv guard — fail fast with ONE clear message instead of 40
# ModuleNotFoundError tracebacks when pytest is run with the system Python
# that doesn't have PMB's deps. (A fresh contributor running `pytest` from
# the repo root without activating the venv would otherwise see a wall of
# import errors and assume the project is broken.)
# ---------------------------------------------------------------------------
def pytest_configure(config):
    # Run fully offline. The embedding / rerank / CLIP models are already in
    # the local HF cache from a prior run, so loading them must NOT touch the
    # network. huggingface_hub shares one httpx client per process; once any
    # test tears that client down, a later model load that still tries a
    # revision check fails with "Cannot send a request, as the client has been
    # closed" — flakily, depending on test order. Offline mode reads the cache
    # directly and skips that check, making model loads deterministic.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    missing = []
    for mod in ("numpy", "fastmcp", "typer"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if not missing:
        return
    is_win = os.name == "nt"
    runner = (".venv\\Scripts\\python.exe" if is_win else ".venv/bin/python")
    bar = "═" * 64
    msg = "\n".join([
        "",
        bar,
        f"  PMB's test deps are missing: {', '.join(missing)}",
        "",
        "  You're probably running pytest with the SYSTEM Python.",
        "  Run the tests through the project venv instead:",
        "",
        f"      {runner} -m pytest",
        "",
        "  First-time setup (from the repo root):",
        "      python -m venv .venv",
        ("      .venv\\Scripts\\pip install -e ." if is_win
         else "      .venv/bin/pip install -e ."),
        bar,
        "",
    ])
    import pytest as _pytest
    _pytest.exit(msg, returncode=1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_test_env(monkeypatch):
    """Per-test isolation that compensates for two pre-existing suite patterns.

    (1) Deterministic recall side-effects. Production defaults
        `recall.touch_async=True`: recall enqueues access_count / importance /
        last_accessed writes to a ~250ms background flusher to batch the SQLite
        write lock under concurrent recalls. Tests assert on those side-effects
        on the very next line — racing the flusher and failing
        nondeterministically under load. Flipping the SCHEMA default to False
        (autouse, so it applies regardless of which local tmp_pmb_home fixture a
        test shadows conftest with) makes recall drain inline.

    (2) PMB_* env-leak isolation. Several tests set PMB_WORKSPACE /
        PMB_TOOL_PROFILE via raw `os.environ[...] = ...` (no cleanup), which
        leaks into later tests and makes workspace-detection flaky depending on
        order (e.g. source='env'/'personal' where 'cwd' is expected). Clearing
        them at the start of every test — via monkeypatch, which also restores
        the prior state afterwards — undoes both inherited leaks and a test's
        own raw set, so nothing propagates onward. PMB_HOME is left alone (the
        tmp_pmb_home fixture owns it).
    """
    try:
        from pmb.config import SCHEMA
        monkeypatch.setattr(SCHEMA["recall.touch_async"], "default", False)
    except Exception:
        pass
    for _k in ("PMB_WORKSPACE", "PMB_TOOL_PROFILE"):
        monkeypatch.delenv(_k, raising=False)


@pytest.fixture
def tmp_pmb_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated PMB_HOME for the test. Restored automatically.

    Use this whenever a test instantiates Engine and you don't want it
    writing to the developer's real `~/.pmb/`.
    """
    home = tmp_path / "pmb_home"
    home.mkdir()
    monkeypatch.setenv("PMB_HOME", str(home))
    return home


@pytest.fixture
def tmp_workspace_dir(tmp_path: Path) -> Path:
    """An empty directory to act as the workspace `cwd` for an Engine."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def isolated_engine(tmp_pmb_home: Path, tmp_workspace_dir: Path):
    """A fresh Engine that writes to a tmp dir and tears down after the test.

    Yields the engine. Calls `engine.close()` on teardown if the method exists.
    """
    # Lazy import so collecting this fixture doesn't load LanceDB.
    from pmb.core.engine import Engine

    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home, rerank_model=None)
    try:
        yield eng
    finally:
        close = getattr(eng, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                # We don't fail teardown because of a cleanup hiccup; the
                # tmp_path fixture deletes the directory anyway.
                pass
