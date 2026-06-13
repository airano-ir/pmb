"""Smoke tests that prove `import pmb` and a handful of leaf modules
stay lightweight.

Why: a single `import pmb` used to load LanceDB, sentence-transformers,
numpy, rank_bm25, fastmcp, yaml and ~hundreds of transitive modules just
because top-level `__init__.py` eagerly imported `Engine`. That made
short CLI invocations slow and gave the test suite an avoidable startup
tax.

Each test launches a fresh subprocess so the assertion is over a clean
module table (the parent pytest process has already imported all of
PMB and would always fail the check).

If a future change re-introduces eager imports of heavy modules, these
tests will be the first to flag it.
"""

from __future__ import annotations

import json
import subprocess
import sys

HEAVY_MODULES = {
    "lancedb",
    "sentence_transformers",
    "rank_bm25",
    "fastmcp",
    "numpy",
    "yaml",
    "anthropic",
    "fastembed",
    "torch",
    "transformers",
}


def _modules_loaded_after(*statements: str) -> set[str]:
    """Spawn a fresh interpreter, run `statements`, return loaded module names.

    Each statement becomes its own line at column 0 - no f-string indent
    mangling. The interpreter is invoked with `-c CODE`, so CODE itself
    must be a self-contained script (top-level only, no indent).
    """
    lines = [
        "import json, sys",
        *statements,
        "print(json.dumps(sorted(sys.modules.keys())))",
    ]
    code = "\n".join(lines)
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise AssertionError(
            f"Subprocess crashed (rc={out.returncode}):\n"
            f"--- code ---\n{code}\n"
            f"--- stderr ---\n{out.stderr}\n"
            f"--- stdout ---\n{out.stdout}"
        )
    # The interpreter may emit warnings to stdout; pick the last JSON line.
    last_line = out.stdout.strip().splitlines()[-1]
    return set(json.loads(last_line))


def _assert_no_heavy(loaded: set[str], *, allowed: set[str] = frozenset()) -> None:
    leaked = (loaded & HEAVY_MODULES) - set(allowed)
    assert not leaked, f"unexpected heavy modules loaded: {sorted(leaked)}"


def test_import_pmb_is_lightweight():
    """Bare `import pmb` must not pull in LanceDB / ST / fastmcp / etc."""
    loaded = _modules_loaded_after("import pmb")
    _assert_no_heavy(loaded)


def test_pmb_version_is_accessible_without_heavy_load():
    """Reading `pmb.__version__` does not trigger lazy attribute resolution."""
    loaded = _modules_loaded_after(
        "import pmb",
        "v = pmb.__version__",
        "assert isinstance(v, str) and v",
    )
    _assert_no_heavy(loaded)


def test_redact_is_lightweight():
    """Pure-stdlib redaction module - no heavy deps."""
    loaded = _modules_loaded_after("from pmb.security.redact import redact")
    _assert_no_heavy(loaded)


def test_config_only_needs_yaml():
    """Config layer uses PyYAML but nothing else heavy."""
    loaded = _modules_loaded_after("from pmb.config import Config")
    # yaml is legitimate here; nothing else heavy should appear.
    _assert_no_heavy(loaded, allowed={"yaml"})


def test_event_store_does_not_load_lancedb():
    """EventStore is SQLite-only - LanceDB / ST should not appear."""
    loaded = _modules_loaded_after("from pmb.core.events import EventStore")
    _assert_no_heavy(loaded, allowed={"yaml"})  # config import is OK


def test_pmb_dir_lists_lazy_attrs():
    """`dir(pmb)` should still show Engine / Workspace so IDEs autocomplete."""
    loaded = _modules_loaded_after(
        "import pmb",
        "names = dir(pmb)",
        "assert 'Engine' in names, names",
        "assert 'Workspace' in names, names",
        "assert 'detect_workspace' in names, names",
    )
    _assert_no_heavy(loaded)


def test_engine_attribute_access_still_works():
    """Compatibility: `from pmb import Engine` must keep working.

    This test does pull in heavy modules - it's the point of the lazy
    accessor. We just check it doesn't raise.
    """
    code = "import pmb; assert pmb.Engine is not None"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, (
        f"pmb.Engine access crashed:\n"
        f"--- stderr ---\n{out.stderr}\n"
        f"--- stdout ---\n{out.stdout}"
    )


def test_unknown_attribute_raises_attribute_error():
    """The lazy resolver must not silently return None for typos."""
    code = "\n".join([
        "import pmb",
        "try:",
        "    pmb.NoSuchThing",
        "except AttributeError:",
        "    print('OK')",
        "else:",
        "    raise SystemExit('expected AttributeError')",
    ])
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False,
    )
    assert out.returncode == 0 and "OK" in out.stdout, (
        f"unexpected:\n{out.stdout}\n{out.stderr}"
    )


def test_pmb_core_lazy_too():
    """`pmb.core` package should also stay light until names are accessed."""
    loaded = _modules_loaded_after("import pmb.core")
    _assert_no_heavy(loaded)
