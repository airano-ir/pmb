"""S1: the daemon-discovery path must stay dependency-light.

`pmb.mcp.registry` (and the daemon-token read) runs on EVERY hook message via
the client. If importing it drags in fastmcp / the mcp SDK / the engine /
sentence-transformers, every user message pays 3–6 s of imports for nothing.
The package `__init__` is PEP-562 lazy so `build_server` resolves only when
someone actually builds a server.

Asserted in a FRESH subprocess because this test process has almost certainly
already imported fastmcp via other tests.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_SRC = str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "src")

_HEAVY = ("fastmcp", "mcp", "pmb.core.engine", "sentence_transformers",
          "lancedb", "torch", "transformers")


def _heavy_after_import(module: str) -> str:
    code = (
        f"import sys; import {module}; "
        f"heavy=[m for m in {_HEAVY!r} if m in sys.modules]; "
        "print(','.join(heavy))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, f"import {module} failed:\n{r.stderr}"
    return r.stdout.strip()


def test_registry_import_pulls_no_heavy_deps():
    pulled = _heavy_after_import("pmb.mcp.registry")
    assert pulled == "", f"pmb.mcp.registry pulled heavy modules: {pulled}"


def test_daemon_import_pulls_no_heavy_deps():
    # the hook client reads the daemon token via this module — keep it light
    pulled = _heavy_after_import("pmb.mcp.daemon")
    assert pulled == "", f"pmb.mcp.daemon pulled heavy modules: {pulled}"


def test_build_server_still_reachable_lazily():
    # PEP-562 attribute access must still resolve the server entry point
    code = (
        "import pmb.mcp; assert callable(pmb.mcp.build_server); "
        "from pmb.mcp import main; assert callable(main); print('ok')"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True, env=env, timeout=180)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr
