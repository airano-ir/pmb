"""Perf rows must survive process shutdown.

`record_call` buffers in memory and only writes once `_PERF_FLUSH_EVERY` rows
have accumulated, so before `install_shutdown_flush` a process that made fewer
calls than that wrote nothing at all, and every process lost its final partial
batch. Since an MCP stdio server is a child process its host terminates with a
signal — and Python does not run `atexit` handlers on SIGTERM — both the
atexit and the signal path are exercised here, in real subprocesses.
"""
from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time

import pytest

CALLS = 3  # deliberately below _PERF_FLUSH_EVERY


def _row_count(db_path) -> int:
    if not db_path.exists():
        return 0
    with sqlite3.connect(db_path) as conn:
        try:
            return conn.execute("select count(*) from mcp_calls").fetchone()[0]
        except sqlite3.OperationalError:
            return 0  # table never created


def _child_source(db_path, *, install: bool, exit_mode: str) -> str:
    return textwrap.dedent(
        f"""
        import sys, time, signal
        from pathlib import Path
        signal.signal(signal.SIGINT, signal.default_int_handler)
        from pmb.mcp.perf import record_call, install_shutdown_flush
        if {install!r}:
            install_shutdown_flush()
        db = Path({str(db_path)!r})
        for i in range({CALLS}):
            record_call(db_path=db, workspace_id="ws", tool_name=f"t{{i}}",
                        duration_ms=1.0)
        print("ready", flush=True)
        if {exit_mode!r} == "clean":
            sys.exit(0)
        time.sleep(30)
        """
    )


def _run_child(tmp_path, *, install: bool, exit_mode: str):
    db_path = tmp_path / f"perf-{install}-{exit_mode}.sqlite"
    proc = subprocess.Popen(
        [sys.executable, "-c", _child_source(db_path, install=install,
                                             exit_mode=exit_mode)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if exit_mode == "clean":
        proc.wait(timeout=30)
    else:
        # wait for the child to report its rows are buffered
        deadline = time.time() + 30
        while time.time() < deadline:
            line = proc.stdout.readline()
            if "ready" in line:
                break
        proc.send_signal(getattr(signal, exit_mode))
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - fix regressed
            proc.kill()
            pytest.fail(f"child survived {exit_mode}")
    return db_path


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
@pytest.mark.parametrize("exit_mode", ["clean", "SIGTERM", "SIGINT"])
def test_partial_batch_is_flushed_on_shutdown(tmp_path, exit_mode):
    db_path = _run_child(tmp_path, install=True, exit_mode=exit_mode)
    assert _row_count(db_path) == CALLS


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
def test_without_the_hook_a_partial_batch_is_lost():
    """Pins the behaviour the fix exists to change."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        db_path = _run_child(Path(tmp), install=False, exit_mode="SIGTERM")
        assert _row_count(db_path) == 0
