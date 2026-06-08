"""Process-wide circuit breaker for flaky / slow backends (issue #12).

After N consecutive failures (timeouts or errors) a backend is "opened" — its
calls are skipped for a cooldown window, so PMB degrades to cheaper local
behaviour instead of hammering a dead backend on every request. A single
success closes it again.

In-memory and process-wide (the MCP server is a single long-lived process),
deliberately dependency-free. Current state is exposed via ``status()`` for the
``stats`` tool / dashboard. The same primitives can wrap any backend
(``"llm"``, ``"reranker"``, ``"embedding"``, …) — wire a call site with
``is_open`` + ``record_failure`` / ``record_success``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class _State:
    failures: int = 0          # consecutive failures since last success
    opened_until: float = 0.0  # epoch until which the breaker stays open
    total_failures: int = 0    # lifetime failures (diagnostics)
    last_error: str = ""


_BREAKERS: dict[str, _State] = {}


def is_open(backend: str) -> bool:
    """True if `backend` is currently disabled (in its cooldown window)."""
    st = _BREAKERS.get(backend)
    return bool(st and time.time() < st.opened_until)


def record_failure(
    backend: str,
    threshold: int = 2,
    cooldown_s: float = 60.0,
    error: str = "",
) -> bool:
    """Note a failure for `backend`. Opens the breaker once `threshold`
    consecutive failures are reached. Returns True if the breaker is now open."""
    st = _BREAKERS.setdefault(backend, _State())
    st.failures += 1
    st.total_failures += 1
    if error:
        st.last_error = str(error)[:200]
    if st.failures >= max(1, int(threshold)):
        st.opened_until = time.time() + float(cooldown_s)
    return time.time() < st.opened_until


def record_success(backend: str) -> None:
    """A successful call closes the breaker and clears the failure streak."""
    st = _BREAKERS.get(backend)
    if st:
        st.failures = 0
        st.opened_until = 0.0


def reset(backend: str | None = None) -> None:
    """Clear one breaker (or all). Mainly for tests / manual recovery."""
    if backend is None:
        _BREAKERS.clear()
    else:
        _BREAKERS.pop(backend, None)


def status() -> dict:
    """Snapshot of all known breakers — for `stats` / dashboard / `pmb health`."""
    now = time.time()
    out: dict[str, dict] = {}
    for name, st in _BREAKERS.items():
        out[name] = {
            "open": now < st.opened_until,
            "failures": st.failures,
            "total_failures": st.total_failures,
            "cooldown_remaining_s": max(0.0, round(st.opened_until - now, 1)),
            "last_error": st.last_error or None,
        }
    return out
