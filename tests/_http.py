"""Shared resilient HTTP helpers for tests that hit a REAL local server.

Integration tests start a ThreadingHTTPServer in a thread and talk to it over
loopback. On shared CI runners the server can be slow to bind, or a request can
stall under load - so a fixed sleep + a tight `urlopen(timeout=5)` flakes
(~50/50, which reddened the whole matrix). These helpers wait for readiness and
retry transient connection/timeout errors with backoff: a genuine failure still
surfaces after the last attempt, a transient stall self-heals.

Use these instead of rolling a one-off urlopen with a tight timeout.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


def wait_ready(port, path: str = "/", *, timeout: float = 30.0,
               probe_timeout: float = 2.0) -> bool:
    """Block until the server answers at `path` (any HTTP status = up), or
    `timeout` seconds elapse. Returns True if it came up."""
    url = f"http://127.0.0.1:{port}{path}"
    for _ in range(max(1, int(timeout / 0.25))):
        try:
            with urllib.request.urlopen(url, timeout=probe_timeout) as r:
                if getattr(r, "status", 200) == 200:
                    return True
        except urllib.error.HTTPError:
            return True  # it answered (even non-200) → server is up
        except OSError:
            time.sleep(0.25)
    return False


def _fetch(url, *, data=None, timeout: float = 30.0, attempts: int = 5) -> bytes:
    """Resilient fetch returning the raw body. Retries transient timeout /
    connection errors with backoff; a real HTTP error (404/500) propagates so
    status assertions still work."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            if data is None:
                req = urllib.request.Request(url)
            else:
                req = urllib.request.Request(
                    url, data=data, method="POST",
                    headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError:
            raise  # a real HTTP response (404/500) - not a transient failure
        except (urllib.error.URLError, OSError) as e:
            last = e
            time.sleep(0.5 * (i + 1))
    raise AssertionError(f"{url} unreachable after {attempts} attempts: {last!r}")


def request_json(port, path, *, payload=None, timeout: float = 30.0,
                 attempts: int = 5):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    raw = _fetch(f"http://127.0.0.1:{port}{path}",
                 data=data, timeout=timeout, attempts=attempts)
    return json.loads(raw.decode("utf-8"))


def get_text(port, path, *, timeout: float = 30.0) -> str:
    """Resilient GET returning the raw response body as text."""
    return _fetch(f"http://127.0.0.1:{port}{path}", timeout=timeout).decode("utf-8")


def get_json(port, path):
    return request_json(port, path)


def post_json(port, path, payload, *, timeout: float = 30.0):
    return request_json(port, path, payload=payload, timeout=timeout)
