"""Local registry of running PMB MCP servers (issue #6).

Why: with the default stdio transport the agent host spawns a fresh
``pmb-mcp`` process per session, and each loads the embedding model + opens
LanceDB — N sessions = N× RAM and N cold starts. There's no way to see how
many are running, and an HTTP ``pmb mcp serve`` could be started twice on the
same port by accident.

This module keeps a tiny JSON registry under ``$PMB_HOME/servers.json`` so:
  * an HTTP server can refuse to start a SECOND instance on a live host:port
    (``find_live_http``) — point clients at the existing one instead;
  * ``pmb mcp status`` can list what's running, with per-process memory.

Best-effort and dependency-light: stdlib only, ``psutil`` used opportunistically
for liveness + RSS. Writes are atomic (temp + os.replace); the occasional lost
update under a concurrent spawn race is self-healing via dead-PID pruning.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def _pmb_home() -> Path:
    return Path(os.environ.get("PMB_HOME") or (Path.home() / ".pmb"))


def registry_path() -> Path:
    return _pmb_home() / "servers.json"


# S8: resolve psutil ONCE. _pid_alive / _rss_mb ran `import psutil` per entry —
# when psutil isn't installed that's a failing import per call, twice per server
# on every hook discovery. Cache the module (or its absence) at first use.
_PSUTIL: object | None = None
_PSUTIL_TRIED = False


def _psutil():
    global _PSUTIL, _PSUTIL_TRIED
    if not _PSUTIL_TRIED:
        _PSUTIL_TRIED = True
        try:
            import psutil as _p  # type: ignore
            _PSUTIL = _p
        except Exception:
            _PSUTIL = None
    return _PSUTIL


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    _ps = _psutil()
    if _ps is not None:
        try:
            return bool(_ps.pid_exists(int(pid)))
        except Exception:
            pass
    if os.name == "nt":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED = 0x1000
            h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED, False, int(pid))
            if h:
                ctypes.windll.kernel32.CloseHandle(h)
                return True
            return False
        except Exception:
            return True  # can't tell → assume alive (best-effort)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return True


def _rss_mb(pid: int | None) -> float | None:
    if not pid:
        return None
    _ps = _psutil()
    if _ps is None:
        return None
    try:
        return _ps.Process(int(pid)).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _load() -> list[dict]:
    try:
        raw = registry_path().read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(entries: list[dict]) -> None:
    path = registry_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        pass  # registry is best-effort; never raise into the server path


def _prune(entries: list[dict]) -> list[dict]:
    """Drop entries whose PID is no longer alive."""
    return [e for e in entries if _pid_alive(e.get("pid"))]


def register_server(
    transport: str,
    host: str | None = None,
    port: int | None = None,
    path: str | None = None,
    workspace: str | None = None,
    pid: int | None = None,
    kind: str = "mcp",
) -> dict:
    """Record THIS process as a running PMB MCP server. Prunes dead entries
    first. Returns the entry (includes the resolved pid).

    `kind` distinguishes a plain MCP server ("mcp") from the persistent memory
    daemon ("daemon") so hooks can find the daemon specifically."""
    entry = {
        "pid": int(pid if pid is not None else os.getpid()),
        "transport": transport,
        "kind": kind,
        "host": host,
        "port": port,
        "path": path,
        "workspace": workspace,
        "pmb_home": str(_pmb_home()),
        "started_at": time.time(),
    }
    entries = _prune(_load())
    # replace any stale entry for the same pid
    entries = [e for e in entries if e.get("pid") != entry["pid"]]
    entries.append(entry)
    _save(entries)
    return entry


def unregister_server(pid: int | None = None) -> None:
    target = int(pid if pid is not None else os.getpid())
    entries = [e for e in _load() if e.get("pid") != target]
    _save(entries)


def list_servers(prune: bool = True, with_rss: bool = True) -> list[dict]:
    """Return registered servers, each annotated with `alive` and `rss_mb`.
    When `prune` is True, dead entries are removed from the registry first.
    `with_rss=False` skips the per-entry psutil RSS syscall for hot callers
    that only need liveness (S8)."""
    entries = _load()
    if prune:
        live = _prune(entries)
        if len(live) != len(entries):
            _save(live)   # prune only removes -> equal length == unchanged content
        entries = live
    out = []
    for e in entries:
        d = dict(e)
        d["alive"] = _pid_alive(e.get("pid"))
        # S8: RSS is a psutil.Process() syscall per entry; only `pmb status`
        # needs it. The hot discovery callers (find_live_daemon / find_live_http,
        # one per hook message) pass with_rss=False and skip it.
        d["rss_mb"] = _rss_mb(e.get("pid")) if with_rss else None
        out.append(d)
    return out


def find_live_http(host: str, port: int) -> dict | None:
    """Return a live streamable-http server entry bound to host:port, if any.
    Used to avoid spawning a second heavy server on the same endpoint."""
    for e in list_servers(prune=True, with_rss=False):
        if (
            e.get("transport") == "streamable-http"
            and e.get("host") == host
            and e.get("port") == int(port)
            and e.get("alive")
        ):
            return e
    return None


def find_live_daemon(pmb_home: str | None = None) -> dict | None:
    """Return the live persistent-memory daemon entry for this PMB_HOME, if any.

    Hooks call this to decide whether to route prepare-context to a warm daemon
    instead of paying a cold per-process Engine. Matches on kind=='daemon' and
    (when given) the same pmb_home, newest first."""
    want_home = str(pmb_home or _pmb_home())
    live = [e for e in list_servers(prune=True, with_rss=False)
            if e.get("kind") == "daemon" and e.get("alive")
            and e.get("port")]
    live = [e for e in live if str(e.get("pmb_home") or want_home) == want_home]
    live.sort(key=lambda e: e.get("started_at") or 0, reverse=True)
    return live[0] if live else None
