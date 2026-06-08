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
from typing import Optional


def _pmb_home() -> Path:
    return Path(os.environ.get("PMB_HOME") or (Path.home() / ".pmb"))


def registry_path() -> Path:
    return _pmb_home() / "servers.json"


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        import psutil  # type: ignore
        return psutil.pid_exists(int(pid))
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


def _rss_mb(pid: Optional[int]) -> Optional[float]:
    if not pid:
        return None
    try:
        import psutil  # type: ignore
        return psutil.Process(int(pid)).memory_info().rss / (1024 * 1024)
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
    host: Optional[str] = None,
    port: Optional[int] = None,
    path: Optional[str] = None,
    workspace: Optional[str] = None,
    pid: Optional[int] = None,
) -> dict:
    """Record THIS process as a running PMB MCP server. Prunes dead entries
    first. Returns the entry (includes the resolved pid)."""
    entry = {
        "pid": int(pid if pid is not None else os.getpid()),
        "transport": transport,
        "host": host,
        "port": port,
        "path": path,
        "workspace": workspace,
        "started_at": time.time(),
    }
    entries = _prune(_load())
    # replace any stale entry for the same pid
    entries = [e for e in entries if e.get("pid") != entry["pid"]]
    entries.append(entry)
    _save(entries)
    return entry


def unregister_server(pid: Optional[int] = None) -> None:
    target = int(pid if pid is not None else os.getpid())
    entries = [e for e in _load() if e.get("pid") != target]
    _save(entries)


def list_servers(prune: bool = True) -> list[dict]:
    """Return registered servers, each annotated with `alive` and `rss_mb`.
    When `prune` is True, dead entries are removed from the registry first."""
    entries = _load()
    if prune:
        live = _prune(entries)
        if len(live) != len(entries):
            _save(live)
        entries = live
    out = []
    for e in entries:
        d = dict(e)
        d["alive"] = _pid_alive(e.get("pid"))
        d["rss_mb"] = _rss_mb(e.get("pid"))
        out.append(d)
    return out


def find_live_http(host: str, port: int) -> Optional[dict]:
    """Return a live streamable-http server entry bound to host:port, if any.
    Used to avoid spawning a second heavy server on the same endpoint."""
    for e in list_servers(prune=True):
        if (
            e.get("transport") == "streamable-http"
            and e.get("host") == host
            and e.get("port") == int(port)
            and e.get("alive")
        ):
            return e
    return None
