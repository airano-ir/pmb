"""Persistent memory daemon (B-phase).

The whole reason this exists: the UserPromptSubmit hook spawns a FRESH
`pmb prepare-context` process per user message, whose new Engine is cold, so
semantic recall is skipped (`RECALL_COLD_SKIP`). `pmb warmup` only warms its
own (short-lived) process. The daemon holds ONE warm Engine + embedding model
+ LanceDB for the whole session and answers prepare-context over a tiny local
HTTP API, so the hook gets real semantic recall in <150ms.

It is the SAME streamable-http MCP server (one warm process, bearer auth,
registry tracking) with three extra internal routes mounted via fastmcp's
`custom_route`. Hooks become thin HTTP clients (see cli/commands/ambient.py)
that fall back to the existing cold path the instant the daemon is absent.
"""
from __future__ import annotations

import os
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import pmb


def _pmb_home() -> Path:
    return Path(os.environ.get("PMB_HOME") or (Path.home() / ".pmb"))


def token_path() -> Path:
    return _pmb_home() / "daemon.token"


def write_daemon_token() -> str:
    """Generate + persist a fresh per-start token (overwrites any old one).
    chmod 600 on POSIX so other local users can't read it."""
    tok = secrets.token_urlsafe(32)
    p = token_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(tok, encoding="utf-8")
    if os.name != "nt":
        try:
            os.chmod(p, 0o600)
        except Exception:
            pass
    return tok


def read_daemon_token() -> Optional[str]:
    try:
        return token_path().read_text(encoding="utf-8").strip() or None
    except Exception:
        return None


# Shared mutable so the request middleware and the idle watcher agree on when
# the last request arrived.
_LAST_REQUEST = {"ts": time.time()}


def _register_internal_routes(mcp, engine) -> None:
    """Mount /internal/health + /internal/hook/* on the fastmcp ASGI app,
    backed by the SAME warm engine. Must be called before http_app()."""
    import anyio
    from starlette.responses import JSONResponse

    @mcp.custom_route("/internal/health", methods=["GET"])
    async def _health(request):  # noqa: ANN001
        return JSONResponse({
            "ok": True,
            "version": pmb.__version__,
            "warm": bool(getattr(engine, "is_warm", lambda: False)()),
            "workspace": engine.workspace.id,
            "pmb_home": str(_pmb_home()),
        })

    @mcp.custom_route("/internal/hook/prepare-context", methods=["POST"])
    async def _prepare(request):  # noqa: ANN001
        _LAST_REQUEST["ts"] = time.time()
        try:
            body = await request.json()
        except Exception:
            body = {}
        msg = str((body or {}).get("message") or "").strip()
        max_chars = int((body or {}).get("max_chars") or 4000)

        def _work() -> str:
            from pmb.hooks import compute_prepare_context_text
            try:
                return compute_prepare_context_text(engine, msg, max_chars) or ""
            except Exception as e:
                try:
                    from pmb.core.errlog import log_error
                    log_error(engine.workspace.db_path, "daemon_hook", e, "prepare")
                except Exception:
                    pass
                return ""

        text = await anyio.to_thread.run_sync(_work)
        return JSONResponse({
            "context": text,
            "version": pmb.__version__,
            "warm": bool(getattr(engine, "is_warm", lambda: False)()),
            "source": "daemon",
        })


def _idle_watcher(idle_exit_min: float, engine) -> None:
    """Exit the process after `idle_exit_min` minutes with no request, so a
    forgotten daemon doesn't hold ~400MB forever. 0 = never exit."""
    if not idle_exit_min or idle_exit_min <= 0:
        return
    limit_s = float(idle_exit_min) * 60.0
    while True:
        time.sleep(min(limit_s / 2.0, 60.0))
        if (time.time() - _LAST_REQUEST["ts"]) >= limit_s:
            try:
                engine.wait_for_writes(timeout=5.0)  # don't drop queued writes
            except Exception:
                pass
            sys.stderr.write("[pmb-daemon] idle timeout reached — exiting.\n")
            os._exit(0)


def _daemon_bearer_middleware(token: str):
    """Bearer middleware that lets /internal/health and CORS preflights through
    (they leak nothing) but requires the token everywhere else."""
    import hmac

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    expected = f"Bearer {token}"

    class _Mw(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            _LAST_REQUEST["ts"] = time.time()
            if request.method == "OPTIONS" or request.url.path in (
                "/internal/health", "/healthz", "/",
            ):
                return await call_next(request)
            got = request.headers.get("authorization", "")
            if not got or not hmac.compare_digest(got, expected):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await call_next(request)

    return _Mw


def run_daemon(
    host: str = "127.0.0.1",
    port: int = 8765,
    path: str = "/mcp",
    idle_exit_min: Optional[float] = None,
) -> int:
    """Foreground daemon runner. Returns a process exit code.

    Refuses to start a second daemon for this PMB_HOME (points at the live one).
    Builds the warm MCP server, mounts internal routes, writes a fresh token,
    recovers the write outbox, registers in the server registry, and serves.
    """
    from pmb.mcp.registry import (
        find_live_daemon,
        register_server,
        unregister_server,
    )
    from pmb.mcp.server import build_server

    existing = find_live_daemon()
    if existing:
        sys.stderr.write(
            f"[pmb-daemon] already running (pid {existing.get('pid')}, "
            f"port {existing.get('port')}). Not starting a second.\n"
        )
        return 0

    try:
        import uvicorn
    except ImportError:
        sys.stderr.write(
            "[pmb-daemon] uvicorn is required. Install: pip install 'uvicorn[standard]'\n"
        )
        return 2

    server = build_server(prewarm=True)
    engine = getattr(server, "_pmb_engine", None)
    if engine is None:
        sys.stderr.write("[pmb-daemon] could not access engine from server.\n")
        return 2

    _register_internal_routes(server, engine)

    # Build the ASGI app from fastmcp, then attach bearer auth.
    app = None
    builder = getattr(server, "http_app", None) or getattr(
        server, "streamable_http_app", None)
    if builder is None:
        sys.stderr.write("[pmb-daemon] fastmcp exposes no http_app builder.\n")
        return 2
    try:
        app = builder(path=path)
    except TypeError:
        app = builder()

    token = write_daemon_token()
    try:
        app.add_middleware(_daemon_bearer_middleware(token))
    except Exception as e:
        sys.stderr.write(f"[pmb-daemon] middleware install failed: {e}\n")

    # Replay any writes left pending by a previous (crashed) process.
    try:
        n = engine.recover_outbox()
        if n:
            sys.stderr.write(f"[pmb-daemon] recovering {n} pending write(s).\n")
    except Exception:
        pass

    if idle_exit_min is None:
        try:
            idle_exit_min = float(engine.config.get("daemon.idle_exit_min"))
        except Exception:
            idle_exit_min = 120.0

    entry = None
    try:
        entry = register_server(
            transport="streamable-http", kind="daemon",
            host=host, port=port, path=path,
            workspace=getattr(server, "name", None),
        )
        import atexit
        atexit.register(unregister_server, entry["pid"])
    except Exception:
        pass

    threading.Thread(target=_idle_watcher, args=(idle_exit_min, engine),
                     daemon=True, name="pmb-daemon-idle").start()

    sys.stderr.write(
        f"[pmb-daemon] warm memory daemon on http://{host}:{port}{path}\n"
        f"  workspace: {engine.workspace.id}  ·  idle-exit: "
        f"{'never' if not idle_exit_min else f'{idle_exit_min:g}min'}\n"
    )
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        if entry is not None:
            try:
                unregister_server(entry["pid"])
            except Exception:
                pass
    return 0
