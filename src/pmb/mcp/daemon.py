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

import pmb


def _pmb_home() -> Path:
    return Path(os.environ.get("PMB_HOME") or (Path.home() / ".pmb"))


def token_path() -> Path:
    return _pmb_home() / "daemon.token"


def write_daemon_token(rotate: bool = False) -> str:
    """Return the local daemon token, persisting it across restarts (S6).

    A REUSED token is what lets `pmb connect --daemon` bake a stable
    `Authorization: Bearer <token>` into an MCP client's config: if the token
    rotated on every daemon start, that static header would go stale the first
    time the daemon idle-exits and restarts. Pass ``rotate=True`` to force a
    fresh secret (e.g. on suspected compromise). chmod 600 on POSIX so other
    local users can't read it."""
    p = token_path()
    if not rotate:
        existing = read_daemon_token()
        if existing:
            return existing
    tok = secrets.token_urlsafe(32)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(tok, encoding="utf-8")
    if os.name != "nt":
        try:
            os.chmod(p, 0o600)
        except Exception:
            pass
    return tok


def read_daemon_token() -> str | None:
    try:
        return token_path().read_text(encoding="utf-8").strip() or None
    except Exception:
        return None


# Shared mutable so the request middleware and the idle watcher agree on when
# the last request arrived.
_LAST_REQUEST = {"ts": time.time()}

# M1: last maintenance-tick summary, surfaced in /internal/health and
# `pmb daemon status`. None until the first tick runs.
_LAST_MAINTENANCE: dict = {"summary": None}

# R11: per-session set of lesson ulids already fired by the PreToolUse guard,
# so a rule interrupts at most once per session (not on every tool call).
_PRETOOL_SEEN: dict[str, set] = {}


def pretool_lessons(engine, excerpt: str, seen: set) -> list:
    """R11 core: lessons that should FIRE for a tool-call excerpt - a STRONG
    match (>= 2 distinctive overlapping tokens incl >= 1 identifier-grade one),
    not yet fired this session, max 2. The guard interrupts the agent, so the
    bar is deliberately higher than ordinary lesson surfacing. Pure function for
    testability; `seen` is mutated with the fired ulids."""
    if not excerpt or not excerpt.strip():
        return []
    try:
        import re as _re

        from pmb.core.text_match import (
            distinctive_tokens,
            is_strong,
            shell_command_names,
        )
    except Exception:
        return []
    # Command name(s) the agent is about to run, extracted STRUCTURALLY (no
    # hardcoded command list). Lets a rule that NAMES a command ('never use git')
    # fire even though 'git' is too short to be a distinctive token.
    cmds = shell_command_names(excerpt)
    q = distinctive_tokens(excerpt)
    try:
        cands = engine.find_lessons(excerpt, limit=6) or []
    except Exception:
        cands = []
    # ALSO pull rules that mention a command we're about to run - distinctive
    # token matching drops short names like 'git', so a bare 'never use git' rule
    # would never even be a candidate. Cheap recent-lesson scan for a raw word hit.
    if cmds:
        have = {L.get("ulid") for L in cands}
        try:
            for L in (engine.find_lessons("", limit=200) or []):
                words = set(_re.findall(r"[a-z0-9_.\-/]+",
                                        (L.get("content") or "").lower()))
                if (cmds & words) and L.get("ulid") not in have:
                    cands.append(L)
                    have.add(L.get("ulid"))
        except Exception:
            pass
    fired = []
    for L in cands:
        u = L.get("ulid")
        if not u or u in seen:
            continue
        content = L.get("content") or ""
        ov = q & distinctive_tokens(content)
        words = set(_re.findall(r"[a-z0-9_.\-/]+", content.lower()))
        cmd_hit = bool(cmds & words)  # the rule names the command we're running
        # Fire on: a rule that NAMES this command, OR two distinctive overlapping
        # tokens, OR one identifier-grade one (record_batch, qwen2.5 - is_strong).
        # The guard interrupts the agent, so the generic bar is high - but a rule
        # naming the exact command we're about to run always qualifies.
        if cmd_hit or len(ov) >= 2 or any(is_strong(t) for t in ov):
            seen.add(u)
            fired.append(L)
        if len(fired) >= 2:
            break
    return fired


def _workspace_matches(engine, requested) -> bool:
    """S4: the daemon serves exactly ONE workspace (its build cwd). A client
    that names a DIFFERENT workspace must be refused so we never inject
    workspace A's memory into workspace B's hooks. A client that names nothing
    (no PMB_WORKSPACE) is served best-effort, the unchanged single-workspace
    contract."""
    if not requested:
        return True
    req = str(requested).strip().lower()
    ws = engine.workspace
    cands = {str(getattr(ws, "id", "")).lower(), str(getattr(ws, "name", "")).lower()}
    return req in cands


def _register_internal_routes(mcp, engine) -> None:
    """Mount /internal/health + /internal/hook/* + /internal/shutdown on the
    fastmcp ASGI app, backed by the SAME warm engine. Call before http_app()."""
    import anyio
    from starlette.responses import JSONResponse

    @mcp.custom_route("/internal/health", methods=["GET"])
    async def _health(request):  # noqa: ANN001
        return JSONResponse({
            "ok": True,
            "version": pmb.__version__,
            "warm": bool(getattr(engine, "is_warm", lambda: False)()),
            "workspace": engine.workspace.id,
            "workspace_name": getattr(engine.workspace, "name", None),
            "pmb_home": str(_pmb_home()),
            "last_maintenance": _LAST_MAINTENANCE["summary"],  # M1
        })

    @mcp.custom_route("/internal/hook/prepare-context", methods=["POST"])
    async def _prepare(request):  # noqa: ANN001
        _LAST_REQUEST["ts"] = time.time()
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not _workspace_matches(engine, (body or {}).get("workspace")):
            return JSONResponse({"error": "workspace_mismatch",
                                 "version": pmb.__version__}, status_code=409)
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

    @mcp.custom_route("/internal/hook/session-restore", methods=["POST"])
    async def _restore(request):  # noqa: ANN001
        _LAST_REQUEST["ts"] = time.time()
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not _workspace_matches(engine, (body or {}).get("workspace")):
            return JSONResponse({"error": "workspace_mismatch",
                                 "version": pmb.__version__}, status_code=409)
        max_chars = int((body or {}).get("max_chars") or 3000)

        def _work() -> str:
            from pmb.hooks import build_session_restore
            try:
                return build_session_restore(
                    engine, minutes=None, include_project=True,
                    max_chars=max_chars) or ""
            except Exception as e:
                try:
                    from pmb.core.errlog import log_error
                    log_error(engine.workspace.db_path, "daemon_hook", e, "restore")
                except Exception:
                    pass
                return ""

        text = await anyio.to_thread.run_sync(_work)
        return JSONResponse({
            "context": text,
            "version": pmb.__version__,
            "source": "daemon",
        })

    @mcp.custom_route("/internal/hook/pretool", methods=["POST"])
    async def _pretool(request):  # noqa: ANN001
        """R11: PreToolUse lesson guard. Fires a matching lesson at TOOL-CALL
        time ("use pnpm, never npm" when the agent is about to run `npm
        install`), even if the agent never called memory. Daemon-served only;
        advisory (never blocks); once per (session, lesson)."""
        _LAST_REQUEST["ts"] = time.time()
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not _workspace_matches(engine, (body or {}).get("workspace")):
            return JSONResponse({"error": "workspace_mismatch",
                                 "version": pmb.__version__}, status_code=409)
        excerpt = str((body or {}).get("excerpt") or "")[:500]
        session = str((body or {}).get("session_id") or "")

        def _work() -> str:
            if not excerpt.strip():
                return ""
            try:
                if not engine.config.get("hooks.pretool_guard"):
                    return ""
            except Exception:
                pass
            try:
                seen = _PRETOOL_SEEN.setdefault(session, set())
                fired = pretool_lessons(engine, excerpt, seen)
                if not fired:
                    return ""
                try:
                    engine._log_lesson_surfaces(fired, query="pretool",
                                                source="pretool_guard")
                except Exception:
                    pass
                lines = ["[pmb] Relevant rule(s) before this action:"]
                lines += [f"  ! {(L.get('content') or '')[:240]}" for L in fired]
                return "\n".join(lines)
            except Exception:
                return ""

        text = await anyio.to_thread.run_sync(_work)
        return JSONResponse({"context": text, "version": pmb.__version__,
                             "source": "daemon"})

    @mcp.custom_route("/internal/shutdown", methods=["POST"])
    async def _shutdown(request):  # noqa: ANN001
        """S3: authenticated shutdown so a client that detects a VERSION
        mismatch can retire the stale daemon and autostart the new build,
        instead of every hook falling cold for up to idle_exit_min."""
        def _drain():
            try:
                engine.wait_for_writes(timeout=5.0)
            except Exception:
                pass
            try:
                from pmb.mcp.perf import flush_perf  # S9: don't drop buffered perf
                flush_perf()
            except Exception:
                pass

        await anyio.to_thread.run_sync(_drain)

        def _bye():
            time.sleep(0.2)
            os._exit(0)

        threading.Thread(target=_bye, daemon=True).start()
        return JSONResponse({"ok": True, "version": pmb.__version__})


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
            try:
                from pmb.mcp.perf import flush_perf  # S9: flush buffered perf
                flush_perf()
            except Exception:
                pass
            sys.stderr.write("[pmb-daemon] idle timeout reached - exiting.\n")
            os._exit(0)


def _maintenance_watcher(engine) -> None:
    """M1: run the self-maintenance tick once per `maintenance_interval_h` of
    uptime, only while idle. Daemon thread; never raises into the server."""
    from pmb.maintenance.tick import run_maintenance_tick, should_run_maintenance
    try:
        if not bool(engine.config.get("daemon.maintenance")):
            return
    except Exception:
        return
    last_tick = time.time()   # don't fire immediately on a fresh start
    while True:
        try:
            interval_s = float(engine.config.get("daemon.maintenance_interval_h")) * 3600.0
            idle_min_s = float(engine.config.get("daemon.maintenance_idle_min")) * 60.0
        except Exception:
            interval_s, idle_min_s = 24 * 3600.0, 300.0
        # wake periodically; the predicate gates the actual run
        time.sleep(min(interval_s / 4.0, 300.0))
        now = time.time()
        if not should_run_maintenance(now, last_tick, interval_s,
                                      _LAST_REQUEST["ts"], idle_min_s):
            continue
        try:
            archive = bool(engine.config.get("daemon.maintenance_archive"))
        except Exception:
            archive = True
        try:
            summary = run_maintenance_tick(engine, archive=archive, now=now)
            _LAST_MAINTENANCE["summary"] = summary
            st = summary.get("steps", {})
            sys.stderr.write(
                f"[pmb-daemon] maintenance: archived "
                f"{st.get('archive_cold', {}).get('archived', 0)}, conflicts "
                f"{st.get('conflicts', {}).get('found', 0)}, declutter-candidates "
                f"{st.get('declutter_dryrun', {}).get('would_archive', 0)}\n"
            )
        except Exception as e:
            sys.stderr.write(f"[pmb-daemon] maintenance tick failed: {e}\n")
        last_tick = now


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


def _acquire_singleton_lock():
    """Atomically claim the per-PMB_HOME daemon singleton.

    Returns (handle, state, home):
      handle - an OPEN file handle to keep alive for the process lifetime (the
               OS lock holds only while the fd stays open), or None;
      state  - "acquired" (we are the singleton), "held" (another live daemon
               owns it -> caller exits), or "error" (could not lock for a
               non-contention reason -> caller falls back to the registry check).

    OS-level + crash-safe: the lock auto-releases when this process exits, so a
    crashed daemon never wedges the slot. Replaces the old check-then-spawn race
    where two concurrent starts both passed find_live_daemon() and produced
    duplicate daemons on 8765/8766."""
    import os as _os
    from pathlib import Path as _P
    home = _os.environ.get("PMB_HOME") or str(_P.home() / ".pmb")
    try:
        _P(home).mkdir(parents=True, exist_ok=True)
        fh = open(_P(home) / "daemon.lock", "a+")  # noqa: SIM115 (held for life)
    except Exception:
        return None, "error", home
    try:
        fh.seek(0)
        if _os.name == "nt":
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            fh.close()
        except Exception:
            pass
        return None, "held", home
    return fh, "acquired", home


def run_daemon(
    host: str = "127.0.0.1",
    port: int = 8765,
    path: str = "/mcp",
    idle_exit_min: float | None = None,
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

    # Bulletproof singleton: an atomic per-PMB_HOME OS lock, taken BEFORE the
    # heavy build_server. find_live_daemon() alone is check-then-spawn - two
    # concurrent starts both pass it, giving duplicate daemons (8765 + 8766 via
    # _free_port). The lock lets exactly one win and auto-releases on crash, so
    # the slot never wedges. The registry check below stays as a friendly
    # fast-path and covers a daemon started before this lock existed.
    _singleton, _lock_state, _home = _acquire_singleton_lock()
    if _lock_state == "held":
        _ex = find_live_daemon()
        sys.stderr.write(
            f"[pmb-daemon] another daemon already owns {_home} "
            f"(pid {_ex.get('pid') if _ex else '?'}). Not starting a second.\n"
        )
        return 0

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
    # Keep the singleton lock handle alive for the daemon's whole lifetime (the
    # OS lock holds only while the fd is open); auto-released on exit/crash.
    engine._daemon_singleton_lock = _singleton

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
    # M1: self-maintenance tick (no-op unless daemon.maintenance is on).
    threading.Thread(target=_maintenance_watcher, args=(engine,),
                     daemon=True, name="pmb-daemon-maintenance").start()

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
