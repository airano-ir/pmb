"""`pmb-hook` - the stdlib-only fast lane for agent lifecycle hooks (S2).

Dispatches the five hook subcommands. For each it tries the warm daemon over
localhost HTTP first (≈10-50 ms, real semantic recall), and falls back to the
existing behaviour only when the daemon is absent:

    prepare-context  daemon → else exec the full `pmb prepare-context` (cold
                     path + autostart, the unchanged contract)
    session-restore  daemon → else exec the full `pmb session-restore`
    track-action     daemon → else INLINE light ambient_log insert (no CLI,
                     no numpy - the PostToolUse hot path runs on every tool call)
    followcheck      daemon → else exec the full `pmb lesson-followcheck`
    autowrite        daemon → else exec the full `pmb autowrite`

Imports are stdlib + the deliberately-light `pmb.mcp.registry` /
`pmb.mcp.daemon` (no engine, no fastmcp). Importing this module must never pull
numpy / typer / lancedb - `tests/test_lazy_mcp_import.py`-style budget applies.
"""
from __future__ import annotations

import json
import os
import re as _re
import sys
import time as _time
import urllib.request
from pathlib import Path

# S10: capture the true process-start instant at import, so the trace can report
# end-to-end wall time (import + dispatch + transport + render), not just the
# daemon's inner compute. Under-reported ~30x before this.
_PROC_T0 = _time.perf_counter()
_TRACE_RE = _re.compile(r"(latency=\d+ms)\]")


def _inject_trace_total(ctx: str, source: str) -> str:
    """Stamp the real end-to-end wall time + the serving source (daemon|cold)
    into the auto-context trace header the daemon/CLI already baked. No-op when
    tracing is off (no `latency=` marker present)."""
    if not ctx or "latency=" not in ctx:
        return ctx
    total = int((_time.perf_counter() - _PROC_T0) * 1000)
    return _TRACE_RE.sub(rf"\1 total={total}ms source={source}]", ctx, count=1)


# ── stdin / payload ─────────────────────────────────────────────────

def _read_stdin_utf8() -> str:
    try:
        return sys.stdin.buffer.read().decode("utf-8", errors="replace")
    except Exception:
        try:
            return sys.stdin.read()
        except Exception:
            return ""


def _extract_prompt(raw: str) -> str:
    """Pull the user message out of the host's JSON hook payload (or accept raw
    text). Mirrors cli/commands/ambient._extract_hook_prompt - kept duplicated
    here so the fast lane stays stdlib-only."""
    if not raw:
        return ""
    s = raw.lstrip()
    if s[:1] in ("{", "["):
        try:
            obj = json.loads(s)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            for k in ("prompt", "message", "user_message", "user_prompt",
                      "text", "content"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    return raw


# ── tiny flag parsing (no argparse - keep it instant) ───────────────

def _opt(args: list[str], name: str, default: str | None = None) -> str | None:
    if name in args:
        i = args.index(name)
        if i + 1 < len(args):
            return args[i + 1]
    return default


def _has(args: list[str], name: str) -> bool:
    return name in args


def _workspace_hint() -> str | None:
    # Cheap, Engine-free workspace signal so the daemon can refuse to serve a
    # different workspace's memory (S4). PMB_WORKSPACE is the reliable case;
    # otherwise None and the daemon best-effort serves its own.
    return os.environ.get("PMB_WORKSPACE") or None


# ── daemon transport ────────────────────────────────────────────────

def _retire_stale_daemon(host: str, port: int, tok: str) -> None:
    """S3: a live daemon answered with a DIFFERENT version (the user upgraded
    pmb but the old daemon still holds the slot). Retire it - rate-limited by a
    stamp - so the cold-path fallback below can autostart the new build. Without
    this, every hook stays cold for up to idle_exit_min after an upgrade."""
    try:
        home = Path(os.environ.get("PMB_HOME") or (Path.home() / ".pmb"))
        stamp = home / "daemon.heal_stamp"
        import time as _t
        now = _t.time()
        try:
            if stamp.exists() and (now - stamp.stat().st_mtime) < 600:
                return
        except Exception:
            pass
        try:
            home.mkdir(parents=True, exist_ok=True)
            stamp.write_text(str(now), encoding="utf-8")
        except Exception:
            pass
        req = urllib.request.Request(
            f"http://{host}:{int(port)}/internal/shutdown", data=b"{}",
            headers={"Authorization": f"Bearer {tok}",
                     "Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=0.5).read()
        except Exception:
            pass
    except Exception:
        pass


def _daemon_request(route: str, payload: dict, timeout: float) -> dict | None:
    """One localhost POST to a live, version-matched daemon. None on ANY
    failure (so the caller falls back). Light imports only. On a version
    mismatch it retires the stale daemon (S3) before returning None."""
    try:
        import pmb
        from pmb.mcp.daemon import read_daemon_token
        from pmb.mcp.registry import find_live_daemon

        info = find_live_daemon()
        if not info or not info.get("port"):
            return None
        host = info.get("host") or "127.0.0.1"
        port = int(info["port"])
        tok = read_daemon_token() or ""
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://{host}:{port}{route}",
            data=body,
            headers={"Authorization": f"Bearer {tok}",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if getattr(r, "status", 200) not in (200, None):
                return None
            out = json.loads(r.read().decode("utf-8"))
        if out.get("version") != pmb.__version__:
            _retire_stale_daemon(host, port, tok)  # upgrade → kill the old one
            return None
        return out
    except Exception:
        return None


# ── full-CLI fallback ───────────────────────────────────────────────

def _full_cli_path() -> str:
    py = Path(sys.executable)
    for c in (py.parent / "pmb.exe", py.parent / "pmb"):
        if c.exists():
            return str(c)
    return "pmb"


def _exec_full_cli(cli_args: list[str], detached: bool = False,
                   trace_source: str | None = None) -> int:
    """Run the full `pmb <cli_args>` and forward its stdout. The slow cold path
    - reached only when the daemon is absent. When `trace_source` is set, stamp
    the honest end-to-end total + source into the trace header (S10)."""
    import subprocess
    cmd = [_full_cli_path(), *cli_args]
    if detached:
        kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
                        "stdin": subprocess.DEVNULL}
        if os.name == "nt":
            kwargs["creationflags"] = 0x08000000 | 0x00000200  # CREATE_NO_WINDOW|NEW_GROUP
        else:
            kwargs["start_new_session"] = True
        try:
            subprocess.Popen(cmd, **kwargs)
        except Exception:
            pass
        return 0
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.stdout:
            out = _inject_trace_total(r.stdout, trace_source) if trace_source else r.stdout
            sys.stdout.write(out)
        return r.returncode
    except Exception:
        return 0


# ── subcommands ─────────────────────────────────────────────────────

def cmd_prepare_context(args: list[str]) -> int:
    max_chars = int(_opt(args, "--max-chars", "4000") or 4000)
    quiet = _has(args, "--quiet") or _has(args, "-q")
    msg = _extract_prompt(_read_stdin_utf8().strip()).strip()
    if not msg:
        return 0
    out = _daemon_request(
        "/internal/hook/prepare-context",
        {"message": msg, "max_chars": max_chars, "workspace": _workspace_hint()},
        timeout=1.2,
    )
    if out is not None:
        ctx = _inject_trace_total(out.get("context") or "", "daemon")
        if ctx.strip():
            sys.stdout.write(ctx if ctx.endswith("\n") else ctx + "\n")
        return 0
    # daemon absent → the full cold path (which also autostarts a daemon). Pass
    # the message as a positional arg since stdin is already consumed here.
    cli = ["prepare-context", msg, "--max-chars", str(max_chars)]
    if quiet:
        cli.append("--quiet")
    return _exec_full_cli(cli, trace_source="cold")


def cmd_session_restore(args: list[str]) -> int:
    max_chars = int(_opt(args, "--max-chars", "3000") or 3000)
    quiet = _has(args, "--quiet") or _has(args, "-q")
    out = _daemon_request(
        "/internal/hook/session-restore",
        {"max_chars": max_chars, "workspace": _workspace_hint()},
        timeout=1.5,
    )
    if out is not None:
        ctx = _inject_trace_total(out.get("context") or "", "daemon")
        if ctx.strip():
            sys.stdout.write(ctx if ctx.endswith("\n") else ctx + "\n")
        return 0
    cli = ["session-restore", "--max-chars", str(max_chars)]
    if quiet:
        cli.append("--quiet")
    return _exec_full_cli(cli, trace_source="cold")


def cmd_track_action(args: list[str]) -> int:
    raw = _read_stdin_utf8()
    if not raw.strip():
        return 0
    # No daemon round-trip here: track-action is a pure same-DB SQLite insert on
    # the PostToolUse hot path (every tool call). The dependency-light
    # ambient_log path (no engine, no numpy, no CLI) is already the fast lane -
    # a localhost POST would only add latency for zero benefit.
    try:
        payload = json.loads(raw)
    except Exception:
        return 0
    tool = payload.get("tool_name") or payload.get("tool") or ""
    if not tool:
        return 0
    ti = payload.get("tool_input") or {}
    tr = payload.get("tool_response") or payload.get("tool_result") or {}
    target = (ti.get("file_path") or ti.get("path") or ti.get("command")
              or ti.get("notebook_path") or ti.get("url") or "")
    command = ti.get("command") or ""
    status = ""
    if isinstance(tr, dict):
        if tr.get("success") is True or tr.get("is_error") is False:
            status = "ok"
        elif tr.get("success") is False or tr.get("is_error") is True:
            status = "error"
        elif "exit_code" in tr:
            status = "ok" if tr.get("exit_code") == 0 else str(tr.get("exit_code"))
    try:
        from pmb.core.ambient_log import insert_agent_action
        from pmb.core.workspace import detect_workspace
        ws = detect_workspace()
        insert_agent_action(
            ws.db_path, ws.id, tool=tool, target=str(target),
            status=status, command=str(command),
            session_id=payload.get("session_id"),
        )
    except Exception:
        pass
    return 0


def cmd_pretool(args: list[str]) -> int:
    """R11: PreToolUse lesson guard. Reads the PostToolUse-shaped payload, asks
    the warm daemon whether a lesson should fire for this tool call, and prints
    it as additionalContext. Daemon-served ONLY - no cold fallback (the guard is
    an accelerator, not a contract); silent no-op if no daemon."""
    raw = _read_stdin_utf8()
    if not raw.strip():
        return 0
    try:
        payload = json.loads(raw)
    except Exception:
        return 0
    tool = payload.get("tool_name") or payload.get("tool") or ""
    ti = payload.get("tool_input") or {}
    # a compact excerpt of what the tool is about to do
    parts = [tool]
    for k in ("command", "file_path", "path", "url", "content", "query",
              "pattern", "prompt", "description"):
        v = ti.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v)
    excerpt = " ".join(parts)[:500].strip()
    if not excerpt:
        return 0
    out = _daemon_request(
        "/internal/hook/pretool",
        {"excerpt": excerpt, "session_id": payload.get("session_id"),
         "workspace": _workspace_hint()},
        timeout=0.6,
    )
    if out is not None:
        ctx = out.get("context") or ""
        if ctx.strip():
            sys.stdout.write(ctx if ctx.endswith("\n") else ctx + "\n")
    return 0


def cmd_read_guard(args: list[str]) -> int:
    """PreToolUse(Read): ask the daemon whether this is a redundant re-read of an
    unchanged, recently-read file; if so, emit a deny so the file is not dumped
    into context again. Daemon-served only; silent allow without a daemon."""
    raw = _read_stdin_utf8()
    if not raw.strip():
        return 0
    try:
        payload = json.loads(raw)
    except Exception:
        return 0
    if (payload.get("tool_name") or payload.get("tool") or "") != "Read":
        return 0
    ti = payload.get("tool_input") or {}
    fp = ti.get("file_path") or ti.get("path") or ""
    if not fp:
        return 0
    try:
        from pmb.hooks.read_guard import deny_payload, file_sha1
        sha = file_sha1(fp)
    except Exception:
        return 0
    out = _daemon_request(
        "/internal/hook/readguard",
        {"file_path": fp, "sha1": sha, "session_id": payload.get("session_id"),
         "workspace": _workspace_hint()},
        timeout=0.6,
    )
    if out and out.get("decision") == "deny":
        sys.stdout.write(json.dumps(deny_payload(out.get("reason") or "")) + "\n")
    return 0


def cmd_followcheck(args: list[str]) -> int:
    window = _opt(args, "--window", "30") or "30"
    out = _daemon_request(
        "/internal/hook/followcheck",
        {"window": int(window), "workspace": _workspace_hint()},
        timeout=1.0,
    )
    if out is not None:
        return 0
    return _exec_full_cli(["lesson-followcheck", "--window", str(window), "--quiet"],
                          detached=True)


def cmd_autowrite(args: list[str]) -> int:
    window = _opt(args, "--window", "30") or "30"
    out = _daemon_request(
        "/internal/hook/autowrite",
        {"window": int(window), "workspace": _workspace_hint()},
        timeout=1.0,
    )
    if out is not None:
        return 0
    return _exec_full_cli(["autowrite", "--window", str(window), "--quiet"],
                          detached=True)


def cmd_capture_exploration(args: list[str]) -> int:
    """Stop hook: pull transcript_path from the payload and hand off to the cold
    `pmb capture-exploration` (gated off by default via memo.autocapture_enabled)."""
    raw = _read_stdin_utf8()
    tpath = ""
    try:
        s = raw.lstrip()
        if s[:1] == "{":
            tpath = (json.loads(s) or {}).get("transcript_path", "") or ""
    except Exception:
        tpath = ""
    if not tpath:
        return 0
    return _exec_full_cli(
        ["capture-exploration", "--transcript", tpath, "--quiet"], detached=True,
    )


_DISPATCH = {
    "prepare-context": cmd_prepare_context,
    "session-restore": cmd_session_restore,
    "track-action": cmd_track_action,
    "pretool": cmd_pretool,
    "followcheck": cmd_followcheck,
    "lesson-followcheck": cmd_followcheck,  # accept the full-CLI name too
    "autowrite": cmd_autowrite,
    "capture-exploration": cmd_capture_exploration,
    "read-guard": cmd_read_guard,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        sys.stderr.write("usage: pmb-hook <prepare-context|session-restore|"
                         "track-action|followcheck|autowrite> [flags]\n")
        return 2
    sub, rest = argv[0], argv[1:]
    fn = _DISPATCH.get(sub)
    if fn is None:
        sys.stderr.write(f"pmb-hook: unknown subcommand {sub!r}\n")
        return 2
    try:
        return int(fn(rest) or 0)
    except Exception:
        # A hook must never crash the agent turn.
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
