"""`pmb daemon` - manage the persistent memory daemon (B-phase).

The daemon holds ONE warm Engine + embedding model + LanceDB so hook-based
auto-recall gets real semantic recall instead of the per-process cold skip.
`start` spawns it detached; `run` is the foreground worker `start` launches.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

import typer

from pmb.cli._common import console

daemon_app = typer.Typer(help="Manage the persistent memory daemon (warm hooks).")


def _health_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/internal/health"


def _probe(host: str, port: int, timeout: float = 0.5) -> dict | None:
    try:
        with urllib.request.urlopen(_health_url(host, port), timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


@daemon_app.command()
def start(
    port: int = typer.Option(8765, "--port"),
    host: str = typer.Option("127.0.0.1", "--host"),
):
    """Start the memory daemon in the background (idempotent)."""
    from pmb.mcp.registry import find_live_daemon
    live = find_live_daemon()
    if live:
        console.print(
            f"[green]Daemon already running[/] (pid {live.get('pid')}, "
            f"port {live.get('port')}).")
        return

    flags = 0
    kwargs: dict = {}
    if os.name == "nt":
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        flags = 0x00000008 | 0x00000200
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(
        [sys.executable, "-m", "pmb.cli", "daemon", "run",
         "--port", str(port), "--host", host],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL, **kwargs,
    )

    with console.status("starting memory daemon…"):
        for _ in range(50):  # ~5s for uvicorn to bind (model warms async after)
            time.sleep(0.1)
            data = _probe(host, port)
            if data:
                console.print(
                    f"[green]✓ Daemon up[/] v{data.get('version')} · "
                    f"workspace {data.get('workspace')} · "
                    f"warm={data.get('warm')}\n"
                    f"[dim]Hooks will use it on the next message. "
                    f"Embedding model finishes warming in the background.[/]")
                return
    console.print(
        "[yellow]Daemon spawned but health didn't respond yet.[/] "
        "Check `pmb daemon status` in a moment.")


@daemon_app.command()
def run(
    port: int = typer.Option(8765, "--port"),
    host: str = typer.Option("127.0.0.1", "--host"),
    idle_exit_min: float | None = typer.Option(
        None, "--idle-exit-min",
        help="Exit after N idle minutes (default from config; 0 = never)."),
):
    """Run the daemon in the FOREGROUND (this is what `start` launches)."""
    from pmb.mcp.daemon import run_daemon
    raise typer.Exit(run_daemon(host=host, port=port, idle_exit_min=idle_exit_min))


@daemon_app.command()
def stop():
    """Stop the running memory daemon."""
    from pmb.mcp.registry import find_live_daemon, unregister_server
    live = find_live_daemon()
    if not live:
        console.print("[yellow]No daemon running.[/]")
        return
    pid = int(live.get("pid"))
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           capture_output=True)
        else:
            import signal
            os.kill(pid, signal.SIGTERM)
    except Exception as e:
        console.print(f"[red]Failed to stop pid {pid}:[/] {e}")
        return
    try:
        unregister_server(pid)
    except Exception:
        pass
    console.print(f"[green]✓ Stopped daemon[/] (pid {pid}).")


@daemon_app.command()
def status(
    port: int = typer.Option(8765, "--port"),
    host: str = typer.Option("127.0.0.1", "--host"),
):
    """Show whether the memory daemon is running and warm."""
    from pmb.mcp.registry import find_live_daemon
    live = find_live_daemon()
    if not live:
        console.print(
            "[yellow]No daemon running.[/] Start one with "
            "[cyan]pmb daemon start[/] for warm hook recall.")
        return
    data = _probe(live.get("host") or host, int(live.get("port") or port))
    up_s = max(0.0, time.time() - float(live.get("started_at") or time.time()))
    rss = live.get("rss_mb")
    console.print(
        f"[green]Daemon running[/] · pid {live.get('pid')} · "
        f"port {live.get('port')} · uptime {up_s/60:.0f}min"
        + (f" · RSS {rss:.0f}MB" if rss else ""))
    if data:
        console.print(
            f"  v{data.get('version')} · workspace {data.get('workspace')} · "
            f"warm={data.get('warm')}"
            + ("" if data.get("warm") else " [dim](embedding model still loading)[/]"))
    else:
        console.print("  [yellow]health endpoint did not respond[/]")


@daemon_app.command()
def restart(
    port: int = typer.Option(8765, "--port"),
    host: str = typer.Option("127.0.0.1", "--host"),
):
    """Stop the daemon (if any) and start a fresh one."""
    stop()
    time.sleep(0.5)
    start(port=port, host=host)
