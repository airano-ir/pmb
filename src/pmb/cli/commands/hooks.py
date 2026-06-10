"""`pmb hooks ...` and `pmb mcp ...` — lifecycle-hook install + MCP server.

Extracted from cli/main.py (no behavior change)."""

from __future__ import annotations

from typing import Optional

import typer
from rich.panel import Panel
from rich.table import Table

from pmb.cli._common import _agent_toggles_from_config, _humanize_time, console  # noqa: F401

hooks_app = typer.Typer(
    help="Install force-feeding session-start hooks into your agent's "
         "config so PMB context arrives BEFORE the model thinks.",
)


# ═══════════════════════════════════════════════════════════════════════
# pmb mcp serve — expose the MCP server over HTTP for team-shared mode.
# One persistent process on a homelab box / Tailscale node serves every
# developer's agent. Same workspace, same memory, no per-machine state.
# ═══════════════════════════════════════════════════════════════════════

mcp_app = typer.Typer(
    help="Run the MCP server. Stdio (per-developer) is the default; "
         "streamable-http exposes one shared instance to a team.",
)


@mcp_app.command("serve")
def mcp_serve_cmd(
    transport: str = typer.Option(
        "streamable-http", "--transport",
        help="streamable-http (HTTP for team-shared) or stdio (one-shot).",
    ),
    host: str = typer.Option(
        "127.0.0.1", "--host",
        help="Bind address. Use 0.0.0.0 to accept connections from the LAN / "
             "Tailscale mesh.",
    ),
    port: int = typer.Option(
        8765, "--port",
        help="Bind port (default 8765). Make sure it doesn't collide with "
             "`pmb dashboard`.",
    ),
    path: str = typer.Option(
        "/mcp", "--path",
        help="Mount path for the streamable-http transport.",
    ),
    workspace: Optional[str] = typer.Option(
        None, "--workspace", "-w",
        help="Force a specific workspace id (default: cwd-based detection).",
    ),
    bearer_token: Optional[str] = typer.Option(
        None, "--bearer-token", "--token",
        envvar="PMB_MCP_BEARER_TOKEN",
        help="Shared secret required in `Authorization: Bearer <token>`. "
             "Strongly recommended if `--host` is not 127.0.0.1.",
    ),
):
    """Run the PMB MCP server.

    Examples:

      # Local stdio (what `pmb connect` wires by default — usually no
      # need to run this manually)
      pmb mcp serve --transport stdio

      # Team-shared HTTP on Tailscale mesh
      pmb mcp serve --transport streamable-http --host 0.0.0.0 --port 8765 \\
                   --workspace team --bearer-token <secret>

      # Then on each developer's machine:
      pmb connect claude-code --remote http://memo.local:8765/mcp \\
                              --bearer-token <secret>
    """
    import os

    if transport not in ("stdio", "streamable-http", "http", "https"):
        console.print(f"[red]Unknown transport {transport!r}.[/]")
        raise typer.Exit(2)
    if transport in ("http", "https"):
        transport = "streamable-http"

    # Hand off to the existing server entrypoint via env-vars so we share
    # one code path with `pmb-mcp` when it's spawned by an agent.
    if workspace:
        os.environ["PMB_WORKSPACE"] = workspace
    os.environ["PMB_MCP_TRANSPORT"] = transport
    os.environ["PMB_MCP_HOST"] = host
    os.environ["PMB_MCP_PORT"] = str(port)
    os.environ["PMB_MCP_PATH"] = path
    if bearer_token:
        os.environ["PMB_MCP_BEARER_TOKEN"] = bearer_token

    if transport == "streamable-http":
        # Issue #6 singleton: don't start a second heavy server on a live port.
        try:
            from pmb.mcp.registry import find_live_http
            _existing = find_live_http(host, port)
        except Exception:
            _existing = None
        if _existing:
            console.print(Panel.fit(
                f"A PMB MCP server is already running here:\n"
                f"  url: http://{host}:{port}{path}\n"
                f"  pid: {_existing.get('pid')}\n\n"
                f"Not starting a second (saves a model + LanceDB load).\n"
                f"Point clients at the URL above, or stop it first.\n"
                f"See: [cyan]pmb mcp status[/]",
                title="PMB · mcp serve — already running",
            ))
            return
        url = f"http://{host}:{port}{path}"
        auth = (
            f"[green]bearer-token enabled[/]"
            if bearer_token
            else "[yellow]UNAUTHENTICATED — bind to 127.0.0.1 or set --bearer-token[/]"
        )
        console.print(Panel.fit(
            f"PMB MCP server starting\n"
            f"  transport: streamable-http\n"
            f"  url:       {url}\n"
            f"  auth:      {auth}\n"
            f"  workspace: {os.environ.get('PMB_WORKSPACE', '(cwd-detected)')}\n\n"
            f"Wire an agent to it:\n"
            f"  [cyan]pmb connect claude-code --remote {url}"
            + (f" --bearer-token <secret>" if bearer_token else "") + "[/]\n\n"
            f"Stop with Ctrl-C.",
            title="PMB · mcp serve",
        ))
    else:
        console.print("[dim]Running stdio MCP server. Use Ctrl-D / Ctrl-C to stop.[/]")

    from pmb.mcp.server import main as _server_main
    _server_main()


@mcp_app.command("status")
def mcp_status_cmd():
    """Show PMB MCP servers currently registered as running (issue #6).

    stdio servers are spawned per-session by your agent; a shared one is
    started with `pmb mcp serve`. Each row shows the process + its memory so
    you can spot duplicate heavy servers eating RAM."""
    from pmb.mcp.registry import list_servers
    servers = list_servers(prune=True)
    if not servers:
        console.print("[yellow]No PMB MCP servers registered as running.[/]")
        console.print(
            "[dim]Agents spawn a stdio server per session; start a shared one "
            "with `pmb mcp serve`.[/]"
        )
        return
    table = Table(show_header=True, header_style="bold magenta",
                  title="PMB MCP servers")
    table.add_column("PID")
    table.add_column("Transport")
    table.add_column("Endpoint")
    table.add_column("Workspace")
    table.add_column("RSS", justify="right")
    table.add_column("Alive", justify="center")
    for s in servers:
        endpoint = (
            f"http://{s.get('host')}:{s.get('port')}{s.get('path') or ''}"
            if s.get("transport") == "streamable-http" else "stdio"
        )
        rss = f"{s['rss_mb']:.0f} MB" if s.get("rss_mb") is not None else "—"
        alive = "[green]✓[/]" if s.get("alive") else "[red]✗[/]"
        table.add_row(
            str(s.get("pid")), s.get("transport") or "?", endpoint,
            str(s.get("workspace") or "—"), rss, alive,
        )
    console.print(table)
    http_n = sum(1 for s in servers if s.get("transport") == "streamable-http")
    stdio_n = len(servers) - http_n
    console.print(
        f"\n[dim]{len(servers)} server(s): {stdio_n} stdio, {http_n} http. "
        f"Each stdio server holds its own model + LanceDB in RAM.[/]"
    )


@mcp_app.command("perf")
def mcp_perf_cmd(
    days: float = typer.Option(7.0, "--days", help="Look-back window in days."),
):
    """Per-tool MCP latency + error report (reads the mcp_calls table).

    The measurement loop for the token-diet / daemon work: shows whether tools
    actually got faster. p50/p95 in ms, error and client-timeout rates."""
    import sqlite3
    import time

    from pmb.core.workspace import detect_workspace
    ws = detect_workspace()
    if not ws.db_path.exists():
        console.print("[yellow]No workspace database yet.[/]")
        return
    cutoff = time.time() - days * 86400.0
    try:
        with sqlite3.connect(str(ws.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT tool_name, duration_ms, success, outcome, client_timeout "
                "FROM mcp_calls WHERE timestamp >= ?", (cutoff,)).fetchall()
    except Exception as e:
        console.print(f"[yellow]No MCP perf data ({e}).[/]")
        return
    if not rows:
        console.print(f"[yellow]No MCP calls in the last {days:g} day(s).[/] "
                      "[dim]Perf is recorded once tools are called via MCP.[/]")
        return

    from collections import defaultdict
    agg: dict[str, dict] = defaultdict(
        lambda: {"durs": [], "err": 0, "to": 0, "n": 0})
    for r in rows:
        a = agg[r["tool_name"]]
        a["n"] += 1
        a["durs"].append(float(r["duration_ms"] or 0.0))
        if r["outcome"] == "error" or (r["success"] is not None and not r["success"]):
            a["err"] += 1
        if r["client_timeout"]:
            a["to"] += 1

    def _pct(xs, p):
        if not xs:
            return 0.0
        s = sorted(xs)
        i = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
        return s[i]

    table = Table(show_header=True, header_style="bold magenta",
                  title=f"MCP tool performance (last {days:g}d)")
    table.add_column("Tool")
    table.add_column("Calls", justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("Err%", justify="right")
    table.add_column("Timeouts", justify="right")
    for name, a in sorted(agg.items(), key=lambda kv: -kv[1]["n"]):
        err_pct = 100.0 * a["err"] / a["n"] if a["n"] else 0.0
        table.add_row(
            name, str(a["n"]),
            f"{_pct(a['durs'], 50):.0f}", f"{_pct(a['durs'], 95):.0f}",
            (f"[red]{err_pct:.0f}[/]" if err_pct >= 5 else f"{err_pct:.0f}"),
            str(a["to"]) if a["to"] else "—",
        )
    console.print(table)
    total = sum(a["n"] for a in agg.values())
    console.print(f"[dim]{total} call(s) across {len(agg)} tool(s).[/]")


@hooks_app.command("install")
def hooks_install_cmd(
    agent: str = typer.Argument(
        ...,
        help="claude-code | codex | cursor (cursor: unsupported, prints why).",
    ),
):
    """Install PMB's lifecycle hooks for AGENT.

    For claude-code this wires THREE hooks:
      • UserPromptSubmit → pmb prepare-context  (auto-recall: inject
        lessons / decisions / recall / project context per turn)
      • SessionStart     → pmb session-restore  (rebuild context after a
        compaction / resume)
      • Stop             → pmb lesson-followcheck (infer which lessons
        were actually followed — feeds the adherence dashboard)

    For codex it wires the per-turn context injector only.

    Examples:
        pmb hooks install claude-code
        pmb hooks install codex
    """
    from pmb.cli.hooks import install_hook
    try:
        result = install_hook(agent)
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2)

    act = result.get("action")
    if act == "not_supported":
        console.print(
            f"[yellow]Hooks not supported for {result['agent']}.[/]\n"
            f"  {result.get('reason','')}"
        )
        return
    if act == "mcp_only":
        console.print(Panel.fit(
            f"[yellow]Ambient not available for [bold]{result['agent']}[/][/]\n"
            f"{result.get('reason','')}\n\n"
            f"[dim]Run [cyan]pmb hooks capabilities[/] to see what each agent "
            f"supports.[/]",
            title="PMB · MCP-only agent",
        ))
        return

    # claude-code returns a multi-event `actions` list; codex returns a
    # single `action`/`command`.
    if result.get("actions"):
        lines = "\n".join(
            f"  • {a['event']}: {a['action']}" for a in result["actions"]
        )
        console.print(Panel.fit(
            f"[green]Hooks installed[/] for [bold]{result['agent']}[/]\n"
            f"  config: {result.get('path','?')}\n{lines}\n\n"
            "Restart the agent. Now, every turn:\n"
            "  · context is injected BEFORE the model thinks (auto-recall)\n"
            "  · after a compaction it rebuilds where you left off\n"
            "  · each action is observed, and the turn is journaled if the\n"
            "    agent didn't record it itself (ambient memory, on by default)\n"
            "  · at turn end it scores lesson follow-through\n"
            "— the adherence problem, handled at the protocol level.\n"
            "  [dim](ambient off: pmb config set autowrite.enabled false ·\n"
            "   undo its writes: pmb forget-auto)[/]",
            title="PMB · hooks installed",
        ))
    else:
        colour = "green" if act in ("installed", "created", "updated") else "yellow"
        console.print(Panel.fit(
            f"[{colour}]Hook {act}[/] for [bold]{result['agent']}[/]\n"
            f"  config:  {result.get('path','?')}\n"
            f"  command: {result.get('command','?')}\n\n"
            "Restart the agent. Every new session now reads PMB BEFORE it\n"
            "starts thinking — adherence problem solved at the protocol level.",
            title="PMB · hook installed",
        ))


@hooks_app.command("list")
def hooks_list_cmd():
    """Show which lifecycle hooks are currently installed."""
    from pmb.cli.hooks import list_installed
    rows = list_installed()
    t = Table(show_header=True, header_style="bold magenta", title="PMB hooks")
    t.add_column("Agent"); t.add_column("Event"); t.add_column("Installed")
    t.add_column("Path")
    for r in rows:
        mark = "[green]✓[/]" if r["installed"] else "[dim]–[/]"
        t.add_row(r["agent"], r.get("event", "?"), mark, r["path"])
    console.print(t)


@hooks_app.command("capabilities")
def hooks_capabilities_cmd():
    """Show which ambient mechanism each agent supports.

    Ambient memory needs to OBSERVE the agent's actions. Hosts differ:
    Claude Code has rich hooks, Codex has a parseable action log, and most
    others expose only MCP (so auto-recall works but ambient auto-write
    can't — nothing to observe).
    """
    from pmb.cli.hooks import capability_report
    t = Table(show_header=True, header_style="bold magenta",
              title="PMB ambient capabilities")
    t.add_column("Agent"); t.add_column("Mechanism"); t.add_column("Ambient")
    t.add_column("What works")
    cap_colour = {"hooks": "green", "rollout": "cyan", "mcp-only": "yellow"}
    for r in capability_report():
        amb = "[green]✓ full[/]" if r["ambient"] else "[yellow]recall only[/]"
        col = cap_colour.get(r["capability"], "dim")
        t.add_row(r["agent"], f"[{col}]{r['capability']}[/]", amb, r["details"])
    console.print(t)
    console.print(
        "\n[dim]hooks = Claude Code · rollout = Codex · mcp-only = the rest "
        "(auto-recall via `pmb connect`, no action observation).[/]"
    )


@hooks_app.command("uninstall")
def hooks_uninstall_cmd(
    agent: str = typer.Argument(..., help="claude-code | codex"),
):
    """Remove PMB's session-start hook for AGENT."""
    from pmb.cli.hooks import uninstall_hook
    try:
        r = uninstall_hook(agent)
    except ValueError as e:
        console.print(f"[red]{e}[/]"); raise typer.Exit(2)
    if r["action"] == "not_installed":
        console.print(f"[dim]Nothing to remove for {r['agent']}.[/]")
    else:
        console.print(f"[green]Removed[/] hook for {r['agent']} ({r.get('path','?')})")


