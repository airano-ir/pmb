"""`pmb hooks ...` and `pmb mcp ...` — lifecycle-hook install + MCP server.

Extracted from cli/main.py (no behavior change)."""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.table import Table
from rich.panel import Panel
from rich.markup import escape as esc

from pmb.core.engine import Engine
from pmb.core.workspace import detect_workspace
from pmb.cli._common import console, _humanize_time, _agent_toggles_from_config  # noqa: F401

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
    import os, sys

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


