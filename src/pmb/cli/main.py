"""
PMB CLI.

Common commands:
- pmb                                 - show this workspace's status dashboard
- pmb init [--name NAME]              - initialize a workspace in the current folder
- pmb stats                           - workspace statistics
- pmb list [--limit N] [--type T]     - list recent events
- pmb remember "query" "response"     - add a Q/A pair
- pmb recall "query" [-k 5]           - search memory
- pmb pin ULID                        - pin an event (kept, never auto-archived)
- pmb forget ULID                     - archive an event
- pmb workspaces                      - list all workspaces
- pmb workspace use NAME              - switch the default workspace
"""

from __future__ import annotations

import typer

import pmb.cli.commands.capture  # noqa: F401  (registers capture root commands)

# The root Typer app, console, and shared helpers live in pmb.cli._common so
# the per-area command modules can register onto the same app without an import
# cycle. (Re-exported here for the many in-file commands that still use them.)
from pmb.cli._common import (  # noqa: E402
    app,
)
from pmb.cli.commands.graph import graph_app

app.add_typer(graph_app, name="graph")
from pmb.cli.commands.health import health_app

app.add_typer(health_app, name="health")
from pmb.cli.commands.config import config_app

app.add_typer(config_app, name="config")
# Ollama subcommand - for fully-local LLM ops (no Anthropic / OpenAI key)
from pmb.cli.ollama_cmd import app as ollama_app

app.add_typer(ollama_app, name="ollama")
import pmb.cli.commands.maintenance  # noqa: F401  (registers maintenance root commands)
from pmb.cli.commands.workspace import workspace_app

app.add_typer(workspace_app, name="workspace")
from pmb.cli.commands.index import index_app

app.add_typer(index_app, name="index")
from pmb.cli.commands.snapshot import snapshot_app

app.add_typer(snapshot_app, name="snapshot")
import pmb.cli.commands.ambient  # noqa: F401  (registers ambient @app.command root commands)
import pmb.cli.commands.manage  # noqa: F401  (registers manage root commands)
from pmb.cli.commands.hooks import hooks_app, mcp_app

app.add_typer(hooks_app, name="hooks")
app.add_typer(mcp_app, name="mcp")
from pmb.cli.commands.daemon import daemon_app

app.add_typer(daemon_app, name="daemon")
from pmb.cli.commands.lang import lang_app

app.add_typer(lang_app, name="lang")


# ── Group the flat command list into ordered help panels ─────────────────────
# `pmb --help` otherwise dumps ~70 commands as one wall. These panels turn it
# into a guided map. Centralized here (one map) so adding a command is one line,
# not a per-decorator `rich_help_panel=` scattered across files. Anything not
# listed falls to "More" rather than being dropped.
_HELP_COMMAND_PANELS: list[tuple[str, list[str]]] = [
    ("✦ Setup & agents", [
        "setup", "connect", "doctor", "ambient-watch", "codex-notify"]),
    ("Capture", [
        "remember", "fact", "note", "learn", "distill", "lessons", "watch",
        "sync", "import", "tag", "untag", "tags", "tagged", "ttl"]),
    ("Recall & explore", [
        "recall", "why", "overview", "audit", "timeline", "insights", "digest",
        "history", "correlate", "reminders", "goals", "session"]),
    ("Maintain memory", [
        "pin", "forget", "forget-topic", "forget-auto", "decay", "declutter",
        "dedupe", "prune-expired", "prune-graph", "compact", "reindex",
        "regraph", "repair-keyed", "rehearse", "consolidate", "reflect",
        "arcs", "feedback", "migrate-workspaces", "schedule"]),
    ("Workspaces & data", [
        "init", "workspaces", "stats", "list", "export", "dashboard", "tui",
        "tune", "warmup"]),
    ("Agent internals", [
        "prepare-context", "auto-context", "session-restore",
        "lesson-followcheck", "track-action", "autowrite"]),
]
_HELP_GROUP_PANELS: dict[str, str] = {
    "hooks": "✦ Setup & agents", "mcp": "✦ Setup & agents",
    "daemon": "✦ Setup & agents",
    "workspace": "Workspaces & data", "index": "Workspaces & data",
    "snapshot": "Workspaces & data",
    "graph": "More", "health": "More", "config": "More",
    "ollama": "More", "lang": "More",
}


def _cmd_name(info) -> str:
    cb = getattr(info, "callback", None)
    return info.name or (cb.__name__.replace("_", "-") if cb else "")


def _organize_help() -> None:
    """Assign each command/group a rich help panel + reorder so the panels
    render in the declared order (Rich uses first-appearance order)."""
    by_name = {_cmd_name(i): i for i in app.registered_commands}
    ordered, seen = [], set()
    for panel, names in _HELP_COMMAND_PANELS:
        for nm in names:
            info = by_name.get(nm)
            if info is not None and nm not in seen:
                info.rich_help_panel = panel
                ordered.append(info)
                seen.add(nm)
    for info in app.registered_commands:        # anything unlisted -> "More"
        if _cmd_name(info) not in seen:
            info.rich_help_panel = "More"
            ordered.append(info)
    app.registered_commands = ordered
    for g in app.registered_groups:             # sub-typers join their panel
        panel = _HELP_GROUP_PANELS.get(g.name)
        if panel:
            g.rich_help_panel = panel


_organize_help()


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context):
    """PMB - local-first memory for AI agents. Bare `pmb` opens the command
    palette (in a terminal) or the status dashboard (piped)."""
    # Any subcommand (and `--help`, which Click handles before this body)
    # passes through untouched.
    if ctx.invoked_subcommand is not None:
        return

    import sys

    from pmb.cli._common import console

    # Bare `pmb` in a real terminal opens the command palette: type what you
    # want, the list filters live by intent, Enter runs it. Non-interactive
    # (piped, CI, scripts) or any palette error falls back to the static status
    # dashboard, so `pmb | cat` and automation keep their plain output.
    try:
        if sys.stdin.isatty() and console.is_terminal:
            # First ever run: the ✦ boot animation + a one-screen welcome, then
            # stop. Every run after that goes straight to the command palette.
            from pmb.cli.intro import maybe_first_run
            if maybe_first_run(console):
                return
            from pmb.cli.palette import command_catalog, run_palette
            chosen = run_palette(console, command_catalog(app))
            if chosen:
                import subprocess
                console.print(f"[dim]>[/] [magenta]✦[/] [bold]pmb {chosen}[/]")
                subprocess.run([sys.argv[0], *chosen.split()])
                return
            # Esc / Ctrl-C falls through to the status dashboard below.
    except Exception:
        pass
    from pmb.cli.status_panel import render_status
    render_status(console)


if __name__ == "__main__":
    app()
