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


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context):
    """PMB — local-first memory for AI agents. Run with no command for status."""
    # Only render the status dashboard for a bare `pmb`; any subcommand
    # (and `--help`, which Click handles before this body) passes through.
    if ctx.invoked_subcommand is None:
        from pmb.cli._common import console
        from pmb.cli.status_panel import render_status
        render_status(console)


if __name__ == "__main__":
    app()
