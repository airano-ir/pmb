"""
PMB CLI.

Commands:
- pmb init [--name NAME]              - инициализация workspace в текущей папке
- pmb stats                           - статистика workspace
- pmb list [--limit N] [--type T]     - список последних событий
- pmb remember "query" "response"     - добавить Q/A
- pmb recall "query" [-k 5]           - поиск
- pmb pin ULID                        - закрепить
- pmb forget ULID                     - архивировать
- pmb workspaces                      - все workspaces
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markup import escape as esc

from pmb.core.engine import Engine
from pmb.core.workspace import detect_workspace, list_workspaces, Workspace

# The root Typer app, console, and shared helpers live in pmb.cli._common so
# the per-area command modules can register onto the same app without an import
# cycle. (Re-exported here for the many in-file commands that still use them.)
from pmb.cli._common import (  # noqa: E402
    app, console, _humanize_time, _open_config, _agent_toggles_from_config,
    _parse_duration, _apply_ttl,
)


import pmb.cli.commands.capture  # noqa: F401  (registers capture root commands)

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
import pmb.cli.commands.manage  # noqa: F401  (registers manage root commands)

import pmb.cli.commands.ambient  # noqa: F401  (registers ambient @app.command root commands)


from pmb.cli.commands.hooks import hooks_app, mcp_app
app.add_typer(hooks_app, name="hooks")
app.add_typer(mcp_app, name="mcp")


if __name__ == "__main__":
    app()
