"""`pmb index ...` — extracted from cli/main.py (no behavior change)."""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import List, Optional

import typer
from rich.table import Table
from rich.panel import Panel
from rich.markup import escape as esc

from pmb.core.engine import Engine
from pmb.core.workspace import detect_workspace, list_workspaces, Workspace
from pmb.cli._common import app, console, _humanize_time, _open_config, _agent_toggles_from_config  # noqa: F401

index_app = typer.Typer(help="Index external content into PMB memory (PDFs, code projects).")


@index_app.command("pdf")
def index_pdf_cmd(
    path: str = typer.Argument(..., help="Path to a PDF file OR a directory."),
    recurse: bool = typer.Option(
        False, "--recurse", "-r",
        help="If PATH is a directory, find PDFs recursively.",
    ),
    importance: float = typer.Option(
        0.6, "--importance", help="Per-chunk importance (0.0-1.0)",
        min=0.0, max=1.0,
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Re-ingest even if a PDF with the same content hash is already in memory.",
    ),
):
    """Extract text from a PDF and persist it as searchable PMB events.

    Re-ingesting the same file is a no-op (idempotent via SHA1 of content).

    Examples:
      pmb index pdf paper.pdf
      pmb index pdf ~/Documents/research/ --recurse
      pmb index pdf paper.pdf --force      # force re-index
    """
    from pmb.ingest.pdf import ingest_pdf, ingest_pdfs
    from pmb.core.engine import Engine
    eng = Engine()
    p = Path(path)
    if p.is_dir():
        result = ingest_pdfs(eng, p, recurse=recurse,
                             importance=importance, force=force)
        console.print(Panel.fit(
            f"[green]Indexed {result['n_files']} PDF{'s' if result['n_files']!=1 else ''}[/]\n"
            f"  chunks: {result['n_chunks']}\n"
            f"  skipped (already indexed): {result['n_skipped']}",
            title="PMB · index pdf"
        ))
    else:
        result = ingest_pdf(eng, p, importance=importance, force=force)
        if result.get("error"):
            console.print(f"[red]Error:[/] {result['error']}")
            raise typer.Exit(1)
        if result.get("skipped"):
            console.print(f"[yellow]Skipped[/] {p.name}: {result.get('reason','')}")
        else:
            console.print(
                f"[green]Indexed[/] {p.name}: "
                f"{result['n_pages']} pages → {result['n_chunks']} chunks "
                f"({result['duration_ms']} ms, backend {result['backend']})"
            )
    # Block briefly so the user sees the count update in pmb stats.
    try:
        eng.wait_for_writes(timeout=120)
    except Exception:
        pass


@index_app.command("project")
def index_project_cmd(
    path: str = typer.Argument(
        ".", help="Path to the project root (default: current directory).",
    ),
    importance: float = typer.Option(
        0.55, "--importance", help="Per-file importance (0.0-1.0)",
        min=0.0, max=1.0,
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Re-index every file even if its SHA1 hash is already in memory.",
    ),
    max_files: int = typer.Option(
        5000, "--max-files",
        help="Cap on number of files to walk. Safety against giant repos.",
    ),
):
    """Scan a project directory and persist per-file structure (symbols + imports)
    as PMB events.

    The agent can then recall things like:
      • "where is the auth flow"
      • "what files import LanceDB"
      • "show me the recall pipeline"

    Re-running on the same project is a no-op for unchanged files
    (idempotent via per-file SHA1). Respects .gitignore.

    Examples:
      pmb index project                       # current dir
      pmb index project ~/code/myrepo
      pmb index project ~/code/myrepo --force # re-index all files
    """
    from pmb.ingest.project import index_project
    from pmb.core.engine import Engine
    eng = Engine()
    result = index_project(eng, Path(path),
                           importance=importance, force=force,
                           max_files=max_files)
    if result.get("error"):
        console.print(f"[red]Error:[/] {result['error']}")
        raise typer.Exit(1)
    by_lang = ", ".join(
        f"{k}={v}" for k, v in sorted(
            result.get("by_language", {}).items(), key=lambda x: -x[1]
        )[:8]
    )
    console.print(Panel.fit(
        f"[green]Indexed project[/] [bold]{result['project_name']}[/]\n"
        f"  path:       {result['project_path']}\n"
        f"  files seen: {result['n_files_seen']}\n"
        f"  indexed:    {result['n_indexed']}\n"
        f"  unchanged:  {result['n_skipped']}\n"
        f"  by lang:    {by_lang}\n"
        f"  duration:   {result['duration_ms']} ms",
        title="PMB · index project"
    ))
    try:
        eng.wait_for_writes(timeout=120)
    except Exception:
        pass


