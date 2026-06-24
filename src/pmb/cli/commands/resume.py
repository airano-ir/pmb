"""`pmb resume ...` - committable, human-editable resume note.

A markdown snapshot of "where we are now" sourced from PMB's typed memory
(open goals, decisions, lessons, recent activity, project structure). Inspired
by Recall's `.recall/context.md` workflow but structured, not extractive.
"""
from __future__ import annotations

import typer
from rich.panel import Panel

from pmb.cli._common import app, console, loading  # noqa: F401

resume_app = typer.Typer(
    help="Resume note: committable markdown of 'where we are now'.",
)


@resume_app.command("save")
def resume_save(
    path: str = typer.Option(
        ".pmb/resume.md", "--path", help="Output markdown path.",
    ),
    project: str = typer.Option(
        "", "--project", help="Project root (default: cwd).",
    ),
):
    """Write `.pmb/resume.md` from the current memory state.

    Pulls open goals, recent decisions, lessons, recent activity, latest
    exploration conclusions, and project structure. Content below
    PMB-RESUME-MARKER in an existing file is preserved across regenerations,
    so hand-edited additions survive.
    """
    from pmb.core.engine import Engine
    from pmb.resume.builder import save_resume
    with loading("building resume note…"):
        res = save_resume(Engine(), path=path, project=project or None)
    if not res.get("saved"):
        console.print(f"[red]Error:[/] {res.get('error', 'unknown')}")
        raise typer.Exit(1)
    console.print(Panel.fit(
        f"[green]Wrote[/] {res['path']}  ({res['bytes']} bytes)\n"
        "Commit it to share session continuity with your team.",
        title="PMB · resume save",
    ))


@resume_app.command("show")
def resume_show(
    path: str = typer.Option(
        ".pmb/resume.md", "--path", help="Resume markdown path.",
    ),
):
    """Print the current resume note to stdout."""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        console.print(f"[yellow]Not found:[/] {p}. Run `pmb resume save` first.")
        raise typer.Exit(1)
    console.print(p.read_text(encoding="utf-8", errors="replace"))


@resume_app.command("install")
def resume_install(
    path: str = typer.Option(
        ".pmb/resume.md", "--path", help="Output markdown path.",
    ),
):
    """Enable auto-write of `.pmb/resume.md` at every turn end (Stop hook).

    Sets `resume.auto_save_enabled=true` and stores the output path. The
    existing PMB Stop hook picks this up; no settings.json change needed.
    """
    from pmb.core.engine import Engine
    eng = Engine()
    try:
        eng.config.set("resume.auto_save_enabled", True)
        eng.config.set("resume.path", path)
    except Exception as e:
        console.print(f"[red]Could not write config:[/] {e}")
        raise typer.Exit(1) from None
    console.print(Panel.fit(
        f"[green]Auto-save enabled.[/]\n"
        f"  path: {path}\n"
        "Resume will refresh at every turn end.",
        title="PMB · resume install",
    ))
