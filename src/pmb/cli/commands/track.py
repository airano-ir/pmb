"""`pmb track ...` - git-tied semantic change tracking + module purpose summaries.

Layers a semantic memory on top of the structural `pmb index project`:
  * `pmb track changes`  - summarise the INTENT of new git commits (the "why")
  * `pmb track modules`  - one-line "what this file does" per indexed module
  * `pmb track install`  - git post-commit hook so `changes` runs automatically
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.panel import Panel

from pmb.cli._common import (  # noqa: F401
    app,
    console,
    loading,
)

track_app = typer.Typer(
    help="Semantic project tracking: what changed and WHY (git-tied), plus per-module purpose."
)

# Marker so `install` is idempotent and never clobbers a user's existing hook.
_HOOK_MARKER = "# >>> PMB track (pmb track install) >>>"
_HOOK_END = "# <<< PMB track <<<"


@track_app.command("changes")
def track_changes_cmd(
    path: str = typer.Argument(".", help="Path inside the git repo (default: current dir)."),
    since: str = typer.Option(
        None, "--since",
        help="Track commits after this ref/SHA (default: last cursor, else last --max-commits).",
    ),
    backend: str = typer.Option(
        "auto", "--backend",
        help="LLM backend: auto | claude | anthropic | ollama.",
    ),
    model: str = typer.Option(
        None, "--model",
        help="Model alias/id. Empty = backend default (Haiku).",
    ),
    max_commits: int = typer.Option(
        20, "--max-commits", help="Cap on commits processed per run.",
    ),
):
    """Read new git commits, summarise the INTENT of each (Haiku by default),
    and store it as memory linked to the files it touched.

    Layers on git: keeps the WHY, not the raw diff. Idempotent via a per-repo
    cursor, so re-running only processes commits made since last time.

    Examples:
      pmb track changes
      pmb track changes --since HEAD~5
      pmb track changes --backend ollama        # fully offline
    """
    from pmb.core.engine import Engine
    from pmb.ingest.track import track_changes
    eng = Engine()
    with loading("reading commits + summarising intent…"):
        result = track_changes(
            eng, Path(path), since=since, backend=backend,
            model=model, max_commits=max_commits,
        )
    if result.get("error"):
        console.print(f"[red]Error:[/] {result['error']}")
        raise typer.Exit(1)
    if result.get("n_commits", 0) == 0:
        console.print(f"[yellow]Nothing new[/] - {result.get('message', 'no commits')}")
        return
    lines = "\n".join(
        f"  [dim]{t['commit']}[/]  {t['subject'][:60]}"
        for t in result["tracked"][:10]
    )
    console.print(Panel.fit(
        f"[green]Tracked {result['n_commits']} commit(s)[/] in [bold]{result['project_name']}[/]\n"
        f"{lines}\n"
        f"  cursor → {result['cursor']}   ({result['duration_ms']} ms)",
        title="PMB · track changes",
    ))
    try:
        eng.wait_for_writes(timeout=120)
    except Exception:
        pass


@track_app.command("modules")
def track_modules_cmd(
    path: str = typer.Argument(".", help="Project root (run `pmb index project` first)."),
    backend: str = typer.Option(
        "auto", "--backend", help="LLM backend: auto | claude | anthropic | ollama.",
    ),
    model: str = typer.Option(
        None, "--model", help="Model alias/id. Empty = backend default (Haiku).",
    ),
    limit: int = typer.Option(
        200, "--limit", help="Max files to summarise this run (cost cap).",
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-summarise files that already have a summary.",
    ),
):
    """For each indexed file, store a one-line 'what this module does' summary.

    The structural index (`pmb index project`) says which symbols exist; this
    says WHY each file exists, so recall surfaces purpose next to structure.

    Examples:
      pmb index project && pmb track modules
      pmb track modules --backend ollama --limit 50
    """
    from pmb.core.engine import Engine
    from pmb.ingest.track import summarize_modules
    eng = Engine()
    with loading("summarising modules…"):
        result = summarize_modules(
            eng, Path(path), backend=backend, model=model,
            limit=limit, force=force,
        )
    if result.get("error"):
        console.print(f"[red]Error:[/] {result['error']}")
        raise typer.Exit(1)
    if result.get("n_summarized", 0) == 0:
        console.print(f"[yellow]Nothing to do[/] - {result.get('message', '')}")
        return
    over = result.get("n_over_cap", 0)
    cap_note = f"\n  [yellow]{over} more over --limit; run again to continue[/]" if over else ""
    console.print(Panel.fit(
        f"[green]Summarised {result['n_summarized']} module(s)[/]\n"
        f"  path:     {result['project_path']}\n"
        f"  duration: {result['duration_ms']} ms{cap_note}",
        title="PMB · track modules",
    ))
    try:
        eng.wait_for_writes(timeout=120)
    except Exception:
        pass


@track_app.command("install")
def track_install_cmd(
    path: str = typer.Argument(".", help="Path inside the git repo."),
    backend: str = typer.Option(
        "auto", "--backend", help="Backend the hook will use (auto | claude | anthropic | ollama).",
    ),
):
    """Install a git post-commit hook that runs `pmb track changes` after every
    commit, so change-intent memory stays current with no manual step.

    Runs in the background so it never delays your commit. Requires `pmb` on
    PATH. Will not clobber an existing post-commit hook.
    """
    from pmb.ingest.track import _repo_root
    root = _repo_root(path)
    if not root:
        console.print(f"[red]Error:[/] not a git repository: {Path(path).resolve()}")
        raise typer.Exit(1)

    hooks_dir = Path(root) / ".git" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook = hooks_dir / "post-commit"

    block = (
        f"{_HOOK_MARKER}\n"
        f"pmb track changes --backend {backend} >/dev/null 2>&1 &\n"
        f"{_HOOK_END}\n"
    )

    if hook.exists():
        existing = hook.read_text(encoding="utf-8", errors="replace")
        if _HOOK_MARKER in existing:
            console.print("[yellow]Already installed[/] - PMB block present in post-commit hook.")
            return
        # Respect the user's existing hook: append our block rather than clobber.
        new = existing.rstrip("\n") + "\n\n" + block
        hook.write_text(new, encoding="utf-8")
        console.print(Panel.fit(
            "[green]Appended[/] PMB track to your existing post-commit hook\n"
            f"  {hook}",
            title="PMB · track install",
        ))
    else:
        hook.write_text("#!/bin/sh\n" + block, encoding="utf-8")
        try:
            import os
            os.chmod(hook, 0o755)
        except OSError:
            pass
        console.print(Panel.fit(
            "[green]Installed[/] git post-commit hook\n"
            f"  {hook}\n"
            "  runs [bold]pmb track changes[/] in the background after each commit",
            title="PMB · track install",
        ))
