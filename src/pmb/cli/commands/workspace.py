"""`pmb workspace ...` — extracted from cli/main.py (no behavior change)."""

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

workspace_app = typer.Typer(
    help="Git-backed workspace sync: push / pull / clone your memory to any remote."
)


def _sync_for_current_workspace():
    """Build a WorkspaceGitSync bound to the current workspace's storage dir."""
    from pmb.core.git_sync import WorkspaceGitSync
    ws = detect_workspace()
    ws.ensure_dirs()
    return WorkspaceGitSync(ws.storage_dir), ws


@workspace_app.command("init")
def workspace_init(
    remote: Optional[str] = typer.Option(
        None, "--remote", "-r",
        help="Git remote URL (e.g. git@github.com:you/my-memory.git). Optional - "
             "you can add it later or push locally only.",
    ),
    branch: str = typer.Option("main", "--branch", "-b"),
    lean: bool = typer.Option(
        False, "--lean",
        help="Don't track derived caches (bm25, vocab bridges). Smaller repo; "
             "rebuilt locally on first recall after a clone/pull.",
    ),
):
    """Turn the current workspace into a git repo (optionally with a remote)."""
    sync, ws = _sync_for_current_workspace()
    res = sync.init(remote=remote, branch=branch, include_cache=not lean)
    color = "green" if res.ok else "red"
    console.print(Panel.fit(
        f"[{color}]{res.detail}[/]\n"
        f"workspace: [cyan]{ws.name}[/] ({ws.id[:12]})\n"
        f"storage:   {ws.storage_dir}\n"
        f"remote:    {remote or '- (none yet)'}\n"
        f"mode:      {'lean (caches rebuilt locally)' if lean else 'full (caches synced)'}",
        title="workspace init",
    ))
    if not res.ok:
        raise typer.Exit(1)
    if remote:
        console.print("[dim]Next: `pmb workspace push` to upload your memory.[/]")


@workspace_app.command("push")
def workspace_push(
    remote: str = typer.Option("origin", "--remote", "-r"),
    branch: str = typer.Option("main", "--branch", "-b"),
    message: Optional[str] = typer.Option(None, "--message", "-m"),
    lean: bool = typer.Option(False, "--lean", help="Exclude derived caches."),
):
    """Commit and push the current workspace's memory to its git remote."""
    sync, ws = _sync_for_current_workspace()
    res = sync.push(remote=remote, branch=branch, message=message, include_cache=not lean)
    color = "green" if res.ok else "red"
    console.print(f"[{color}]workspace push:[/] {res.detail}")
    if not res.ok:
        raise typer.Exit(1)


@workspace_app.command("pull")
def workspace_pull(
    remote: str = typer.Option("origin", "--remote", "-r"),
    branch: str = typer.Option("main", "--branch", "-b"),
    ours: bool = typer.Option(
        False, "--ours",
        help="On binary conflict keep LOCAL memory (default: remote wins).",
    ),
):
    """Pull workspace memory from the git remote (remote wins on conflict)."""
    sync, ws = _sync_for_current_workspace()
    res = sync.pull(remote=remote, branch=branch, strategy="ours" if ours else "theirs")
    color = "green" if res.ok else "red"
    console.print(f"[{color}]workspace pull:[/] {res.detail}")
    if res.extra and res.extra.get("hint"):
        console.print(f"[dim]{res.extra['hint']}[/]")
    if not res.ok:
        raise typer.Exit(1)
    console.print("[dim]If you ran with --lean previously, the next recall rebuilds "
                  "the BM25 index automatically.[/]")


@workspace_app.command("status")
def workspace_status():
    """Show git status of the current workspace (branch, remote, dirty)."""
    sync, ws = _sync_for_current_workspace()
    res = sync.status()
    if not res.ok:
        console.print(f"[yellow]{res.detail}[/]")
        return
    e = res.extra or {}
    console.print(Panel.fit(
        f"workspace: [cyan]{ws.name}[/] ({ws.id[:12]})\n"
        f"branch:    {e.get('branch') or '-'}\n"
        f"remote:    {e.get('remote') or '- (none)'}\n"
        f"state:     {'[yellow]dirty[/]' if e.get('dirty') else '[green]clean[/]'} "
        f"({e.get('n_changed', 0)} change(s))\n"
        f"last:      {e.get('last_commit') or '-'}",
        title="workspace status",
    ))


@workspace_app.command("export")
def workspace_export(
    out: str = typer.Argument(..., help="Output bundle path, e.g. memory.enc"),
    key_file: Optional[Path] = typer.Option(
        None, "--key-file",
        help="Encrypt with a raw 32-byte key file instead of a passphrase.",
    ),
):
    """Encrypt the current workspace into a single portable bundle.

    Safe to back up anywhere - Dropbox, a USB stick, even a PUBLIC git repo:
    the bundle is authenticated-encrypted (AES + HMAC), so the storage host
    only ever sees ciphertext. Restore with `pmb workspace import`.
    """
    from pmb.core.encryption import export_workspace, EncryptionUnavailable
    _, ws = _sync_for_current_workspace()
    passphrase = None
    if not key_file:
        import getpass
        passphrase = getpass.getpass("Passphrase to encrypt workspace: ")
        confirm = getpass.getpass("Confirm passphrase: ")
        if passphrase != confirm:
            console.print("[red]Passphrases do not match.[/]")
            raise typer.Exit(2)
        if not passphrase:
            console.print("[red]Empty passphrase rejected.[/]")
            raise typer.Exit(2)
    try:
        res = export_workspace(ws.storage_dir, Path(out),
                               passphrase=passphrase, key_file=key_file)
    except EncryptionUnavailable as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2)
    color = "green" if res.ok else "red"
    console.print(f"[{color}]workspace export:[/] {res.detail}")
    if not res.ok:
        raise typer.Exit(1)


@workspace_app.command("import")
def workspace_import(
    bundle: str = typer.Argument(..., help="Encrypted bundle path"),
    name: str = typer.Argument(..., help="Local workspace id to create"),
    key_file: Optional[Path] = typer.Option(
        None, "--key-file", help="Decrypt with the matching 32-byte key file.",
    ),
):
    """Decrypt a bundle into ~/.pmb/workspaces/<name>."""
    from pmb.core.encryption import import_workspace, EncryptionUnavailable
    from pmb.core.workspace import DEFAULT_PMB_HOME
    import os as _os
    pmb_home = Path(_os.environ.get("PMB_HOME", DEFAULT_PMB_HOME))
    dest = pmb_home / "workspaces" / name
    passphrase = None
    if not key_file:
        import getpass
        passphrase = getpass.getpass("Passphrase to decrypt workspace: ")
    try:
        res = import_workspace(Path(bundle), dest,
                               passphrase=passphrase, key_file=key_file)
    except EncryptionUnavailable as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2)
    color = "green" if res.ok else "red"
    console.print(f"[{color}]workspace import:[/] {res.detail}")
    if not res.ok:
        raise typer.Exit(1)
    console.print(
        f"[dim]Use it: [bold]PMB_WORKSPACE={name} pmb stats[/][/]"
    )


@workspace_app.command("clone")
def workspace_clone(
    url: str = typer.Argument(..., help="Git URL of a workspace repo"),
    name: str = typer.Argument(..., help="Local workspace id to create"),
):
    """Clone a remote workspace into ~/.pmb/workspaces/<name>."""
    from pmb.core.git_sync import clone_workspace
    from pmb.core.workspace import DEFAULT_PMB_HOME
    import os as _os
    pmb_home = Path(_os.environ.get("PMB_HOME", DEFAULT_PMB_HOME))
    res = clone_workspace(url, name, pmb_home)
    color = "green" if res.ok else "red"
    console.print(f"[{color}]workspace clone:[/] {res.detail}")
    if not res.ok:
        raise typer.Exit(1)
    console.print(
        f"[dim]Use it: [bold]PMB_WORKSPACE={name} pmb stats[/] or "
        f"[bold]pmb connect claude-code --workspace {name}[/][/]"
    )


