"""`pmb snapshot ...` - extracted from cli/main.py (no behavior change)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.markup import escape as esc
from rich.table import Table

from pmb.cli._common import (  # noqa: F401
    _agent_toggles_from_config,
    _humanize_time,
    _open_config,
    app,
    console,
)
from pmb.core.workspace import detect_workspace

snapshot_app = typer.Typer(help="Local, offline snapshots of your workspace (no cloud).")


def _snapshots_dir(ws) -> Path:
    return ws.storage_dir / "snapshots"


def _checkpoint_ws_sqlite(ws) -> None:
    """Best-effort WAL checkpoint so the copied events.sqlite is complete."""
    import sqlite3 as _sq
    db = ws.db_path
    if not db.exists():
        return
    try:
        con = _sq.connect(str(db), timeout=2.0)
        try:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            con.close()
    except Exception:
        pass


@snapshot_app.command("create")
def snapshot_create(
    note: str | None = typer.Option(None, "--note", "-m", help="Label for this snapshot"),
):
    """Copy the current workspace to a timestamped local snapshot."""
    import shutil
    ws = detect_workspace()
    ws.ensure_dirs()
    _checkpoint_ws_sqlite(ws)
    snaps = _snapshots_dir(ws)
    snaps.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(time.time()))
    dest = snaps / stamp
    if dest.exists():
        console.print(f"[red]Snapshot {stamp} already exists. Try again in a second.[/]")
        raise typer.Exit(1)
    shutil.copytree(ws.storage_dir, dest, ignore=shutil.ignore_patterns("snapshots"))
    (dest / "snapshot.json").write_text(json.dumps({
        "id": stamp, "created_at": _humanize_time(time.time()),
        "note": note or "", "workspace": ws.id,
    }, indent=2), encoding="utf-8")
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    console.print(f"[green]Snapshot[/] [cyan]{stamp}[/] created ({size/1024:.0f} KB)"
                  + (f" - {esc(note)}" if note else ""))
    console.print(f"[dim]{dest}[/]")


@snapshot_app.command("list")
def snapshot_list():
    """List local snapshots."""
    ws = detect_workspace()
    snaps = _snapshots_dir(ws)
    items = sorted([d for d in snaps.iterdir() if d.is_dir()], reverse=True) if snaps.exists() else []
    if not items:
        console.print("[yellow]No snapshots yet.[/] Create one: [cyan]pmb snapshot create[/]")
        return
    t = Table(show_header=True, header_style="bold magenta", title=f"Snapshots ({len(items)})")
    t.add_column("ID", style="cyan"); t.add_column("Created", style="dim")
    t.add_column("Size", justify="right"); t.add_column("Note")
    for d in items:
        man = {}
        mf = d / "snapshot.json"
        if mf.exists():
            try:
                man = json.loads(mf.read_text(encoding="utf-8"))
            except Exception:
                pass
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        t.add_row(d.name, man.get("created_at", "?"), f"{size/1024:.0f} KB", esc(man.get("note", "")))
    console.print(t)


@snapshot_app.command("restore")
def snapshot_restore(
    snapshot_id: str = typer.Argument(..., help="Snapshot ID (see `pmb snapshot list`)"),
    yes: bool = typer.Option(False, "--yes", "-y"),
):
    """Restore the workspace from a snapshot (current state is backed up first)."""
    import shutil
    ws = detect_workspace()
    src = _snapshots_dir(ws) / snapshot_id
    if not src.is_dir():
        console.print(f"[red]No snapshot[/] {snapshot_id}. See `pmb snapshot list`.")
        raise typer.Exit(2)
    console.print(f"[yellow]This replaces current memory with snapshot {snapshot_id}.[/]")
    if not yes and not typer.confirm("Continue? (current state is auto-backed-up)"):
        console.print("[yellow]Cancelled.[/]")
        return
    _checkpoint_ws_sqlite(ws)
    safety = _snapshots_dir(ws) / f"pre-restore-{time.strftime('%Y%m%d-%H%M%S', time.gmtime(time.time()))}"
    shutil.copytree(ws.storage_dir, safety, ignore=shutil.ignore_patterns("snapshots"))
    for item in src.iterdir():
        if item.name == "snapshot.json":
            continue
        target = ws.storage_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    console.print(f"[green]Restored[/] snapshot {snapshot_id}. "
                  f"[dim](previous state saved as {safety.name})[/]")
    console.print("[dim]Restart any running PMB/agent so it reopens the restored DB.[/]")


