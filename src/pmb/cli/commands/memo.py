"""`pmb memory ...` - inspect / manage the Memory Delta ledger."""
from __future__ import annotations

import sqlite3
import time

import typer
from rich.panel import Panel
from rich.table import Table

from pmb.cli._common import app, console  # noqa: F401

memory_app = typer.Typer(help="Memory Delta ledger: handles, rehydrate, status.")


@memory_app.command("ledger")
def memory_ledger(
    session: str = typer.Option(
        "", "--session", help="Session id to inspect (default: most recent).",
    ),
    limit: int = typer.Option(40, "--limit"),
):
    """Show what PMB believes the agent has already seen this session."""
    from pmb.core.engine import Engine
    from pmb.memo.ledger import active_handles, ensure_table
    eng = Engine()
    sid = session
    with sqlite3.connect(eng.workspace.db_path) as conn:
        ensure_table(conn)
        if not sid:
            row = conn.execute(
                "SELECT session_id FROM memory_ledger WHERE workspace_id=? "
                "ORDER BY last_seen_at DESC LIMIT 1", (eng.workspace.id,),
            ).fetchone()
            sid = row[0] if row else ""
        if not sid:
            console.print("[yellow]Empty ledger (no Memory Delta data yet).[/]")
            return
        rows = active_handles(conn, eng.workspace.id, sid, limit=limit)
    console.print(Panel.fit(
        f"session: {sid}   items: {len(rows)}",
        title="PMB · memory ledger",
    ))
    if not rows:
        console.print("[dim]No handles in this session.[/]")
        return
    t = Table(show_header=True, header_style="bold magenta")
    t.add_column("handle"); t.add_column("kind"); t.add_column("ref_id")
    t.add_column("age (s)", justify="right")
    now = time.time()
    for r in rows:
        t.add_row(r["handle"], r["kind"], str(r["ref_id"])[:20],
                  f"{int(now - r['last_seen_at'])}")
    console.print(t)


@memory_app.command("rehydrate")
def memory_rehydrate(
    session: str = typer.Argument(..., help="Session id to clear."),
):
    """Wipe a session's ledger (Context Rebase). Use this manually if a /compact
    happened and the SessionStart hook didn't catch it."""
    from pmb.core.engine import Engine
    from pmb.memo.ledger import ensure_table, rehydrate
    eng = Engine()
    with sqlite3.connect(eng.workspace.db_path) as conn:
        ensure_table(conn)
        n = rehydrate(conn, eng.workspace.id, session)
        conn.commit()
    console.print(f"[green]Cleared {n} ledger row(s)[/] for session {session}.")
