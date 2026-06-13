"""`pmb graph ...` - inspect the association graph (entities + edges)."""

from __future__ import annotations

import typer
from rich.panel import Panel
from rich.table import Table

from pmb.cli._common import console
from pmb.core.engine import Engine

graph_app = typer.Typer(help="Inspect the association graph (entities + edges).")


@graph_app.command("stats")
def graph_stats():
    """Counts of entities and edges in this workspace."""
    eng = Engine()
    s = eng.graph_stats()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric"); table.add_column("Value", justify="right")
    table.add_row("Entities", str(s["n_entities"]))
    table.add_row("Edges", str(s["n_edges"]))
    console.print(table)
    if s["by_kind"]:
        kt = Table(show_header=True, header_style="bold magenta", title="By kind")
        kt.add_column("Kind"); kt.add_column("Count", justify="right")
        for k, n in sorted(s["by_kind"].items(), key=lambda x: -x[1]):
            kt.add_row(k, str(n))
        console.print(kt)


@graph_app.command("top")
def graph_top(
    kind: str | None = typer.Option(None, "--kind", help="file | tech | concept"),
    limit: int = typer.Option(20, "-n", "--limit"),
):
    """Most-mentioned entities."""
    eng = Engine()
    ents = eng.graph_top_entities(kind=kind, limit=limit)
    if not ents:
        console.print("[yellow]No entities yet. Add events or run `pmb graph rebuild`.[/]")
        return
    t = Table(show_header=True, header_style="bold magenta")
    t.add_column("Kind"); t.add_column("Name"); t.add_column("Mentions", justify="right")
    for e in ents:
        t.add_row(e["kind"], e["name"], str(e["n_mentions"]))
    console.print(t)


@graph_app.command("neighbors")
def graph_neighbors(
    name: str = typer.Argument(..., help="Entity name to look up"),
    kind: str | None = typer.Option(None, "--kind"),
    top_k: int = typer.Option(10, "-k"),
):
    """Strongest neighbors of an entity by co-occurrence weight."""
    eng = Engine()
    r = eng.graph_neighbors(name, kind=kind, top_k=top_k)
    if not r["entity"]:
        console.print(f"[yellow]Entity {name!r} not in graph.[/]")
        return
    primary = r["entity"]
    console.print(Panel.fit(
        f"[bold]{primary['kind']}[/]: [cyan]{primary['name']}[/]\n"
        f"mentions: {primary['n_mentions']}",
        title="Entity",
    ))
    t = Table(show_header=True, header_style="bold magenta")
    t.add_column("Kind"); t.add_column("Name"); t.add_column("Weight", justify="right")
    for nbr in r["neighbors"]:
        e = nbr["entity"]
        t.add_row(e["kind"], e["name"], str(nbr["weight"]))
    console.print(t)


@graph_app.command("rebuild")
def graph_rebuild():
    """Reindex graph from all active events (run once after upgrading)."""
    eng = Engine()
    r = eng.graph_rebuild_from_events()
    console.print(
        f"[green]Reindexed[/] {r['n_events_indexed']} events. "
        f"Entities: {r['n_entities']}, Edges: {r['n_edges']}"
    )
