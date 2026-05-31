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
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from pmb.core.engine import Engine
from pmb.core.workspace import detect_workspace, list_workspaces, Workspace

# UTF-8 на Windows. Guard against non-stdlib stdout (Textual, pytest capture).
try:
    _enc = getattr(sys.stdout, "encoding", None)
    if _enc and _enc.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_show_locals=False)
console = Console()


def _humanize_time(ts: Optional[float]) -> str:
    if ts is None:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts))


@app.command()
def dashboard(
    port: int = typer.Option(8765, "--port", "-p", help="Port to bind to"),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address (default localhost)"),
):
    """Launch the local web dashboard.

    Opens http://127.0.0.1:8765 with:
      • workspace stats
      • event timeline
      • entity graph (filterable by kind)
      • narrative arcs
      • recall debugger (paste query → see scoring breakdown)
      • per-event inspector (entities, derived facts, edges, pin/archive/feedback)

    Bound to localhost by default. No external dependencies - pure stdlib.
    """
    eng = Engine()
    try:
        from pmb.dashboard.server import run_dashboard
    except Exception as e:
        console.print(f"[red]Failed to import dashboard:[/] {e}")
        return
    run_dashboard(eng, host=host, port=port)


@app.command()
def init(
    name: Optional[str] = typer.Option(None, "--name", help="Custom workspace name"),
):
    """Инициализировать workspace в текущей директории."""
    ws = detect_workspace()
    if name:
        ws.name = name
    ws.ensure_dirs()
    ws.save_meta()

    # Также положим .pmb/workspace.yaml в проект (опционально)
    local_config = Path.cwd() / ".pmb"
    local_config.mkdir(exist_ok=True)
    config_file = local_config / "workspace.yaml"
    if not config_file.exists():
        config_file.write_text(
            f"id: {ws.id}\nname: {ws.name}\n", encoding="utf-8"
        )

    console.print(Panel.fit(
        f"[bold green]Workspace initialized[/]\n\n"
        f"  ID:     [cyan]{ws.id}[/]\n"
        f"  Name:   [cyan]{ws.name}[/]\n"
        f"  Root:   {ws.root}\n"
        f"  Source: {ws.source}\n"
        f"  Storage: {ws.storage_dir}\n",
        title="PMB",
    ))


@app.command()
def warmup():
    """P1-1: Eagerly load model + BM25 + LanceDB so the next recall is fast.

    Without this the first query pays ~1-2s cold-start cost. Useful when
    integrating PMB into a latency-sensitive flow (voice assistant) - call
    `pmb warmup` at startup and the actual user-facing query is hot.
    """
    eng = Engine()
    console.print("[dim]Warming up PMB (model + BM25 + LanceDB)...[/]")
    result = eng.warmup(with_first_query=True)
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Stage")
    table.add_column("ms")
    table.add_row("Model load",       f"{result['model_load_ms']:.1f}")
    table.add_row("BM25 load",        f"{result['bm25_load_ms']:.1f}")
    table.add_row("LanceDB open",     f"{result['lance_open_ms']:.1f}")
    table.add_row("First probe query", f"{result['first_query_ms']:.1f}")
    table.add_row("[bold]Total[/]",   f"[bold]{result['total_ms']:.1f}[/]")
    console.print(table)
    console.print("[green]Engine warm.[/] Next recall will be fast.")


@app.command()
def stats():
    """Статистика текущего workspace."""
    eng = Engine()
    s = eng.stats()
    ws = s["workspace"]
    ev = s["events"]

    console.print(Panel.fit(
        f"[bold]Workspace[/]: [cyan]{ws['name']}[/] ({ws['id']})\n"
        f"  root: {ws['root']}\n"
        f"  source: {ws['source']}\n"
        f"  created: {ws['created_at']}\n",
        title="PMB Stats",
    ))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Total events", str(ev["total"]))
    table.add_row("Active", f"[green]{ev['active']}[/]")
    table.add_row("Archived", f"[dim]{ev['archived']}[/]")
    table.add_row("Search index size", str(s["search_index_size"]))
    table.add_row("Oldest", _humanize_time(ev["oldest_timestamp"]))
    table.add_row("Newest", _humanize_time(ev["newest_timestamp"]))
    console.print(table)

    if ev["by_type"]:
        type_table = Table(show_header=True, header_style="bold magenta", title="Events by type")
        type_table.add_column("Type")
        type_table.add_column("Count")
        for t, n in sorted(ev["by_type"].items(), key=lambda x: -x[1]):
            type_table.add_row(t, str(n))
        console.print(type_table)


@app.command(name="list")
def list_cmd(
    limit: int = typer.Option(20, "-n", "--limit"),
    event_type: Optional[str] = typer.Option(None, "--type"),
):
    """Последние события в текущем workspace."""
    eng = Engine()
    events = eng.events.list_active(eng.workspace.id, limit=limit, event_type=event_type)

    if not events:
        console.print("[yellow]No events yet. Use `pmb remember` or recall to start filling memory.[/]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Time", style="dim")
    table.add_column("Type", style="cyan")
    table.add_column("Importance", justify="right")
    table.add_column("Access", justify="right")
    table.add_column("Content")
    for e in events:
        content = e.content[:80] + "..." if len(e.content) > 80 else e.content
        table.add_row(
            _humanize_time(e.timestamp),
            e.event_type,
            f"{e.importance:.2f}",
            str(e.access_count),
            content,
        )
    console.print(table)


@app.command()
def remember(
    query: str = typer.Argument(...),
    response: str = typer.Argument(...),
    importance: float = typer.Option(0.5, "--importance", "-i"),
):
    """Добавить Q/A в память вручную."""
    eng = Engine()
    ulid = eng.remember(query=query, response=response, importance=importance)
    console.print(f"[green]Stored[/] ULID: [cyan]{ulid}[/]")


@app.command()
def fact(
    text: str = typer.Argument(..., help="The factual statement to record"),
    importance: float = typer.Option(0.7, "--importance", "-i"),
):
    """Record a standalone fact (project decision, preference, setting)."""
    eng = Engine()
    ulid = eng.record_fact(fact=text, importance=importance)
    console.print(f"[green]Fact stored[/] ULID: [cyan]{ulid}[/]")


@app.command()
def note(
    text: str = typer.Argument(..., help="The note to remember, in quotes"),
    importance: float = typer.Option(0.6, "--importance", "-i"),
    pin: bool = typer.Option(False, "--pin", help="Pin it (max importance, never auto-archived)"),
):
    """Instant capture - jot a memory straight from the terminal, no agent.

    The lowest-friction way to feed PMB:
      pmb note "decided to use Postgres for JSONB"
      pmb note "Anna's birthday is March 3" --pin
    """
    eng = Engine()
    ulid = eng.record_fact(
        fact=text,
        importance=0.95 if pin else importance,
        metadata={"source": "cli-note"},
    )
    if pin:
        try:
            eng.pin(ulid)
        except Exception:
            pass
    console.print(f"[green]Noted[/] [cyan]{ulid}[/]" + (" [yellow](pinned)[/]" if pin else ""))


@app.command()
def audit(
    limit: int = typer.Option(2000, "-n", "--limit", help="Max events to scan"),
):
    """'What does PMB know about me?' - a grouped, read-only view of memory.

    Shows everything stored, grouped by type and by source, so you can see
    at a glance what PMB has accumulated and where it came from. Local,
    read-only, no model load.
    """
    from collections import Counter
    from pmb.provenance import source_key, describe_source
    eng = Engine()
    events = eng.events.list_active(eng.workspace.id, limit=limit)

    if not events:
        console.print("[yellow]Memory is empty. Use `pmb note` or connect an agent.[/]")
        return

    console.print(Panel.fit(
        f"[bold]{len(events)}[/] active memories in workspace "
        f"[cyan]{eng.workspace.name}[/]",
        title="PMB audit - what I know about you",
    ))

    by_type = Counter(e.event_type for e in events)
    by_source = Counter(source_key(e.metadata) for e in events)
    pinned = [e for e in events if e.importance >= 0.9]

    t1 = Table(show_header=True, header_style="bold magenta", title="By type")
    t1.add_column("Type", style="cyan"); t1.add_column("Count", justify="right")
    for t, n in by_type.most_common():
        t1.add_row(t, str(n))
    console.print(t1)

    t2 = Table(show_header=True, header_style="bold magenta", title="By source (where it came from)")
    t2.add_column("Source", style="cyan"); t2.add_column("Count", justify="right")
    for s, n in by_source.most_common():
        t2.add_row(s, str(n))
    console.print(t2)

    # Highest-importance / pinned facts - the things PMB treats as core about you
    if pinned:
        t3 = Table(show_header=True, header_style="bold magenta",
                   title=f"Core / pinned facts (importance ≥ 0.9) - top {min(15, len(pinned))}")
        t3.add_column("Content"); t3.add_column("From", style="dim")
        for e in sorted(pinned, key=lambda x: -x.importance)[:15]:
            content = e.content[:70] + ("…" if len(e.content) > 70 else "")
            t3.add_row(content, describe_source(e.metadata))
        console.print(t3)

    console.print(
        "[dim]Tip: `pmb recall \"<query>\"` to search · `pmb forget <ulid>` to archive "
        "· `pmb export` (coming) for full dump.[/]"
    )


@app.command()
def watch(
    path: str = typer.Argument(..., help="File or directory to watch (e.g. ~/journal.md)"),
    interval: float = typer.Option(5.0, "--interval", help="Poll interval seconds"),
    once: bool = typer.Option(False, "--once", help="Single pass then exit (cron-friendly)"),
    importance: float = typer.Option(0.5, "--importance", "-i"),
):
    """Auto-capture: ingest new paragraphs from a notes file/folder into memory.

    Turns existing note-taking into memory with zero extra effort:
      pmb watch ~/journal.md          # poll forever (Ctrl+C to stop)
      pmb watch ~/notes/ --once       # single pass (run from cron/Task Scheduler)

    Content-hash dedup: editing old text won't re-ingest; only new paragraphs
    are added.
    """
    from pmb.ingest.watch import scan_new_chunks, load_state, save_state
    eng = Engine()
    target = Path(path).expanduser()
    if not target.exists():
        console.print(f"[red]Path not found:[/] {target}")
        raise typer.Exit(2)

    state_path = eng.workspace.storage_dir / "watch_state.json"

    def _pass() -> int:
        seen = load_state(state_path)
        new_items, updated = scan_new_chunks(target, seen)
        if new_items:
            eng.record_batch_bulk([
                {"type": "fact", "content": it["content"], "importance": importance,
                 "metadata": {"source": "watch", "file": it["file"]}}
                for it in new_items
            ])
            save_state(state_path, updated)
        return len(new_items)

    if once:
        n = _pass()
        console.print(f"[green]Ingested[/] {n} new paragraph(s) from {target}")
        if n:
            eng.regraph()
        return

    console.print(f"[cyan]Watching[/] {target} (every {interval:.0f}s). Ctrl+C to stop.")
    import time as _time
    try:
        while True:
            n = _pass()
            if n:
                console.print(f"  [green]+{n}[/] new paragraph(s) ingested")
            _time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped watching. Run `pmb regraph` to refresh the entity graph.[/]")


@app.command()
def recall(
    query: str = typer.Argument(...),
    top_k: int = typer.Option(5, "-k", "--top"),
    rerank: bool = typer.Option(False, "--rerank",
                                help="Cross-encoder reranker over top-25 (adds ~80MB model + ~100ms)"),
):
    """Поиск релевантной памяти."""
    eng = Engine(
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2" if rerank else None,
    )
    pack = eng.recall(query=query, top_k=top_k, rerank=rerank)

    console.print(f"\n[bold]Query:[/] {query}")
    console.print(f"[dim]Workspace: {pack.workspace_name} | "
                  f"Search took {pack.elapsed_ms:.1f}ms | "
                  f"Total in workspace: {pack.n_total_in_workspace}[/]\n")

    if not pack.results:
        console.print("[yellow]No matches.[/]")
        return

    from pmb.provenance import describe_source
    for i, r in enumerate(pack.results, 1):
        ts = _humanize_time(r.timestamp)
        sigs = (f"score={r.score:.2f} bm25={r.bm25_score:.2f} "
                f"vec={r.vec_score:.2f} imp={r.importance:.2f}")
        title = f"#{i}  [{r.event_type}]  {ts}  ({sigs})"
        content = r.content[:500] + "..." if len(r.content) > 500 else r.content
        # Source attribution: show WHERE this memory came from (trust feature).
        src = describe_source(r.metadata)
        subtitle = f"[dim]{r.ulid}  ·  from: {src}[/]"
        console.print(Panel(
            content,
            title=title,
            subtitle=subtitle,
            title_align="left",
        ))


@app.command()
def why(
    query: str = typer.Argument(...),
    top_k: int = typer.Option(3, "-k", "--top", help="Explain the top-K results"),
):
    """Explain WHY recall ranked results the way it did - full PAMVR trace.

    Most memory tools are a black box: you get results, no idea why. `pmb why`
    shows, per result, exactly which of the 14 predicate-aware reranking rules
    fired and the multiplier each contributed. Great for debugging a miss or
    just understanding the engine.

      pmb why "where do I live now"
    """
    from pmb.reasoning.pamvr import explain_pamvr
    eng = Engine()
    pack = eng.recall(query=query, top_k=top_k)

    console.print(f"\n[bold]Why these results for:[/] {query}")
    console.print(f"[dim]Workspace {pack.workspace_name} · {pack.elapsed_ms:.1f}ms · "
                  f"{pack.n_total_in_workspace} events[/]\n")

    if not pack.results:
        console.print("[yellow]No matches - nothing to explain.[/]")
        return

    bridges = getattr(eng, "_vocab_bridges", None)
    for i, r in enumerate(pack.results, 1):
        ev = eng.events.get_by_ulid(r.ulid)
        content = r.content[:160] + ("…" if len(r.content) > 160 else "")
        console.print(f"[bold cyan]#{i}[/]  {content}")
        console.print(
            f"   [dim]final score {r.score:.3f}  "
            f"(bm25 {r.bm25_score:.2f} · vec {r.vec_score:.2f} · imp {r.importance:.2f})[/]"
        )
        if ev is None:
            console.print("   [dim](event detail unavailable)[/]\n")
            continue
        exp = explain_pamvr(query, ev, base_score=1.0, vocab_bridges=bridges)
        if not exp["rules_fired"]:
            console.print("   [dim]no PAMVR rules fired (pure BM25+vector match)[/]\n")
            continue
        for step in exp["rules_fired"]:
            mult = step["mult"]
            arrow = "[green]▲[/]" if mult > 1.0 else "[red]▼[/]"
            console.print(f"     {arrow} {step['rule']:<40} ×{mult:.2f}")
        net = exp["net_multiplier"]
        net_color = "green" if net >= 1.0 else "red"
        console.print(f"   [bold]net PAMVR multiplier: [{net_color}]×{net:.2f}[/][/]\n")


@app.command()
def pin(ulid: str = typer.Argument(...)):
    """Закрепить событие - высокая importance, не архивируется автоматом."""
    eng = Engine()
    eng.pin(ulid)
    console.print(f"[green]Pinned[/] {ulid}")


@app.command()
def forget(ulid: str = typer.Argument(...)):
    """Заархивировать событие. Не удаляется навсегда - можно unforget."""
    eng = Engine()
    eng.forget(ulid)
    console.print(f"[yellow]Archived[/] {ulid}")


@app.command()
def sync(
    days: Optional[int] = typer.Option(None, "--days",
                                       help="Sync commits from last N days (default: since last sync)"),
):
    """Захватить git commits в memory."""
    eng = Engine()
    since = None
    if days:
        since = time.time() - days * 86400

    result = eng.sync_git(since_timestamp=since)
    if result.get("error"):
        console.print(f"[red]{result['error']}[/]")
        return

    captured = result.get("captured", 0)
    skipped = result.get("skipped_existing", 0)
    branch = result.get("branch", "?")
    console.print(
        f"[green]✓[/] git sync done. "
        f"captured: [bold]{captured}[/], skipped (already imported): {skipped}, "
        f"branch: [cyan]{branch}[/]"
    )


@app.command()
def session(
    action: str = typer.Argument(..., help="start | end | current"),
    name: Optional[str] = typer.Argument(None, help="Session name (for start)"),
):
    """Управление сессиями."""
    eng = Engine()
    if action == "start":
        s = eng.session_start(name)
        console.print(f"[green]✓[/] Session started: [cyan]{s['name']}[/] (id={s['id']})")
    elif action == "end":
        s = eng.session_end()
        if s:
            console.print(f"[yellow]Session ended:[/] {s['name']} (id={s['id']})")
        else:
            console.print("[dim]No active session.[/]")
    elif action == "current":
        s = eng.session_current()
        if s:
            duration = (time.time() - s["started_at"]) / 60.0
            console.print(
                f"[cyan]Current session:[/] {s['name']} (id={s['id']}, "
                f"running {duration:.0f} min)"
            )
        else:
            console.print("[dim]No active session.[/]")
    else:
        console.print(f"[red]Unknown action: {action}[/]")


@app.command()
def rehearse(
    importance: float = typer.Option(0.5, "--min-importance",
                                     help="Only rehearse events at or above this importance"),
    idle_days: float = typer.Option(7.0, "--idle-days",
                                    help="Skip events accessed within N days"),
    max_n: int = typer.Option(20, "-n", "--max-n",
                              help="Cap on rehearsals per run"),
):
    """Spaced-repetition rehearsal - refresh important but idle memories.

    Like the brain's testing effect: a memory you haven't touched in a
    while drifts down; running rehearsal periodically (cron weekly) keeps
    high-importance facts above the noise floor.
    """
    eng = Engine()
    result = eng.rehearse(
        importance_threshold=importance,
        min_idle_days=idle_days,
        max_rehearse=max_n,
    )
    console.print(
        f"[cyan]Rehearsed[/] {result['n_rehearsed']} / {result['n_candidates']} "
        f"eligible memories in {result['elapsed_seconds']:.1f}s "
        f"({result['n_failed']} failed)"
    )


@app.command()
def reindex():
    """Re-embed all events with the current model.

    Run this after switching the embedding model (e.g. when upgrading from
    English-only to multilingual). All vector embeddings are regenerated
    from event content.

    Time: ~1ms/event on CPU. 1000 events ≈ 2 minutes.
    """
    eng = Engine()
    console.print("[yellow]Re-embedding all events with current model...[/]")
    result = eng.reindex_embeddings()
    console.print(
        f"[cyan]Reindexed[/] {result['n_events']} events in "
        f"[yellow]{result['elapsed_seconds']}s[/]"
    )


@app.command()
def dedupe(
    threshold: float = typer.Option(
        0.92, "--threshold", "-t",
        help="Cosine threshold for cluster merge. Higher = more conservative.",
    ),
    types: Optional[str] = typer.Option(
        None, "--types",
        help="Comma-separated event_types to include (default: all).",
    ),
    run_pending: bool = typer.Option(
        False, "--run-pending",
        help="Also drain dedup_pending queue via LLM verify (L2.5).",
    ),
    backend: str = typer.Option(
        "auto", "--backend",
        help="LLM backend for --run-pending: auto / ollama / anthropic / none",
    ),
    list_pending: bool = typer.Option(
        False, "--list-pending",
        help="Just show pending borderline pairs, don't merge.",
    ),
    undo: bool = typer.Option(
        False, "--undo",
        help="Restore events archived by previous dedup runs.",
    ),
):
    """Improvement U: multi-layer dedup.

    Default: one-shot sweep of all active events. Clusters by cosine ≥
    threshold within each event_type; archives losers with metadata
    pointer back to the winner (reversible via --undo).

    With --run-pending: drain the borderline queue (L2.5) by asking a local
    or cloud LLM whether each pair describes the same fact.

    With --list-pending: just inspect the queue, no merges.
    With --undo: restore events archived by previous dedup runs.
    """
    eng = Engine()
    if undo:
        n = eng.dedupe_undo()
        console.print(f"[cyan]Restored[/] {n} previously-merged events")
        return
    if list_pending:
        pairs = eng.dedupe_list_pending(limit=100)
        if not pairs:
            console.print("[dim]No pending borderline pairs.[/]")
            return
        for p in pairs:
            console.print(
                f"  [{p['similarity']:.3f}] {p['event_type']}: "
                f"{p['new_content'][:50]} <-> {p['candidate_content'][:50]}"
            )
        console.print(f"[cyan]{len(pairs)}[/] pending pairs")
        return
    if run_pending:
        console.print(f"[yellow]Running LLM verify ({backend})...[/]")
        result = eng.dedupe_run_pending(backend=backend)
        console.print(
            f"[cyan]Processed[/] {result['n_processed']}: "
            f"merged={result['n_merged']}, kept={result['n_kept']}, "
            f"skipped={result['n_skipped']}"
        )
        return
    type_list = [t.strip() for t in types.split(",")] if types else None
    console.print(f"[yellow]Sweeping (threshold={threshold:.2f})...[/]")
    result = eng.dedupe_sweep(threshold=threshold, event_types=type_list)
    console.print(
        f"[cyan]Dedupe done.[/] clusters={result['n_clusters']}, "
        f"merged={result['n_merged']}, by_type={result['by_type']}"
    )


@app.command(name="prune-graph")
def prune_graph(
    max_weight: int = typer.Option(
        1, "--max-weight", "-w",
        help="Edges with weight <= this AND older than --days are removed.",
    ),
    days: float = typer.Option(
        30.0, "--days", "-d",
        help="Only prune edges older than this many days.",
    ),
    keep_orphans: bool = typer.Option(
        False, "--keep-orphans",
        help="Don't drop entities that become orphaned after edge pruning.",
    ),
):
    """Improvement V: prune weak co-occurrence edges to keep recall fast.

    On a workspace with 10k+ events, the entity graph can grow to hundreds
    of thousands of edges. Most are one-off co-mentions that don't help
    recall. This command drops them (and any orphan entities left behind).

    Reversible? No - but events and embeddings aren't touched, so a
    `pmb regraph` rebuilds everything.

    Time: ~50ms even on a workspace with 100k edges.
    """
    eng = Engine()
    console.print(
        f"[yellow]Pruning edges weight≤{max_weight}, older than {days}d...[/]"
    )
    result = eng.prune_graph(
        max_weight=max_weight,
        older_than_days=days,
        also_drop_orphan_entities=not keep_orphans,
    )
    console.print(
        f"[cyan]Edges:[/] {result['edges_before']} → {result['edges_after']} "
        f"(pruned {result['n_edges_pruned']}). "
        f"Orphan entities dropped: {result['n_entities_pruned']}."
    )


@app.command()
def tui():
    """Improvement PP: full-workspace TUI - Memory / Recall / Stats / Dedup / Tune.

    5-tab terminal workspace:
      [1] Memory   - paginated event browser, filter, detail pane
      [2] Recall   - interactive query playground with score breakdown
      [3] Stats    - live MCP perf (auto-refresh 2s)
      [4] Dedup    - borderline duplicate pairs awaiting decision
      [5] Tune     - all 67 settings, click to edit, type-validated

    Hotkeys: 1-5 switch tabs · / filter · r reload · q quit · ? help

    Best-of from k9s / lazygit / mem0 / htop dashboards, terminal-native.
    """
    eng = Engine()
    try:
        from pmb.cli.tui_workspace import run_workspace_tui
    except ImportError:
        console.print(
            "[red]Textual not installed.[/] Install: "
            "[cyan].venv/Scripts/pip install textual[/]"
        )
        raise typer.Exit(1)
    run_workspace_tui(eng)


@app.command()
def tune():
    """Improvement OO: interactive TUI for fine-tuning all PMB settings.

    Browse 67 settings across 9 categories (recall, dedup, embedding, …).
    See current value + source (workspace/global/default), type, valid range.
    Edit live with type validation. Press 'd' to reset to default.

    Writes to per-workspace `config.yaml`. Press 'q' to quit.

    Examples of what you can tune:
      - recall.top_k, recall.rerank, recall.graph_boost
      - dedup.cosine_high (0.92 default - lower = more aggressive merging)
      - embedding.backend (sentence-transformers vs fastembed)
      - mcp.record_batch_async, recall.adaptive_decompose
    """
    eng = Engine()
    try:
        from pmb.cli.tui_config import run_tui
    except ImportError:
        console.print(
            "[red]Textual not installed.[/] Install with: "
            "[cyan].venv/Scripts/pip install textual[/]"
        )
        raise typer.Exit(1)
    run_tui(eng.config)


@app.command()
def regraph():
    """Wipe and rebuild the entity/edge graph from active events.

    Use this after the entity extractor has been improved (new stop-lists,
    path guards, cross-kind dedup, etc.) so old garbage nodes - question
    words, path components, dialogue roles, code identifiers misclassified
    as people - get removed without touching the event log itself.

    Safe: only touches `graph_entities`, `graph_event_entities`, `graph_edges`,
    and the workspace-scoped `known_persons` dictionary. Events / embeddings /
    facts / goals / reflections are untouched.

    Time: ~1ms/event. 1000 events ≈ 1 second.
    """
    eng = Engine()
    console.print("[yellow]Rebuilding graph from active events...[/]")
    result = eng.regraph()
    console.print(
        f"[cyan]Regraphed[/] {result['events_reindexed']} events → "
        f"{result['entities_created']} entity links"
    )


@app.command()
def reflect(
    limit: int = typer.Option(20, "-n", "--limit",
                              help="Max events to reflect on this run"),
    max_age_days: float = typer.Option(30.0, "--max-age-days",
                                        help="Only consider events newer than this"),
    backend: str = typer.Option("auto", "--backend",
                                help="LLM backend: auto / claude / anthropic / ollama"),
    source: Optional[str] = typer.Option(None, "--source",
                                          help="Reflect on a single event by ULID"),
):
    """PMB v2 - Reflective Memory.

    For each event, an LLM asks itself 'why does this matter? what does
    it imply? what questions might this answer?'. The answers are stored
    as new searchable events that bridge multi-hop queries.

    Run periodically in background (cron / idle hook). Recall stays fast
    - all LLM work happens here.
    """
    eng = Engine()
    if source:
        out = eng.reflect_event(source, backend=backend)
        if out is None:
            console.print("[yellow]Nothing to do[/] (already reflected, or no LLM, or source missing)")
            return
        console.print(f"[cyan]Reflected[/] on {source}")
        console.print(f"  significance: [white]{out['significance']}[/]")
        if out.get("might_answer"):
            console.print("  might answer:")
            for q in out["might_answer"][:5]:
                console.print(f"    • {q}")
        return

    result = eng.reflect_batch(
        limit=limit, max_age_days=max_age_days, backend=backend,
    )
    if result.get("skipped") == "no_llm":
        console.print("[yellow]No LLM backend available.[/] "
                      "Install Claude CLI / Ollama / set ANTHROPIC_API_KEY.")
        return
    console.print(
        f"[cyan]Reflected[/] {result['n_reflected']} / {result['n_candidates']} "
        f"events ([yellow]{result['n_failed']}[/] failed)"
    )
    for s in result.get("samples", []):
        console.print(f"\n  source: [dim]{s['source_preview']}[/]")
        console.print(f"  → {s['significance']}")
        if s.get("might_answer"):
            for q in s["might_answer"][:2]:
                console.print(f"    might answer: {q}")


@app.command()
def arcs(
    action: str = typer.Argument("list", help="list / cluster / show"),
    arc_id: Optional[int] = typer.Argument(None, help="arc id when action=show"),
    limit: int = typer.Option(20, "-n", "--limit"),
    backend: str = typer.Option("auto", "--backend"),
    status: str = typer.Option("active", "--status", help="active / closed / all"),
):
    """PMB v2 - narrative arcs.

    Arcs are LLM-clustered story threads ("Postgres adoption journey",
    "Alice's onboarding"). Recall uses them to answer 'tell me about X'
    style questions with full narrative context.

      pmb arcs list             # list active arcs
      pmb arcs cluster          # run a clustering pass over recent unassigned events
      pmb arcs show 7           # detail of arc id 7
    """
    eng = Engine()
    if action == "list":
        status_filter = None if status == "all" else status
        items = eng.list_arcs(status=status_filter, limit=limit)
        if not items:
            console.print("[dim]No arcs yet. Run `pmb arcs cluster` to build them.[/]")
            return
        for a in items:
            console.print(
                f"[cyan]#{a['id']}[/] [white]{a['title']}[/] "
                f"([yellow]{a['n_events']}[/] events, {a['status']})"
            )
            if a['summary']:
                console.print(f"    {a['summary'][:280]}")
    elif action == "cluster":
        result = eng.cluster_events_into_arcs(limit=limit, backend=backend)
        if result.get("skipped") == "no_llm":
            console.print("[yellow]No LLM backend available.[/]")
            return
        console.print(
            f"[cyan]Clustered[/] {result['n_candidates']} candidates: "
            f"joined={result['n_joined']}, "
            f"created={result['n_created']}, "
            f"ignored={result['n_ignored']}, "
            f"summaries={result.get('n_summaries_updated', 0)}"
        )
    elif action == "show":
        if arc_id is None:
            console.print("[red]pmb arcs show <id> - id required[/]")
            return
        detail = eng.arc_detail(arc_id)
        if not detail:
            console.print(f"[red]Arc #{arc_id} not found in this workspace[/]")
            return
        console.print(f"[cyan]#{detail['id']}[/] [white]{detail['title']}[/]")
        if detail['summary']:
            console.print(f"\n[dim]Summary:[/] {detail['summary']}\n")
        for e in detail['events']:
            console.print(f"  • {e['preview']}")
    else:
        console.print(f"[red]Unknown arcs action: {action}[/]")


@app.command()
def decay(
    days: float = typer.Option(1.0, "--days", help="Days since last decay"),
):
    """Применить forgetting curve. Понижает importance, архивирует устаревшее."""
    eng = Engine()
    result = eng.apply_daily_decay(days_since=days)
    console.print(
        f"[cyan]Decay applied:[/] {result['n_decayed']}/{result['n_active_processed']} events decayed, "
        f"{result['n_archived']} archived (factor={result['decay_factor']:.4f})"
    )


@app.command()
def correlate(
    file_path: str = typer.Argument(...),
    top_k: int = typer.Option(10, "-k"),
):
    """Файлы которые часто меняются вместе с указанным."""
    eng = Engine()
    pairs = eng.file_correlations(file_path, top_k)
    if not pairs:
        console.print(f"[yellow]No correlations found for {file_path}[/]")
        return

    table = Table(show_header=True, header_style="bold magenta",
                  title=f"Files co-modified with {file_path}")
    table.add_column("File")
    table.add_column("Co-occurrences", justify="right")
    for f, c in pairs:
        table.add_row(f, str(c))
    console.print(table)


@app.command()
def history(file_path: str = typer.Argument(...)):
    """История commit'ов для файла."""
    eng = Engine()
    items = eng.file_history(file_path)
    if not items:
        console.print(f"[yellow]No git history found for {file_path}[/]")
        return

    table = Table(show_header=True, header_style="bold magenta",
                  title=f"Commit history: {file_path}")
    table.add_column("Date")
    table.add_column("SHA")
    table.add_column("Author")
    table.add_column("Subject")
    for item in items:
        ts = _humanize_time(item["timestamp"])
        table.add_row(ts, item.get("sha") or "?",
                      item.get("author", "?"),
                      item.get("subject", "")[:60])
    console.print(table)


@app.command()
def feedback(
    ulid: str = typer.Argument(..., help="ULID of the recall hit to judge"),
    verdict: str = typer.Argument(..., help="useful | wrong | irrelevant"),
    query: Optional[str] = typer.Option(None, "-q", "--query",
                                        help="The query you were running"),
    expected_ulid: Optional[str] = typer.Option(None, "-e", "--expected",
                                                help="ULID that should have been returned"),
):
    """Record real recall feedback. This drives adaptive importance over time."""
    eng = Engine()
    try:
        eng.record_recall_feedback(
            ulid, verdict, query=query, expected_ulid=expected_ulid,
        )
    except ValueError as e:
        console.print(f"[red]Invalid verdict:[/] {e}")
        console.print("[dim]Valid: useful | wrong | irrelevant[/]")
        raise typer.Exit(code=2)
    except LookupError as e:
        console.print(f"[red]{e}[/]")
        console.print("[dim]Tip: `pmb list -n 20` to see recent ULIDs.[/]")
        raise typer.Exit(code=2)
    color = {"useful": "green", "wrong": "yellow", "irrelevant": "red"}.get(verdict, "white")
    console.print(f"[{color}]Recorded[/] {verdict} for [cyan]{ulid}[/]"
                  + (f" (expected [cyan]{expected_ulid}[/])" if expected_ulid else ""))


graph_app = typer.Typer(help="Inspect the association graph (entities + edges).")
app.add_typer(graph_app, name="graph")


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
    kind: Optional[str] = typer.Option(None, "--kind", help="file | tech | concept"),
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
    kind: Optional[str] = typer.Option(None, "--kind"),
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


health_app = typer.Typer(help="Health checks: self-test, trends, conflicts, user feedback.")
app.add_typer(health_app, name="health")


@health_app.command("feedback")
def health_feedback():
    """Show summary of recorded recall feedback (the real-user metric)."""
    eng = Engine()
    s = eng.feedback_summary()
    if s["verdict"] == "no_data":
        console.print("[yellow]No feedback recorded yet. Use `pmb feedback ULID useful|wrong|irrelevant`.[/]")
        return

    color = {"healthy": "green", "mixed": "yellow", "poor": "red"}[s["verdict"]]
    rate = s["useful_rate"]
    rate_str = f"{rate:.1%}" if rate is not None else "-"

    console.print(Panel.fit(
        f"Verdict: [{color}]{s['verdict']}[/]\n"
        f"Total feedback entries: {s['total']}\n"
        f"  useful: [green]{s['useful']}[/]\n"
        f"  wrong: [yellow]{s['wrong']}[/]\n"
        f"  irrelevant: [red]{s['irrelevant']}[/]\n"
        f"useful_rate: [{color}]{rate_str}[/]\n"
        f"unique queries: {s['n_unique_queries']}",
        title="User Recall Feedback",
    ))

    if s["most_wrong_events"]:
        t = Table(show_header=True, header_style="bold magenta", title="Most-flagged-wrong events")
        t.add_column("ULID")
        t.add_column("Wrong count", justify="right")
        for ulid, n in s["most_wrong_events"]:
            t.add_row(ulid, str(n))
        console.print(t)


@health_app.command("run")
def health_run(
    n: int = typer.Option(20, "-n", "--n-samples"),
    min_age_days: float = typer.Option(1.0, "--min-age", help="Skip events younger than this"),
    no_adaptive: bool = typer.Option(False, "--no-adaptive",
                                     help="Skip adaptive boost"),
):
    """Запустить self-test: система задаёт сама себе вопросы из старой памяти."""
    eng = Engine()
    result = eng.run_self_test(
        n_samples=n, min_age_days=min_age_days,
        apply_adaptive=not no_adaptive,
    )
    st = result["self_test"]

    if st["n_tested"] == 0:
        if st.get("empty_reason") == "all_events_younger_than_min_age":
            console.print(
                f"[yellow]Skipped self-test:[/] "
                f"all {st['n_too_recent']} content events are younger than "
                f"{st['eligible_min_age_days']} days. "
                f"Lower with `--min-age 0` for a fresh workspace."
            )
        else:
            console.print(
                f"[yellow]Skipped self-test:[/] "
                f"no qa/fact/git events in workspace ({st['n_total_active']} total)."
            )
        return

    table = Table(show_header=True, header_style="bold magenta", title="Self-Test Results")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Tested", str(st["n_tested"]))
    table.add_row("Total active in workspace", str(st["n_total_active"]))
    table.add_row("accuracy@1", f"[green]{st['accuracy_at_1']:.1%}[/]")
    table.add_row("accuracy@3", f"[green]{st['accuracy_at_3']:.1%}[/]")
    table.add_row("accuracy@5", f"[green]{st['accuracy_at_5']:.1%}[/]")
    if st["avg_rank"] is not None:
        table.add_row("avg rank when found", f"{st['avg_rank']:.2f}")
    console.print(table)

    # Session-aware loose metric (closer to real-world recall)
    if st.get("session_accuracy_at_5") is not None:
        console.print(
            f"\n[dim]session-aware acc@5 (any same-session hit counts):[/] "
            f"[cyan]{st['session_accuracy_at_5']:.1%}[/] "
            f"(coverage {st.get('session_coverage', 0):.0%})"
        )

    if result.get("adaptive"):
        a = result["adaptive"]
        console.print(
            f"[cyan]Adaptive boost:[/] {a['n_boosted']} events boosted "
            f"({a['n_superboosted']} super-boosted after repeated failures)"
        )

    if result.get("feedback_adaptive"):
        fa = result["feedback_adaptive"]
        if fa["n_feedback_entries"] > 0:
            console.print(
                f"[cyan]User feedback ({fa['n_feedback_entries']} entries):[/] "
                f"promoted {fa['n_promoted_useful']} useful + "
                f"{fa['n_promoted_expected']} expected, "
                f"demoted {fa['n_demoted_wrong']} wrong"
            )

    if st["failed_queries"]:
        console.print(f"\n[yellow]Failed queries (saved {len(st['failed_queries'])}):[/]")
        for f in st["failed_queries"][:3]:
            console.print(f"  Q: '{f['query']}'")
            console.print(f"     expected: {f['expected_content_preview'][:80]}")


@health_app.command("trend")
def health_trend():
    """Показать trend self-test accuracy за время."""
    eng = Engine()
    t = eng.health_trend()
    if t["verdict"] == "insufficient":
        console.print(f"[yellow]Need more self-test runs (have {t['n_runs']}).[/]")
        return

    color = {"stable": "green", "degrading": "red", "improving": "cyan"}[t["verdict"]]
    console.print(Panel.fit(
        f"Verdict: [{color}]{t['verdict']}[/]\n"
        f"Runs: {t['n_runs']}\n"
        f"First acc@5: {t['first_run_acc5']:.1%}\n"
        f"Last acc@5:  {t['last_run_acc5']:.1%}\n"
        f"Δ: {t['delta_pp']:+.1f}pp",
        title="Health Trend",
    ))


@health_app.command("conflicts")
def health_conflicts(
    resolve: bool = typer.Option(False, "--resolve",
                                  help="Auto-archive obvious supersede conflicts"),
):
    """Найти противоречия между фактами разного времени."""
    eng = Engine()
    conflicts = eng.detect_conflicts()
    if not conflicts:
        console.print("[green]No conflicts detected.[/]")
        return

    table = Table(show_header=True, header_style="bold magenta", title="Conflicts")
    table.add_column("Key")
    table.add_column("Older")
    table.add_column("Newer")
    table.add_column("Resolution")
    table.add_column("Confidence")
    for c in conflicts[:20]:
        table.add_row(
            c["key"],
            c["older_value"][:40],
            c["newer_value"][:40],
            c["suggested_resolution"],
            f"{c['confidence']:.2f}",
        )
    console.print(table)

    if resolve:
        action = eng.auto_resolve_conflicts(dry_run=False)
        console.print(
            f"\n[yellow]Auto-resolved:[/] {action['n_archived']} conflicts archived "
            f"({action['n_supersede_candidates']} candidates total)"
        )


@app.command()
def consolidate(
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="Cluster and call LLM, but don't store/archive"),
    auto_if_due: bool = typer.Option(
        False, "--if-due",
        help="Only consolidate when auto-trigger thresholds are met (per config)",
    ),
    backend: str = typer.Option("auto", "--backend",
                                help="auto | claude | anthropic | ollama. Auto prefers `claude` CLI if installed (no key needed), then Anthropic API, then Ollama."),
    model: Optional[str] = typer.Option(None, "--model",
                                        help="Override default model (e.g. 'haiku' for claude/anthropic, 'llama3.1:8b' for ollama)"),
    since_days: float = typer.Option(14.0, "--since-days"),
    threshold: float = typer.Option(0.5, "--threshold",
                                    help="Cosine similarity threshold for clustering (MiniLM tends to give 0.3-0.6 for related short facts)"),
    min_size: int = typer.Option(3, "--min-size",
                                 help="Minimum cluster size to consider"),
    max_clusters: int = typer.Option(10, "--max-clusters"),
):
    """LLM-based sleep-stage consolidation.

    No API key needed if `claude` CLI is in PATH - uses your existing
    Claude Code login. Alternative backends: ANTHROPIC_API_KEY for direct
    API, or Ollama for fully local.

    Clusters related recent memories, asks an LLM to extract one underlying
    rule per cluster, stores it as a high-importance fact, archives the
    source events.
    """
    eng = Engine()
    if auto_if_due:
        decision = eng.consolidation_due()
        if not decision.get("should_run"):
            console.print(
                f"[yellow]Skipping:[/] {decision.get('reason')}. "
                f"Run without --if-due to force, or lower thresholds via "
                f"`pmb config set consolidate.auto_min_new_events ...`."
            )
            return
        console.print(f"[cyan]Auto-trigger fired:[/] {decision['reason']}")
    try:
        result = eng.consolidate(
            dry_run=dry_run,
            backend=backend,
            model=model,
            since_days=since_days,
            similarity_threshold=threshold,
            min_cluster_size=min_size,
            max_clusters=max_clusters,
        )
    except ImportError as e:
        console.print(f"[red]Missing dependency:[/] {e}")
        console.print("[dim]Install with: pip install -e .[consolidate][/]")
        raise typer.Exit(code=2)
    except RuntimeError as e:
        console.print(f"[red]Consolidation backend unavailable:[/]\n{e}")
        raise typer.Exit(code=2)
    except Exception as e:
        console.print(f"[red]Consolidation failed:[/] {e}")
        raise typer.Exit(code=1)

    if result["n_clusters_found"] == 0:
        console.print(f"[yellow]No clusters found[/] "
                      f"(threshold={threshold}, since={since_days}d, min_size={min_size}). "
                      f"Try lowering --threshold or --min-size.")
        return

    table = Table(show_header=True, header_style="bold magenta",
                  title="Consolidation Results" + (" (dry-run)" if dry_run else ""))
    table.add_column("Anchor", style="dim", width=22)
    table.add_column("Size", justify="right")
    table.add_column("Sim", justify="right")
    table.add_column("Conf", justify="right")
    table.add_column("Result")
    table.add_column("Summary", overflow="fold")
    for r in result["results"]:
        status = "[green]stored[/]" if r["consolidated"] else "[dim]skipped[/]"
        table.add_row(
            r["cluster_anchor"][:20],
            str(r["cluster_size"]),
            f"{r['avg_similarity']:.2f}",
            f"{r['confidence']:.2f}",
            status,
            r["summary"] if r["consolidated"] else r["reasoning"],
        )
    console.print(table)
    console.print(
        f"[cyan]{result['n_consolidated']}[/] consolidated, "
        f"[yellow]{result['n_archived']}[/] source events archived "
        f"({result['n_clusters_found']} clusters total)"
        + (" [dim](dry-run, no writes)[/]" if dry_run else "")
    )


@app.command()
def compact(
    dry_run: bool = typer.Option(False, "--dry-run"),
    age_days: int = typer.Option(30, "--age", help="Min age of archived events to move"),
):
    """Compact storage: переместить старые archived events в cold storage + VACUUM."""
    eng = Engine()
    result = eng.compact(dry_run=dry_run, age_days=age_days)
    if result["dry_run"]:
        console.print(f"[cyan]Dry run:[/] would move {result['moved_to_cold']} events")
    else:
        saved_kb = result.get("size_saved", 0) / 1024.0
        console.print(
            f"[green]✓[/] moved [bold]{result['moved_to_cold']}[/] events to cold storage. "
            f"DB size: {result['main_size_before']/1024:.1f}KB → {result['main_size_after']/1024:.1f}KB "
            f"(saved {saved_kb:.1f}KB)"
        )


@app.command()
def schedule():
    """Generate OS scheduler config (cron / Windows schtasks)."""
    from pmb.maintenance.scheduler import generate_scheduler_config
    cfg = generate_scheduler_config()
    if not cfg.get("supported"):
        console.print(f"[red]Unsupported OS: {cfg['os']}[/]")
        return
    console.print(Panel.fit(
        "\n".join(cfg["install_steps"]),
        title=f"Scheduler config ({cfg['type']})",
    ))


@app.command()
def doctor(
    remote: Optional[str] = typer.Option(
        None, "--remote",
        help="Print an SSH-tunneled MCP config for a remote PMB. "
             "Format: user@host:/abs/path/to/repo",
    ),
):
    """Диагностика установки и runtime state.

    With --remote user@host:/path, also prints a `.mcp.json` snippet that
    tunnels MCP over ssh - for the case where PMB and Ollama live on
    a server and the agent (Claude Code / Cursor) runs locally.
    """
    from pmb.cli.doctor import print_doctor
    rc = print_doctor(console, remote=remote)
    if rc != 0:
        raise typer.Exit(code=rc)


config_app = typer.Typer(help="Inspect and tune PMB knobs from the console.")
app.add_typer(config_app, name="config")


# Ollama subcommand - for fully-local LLM ops (no Anthropic / OpenAI key)
from pmb.cli.ollama_cmd import app as ollama_app
app.add_typer(ollama_app, name="ollama")


def _open_config():
    """Construct a Config bound to the current workspace + global home."""
    from pmb.config import Config, SCHEMA  # noqa: F401
    ws = detect_workspace()
    ws.ensure_dirs()
    return Config(workspace_dir=ws.storage_dir, pmb_home=ws.pmb_home), ws


@config_app.command("list")
def config_list(
    only_overridden: bool = typer.Option(
        False, "--only-overridden",
        help="Show only keys whose value differs from the schema default",
    ),
):
    """Print every config key, its value, and where the value comes from."""
    from pmb.config import SCHEMA
    cfg, ws = _open_config()
    console.print(Panel.fit(
        f"workspace: [cyan]{ws.name}[/] ({ws.id[:12]})\n"
        f"  workspace yaml: {cfg.workspace_path}\n"
        f"  global yaml:    {cfg.global_path}",
        title="PMB config sources",
    ))
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Key", overflow="fold")
    table.add_column("Value", overflow="fold")
    table.add_column("Source", style="dim")
    table.add_column("Default", style="dim", overflow="fold")
    for key, setting in SCHEMA.items():
        value = cfg.get(key)
        src = cfg.source_of(key)
        if only_overridden and src == "default":
            continue
        default_repr = repr(setting.default)
        table.add_row(key, repr(value), src, default_repr)
    console.print(table)


@config_app.command("get")
def config_get(key: str):
    """Read a single key."""
    from pmb.config import SCHEMA
    cfg, _ = _open_config()
    if key not in SCHEMA:
        console.print(f"[red]Unknown key:[/] {key}")
        console.print(f"[dim]Try: pmb config list[/]")
        raise typer.Exit(code=2)
    console.print(
        f"[cyan]{key}[/] = {cfg.get(key)!r}  "
        f"[dim](source: {cfg.source_of(key)}; default: {SCHEMA[key].default!r})[/]"
    )
    console.print(f"[dim]{SCHEMA[key].help}[/]")


@config_app.command("set")
def config_set(
    key: str,
    value: str,
    glob: bool = typer.Option(
        False, "--global", help="Write to global ~/.pmb/config.yaml instead of workspace",
    ),
):
    """Update a config value. Writes to per-workspace YAML by default."""
    from pmb.config import SCHEMA
    cfg, _ = _open_config()
    if key not in SCHEMA:
        console.print(f"[red]Unknown key:[/] {key}")
        raise typer.Exit(code=2)
    try:
        stored = cfg.set_global(key, value) if glob else cfg.set_workspace(key, value)
    except ValueError as e:
        console.print(f"[red]Rejected:[/] {e}")
        raise typer.Exit(code=2)
    where = "global" if glob else "workspace"
    console.print(f"[green]Set[/] {key} = {stored!r}  [dim]({where})[/]")


@config_app.command("reset")
def config_reset(
    key: Optional[str] = typer.Argument(None, help="Key to reset (omit to reset ALL workspace overrides)"),
):
    """Remove a key from the workspace YAML, returning it to global/default."""
    cfg, _ = _open_config()
    if key:
        from pmb.config import SCHEMA
        if key not in SCHEMA:
            console.print(f"[red]Unknown key:[/] {key}")
            raise typer.Exit(code=2)
        cfg.reset_workspace(key)
        console.print(f"[yellow]Reset[/] {key} in workspace; now {cfg.get(key)!r} ({cfg.source_of(key)})")
    else:
        cfg.reset_workspace(None)
        console.print("[yellow]Reset[/] all workspace overrides.")


@config_app.command("edit")
def config_edit(
    glob: bool = typer.Option(False, "--global"),
):
    """Open the YAML in $EDITOR."""
    cfg, _ = _open_config()
    path = cfg.global_path if glob else cfg.workspace_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# PMB config - run `pmb config list` for keys\n", encoding="utf-8")
    editor = os.environ.get("EDITOR") or ("notepad" if os.name == "nt" else "nano")
    os.system(f'{editor} "{path}"')


@app.command()
def connect(
    agent: Optional[str] = typer.Argument(
        None,
        help="claude-code | cursor | codex | windsurf | gemini | vscode | "
             "zed | opencode | continue",
    ),
    scope: str = typer.Option("project", "--scope",
                              help="project | global (where the agent supports both)"),
    remote: Optional[str] = typer.Option(
        None, "--remote",
        help="SSH-tunnel: user@host:/abs/path/to/repo (memory + Ollama on server)",
    ),
    name: Optional[str] = typer.Option(
        None, "--name", help="MCP entry name (default: pmb or pmb-remote)",
    ),
    workspace: Optional[str] = typer.Option(
        None, "--workspace", "-w",
        help="Force a SPECIFIC workspace id (override cwd-based detection). "
             "Use this to SHARE one memory across multiple AI clients - e.g. "
             "Claude Code + Cursor both pointing at 'personal' workspace.",
    ),
    pmb_home: Optional[Path] = typer.Option(
        None, "--pmb-home",
        help="Override PMB_HOME (where workspaces live on disk). Useful for "
             "multi-user shared memory on a NAS / Dropbox path.",
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config-path",
        help="Override the agent's MCP config file location (for editors "
             "whose config lives somewhere our default guess doesn't cover).",
    ),
    list_agents: bool = typer.Option(
        False, "--list", help="List every supported agent and its config file, then exit.",
    ),
    probe: bool = typer.Option(False, "--probe", help="Spawn pmb-mcp briefly to verify it starts"),
):
    """Add a `pmb` entry to your agent's MCP config without touching other entries.

    Supports 9 agents: claude-code, cursor, codex, windsurf, gemini, vscode,
    zed, opencode, continue.

    SHARING ONE MEMORY ACROSS AI CLIENTS:

      pmb connect claude-code --workspace personal
      pmb connect cursor      --workspace personal
      pmb connect zed         --workspace personal

      → all clients now read/write the same workspace `personal`.
      → records from one are immediately visible to the others.
    """
    from pmb.cli.connect import (
        connect as do_connect, probe_mcp,
        JSON_AGENT_SPECS, supported_agents,
    )

    if list_agents:
        t = Table(show_header=True, header_style="bold magenta", title="Supported agents")
        t.add_column("Agent"); t.add_column("Config / notes")
        t.add_row("claude-code", "Claude Code - .mcp.json (project) / ~/.claude.json (global)")
        t.add_row("cursor", "Cursor - <project>/.cursor/mcp.json or ~/.cursor/mcp.json")
        t.add_row("codex", "OpenAI Codex CLI - ~/.codex/config.toml")
        for aid in sorted(JSON_AGENT_SPECS):
            t.add_row(aid, JSON_AGENT_SPECS[aid].docs)
        console.print(t)
        console.print(f"\n[dim]{len(supported_agents())} agents total.[/]")
        return

    if not agent:
        console.print("[red]Missing AGENT.[/] Run [cyan]pmb connect --list[/] to see options.")
        raise typer.Exit(code=2)

    try:
        result = do_connect(
            agent, cwd=Path.cwd(), scope=scope, remote=remote, name_override=name,
            workspace_id=workspace, pmb_home=pmb_home, config_path=config_path,
        )
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(code=2)

    color = "green" if result["action"] == "added" else "yellow"
    console.print(Panel.fit(
        f"[{color}]{result['action']}[/] entry [cyan]{result['entry_name']}[/] for "
        f"[bold]{result['agent']}[/] ({result['scope']})\n"
        f"config file: {result['config_path']}\n\n"
        + json.dumps(result["entry"], indent=2, ensure_ascii=False),
        title="MCP connected",
    ))

    # Show instruction-rules result so user knows the AI will follow PMB rules
    rules_info = result.get("instruction_rules") or []
    if rules_info:
        for r in rules_info:
            if "error" in r:
                console.print(f"[red]Instructions: {r['error']}[/]")
            else:
                act = r.get("action", "?")
                console.print(
                    f"[green]Instructions {act}:[/] {r['path']}\n"
                    "  → AI will follow PMB rules automatically (recall first, record_fact proactively)"
                )

    if probe:
        ok, msg = probe_mcp()
        verdict_color = "green" if ok else "red"
        console.print(f"[{verdict_color}]Probe:[/] {msg}")
        if not ok:
            raise typer.Exit(code=1)

    console.print(
        "\n[dim]Restart your agent so it picks up the new MCP entry. "
        "Run `pmb doctor` to see the full local state.[/]"
    )
    console.print(
        "\n[yellow]Note:[/] [dim]The first recall after agent restart takes "
        "~30-60s while the embedding model loads. Subsequent recalls return "
        "in <100ms. To pre-warm before connecting, run [bold]pmb warmup[/] "
        "first.[/]"
    )


workspace_app = typer.Typer(
    help="Git-backed workspace sync: push / pull / clone your memory to any remote."
)
app.add_typer(workspace_app, name="workspace")


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


@app.command("import")
def import_cmd(
    source: str = typer.Argument(..., help="chatgpt | claude | mem0 | markdown"),
    path: str = typer.Argument(..., help="Path to the export file or directory"),
    roles: str = typer.Option(
        "user", "--roles",
        help="For chat sources: which roles to import (comma-separated: "
             "user,assistant). Default 'user' keeps signal high.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Parse and preview, but don't write to memory.",
    ),
):
    """Import existing memory from another tool - no more empty cold start.

    Examples:
      pmb import chatgpt ~/Downloads/conversations.json
      pmb import claude  ~/Downloads/claude-export/
      pmb import mem0    mem0_dump.json
      pmb import markdown ~/notes/        # Obsidian vault, plain notes

    After import, the entity graph is rebuilt automatically so recall works
    immediately.
    """
    from pmb.ingest import parse_source
    role_set = {r.strip().lower() for r in roles.split(",") if r.strip()}
    try:
        result = parse_source(source, Path(path), roles=role_set)
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2)

    if result.notes:
        for n in result.notes:
            console.print(f"[yellow]note:[/] {n}")

    console.print(
        f"[cyan]Parsed[/] {result.n_parsed} item(s) from [bold]{result.source}[/] "
        f"([dim]{result.skipped} skipped[/])"
    )
    if result.n_parsed == 0:
        console.print("[yellow]Nothing to import.[/]")
        raise typer.Exit(0 if not result.notes else 1)

    # Preview a few
    for it in result.items[:3]:
        preview = it["content"][:100].replace("\n", " ")
        console.print(f"  [dim]·[/] {preview}…")

    if dry_run:
        console.print("[dim](dry-run - nothing written)[/]")
        return

    eng = Engine()
    console.print(f"[yellow]Importing {result.n_parsed} items (bulk mode)…[/]")
    out = eng.record_batch_bulk(result.items)
    console.print(
        f"[green]Imported[/] {out.get('n_ok', 0)} ok, "
        f"{out.get('n_failed', 0)} failed."
    )
    console.print("[yellow]Rebuilding entity graph…[/]")
    rg = eng.regraph()
    console.print(
        f"[green]Done.[/] Graph: {rg.get('entities_created', 0)} entity links "
        f"from {rg.get('events_reindexed', 0)} events. "
        f"Run [cyan]pmb stats[/] to see your imported memory."
    )


@app.command()
def workspaces():
    """Все известные workspaces."""
    items = list_workspaces()
    if not items:
        console.print("[yellow]No workspaces yet. Run `pmb init` in a project.[/]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Root")
    table.add_column("Source")
    table.add_column("Created")
    for w in items:
        table.add_row(w.id[:12], w.name, str(w.root), w.source, w.created_at[:10])
    console.print(table)


if __name__ == "__main__":
    app()
