"""`pmb` maintenance root commands — extracted from cli/main.py (no behavior change).

cli/main.py imports this module so these @app.command registrations run."""

from __future__ import annotations

import time
from typing import Optional

import typer
from rich.markup import escape as esc
from rich.panel import Panel
from rich.table import Table

from pmb.cli._common import (  # noqa: F401
    _agent_toggles_from_config,
    _apply_ttl,
    _humanize_time,
    _open_config,
    _parse_duration,
    app,
    console,
)
from pmb.core.engine import Engine


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
    action: str = typer.Argument(..., help="start | end | current | brief"),
    name: Optional[str] = typer.Argument(None, help="Session name (for start)"),
):
    """Управление сессиями. `brief` = digest of what was decided/done this
    session (re-orient after a long session / context loss)."""
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
    elif action == "brief":
        b = eng.session_brief()
        if b.get("empty"):
            console.print(f"[yellow]Nothing recorded in {esc(str(b['scope']))}.[/]")
            return
        hdr = f"[bold]{b['n_events']}[/] memories · {esc(str(b['scope']))}"
        if b.get("duration_min") is not None:
            hdr += (f" · {esc(str(b.get('session_name') or 'session'))} "
                    f"({b['duration_min']} min)")
        console.print(Panel.fit(hdr, title="Session brief"))

        def _sb(title, items, marker=""):
            if not items:
                return
            console.print(f"\n[bold magenta]{title}[/] ({len(items)})")
            for it in items:
                console.print(f"  {marker}[dim]{it['when']}[/] {esc(it['content'])}")

        _sb("Decisions", b["decisions"])
        _sb("Done", b["done"])
        _sb("Lessons", b["lessons"], "[magenta]★[/] ")
        _sb("Failures", b["failures"], "[red]⚠[/] ")
        _sb("Goals", b["goals"])
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





