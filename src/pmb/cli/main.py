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
    export: str = typer.Option(
        None, "--export", "-e",
        help="Don't serve - write a self-contained HTML file (open offline / share)."),
    open_browser: bool = typer.Option(
        False, "--open", help="Open the dashboard (or export) in your browser."),
):
    """Launch the local web dashboard - an interactive map of your memory.

    Serves http://127.0.0.1:8765 with the memory graph, a git-style timeline of
    what was captured and when, workspace stats, recall debugger, and a
    per-event inspector. Bound to localhost; pure stdlib, no external deps.

    `--export memory.html` writes a single self-contained file instead of
    serving - useful to open offline or share.
    """
    eng = Engine()
    if export:
        from pmb.dashboard.viz import build_memory_html
        html = build_memory_html(eng)
        p = Path(export)
        p.write_text(html, encoding="utf-8")
        try:
            s = eng.graph_stats()
            console.print(
                f"[green]done[/] wrote [bold]{p}[/] - "
                f"{s['n_entities']} entities, {s['n_edges']} connections")
        except Exception:
            console.print(f"[green]done[/] wrote [bold]{p}[/]")
        if open_browser:
            import webbrowser
            webbrowser.open(p.resolve().as_uri())
        return
    try:
        from pmb.dashboard.server import run_dashboard
    except Exception as e:
        console.print(f"[red]Failed to import dashboard:[/] {e}")
        return
    if open_browser:
        import webbrowser, threading
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{host}:{port}")).start()
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
    ttl: Optional[str] = typer.Option(
        None, "--ttl",
        help="Auto-expire after a duration (30d, 12h, 2w, 3mo). Archived later "
             "by `pmb prune-expired`; never affects recall.",
    ),
):
    """Record a standalone fact (project decision, preference, setting)."""
    eng = Engine()
    ulid = eng.record_fact(fact=text, importance=importance)
    if ttl:
        _apply_ttl(eng, ulid, ttl)
    console.print(f"[green]Fact stored[/] ULID: [cyan]{ulid}[/]")


@app.command()
def note(
    text: str = typer.Argument(..., help="The note to remember, in quotes"),
    importance: float = typer.Option(0.6, "--importance", "-i"),
    pin: bool = typer.Option(False, "--pin", help="Pin it (max importance, never auto-archived)"),
    ttl: Optional[str] = typer.Option(
        None, "--ttl",
        help="Auto-expire after a duration (30d, 12h, 2w, 3mo). Archived later "
             "by `pmb prune-expired`; never affects recall.",
    ),
):
    """Instant capture - jot a memory straight from the terminal, no agent.

    The lowest-friction way to feed PMB:
      pmb note "decided to use Postgres for JSONB"
      pmb note "Anna's birthday is March 3" --pin
      pmb note "spike: try Redis cache" --ttl 14d
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
    if ttl:
        _apply_ttl(eng, ulid, ttl)
    console.print(f"[green]Noted[/] [cyan]{ulid}[/]" + (" [yellow](pinned)[/]" if pin else ""))


@app.command()
def learn(
    lesson: str = typer.Argument(..., help="A reusable lesson, in quotes"),
    importance: float = typer.Option(0.85, "--importance", "-i"),
    failed: bool = typer.Option(
        False, "--failed",
        help="Record a FAILURE (negative memory): 'tried X, it didn't work'. "
             "Surfaces with a warning so you/the agent don't repeat it.",
    ),
    ttl: Optional[str] = typer.Option(
        None, "--ttl",
        help="Auto-expire after a duration (e.g. 90d). Rarely needed for "
             "lessons; useful for time-bound rules.",
    ),
):
    """Teach PMB a durable LESSON - a correction or technique to apply going
    forward, not just a one-off fact.

    Where `note` records what happened, `learn` records how to work better:
      pmb learn "this repo uses pnpm, never npm"
      pmb learn "always run `make fmt` before committing"
      pmb learn --failed "tried bumping numpy to 2.x - broke lancedb, stay on 1.x"

    Lessons/failures are stored at high importance (0.85) and tagged so
    `pmb lessons`, `pmb audit`, and recall treat them specially. Unlike a
    keyword-only store, they are retrieved by PMB's hybrid + predicate-aware
    ranker - the right one surfaces, not a lookalike.
    """
    eng = Engine()
    kind = "failure" if failed else "lesson"
    ulid = eng.record_fact(
        fact=lesson,
        importance=importance,
        metadata={"source": "lesson", "kind": kind},
    )
    if ttl:
        _apply_ttl(eng, ulid, ttl)
    word = "Recorded failure" if failed else "Learned"
    console.print(f"[green]{word}[/] [cyan]{ulid}[/] [dim]({kind}, importance {importance})[/]")
    console.print("[dim]Tip: `pmb lessons` to review · `pmb consolidate` to distill "
                  "lessons from recent sessions via LLM.[/]")


@app.command()
def distill(
    session: Optional[str] = typer.Option(
        None, "--session",
        help="Session id to distill (default: most recent session's events).",
    ),
    backend: str = typer.Option("auto", "--backend",
                                help="auto | claude | anthropic | ollama"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview, don't store"),
):
    """Auto-distill durable LESSONS & FAILURES from a session via an LLM.

    Reads what the agent did/decided/was-corrected-on in a session and extracts
    reusable rules ("use pnpm, never npm") and failures ("numpy 2.x broke
    lancedb") - so the agent works better next time. Zero-command version:
    `pmb config set lessons.auto_distill_on_session_end true` runs this
    automatically on `pmb session end`.

    Needs an LLM backend (claude CLI / Anthropic key / Ollama). Off the recall
    path - cannot affect recall quality or speed.
    """
    eng = Engine()
    res = eng.distill_lessons(session_id=session, backend=backend, dry_run=dry_run)
    if res.get("skipped") == "no_llm":
        console.print("[yellow]No LLM backend available.[/] Install Claude CLI / "
                      "Ollama, or set ANTHROPIC_API_KEY.")
        if res.get("detail"):
            console.print(f"[dim]{res['detail']}[/]")
        return
    if res.get("skipped") == "no_events":
        console.print("[yellow]No session events to distill.[/]")
        return
    cands = res.get("candidates") or []
    if not cands:
        console.print("[dim]No durable lessons found in this session.[/]")
        return
    for c in cands:
        mark = "[red]⚠[/]" if c["type"] == "failure" else "[magenta]★[/]"
        console.print(f"  {mark} {c['content']}")
    if dry_run:
        console.print(f"[dim](dry-run) {len(cands)} candidate(s), nothing stored.[/]")
    else:
        console.print(f"[green]Distilled[/] {res.get('n_recorded', 0)} new "
                      f"lesson(s)/failure(s) into memory.")


@app.command()
def lessons(
    limit: int = typer.Option(50, "-n", "--limit"),
):
    """List the durable lessons PMB has learned (procedural memory).

    These are the corrections and techniques that make the agent work better
    over time - the 'don't repeat this mistake' layer.
    """
    eng = Engine()
    events = eng.events.list_active(eng.workspace.id, limit=2000)
    lessons = [
        e for e in events
        if (e.metadata or {}).get("kind") == "lesson"
        or (e.metadata or {}).get("source") == "lesson"
    ]
    if not lessons:
        console.print(
            "[yellow]No lessons yet.[/] Teach one: "
            "[cyan]pmb learn \"this repo uses pnpm, never npm\"[/]\n"
            "[dim]Or distill from recent work: `pmb consolidate`.[/]"
        )
        return
    lessons.sort(key=lambda e: (-e.importance, -e.timestamp))
    t = Table(show_header=True, header_style="bold magenta",
              title=f"Lessons & failures ({len(lessons)})")
    t.add_column("", width=2); t.add_column("Lesson / failure")
    t.add_column("When", style="dim")
    for e in lessons[:limit]:
        is_fail = (e.metadata or {}).get("kind") == "failure"
        mark = "[red]⚠[/]" if is_fail else "[magenta]★[/]"
        content = e.content[:90] + ("…" if len(e.content) > 90 else "")
        t.add_row(mark, content, _humanize_time(e.timestamp))
    console.print(t)


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
    from pmb.memory_quality import is_stale, confidence_from
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

    # Memory health - hygiene signals (read-only, no ranking change)
    now = time.time()
    n_stale = sum(1 for e in events
                  if is_stale(e.timestamp, now, access_count=e.access_count))
    n_lessons = sum(1 for e in events if (e.metadata or {}).get("kind") == "lesson")
    n_failures = sum(1 for e in events if (e.metadata or {}).get("kind") == "failure")
    n_lowconf = sum(1 for e in events if confidence_from(e.metadata) < 0.6)
    try:
        n_conflicts = len(eng.detect_conflicts())
    except Exception:
        n_conflicts = 0
    th = Table(show_header=True, header_style="bold magenta", title="Memory health")
    th.add_column("Signal"); th.add_column("Count", justify="right")
    th.add_row("Lessons (procedural)", str(n_lessons))
    th.add_row("Failures (don't-repeat)", str(n_failures))
    th.add_row("Possibly stale (>180d, rarely used)",
               f"[yellow]{n_stale}[/]" if n_stale else "0")
    th.add_row("Low-confidence (agent-inferred)", str(n_lowconf))
    th.add_row("Conflicts detected", f"[yellow]{n_conflicts}[/]" if n_conflicts else "0")
    console.print(th)

    console.print(
        "[dim]Tip: `pmb recall \"<query>\"` to search · `pmb lessons` for lessons · "
        "`pmb forget <ulid>` to archive · `pmb health conflicts` to resolve.[/]"
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
    from pmb.memory_quality import freshness_label, confidence_from, confidence_label
    now = time.time()
    for i, r in enumerate(pack.results, 1):
        ts = _humanize_time(r.timestamp)
        sigs = (f"score={r.score:.2f} bm25={r.bm25_score:.2f} "
                f"vec={r.vec_score:.2f} imp={r.importance:.2f}")
        kind = (r.metadata or {}).get("kind", "")
        marker = "[red]⚠ FAILURE[/] " if kind == "failure" else (
                 "[magenta]★ LESSON[/] " if kind == "lesson" else "")
        title = f"#{i}  {marker}[{r.event_type}]  {ts}  ({sigs})"
        content = r.content[:500] + "..." if len(r.content) > 500 else r.content
        # Trust signals (display only - no effect on ranking):
        src = describe_source(r.metadata)
        conf = confidence_from(r.metadata)
        stale = freshness_label(r.timestamp, now,
                                access_count=getattr(r, "access_count", 0) or 0)
        sub = f"{r.ulid}  ·  from: {src}  ·  confidence: {confidence_label(conf)}"
        if stale:
            sub += f"  ·  [yellow]⚠ {stale}[/]"
        console.print(Panel(
            content,
            title=title,
            subtitle=f"[dim]{sub}[/]",
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
def overview(
    topic: str = typer.Argument(..., help="Topic to summarize, in quotes"),
    max_events: Optional[int] = typer.Option(
        None, "-n", "--max-events",
        help="How many memories to synthesize (default: config overview.max_events)."),
):
    """'What do I know about <topic>?' - a structured overview from memory.

    Aggregates the relevant memories into key facts & decisions, lessons,
    failures, open goals, a timeline and related topics. No LLM, fully local -
    great for getting up to speed on a project/feature before you start.

      pmb overview "authentication"
    """
    eng = Engine()
    if max_events is None:
        try:
            max_events = int(eng.config.get("overview.max_events"))
        except Exception:
            max_events = 40
    ov = eng.topic_overview(topic, max_events=max_events)
    if ov.get("empty"):
        console.print(f"[yellow]No memories about[/] '{esc(topic)}'.")
        return
    span = ov.get("span") or {}
    console.print(Panel.fit(
        f"[bold]{ov['n_memories']}[/] memories about [cyan]{esc(topic)}[/]"
        + (f"  ·  {span.get('from')} -> {span.get('to')}" if span else ""),
        title="PMB overview",
    ))

    def _sect(title, items, marker=""):
        if not items:
            return
        console.print(f"\n[bold magenta]{title}[/] ({len(items)})")
        for it in items:
            d = f"[dim]{it.get('date')}[/] " if it.get("date") else ""
            console.print(f"  {marker}{d}{esc(it['content'])}")

    _sect("Facts & decisions", ov["facts"])
    _sect("Lessons", ov["lessons"], "[magenta]★[/] ")
    _sect("Failures", ov["failures"], "[red]⚠[/] ")
    _sect("Goals", ov["goals"])
    if ov.get("related_topics"):
        console.print("\n[bold magenta]Related topics:[/] "
                      + ", ".join(esc(t) for t in ov["related_topics"]))


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


def _agent_toggles_from_config() -> Optional[dict]:
    """Read the agent.* proactive-logging toggles from config (for
    `pmb connect --active` / `pmb setup`). Returns None on any error so the
    rules just fall back to all-categories-on."""
    try:
        cfg, _ = _open_config()
        keys = ("active_mode", "log_decisions", "log_completed", "log_lessons",
                "log_failures", "log_goals", "apply_lessons", "context_continuity")
        return {k: cfg.get(f"agent.{k}") for k in keys}
    except Exception:
        return None


@config_app.command("list")
def config_list(
    only_overridden: bool = typer.Option(
        False, "--only-overridden",
        help="Show only keys whose value differs from the schema default",
    ),
    pro: bool = typer.Option(
        False, "--pro",
        help="Show pro / advanced knobs in addition to the default-tier keys.",
    ),
    all_: bool = typer.Option(
        False, "--all",
        help="Show every config key, including experimental/internal knobs. "
             "Alias for --pro.",
    ),
):
    """Print PMB config keys with values and source.

    By default shows the curated DEFAULT tier (~25 keys most users care
    about). Pass --pro (or --all) to see every knob, including internal
    tunables and experimental flags.
    """
    from pmb.config import SCHEMA, is_default_tier
    show_all = pro or all_
    cfg, ws = _open_config()
    n_default = sum(1 for k in SCHEMA if is_default_tier(k))
    n_pro = len(SCHEMA) - n_default
    hidden_hint = (
        "" if show_all
        else f"  (+ {n_pro} pro knobs hidden — pass --pro to see them)"
    )
    console.print(Panel.fit(
        f"workspace: [cyan]{ws.name}[/] ({ws.id[:12]})\n"
        f"  workspace yaml: {cfg.workspace_path}\n"
        f"  global yaml:    {cfg.global_path}\n"
        f"  showing: {'all '+str(len(SCHEMA))+' keys' if show_all else str(n_default)+' default-tier keys'}"
        f"{hidden_hint}",
        title="PMB config sources",
    ))
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Tier", style="dim", width=4)
    table.add_column("Key", overflow="fold")
    table.add_column("Value", overflow="fold")
    table.add_column("Source", style="dim")
    table.add_column("Default", style="dim", overflow="fold")
    for key, setting in SCHEMA.items():
        is_def = is_default_tier(key)
        if not show_all and not is_def:
            continue
        value = cfg.get(key)
        src = cfg.source_of(key)
        if only_overridden and src == "default":
            continue
        tier_lbl = "[green]●[/]" if is_def else "[dim]○[/]"
        default_repr = repr(setting.default)
        table.add_row(tier_lbl, key, repr(value), src, default_repr)
    console.print(table)
    if not show_all:
        console.print(
            "\n[dim]● default-tier  ○ pro-tier  ·  `pmb config list --pro` to "
            "see everything[/]"
        )


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
        help="Connect to a remote PMB server. Two forms:\n"
             "  • SSH: user@host:/abs/path/to/repo (stdio over SSH tunnel)\n"
             "  • HTTP: http://host:8765/mcp (team-shared streamable-http)",
    ),
    bearer_token: Optional[str] = typer.Option(
        None, "--bearer-token", "--token",
        envvar="PMB_MCP_BEARER_TOKEN",
        help="Shared secret for the remote HTTP server (if it was started "
             "with `pmb mcp serve --bearer-token <secret>`).",
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
    active: bool = typer.Option(
        False, "--active",
        help="Proactive logging: the agent records its own decisions / lessons / "
             "what it did during coding, without waiting for 'remember'. Recall "
             "stays lazy. Default rules are conservative (PMB off until a trigger).",
    ),
    rules_only: bool = typer.Option(
        False, "--rules-only", "--update",
        help="Only refresh the AGENTS.md / CLAUDE.md rules block — don't touch "
             "the MCP config file. Use after a PMB update to bring the agent "
             "instructions in sync with the latest READ-FIRST guidance, without "
             "duplicating or re-adding the MCP server entry.",
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

    _toggles = _agent_toggles_from_config()
    effective_active = active or bool((_toggles or {}).get("active_mode"))

    # --rules-only: skip MCP config wiring, just refresh the markdown rules
    # block. Useful after a PMB upgrade to bring CLAUDE.md / AGENTS.md in
    # sync with the latest READ-FIRST instructions without re-adding the
    # MCP server entry.
    if rules_only:
        from pmb.cli.connect import (
            install_agent_rules,
            instruction_paths_for_agent,
        )
        try:
            paths = instruction_paths_for_agent(agent, Path.cwd())
        except Exception as e:
            console.print(f"[red]Unknown agent {agent!r}: {e}[/]")
            raise typer.Exit(code=2)
        results = []
        for inst_path in paths:
            try:
                action = install_agent_rules(
                    inst_path, active=effective_active,
                    active_toggles=(_toggles if effective_active else None),
                )
                results.append((inst_path, action, None))
            except Exception as e:
                results.append((inst_path, "error", str(e)))
            break  # global only
        for p, act, err in results:
            if err:
                console.print(f"[red]Failed: {p}[/] — {err}")
            else:
                colour = "green" if act in ("updated", "added") else "yellow"
                console.print(
                    f"[{colour}]Rules {act}:[/] {p}\n"
                    "  → MCP config untouched. Restart your agent to pick up "
                    "the new instructions."
                )
        return

    try:
        result = do_connect(
            agent, cwd=Path.cwd(), scope=scope, remote=remote, name_override=name,
            workspace_id=workspace, pmb_home=pmb_home, config_path=config_path,
            active=effective_active,
            active_toggles=(_toggles if effective_active else None),
            bearer_token=bearer_token,
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


@app.command()
def setup(
    agent: Optional[str] = typer.Argument(
        None, help="Agent to wire (claude-code / codex / cursor / ...). Omit to auto-detect."),
    active: bool = typer.Option(
        False, "--active",
        help="Install proactive-logging rules (agent records its own decisions/lessons)."),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Non-interactive: take the recommended defaults, no prompts."),
):
    """Guided first-time setup - detect your agent and wire PMB in one go.

      pmb setup                       # detect your agent, ask a couple questions
      pmb setup codex --active --yes  # non-interactive

    Defaults are already the effective ones (ablation-tuned) - you only need
    `pmb tune` if you want to change them. Full command list: `docs/COMMANDS.md`.
    """
    import shutil as _shutil
    from pmb.cli.connect import (
        detect_installed_agents, connect as do_connect, supported_agents,
    )
    detected = detect_installed_agents()
    ollama_ok = _shutil.which("ollama") is not None

    console.print(Panel.fit(
        f"Detected agents: [cyan]{', '.join(detected) or 'none found yet'}[/]\n"
        f"Ollama (optional local LLM): "
        + ("[green]installed[/]" if ollama_ok
           else "[dim]not installed - fine, PMB works fully offline without it[/]") + "\n"
        f"Defaults: ablation-tuned (BM25-heavy fusion, reranker off) - ready to use.",
        title="PMB setup",
    ))

    chosen = agent or (detected[0] if detected else None)
    if not chosen:
        if yes:
            console.print("[yellow]No known agent config found.[/] Run your agent "
                          "once, then [cyan]pmb connect --list[/].")
            return
        chosen = typer.prompt("Which agent to connect?", default="claude-code")
    if chosen not in supported_agents():
        console.print(f"[red]Unknown agent:[/] {chosen}. "
                      f"Options: {', '.join(supported_agents())}")
        raise typer.Exit(2)

    if not active and not yes:
        active = typer.confirm(
            "Proactive logging - should the agent record its own decisions/lessons "
            "as it works (not just on 'remember')?",
            default=False,
        )

    _toggles = _agent_toggles_from_config()
    eff_active = active or bool((_toggles or {}).get("active_mode"))
    try:
        result = do_connect(
            chosen, cwd=Path.cwd(), active=eff_active,
            active_toggles=(_toggles if eff_active else None),
        )
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2)

    console.print(
        f"[green]Connected[/] [bold]{chosen}[/] "
        f"({'active logging' if eff_active else 'conservative rules'}).  "
        f"[dim]{result['config_path']}[/]"
    )
    console.print(Panel.fit(
        "[bold]Next:[/]\n"
        "  1. Restart your agent so it loads PMB.\n"
        "  2. [cyan]pmb warmup[/]  - pre-load so the first recall is fast (optional).\n"
        "  3. [cyan]pmb ollama status[/]  - only if you want a fully-local LLM.\n"
        "  4. Commands: [cyan]docs/COMMANDS.md[/] or [cyan]pmb --help[/].",
        title="Done",
    ))


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


index_app = typer.Typer(help="Index external content into PMB memory (PDFs, code projects).")
app.add_typer(index_app, name="index")


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


# ===========================================================================
# Local-use features (private / offline). Categories:
#   C. Recall/research : timeline, insights, tags/collections
#   D. Own-your-data   : forget-topic, ttl/expiry, export, snapshots
#   E. Proactivity     : reminders, digest
#
# Every command below is CLI + display + write-layer ONLY. None of them call
# or alter engine.recall(), so retrieval quality (LoCoMo 94.5%) and latency
# are unaffected by construction. TTL/expiry is enforced by an explicit
# `prune-expired` sweep (archive, reversible), never inside recall.
# ===========================================================================

_DUR_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400,
              "w": 604800, "mo": 2592000, "y": 31536000}
_ALL_EVENTS = 10_000_000  # effectively "no limit" for analytics/export scans


def _parse_duration(text: str) -> Optional[float]:
    """'30d' / '12h' / '2w' / '3mo' / '1y' (or a bare integer = days) -> seconds.

    Returns None if unparseable.
    """
    import re
    t = (text or "").strip().lower()
    if not t:
        return None
    if t.isdigit():
        return float(int(t) * 86400)
    m = re.fullmatch(r"(\d+)\s*(mo|[smhdwy])", t)
    if not m:
        return None
    return float(int(m.group(1)) * _DUR_UNITS[m.group(2)])


def _day_key(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


def _kind_marker(meta: Optional[dict]) -> str:
    kind = (meta or {}).get("kind", "")
    if kind == "failure":
        return "[red]⚠[/] "
    if kind == "lesson":
        return "[magenta]★[/] "
    return ""


def _apply_ttl(eng, ulid: str, duration: str) -> None:
    """Stamp metadata.expires_at on an event. Used by note/learn/fact --ttl."""
    secs = _parse_duration(duration)
    if secs is None:
        console.print(f"[yellow]Ignored bad --ttl:[/] {duration} "
                      "(try 30d, 12h, 2w, 3mo, 1y)")
        return
    ev = eng.events.get_by_ulid(ulid)
    if ev is None:
        return
    meta = dict(ev.metadata or {})
    meta["expires_at"] = time.time() + secs
    eng.events.set_metadata(ulid, meta)
    console.print(f"[dim]TTL set: expires {_humanize_time(meta['expires_at'])}[/]")


@app.command()
def timeline(
    limit: int = typer.Option(60, "-n", "--limit", help="Max events to show"),
    event_type: Optional[str] = typer.Option(None, "--type", help="Filter by event type"),
    days: Optional[float] = typer.Option(None, "--days", help="Only events from the last N days"),
    newest_first: bool = typer.Option(False, "--newest-first",
                                      help="Reverse order (default: oldest -> newest)"),
):
    """Chronological view of your memory - the story of a project, by day.

      pmb timeline                 # last 60 memories, oldest->newest, grouped by day
      pmb timeline --days 7        # just this week
      pmb timeline --type goal     # only goals, over time

    Read-only.
    """
    eng = Engine()
    events = eng.events.list_active(eng.workspace.id, limit=_ALL_EVENTS, event_type=event_type)
    if days is not None:
        cutoff = time.time() - days * 86400
        events = [e for e in events if e.timestamp >= cutoff]
    events.sort(key=lambda e: e.timestamp, reverse=newest_first)
    # keep the most recent `limit`; when oldest-first, that's the tail
    events = events[:limit] if newest_first else events[-limit:]

    if not events:
        console.print("[yellow]No events in range. Add memories with `pmb note` "
                      "or connect an agent.[/]")
        return

    console.print(Panel.fit(
        f"[bold]{len(events)}[/] memories · workspace [cyan]{esc(eng.workspace.name)}[/]"
        + (f" · last {days:g}d" if days else ""),
        title="PMB timeline",
    ))
    last_day = None
    for e in events:
        day = _day_key(e.timestamp)
        if day != last_day:
            console.print(f"\n[bold cyan]{day}[/]")
            last_day = day
        clock = time.strftime("%H:%M", time.gmtime(e.timestamp))
        content = esc(e.content[:100]) + ("…" if len(e.content) > 100 else "")
        console.print(f"  [dim]{clock}[/]  {_kind_marker(e.metadata)}[{e.event_type}] {content}")


@app.command()
def insights():
    """Personal analytics over your memory: size, growth, and top topics.

    Read-only snapshot of what PMB has accumulated:
      • totals + type breakdown
      • how far back memory goes, and recent growth per week
      • most-mentioned topics (from the entity graph)
      • procedural memory (lessons / failures) and goals
    """
    from collections import defaultdict
    eng = Engine()
    s = eng.stats()
    ev = s["events"]
    events = eng.events.list_active(eng.workspace.id, limit=_ALL_EVENTS)

    oldest = ev.get("oldest_timestamp")
    newest = ev.get("newest_timestamp")
    span_days = ((newest - oldest) / 86400.0) if (oldest and newest) else 0.0

    console.print(Panel.fit(
        f"[bold]{ev['active']}[/] active memories ([dim]{ev['archived']} archived[/]) · "
        f"workspace [cyan]{esc(eng.workspace.name)}[/]\n"
        f"memory spans [bold]{span_days:.0f}[/] days "
        f"({_humanize_time(oldest)} -> {_humanize_time(newest)})",
        title="PMB insights",
    ))

    if ev["by_type"]:
        tt = Table(show_header=True, header_style="bold magenta", title="By type")
        tt.add_column("Type", style="cyan"); tt.add_column("Count", justify="right")
        for t, n in sorted(ev["by_type"].items(), key=lambda x: -x[1]):
            tt.add_row(t, str(n))
        console.print(tt)

    week_counts: dict = defaultdict(int)
    for e in events:
        week_counts[time.strftime("%Y-W%W", time.gmtime(e.timestamp))] += 1
    if week_counts:
        recent = sorted(week_counts.items())[-8:]
        peak = max(n for _, n in recent) or 1
        gt = Table(show_header=True, header_style="bold magenta", title="Growth (recent weeks)")
        gt.add_column("Week"); gt.add_column("New", justify="right"); gt.add_column("")
        for wk, n in recent:
            gt.add_row(wk, str(n), f"[green]{'█' * max(1, round(20 * n / peak))}[/]")
        console.print(gt)

    try:
        ents = eng.graph_top_entities(limit=12)
    except Exception:
        ents = []
    if ents:
        et = Table(show_header=True, header_style="bold magenta", title="Top topics (entity graph)")
        et.add_column("Topic", style="cyan"); et.add_column("Kind", style="dim")
        et.add_column("Mentions", justify="right")
        for e in ents:
            et.add_row(esc(str(e["name"])), str(e.get("kind", "")), str(e["n_mentions"]))
        console.print(et)

    n_lessons = sum(1 for e in events if (e.metadata or {}).get("kind") == "lesson")
    n_failures = sum(1 for e in events if (e.metadata or {}).get("kind") == "failure")
    n_goals = sum(1 for e in events if e.event_type == "goal")
    n_pinned = sum(1 for e in events if e.importance >= 0.9)
    hl = Table(show_header=True, header_style="bold magenta", title="Highlights")
    hl.add_column("Signal"); hl.add_column("Count", justify="right")
    hl.add_row("Lessons (procedural)", str(n_lessons))
    hl.add_row("Failures (don't-repeat)", str(n_failures))
    hl.add_row("Goals", str(n_goals))
    hl.add_row("Pinned / core (importance ≥ 0.9)", str(n_pinned))
    console.print(hl)


@app.command()
def digest(
    period: str = typer.Argument("today", help="today | week | month"),
    days: Optional[float] = typer.Option(None, "--days", help="Override: last N days"),
):
    """'What did I tell you recently?' - a quick recap of new memories.

      pmb digest            # since the start of today (UTC)
      pmb digest week       # last 7 days
      pmb digest --days 3   # last 3 days

    Read-only. Good for an end-of-day or weekly review.
    """
    from collections import defaultdict
    now = time.time()
    if days is not None:
        cutoff, label = now - days * 86400, f"last {days:g} days"
    elif period == "today":
        cutoff, label = now - (now % 86400), "today"
    elif period == "week":
        cutoff, label = now - 7 * 86400, "last 7 days"
    elif period == "month":
        cutoff, label = now - 30 * 86400, "last 30 days"
    else:
        console.print(f"[red]Unknown period:[/] {period}. Use today | week | month, or --days N.")
        raise typer.Exit(2)

    eng = Engine()
    events = [e for e in eng.events.list_active(eng.workspace.id, limit=_ALL_EVENTS)
              if e.timestamp >= cutoff]
    if not events:
        console.print(f"[yellow]Nothing new in {label}.[/]")
        return
    events.sort(key=lambda e: e.timestamp)

    console.print(Panel.fit(
        f"[bold]{len(events)}[/] new memories · [cyan]{label}[/] · "
        f"workspace {esc(eng.workspace.name)}",
        title="PMB digest",
    ))
    by_type: dict = defaultdict(list)
    for e in events:
        by_type[e.event_type].append(e)
    for etype, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
        console.print(f"\n[bold magenta]{etype}[/] ({len(items)})")
        for e in items[:30]:
            clock = time.strftime("%m-%d %H:%M", time.gmtime(e.timestamp))
            content = esc(e.content[:110]) + ("…" if len(e.content) > 110 else "")
            console.print(f"  [dim]{clock}[/]  {_kind_marker(e.metadata)}{content}")
        if len(items) > 30:
            console.print(f"  [dim]… and {len(items) - 30} more[/]")


@app.command()
def export(
    fmt: str = typer.Option("markdown", "--format", "-f", help="markdown | json"),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="Write to file (default: stdout)"),
    event_type: Optional[str] = typer.Option(None, "--type", help="Only this event type"),
    include_archived: bool = typer.Option(False, "--include-archived",
                                          help="Also dump archived memories"),
):
    """Export all memory to readable Markdown or JSON - your data, in the open.

      pmb export                          # Markdown to stdout
      pmb export -o memory.md             # Markdown to a file
      pmb export --format json -o mem.json

    Plain, unencrypted, human-readable. For an ENCRYPTED portable bundle, use
    `pmb workspace export` instead.
    """
    from collections import defaultdict
    from pmb.provenance import describe_source
    eng = Engine()
    events = eng.events.list_all(
        eng.workspace.id, limit=_ALL_EVENTS,
        event_type=event_type, include_archived=include_archived,
    )
    events.sort(key=lambda e: e.timestamp)

    if fmt == "json":
        payload = {
            "workspace": {"id": eng.workspace.id, "name": eng.workspace.name},
            "exported_at": _humanize_time(time.time()),
            "n_events": len(events),
            "events": [{
                "ulid": e.ulid, "type": e.event_type, "content": e.content,
                "timestamp": e.timestamp, "time": _humanize_time(e.timestamp),
                "importance": e.importance, "access_count": e.access_count,
                "archived": e.archived_at is not None, "metadata": e.metadata or {},
            } for e in events],
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    elif fmt == "markdown":
        lines = [f"# PMB memory export - {eng.workspace.name}", "",
                 f"_Exported {_humanize_time(time.time())} · {len(events)} memories_", ""]
        by_type: dict = defaultdict(list)
        for e in events:
            by_type[e.event_type].append(e)
        for etype, items in sorted(by_type.items()):
            lines.append(f"## {etype} ({len(items)})")
            lines.append("")
            for e in items:
                kind = (e.metadata or {}).get("kind")
                kmark = " ★lesson" if kind == "lesson" else (" ⚠failure" if kind == "failure" else "")
                tag = " `[archived]`" if e.archived_at is not None else ""
                lines.append(f"- **{_humanize_time(e.timestamp)}**{kmark}{tag} - {e.content}")
                lines.append(f"  - _from: {describe_source(e.metadata)} · "
                             f"importance {e.importance:.2f}_")
            lines.append("")
        text = "\n".join(lines)
    else:
        console.print(f"[red]Unknown format:[/] {fmt}. Use markdown | json.")
        raise typer.Exit(2)

    if out:
        Path(out).expanduser().write_text(text, encoding="utf-8")
        console.print(f"[green]Exported[/] {len(events)} memories -> {out}")
    else:
        print(text)  # raw, no rich markup parsing


@app.command(name="forget-topic")
def forget_topic(
    topic: str = typer.Argument(..., help="Topic / keyword to forget, e.g. 'project-x'"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview only, change nothing"),
    field: str = typer.Option("any", "--in",
                              help="Where to match: any | content | tag | source"),
):
    """Forget everything about a topic in one command (archives, reversible).

      pmb forget-topic project-x          # archive every memory mentioning it
      pmb forget-topic acme --dry-run     # preview first

    Case-insensitive substring match in content (and tags / source metadata).
    Archived, not hard-deleted - restore individually with `pmb unforget` if
    you change your mind.
    """
    eng = Engine()
    needle = topic.strip().lower()
    if not needle:
        console.print("[red]Empty topic.[/]")
        raise typer.Exit(2)
    events = eng.events.list_active(eng.workspace.id, limit=_ALL_EVENTS)

    def _matches(e) -> bool:
        meta = e.metadata or {}
        if field in ("any", "content") and needle in (e.content or "").lower():
            return True
        if field in ("any", "tag"):
            if any(needle in str(t).lower() for t in (meta.get("tags") or [])):
                return True
        if field in ("any", "source"):
            for k in ("source", "file", "project"):
                if needle in str(meta.get(k, "")).lower():
                    return True
        return False

    matched = [e for e in events if _matches(e)]
    if not matched:
        console.print(f"[yellow]No active memories match[/] '{esc(topic)}'.")
        return

    console.print(f"[bold]{len(matched)}[/] memories match '[cyan]{esc(topic)}[/]':")
    for e in matched[:15]:
        content = esc(e.content[:80]) + ("…" if len(e.content) > 80 else "")
        console.print(f"  [dim]{_humanize_time(e.timestamp)}[/] [{e.event_type}] {content}")
    if len(matched) > 15:
        console.print(f"  [dim]… and {len(matched) - 15} more[/]")

    if dry_run:
        console.print(f"[dim](dry-run) Would archive {len(matched)} memories. Nothing changed.[/]")
        return
    if not yes and not typer.confirm(
        f"Archive all {len(matched)} memories about '{topic}'?"
    ):
        console.print("[yellow]Cancelled.[/]")
        return
    for e in matched:
        eng.forget(e.ulid)
    console.print(f"[green]Archived[/] {len(matched)} memories about '{esc(topic)}'. "
                  f"[dim](reversible - archived, not deleted)[/]")


@app.command()
def ttl(
    ulid: str = typer.Argument(..., help="Event ULID"),
    duration: str = typer.Argument(..., help="30d / 12h / 2w / 3mo / 1y - or 'clear' to remove"),
):
    """Set (or clear) an expiry on a memory.

      pmb ttl 018f...  30d     # expire 30 days from now
      pmb ttl 018f...  clear   # remove the expiry

    Expiry is enforced only by `pmb prune-expired` (or a cron job) - it never
    touches recall, so there is zero effect on retrieval speed or quality.
    """
    eng = Engine()
    ev = eng.events.get_by_ulid(ulid)
    if ev is None:
        console.print(f"[red]No event with ULID[/] {ulid}")
        raise typer.Exit(2)
    meta = dict(ev.metadata or {})
    if duration.strip().lower() in ("clear", "none", "off"):
        meta.pop("expires_at", None)
        eng.events.set_metadata(ulid, meta)
        console.print(f"[yellow]Cleared[/] expiry on {ulid}")
        return
    secs = _parse_duration(duration)
    if secs is None:
        console.print(f"[red]Bad duration:[/] {duration}. Try 30d, 12h, 2w, 3mo, 1y.")
        raise typer.Exit(2)
    meta["expires_at"] = time.time() + secs
    eng.events.set_metadata(ulid, meta)
    console.print(f"[green]Set[/] expiry on {ulid} -> {_humanize_time(meta['expires_at'])}")


@app.command(name="prune-expired")
def prune_expired(
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview only, change nothing"),
):
    """Archive memories whose TTL has passed (set via `pmb ttl` / `--ttl`).

    Reversible (archives, doesn't delete). Run from cron / Task Scheduler for
    automatic cleanup. Off the recall path.
    """
    eng = Engine()
    now = time.time()
    events = eng.events.list_active(eng.workspace.id, limit=_ALL_EVENTS)
    expired = [e for e in events
               if isinstance((e.metadata or {}).get("expires_at"), (int, float))
               and e.metadata["expires_at"] <= now]
    if not expired:
        console.print("[green]Nothing expired.[/]")
        return
    console.print(f"[bold]{len(expired)}[/] memories past their TTL:")
    for e in expired[:15]:
        content = esc(e.content[:70]) + ("…" if len(e.content) > 70 else "")
        console.print(f"  [dim]exp {_humanize_time(e.metadata['expires_at'])}[/] {content}")
    if dry_run:
        console.print(f"[dim](dry-run) Would archive {len(expired)}.[/]")
        return
    for e in expired:
        eng.forget(e.ulid)
    console.print(f"[green]Archived[/] {len(expired)} expired memories.")


@app.command()
def tag(
    ulid: str = typer.Argument(..., help="Event ULID"),
    tags: List[str] = typer.Argument(..., help="One or more tags to add"),
):
    """Tag a memory for local organization (collections).

      pmb tag 018f... work urgent
      pmb tagged work          # later: list everything tagged 'work'
    """
    eng = Engine()
    ev = eng.events.get_by_ulid(ulid)
    if ev is None:
        console.print(f"[red]No event with ULID[/] {ulid}")
        raise typer.Exit(2)
    meta = dict(ev.metadata or {})
    current = list(meta.get("tags") or [])
    added = []
    for t in tags:
        t = t.strip()
        if t and t not in current:
            current.append(t); added.append(t)
    meta["tags"] = current
    eng.events.set_metadata(ulid, meta)
    console.print(f"[green]Tagged[/] {ulid} +{added or '(nothing new)'}  "
                  f"[dim](now: {', '.join(esc(t) for t in current) or '-'})[/]")


@app.command()
def untag(
    ulid: str = typer.Argument(...),
    tags: List[str] = typer.Argument(...),
):
    """Remove tag(s) from a memory."""
    eng = Engine()
    ev = eng.events.get_by_ulid(ulid)
    if ev is None:
        console.print(f"[red]No event with ULID[/] {ulid}")
        raise typer.Exit(2)
    meta = dict(ev.metadata or {})
    drop = set(tags)
    current = [t for t in (meta.get("tags") or []) if t not in drop]
    meta["tags"] = current
    eng.events.set_metadata(ulid, meta)
    console.print(f"[yellow]Untagged[/] {ulid}  "
                  f"[dim](now: {', '.join(esc(t) for t in current) or '-'})[/]")


@app.command()
def tags():
    """List all tags in this workspace with counts."""
    from collections import Counter
    eng = Engine()
    events = eng.events.list_active(eng.workspace.id, limit=_ALL_EVENTS)
    counter: Counter = Counter()
    for e in events:
        for t in (e.metadata or {}).get("tags") or []:
            counter[t] += 1
    if not counter:
        console.print("[yellow]No tags yet.[/] Add one: [cyan]pmb tag <ulid> work[/]")
        return
    t = Table(show_header=True, header_style="bold magenta", title="Tags")
    t.add_column("Tag", style="cyan"); t.add_column("Memories", justify="right")
    for name, n in counter.most_common():
        t.add_row(esc(str(name)), str(n))
    console.print(t)


@app.command()
def tagged(
    tag_name: str = typer.Argument(..., help="Tag to filter by"),
    limit: int = typer.Option(50, "-n", "--limit"),
):
    """List memories with a given tag (a local 'collection')."""
    eng = Engine()
    events = eng.events.list_active(eng.workspace.id, limit=_ALL_EVENTS)
    hits = [e for e in events if tag_name in ((e.metadata or {}).get("tags") or [])]
    if not hits:
        console.print(f"[yellow]No memories tagged[/] '{esc(tag_name)}'.")
        return
    hits.sort(key=lambda e: -e.timestamp)
    t = Table(show_header=True, header_style="bold magenta",
              title=f"Tagged '{esc(tag_name)}' ({len(hits)})")
    t.add_column("When", style="dim"); t.add_column("Type", style="cyan"); t.add_column("Content")
    for e in hits[:limit]:
        content = esc(e.content[:80]) + ("…" if len(e.content) > 80 else "")
        t.add_row(_humanize_time(e.timestamp), e.event_type, content)
    console.print(t)


@app.command()
def reminders(
    within: float = typer.Option(7.0, "--within", "-w",
                                 help="Flag goals due within N days as 'soon'"),
    all_goals: bool = typer.Option(False, "--all",
                                   help="Also list dated-later and undated open goals"),
):
    """Proactive reminders: goals that are overdue or due soon.

      pmb reminders              # overdue + due within 7 days
      pmb reminders --within 30  # look a month ahead
      pmb reminders --all        # also show later / undated open goals

    Reads your open goals (status pending / in_progress) and their due dates.
    Set a due date when creating a goal via your agent, or with the MCP tools.
    """
    eng = Engine()
    goals = eng.events.list_active(eng.workspace.id, limit=_ALL_EVENTS, event_type="goal")
    open_goals = [g for g in goals
                  if (g.metadata or {}).get("goal_status") not in ("done", "cancelled")]
    now = time.time()
    soon_cut = now + within * 86400
    overdue, soon, later, undated = [], [], [], []
    for g in open_goals:
        due = (g.metadata or {}).get("due_at")
        if not isinstance(due, (int, float)):
            undated.append(g); continue
        if due < now:
            overdue.append((due, g))
        elif due <= soon_cut:
            soon.append((due, g))
        else:
            later.append((due, g))
    overdue.sort(); soon.sort(); later.sort()

    if not (overdue or soon or (all_goals and (later or undated))):
        console.print("[green]Nothing due.[/] No overdue or upcoming goals.")
        return
    if overdue:
        console.print(f"\n[bold red]Overdue ({len(overdue)})[/]")
        for due, g in overdue:
            ago = (now - due) / 86400.0
            console.print(f"  [red]●[/] {esc(g.content[:80])}  "
                          f"[dim](due {_humanize_time(due)}, {ago:.0f}d ago)[/]")
    if soon:
        console.print(f"\n[bold yellow]Due soon ({len(soon)})[/]")
        for due, g in soon:
            ind = (due - now) / 86400.0
            console.print(f"  [yellow]●[/] {esc(g.content[:80])}  "
                          f"[dim](due {_humanize_time(due)}, in {ind:.0f}d)[/]")
    if all_goals and later:
        console.print(f"\n[bold]Later ({len(later)})[/]")
        for due, g in later:
            console.print(f"  [dim]● {esc(g.content[:80])}  (due {_humanize_time(due)})[/]")
    if all_goals and undated:
        console.print(f"\n[bold]Open, no due date ({len(undated)})[/]")
        for g in undated:
            console.print(f"  [dim]○ {esc(g.content[:80])}[/]")


# --- Local snapshots (offline backup, no cloud) ----------------------------

snapshot_app = typer.Typer(help="Local, offline snapshots of your workspace (no cloud).")
app.add_typer(snapshot_app, name="snapshot")


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
    note: Optional[str] = typer.Option(None, "--note", "-m", help="Label for this snapshot"),
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


# ═══════════════════════════════════════════════════════════════════════
# prepare-context — the line that hooks call to inject memory at session
# start. Reads the user message from stdin or a positional arg, prints a
# compact context block.
# ═══════════════════════════════════════════════════════════════════════

def _read_stdin_utf8() -> str:
    """Read stdin as UTF-8, regardless of the platform locale.

    Critical for the hook path: Claude Code pipes the user message as
    UTF-8, but on Windows `sys.stdin.read()` defaults to the locale
    codepage (cp1251 on a RU system), which mangles Cyrillic/non-ASCII
    into mojibake — and then the intent regexes never match. Read the raw
    bytes and decode UTF-8 explicitly (replace on error so we never crash
    the hook).
    """
    import sys as _sys
    try:
        data = _sys.stdin.buffer.read()
        return data.decode("utf-8", errors="replace")
    except Exception:
        # Fallback to the text stream if .buffer isn't available (e.g.
        # pytest capture replaces stdin with a StringIO).
        try:
            return _sys.stdin.read()
        except Exception:
            return ""


@app.command("prepare-context")
def prepare_context_cmd(
    message: Optional[str] = typer.Argument(
        None,
        help="The user message to prepare context for. If omitted, read from stdin.",
    ),
    stdin_flag: bool = typer.Option(
        False, "--stdin",
        help="Read the message from stdin instead of the positional arg.",
    ),
    max_chars: int = typer.Option(
        4000, "--max-chars",
        help="Hard cap on output size. Hosts truncate long hook output.",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q",
        help="Suppress empty / no-context output (silent if nothing to inject).",
    ),
    legacy: bool = typer.Option(
        False, "--legacy",
        help="Force the pre-auto-recall always-on bundle (project + lessons + "
             "recent + goals on every turn). Useful for A/B vs the new "
             "intent-classified path.",
    ),
):
    """Print a compact PMB context block for a user message.

    Designed to be wired into session-start hooks (`pmb hooks install`).
    Output is plain text — the hook host appends it to the prompt as
    an additional system note before the model thinks.

    The default behaviour goes through `pmb.hooks.auto_recall`:

      - skip silently on trivial input (greetings, acks, <5 chars)
      - PROJECT_PREP / PROJECT_OVERVIEW on known project names
      - PAST_QUERY → recall()  (this is the big one — fixes the
        "agent forgot to call recall" hole)
      - RECENT_QUERY → what_just_happened()
      - GOALS_QUERY → list_goals(in_progress)
      - LESSONS_QUERY → wider find_lessons() window
      - GENERIC_FACTUAL → low-cost recall, surface only if confident
      - lessons always run as a side-dish

    Pass `--legacy` to fall back to the older always-on bundle.
    Pass `auto_recall.enabled=false` in config for the same effect at
    the workspace level.
    """
    import sys as _sys
    if stdin_flag or message is None:
        msg = _read_stdin_utf8().strip()
    else:
        msg = (message or "").strip()
    if not msg:
        if quiet:
            return
        _sys.stdout.write("[pmb hook] no user message; nothing to prepare\n")
        return

    try:
        from pmb.core.engine import Engine
        eng = Engine()
    except Exception as e:
        if quiet:
            return
        _sys.stdout.write(f"[pmb hook] engine init failed: {e}\n")
        return

    # Respect the config switch and the CLI override.
    use_auto = not legacy and bool(eng.config.get("auto_recall.enabled"))

    if use_auto:
        from pmb.hooks import run_auto_context, format_context
        try:
            res = run_auto_context(
                eng, msg,
                min_chars=int(eng.config.get("auto_recall.min_message_chars") or 5),
                recall_top_k=int(eng.config.get("auto_recall.recall_top_k") or 5),
                recall_min_score=float(
                    eng.config.get("auto_recall.recall_min_score") or 0.30
                ),
                surface_decisions=bool(
                    eng.config.get("auto_recall.surface_decisions")
                ),
            )
        except Exception as e:
            if quiet:
                return
            _sys.stdout.write(f"[pmb hook] auto-recall failed: {e}\n")
            return

        if res.skipped or res.is_empty():
            if quiet:
                return
            _sys.stdout.write(
                f"[pmb hook] no context to inject "
                f"(intents={','.join(res.intents)}, "
                f"reason={res.skip_reason or 'no-match'}).\n"
            )
            return

        # Allow either CLI flag or config to bound output size; smaller wins.
        cap = min(
            max_chars,
            int(eng.config.get("auto_recall.budget_chars") or 4000),
        )
        include_trace = bool(eng.config.get("auto_recall.include_trace"))
        text = format_context(res, max_chars=cap, include_trace=include_trace)
        if not text:
            if quiet:
                return
            _sys.stdout.write("[pmb hook] auto-recall produced empty output.\n")
            return
        _sys.stdout.write(text + "\n")
        return

    # ── Legacy path: always-on bundle (kept for --legacy A/B testing) ──
    try:
        out: dict = {}
        det = eng.detect_project_in_text(msg)
        if det:
            ov = eng.project_overview(det["name"])
            out["project_context"] = ov
            try:
                arcs = eng.active_arcs_for_project(det["name"], limit=2)
                if arcs:
                    out["active_arcs"] = arcs
            except Exception:
                pass
        try:
            ls = eng.find_lessons(query=msg, limit=5)
            if ls:
                eng._log_lesson_surfaces(ls, query=msg, source="hook.prepare")
                out["lessons"] = ls
        except Exception:
            pass
        try:
            act = eng.recent_activity(minutes=1440.0, limit=8)
            if act:
                out["recent_activity"] = act
        except Exception:
            pass
        try:
            goals = eng.list_goals(status="in_progress", limit=5)
            if goals:
                out["open_goals"] = goals
        except Exception:
            pass
    except Exception as e:
        if quiet:
            return
        _sys.stdout.write(f"[pmb hook] prepare failed: {e}\n")
        return

    if not out:
        if quiet:
            return
        _sys.stdout.write(
            "[pmb hook] no project / lesson / activity matched. "
            "Answer normally.\n"
        )
        return

    buf: list[str] = []
    buf.append("== PMB context for this turn ==")
    buf.append(f"(matched on message: {msg[:80]!r})")

    pc = out.get("project_context")
    if pc and not pc.get("empty"):
        ent = pc.get("entity") or {}
        buf.append(f"\nProject: {ent.get('name')} ({ent.get('n_mentions')} mentions)")
        kf = pc.get("key_facts", [])[:5]
        if kf:
            buf.append("Key facts:")
            for f in kf:
                buf.append(f"  - {f.get('content','')[:160]}")
        ls_in_pc = pc.get("lessons", [])[:5]
        if ls_in_pc:
            buf.append("Lessons (RULES to follow):")
            for L in ls_in_pc:
                sid = L.get("surface_id")
                tag = f" [surface_id={sid}]" if sid else ""
                buf.append(f"  ! {L.get('content','')[:200]}{tag}")
        dec = pc.get("decisions", [])[:3]
        if dec:
            buf.append("Past decisions:")
            for d in dec:
                buf.append(f"  > {d.get('content','')[:160]}")
        og = pc.get("open_goals", [])[:3]
        if og:
            buf.append("Open goals:")
            for g in og:
                buf.append(f"  * {g.get('content', g.get('title',''))[:120]}")

    ls_flat = out.get("lessons", [])
    if ls_flat and (not pc or not pc.get("lessons")):
        buf.append("\nLessons matching this message:")
        for L in ls_flat[:5]:
            sid = L.get("surface_id")
            tag = f" [surface_id={sid}]" if sid else ""
            buf.append(f"  ! {L.get('content','')[:200]}{tag}")

    ra = out.get("recent_activity", [])
    if ra:
        buf.append("\nRecent activity (last 24h):")
        for a in ra[:5]:
            buf.append(f"  - {a.get('content','')[:120]}")

    arcs = out.get("active_arcs", [])
    if arcs:
        buf.append("\nActive narrative arcs:")
        for a in arcs[:2]:
            buf.append(f"  ~ {a.get('title','')[:120]} ({a.get('n_events')} events)")

    buf.append("")
    buf.append("If a Lesson with [surface_id=N] applies, FOLLOW it and after")
    buf.append("acting call mark_lesson_followed(surface_id=N, followed=True,")
    buf.append("note=\"<one line: what you did>\").")

    text = "\n".join(buf)
    if len(text) > max_chars:
        text = text[: max_chars - 40] + "\n... [context truncated]"
    _sys.stdout.write(text + "\n")


# ─── pmb auto-context — debug-friendly inspection of the hook output ──

@app.command("auto-context")
def auto_context_cmd(
    message: str = typer.Argument(
        ..., help="The message to classify and dispatch on.",
    ),
    show_json: bool = typer.Option(
        False, "--json",
        help="Print the structured AutoContextResult as JSON instead of the "
             "formatted context block.",
    ),
    max_chars: int = typer.Option(
        4000, "--max-chars",
        help="Cap on the formatted context block (ignored when --json).",
    ),
):
    """Inspect what the auto-recall hook would inject for a given message.

    Useful for debugging "why didn't PMB fire here?" / "why did it fire
    on this trivial input?". The classifier is regex-only so behaviour is
    deterministic and reproducible.

    Examples:

      pmb auto-context "fix the recall bug in PMB"
      pmb auto-context "когда я последний раз правил docker-compose"
      pmb auto-context "what are my open goals" --json
    """
    from pmb.core.engine import Engine
    from pmb.hooks import run_auto_context, format_context

    eng = Engine()
    res = run_auto_context(
        eng, message,
        min_chars=int(eng.config.get("auto_recall.min_message_chars") or 5),
        recall_top_k=int(eng.config.get("auto_recall.recall_top_k") or 5),
        recall_min_score=float(
            eng.config.get("auto_recall.recall_min_score") or 0.30
        ),
        surface_decisions=bool(eng.config.get("auto_recall.surface_decisions")),
    )
    if show_json:
        import json
        from dataclasses import asdict
        console.print_json(json.dumps(asdict(res), default=str, ensure_ascii=False))
        return
    text = format_context(res, max_chars=max_chars, include_trace=True)
    if not text:
        console.print(
            f"[dim]no context (intents={','.join(res.intents)}, "
            f"reason={res.skip_reason or 'no-match'}, "
            f"latency={res.latency_ms}ms)[/]"
        )
        return
    # markup=False: the formatted block contains `[surface_id=N]` and
    # `[intents=...]` brackets that Rich would otherwise eat as style tags.
    console.print(text, markup=False, highlight=False)


# ─── pmb session-restore — rebuild context after a compaction ─────────

@app.command("session-restore")
def session_restore_cmd(
    minutes: Optional[int] = typer.Option(
        None, "--minutes", "-m",
        help="Window to summarise (default: config session.brief_minutes). "
             "Used when no active session is bound (e.g. fresh hook process).",
    ),
    max_chars: int = typer.Option(
        4000, "--max-chars",
        help="Hard cap on output size.",
    ),
    no_project: bool = typer.Option(
        False, "--no-project",
        help="Skip the project_overview section (session_brief only).",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q",
        help="Print nothing when there's no session to restore.",
    ),
):
    """Rebuild 'where you left off' after the agent's context compacts.

    Designed for a SessionStart(compact|resume) hook (`pmb hooks install`):
    when Claude Code compacts the conversation, this prints what THIS
    session decided / did / learned + the project overview, so the agent
    picks the thread back up instead of re-asking the user.

    Run it by hand to preview:  pmb session-restore -m 180
    """
    import sys as _sys
    try:
        from pmb.core.engine import Engine
        from pmb.hooks import build_session_restore
        eng = Engine()
    except Exception as e:
        if not quiet:
            _sys.stdout.write(f"[pmb hook] engine init failed: {e}\n")
        return
    try:
        text = build_session_restore(
            eng,
            minutes=float(minutes) if minutes else None,
            include_project=not no_project,
            max_chars=max_chars,
        )
    except Exception as e:
        if not quiet:
            _sys.stdout.write(f"[pmb hook] session-restore failed: {e}\n")
        return
    if not text:
        if not quiet:
            _sys.stdout.write(
                "[pmb hook] nothing to restore (no recent session activity).\n"
            )
        return
    _sys.stdout.write(text + "\n")


# ─── pmb lesson-followcheck — infer follow-through, no model cooperation ──

@app.command("lesson-followcheck")
def lesson_followcheck_cmd(
    window: int = typer.Option(
        30, "--window", "-w",
        help="Minutes back to scan for unconfirmed lesson surfaces.",
    ),
    min_overlap: int = typer.Option(
        2, "--min-overlap",
        help="Distinctive-token overlap required to count a lesson as "
             "followed. Higher = stricter (fewer false positives).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Show what would be marked without writing.",
    ),
    show_json: bool = typer.Option(
        False, "--json", help="Emit the FollowCheckResult as JSON.",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q",
        help="Print nothing (for the Stop hook — it runs silently).",
    ),
):
    """Infer lesson follow-through from recorded activity — no model needed.

    For each lesson that surfaced recently but was never confirmed, check
    whether its distinctive tokens show up in what the agent actually did
    this turn (recorded activity). If so, mark it followed with an honest
    'auto-detected' note. Lessons with no evidence stay unconfirmed — we
    never fabricate a follow.

    Designed for a Stop hook (`pmb hooks install`) so the adherence
    dashboard's follow-rate reflects reality instead of sitting at 0%
    because models don't self-report.

    Preview:  pmb lesson-followcheck --dry-run --json
    """
    import sys as _sys
    try:
        from pmb.core.engine import Engine
        from pmb.hooks import run_followcheck
        eng = Engine()
    except Exception as e:
        if not quiet:
            _sys.stdout.write(f"[pmb hook] engine init failed: {e}\n")
        return
    try:
        res = run_followcheck(
            eng,
            window_minutes=float(window),
            activity_minutes=float(window),
            min_overlap=min_overlap,
            apply=not dry_run,
        )
    except Exception as e:
        if not quiet:
            _sys.stdout.write(f"[pmb hook] followcheck failed: {e}\n")
        return
    if show_json:
        import json
        console.print_json(json.dumps(res.to_dict(), ensure_ascii=False))
        return
    if quiet:
        return
    verb = "would mark" if dry_run else "marked"
    if res.marked_followed:
        console.print(
            f"[green]{verb} {res.marked_followed} lesson(s) followed[/] "
            f"(checked {res.checked})"
        )
        for v in res.verdicts:
            console.print(
                f"  ✓ surface {v.surface_id} · overlap: "
                f"{', '.join(v.overlap[:5])}", markup=False, highlight=False,
            )
    else:
        console.print(
            f"[dim]no follow-through inferred "
            f"(checked {res.checked}, reason={res.skipped_reason or 'no overlap'})[/]"
        )


# ═══════════════════════════════════════════════════════════════════════
# pmb hooks install / list / uninstall — force-feed prepare() into the
# agent's session-start hook so the READ-FIRST workflow is not optional.
# ═══════════════════════════════════════════════════════════════════════

hooks_app = typer.Typer(
    help="Install force-feeding session-start hooks into your agent's "
         "config so PMB context arrives BEFORE the model thinks.",
)
app.add_typer(hooks_app, name="hooks")


# ═══════════════════════════════════════════════════════════════════════
# pmb mcp serve — expose the MCP server over HTTP for team-shared mode.
# One persistent process on a homelab box / Tailscale node serves every
# developer's agent. Same workspace, same memory, no per-machine state.
# ═══════════════════════════════════════════════════════════════════════

mcp_app = typer.Typer(
    help="Run the MCP server. Stdio (per-developer) is the default; "
         "streamable-http exposes one shared instance to a team.",
)
app.add_typer(mcp_app, name="mcp")


@mcp_app.command("serve")
def mcp_serve_cmd(
    transport: str = typer.Option(
        "streamable-http", "--transport",
        help="streamable-http (HTTP for team-shared) or stdio (one-shot).",
    ),
    host: str = typer.Option(
        "127.0.0.1", "--host",
        help="Bind address. Use 0.0.0.0 to accept connections from the LAN / "
             "Tailscale mesh.",
    ),
    port: int = typer.Option(
        8765, "--port",
        help="Bind port (default 8765). Make sure it doesn't collide with "
             "`pmb dashboard`.",
    ),
    path: str = typer.Option(
        "/mcp", "--path",
        help="Mount path for the streamable-http transport.",
    ),
    workspace: Optional[str] = typer.Option(
        None, "--workspace", "-w",
        help="Force a specific workspace id (default: cwd-based detection).",
    ),
    bearer_token: Optional[str] = typer.Option(
        None, "--bearer-token", "--token",
        envvar="PMB_MCP_BEARER_TOKEN",
        help="Shared secret required in `Authorization: Bearer <token>`. "
             "Strongly recommended if `--host` is not 127.0.0.1.",
    ),
):
    """Run the PMB MCP server.

    Examples:

      # Local stdio (what `pmb connect` wires by default — usually no
      # need to run this manually)
      pmb mcp serve --transport stdio

      # Team-shared HTTP on Tailscale mesh
      pmb mcp serve --transport streamable-http --host 0.0.0.0 --port 8765 \\
                   --workspace team --bearer-token <secret>

      # Then on each developer's machine:
      pmb connect claude-code --remote http://memo.local:8765/mcp \\
                              --bearer-token <secret>
    """
    import os, sys

    if transport not in ("stdio", "streamable-http", "http", "https"):
        console.print(f"[red]Unknown transport {transport!r}.[/]")
        raise typer.Exit(2)
    if transport in ("http", "https"):
        transport = "streamable-http"

    # Hand off to the existing server entrypoint via env-vars so we share
    # one code path with `pmb-mcp` when it's spawned by an agent.
    if workspace:
        os.environ["PMB_WORKSPACE"] = workspace
    os.environ["PMB_MCP_TRANSPORT"] = transport
    os.environ["PMB_MCP_HOST"] = host
    os.environ["PMB_MCP_PORT"] = str(port)
    os.environ["PMB_MCP_PATH"] = path
    if bearer_token:
        os.environ["PMB_MCP_BEARER_TOKEN"] = bearer_token

    if transport == "streamable-http":
        url = f"http://{host}:{port}{path}"
        auth = (
            f"[green]bearer-token enabled[/]"
            if bearer_token
            else "[yellow]UNAUTHENTICATED — bind to 127.0.0.1 or set --bearer-token[/]"
        )
        console.print(Panel.fit(
            f"PMB MCP server starting\n"
            f"  transport: streamable-http\n"
            f"  url:       {url}\n"
            f"  auth:      {auth}\n"
            f"  workspace: {os.environ.get('PMB_WORKSPACE', '(cwd-detected)')}\n\n"
            f"Wire an agent to it:\n"
            f"  [cyan]pmb connect claude-code --remote {url}"
            + (f" --bearer-token <secret>" if bearer_token else "") + "[/]\n\n"
            f"Stop with Ctrl-C.",
            title="PMB · mcp serve",
        ))
    else:
        console.print("[dim]Running stdio MCP server. Use Ctrl-D / Ctrl-C to stop.[/]")

    from pmb.mcp.server import main as _server_main
    _server_main()


@hooks_app.command("install")
def hooks_install_cmd(
    agent: str = typer.Argument(
        ...,
        help="claude-code | codex | cursor (cursor: unsupported, prints why).",
    ),
):
    """Install PMB's lifecycle hooks for AGENT.

    For claude-code this wires THREE hooks:
      • UserPromptSubmit → pmb prepare-context  (auto-recall: inject
        lessons / decisions / recall / project context per turn)
      • SessionStart     → pmb session-restore  (rebuild context after a
        compaction / resume)
      • Stop             → pmb lesson-followcheck (infer which lessons
        were actually followed — feeds the adherence dashboard)

    For codex it wires the per-turn context injector only.

    Examples:
        pmb hooks install claude-code
        pmb hooks install codex
    """
    from pmb.cli.hooks import install_hook
    try:
        result = install_hook(agent)
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2)

    act = result.get("action")
    if act == "not_supported":
        console.print(
            f"[yellow]Hooks not supported for {result['agent']}.[/]\n"
            f"  {result.get('reason','')}"
        )
        return

    # claude-code returns a multi-event `actions` list; codex returns a
    # single `action`/`command`.
    if result.get("actions"):
        lines = "\n".join(
            f"  • {a['event']}: {a['action']}" for a in result["actions"]
        )
        console.print(Panel.fit(
            f"[green]Hooks installed[/] for [bold]{result['agent']}[/]\n"
            f"  config: {result.get('path','?')}\n{lines}\n\n"
            "Restart the agent. Now, every turn:\n"
            "  · context is injected BEFORE the model thinks (auto-recall)\n"
            "  · after a compaction it rebuilds where you left off\n"
            "  · at turn end it scores lesson follow-through\n"
            "— the adherence problem, handled at the protocol level.",
            title="PMB · hooks installed",
        ))
    else:
        colour = "green" if act in ("installed", "created", "updated") else "yellow"
        console.print(Panel.fit(
            f"[{colour}]Hook {act}[/] for [bold]{result['agent']}[/]\n"
            f"  config:  {result.get('path','?')}\n"
            f"  command: {result.get('command','?')}\n\n"
            "Restart the agent. Every new session now reads PMB BEFORE it\n"
            "starts thinking — adherence problem solved at the protocol level.",
            title="PMB · hook installed",
        ))


@hooks_app.command("list")
def hooks_list_cmd():
    """Show which lifecycle hooks are currently installed."""
    from pmb.cli.hooks import list_installed
    rows = list_installed()
    t = Table(show_header=True, header_style="bold magenta", title="PMB hooks")
    t.add_column("Agent"); t.add_column("Event"); t.add_column("Installed")
    t.add_column("Path")
    for r in rows:
        mark = "[green]✓[/]" if r["installed"] else "[dim]–[/]"
        t.add_row(r["agent"], r.get("event", "?"), mark, r["path"])
    console.print(t)


@hooks_app.command("uninstall")
def hooks_uninstall_cmd(
    agent: str = typer.Argument(..., help="claude-code | codex"),
):
    """Remove PMB's session-start hook for AGENT."""
    from pmb.cli.hooks import uninstall_hook
    try:
        r = uninstall_hook(agent)
    except ValueError as e:
        console.print(f"[red]{e}[/]"); raise typer.Exit(2)
    if r["action"] == "not_installed":
        console.print(f"[dim]Nothing to remove for {r['agent']}.[/]")
    else:
        console.print(f"[green]Removed[/] hook for {r['agent']} ({r.get('path','?')})")


if __name__ == "__main__":
    app()
