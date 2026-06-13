"""PMB TUI Workspace - multi-tab terminal UI for memory management.

Layout (k9s / lazygit / mem0 dashboard inspired):

  ┌────────────────────────────────────────────────────────────────┐
  │ workspace: pmb │ 19 events │ 92 entities │ 11:30:45            │  ← status
  ├──────────────────────────────────────────────────────────────────┤
  │ [1] Memory  [2] Recall  [3] Stats  [4] Dedup  [5] Tune          │  ← tabs
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  MAIN VIEW (depends on active tab)                               │
  │  - Memory: paginated event list + filter + detail pane           │
  │  - Recall: query input + results + scoring breakdown             │
  │  - Stats:  live MCP perf metrics (auto-refresh)                  │
  │  - Dedup:  borderline pairs with merge/keep actions              │
  │  - Tune:   67 settings editor (from tui_config)                  │
  │                                                                  │
  ├──────────────────────────────────────────────────────────────────┤
  │ 1-5 tabs · / search · r reload · q quit · ? help                 │  ← footer
  └──────────────────────────────────────────────────────────────────┘

Lifted ideas from:
  - k9s:       drill-down navigation, command bindings, resource list
  - lazygit:   master/detail panes, action shortcuts
  - htop:      live refresh, color-coded metrics
  - mem0:      memory browser with filters
"""
from __future__ import annotations

import time

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Static,
    TabbedContent,
    TabPane,
)

from pmb.core.engine import Engine

# ────────────────────────────────────────────────────────────────────────
# Tab 1: Memory browser
# ────────────────────────────────────────────────────────────────────────

class MemoryTab(Vertical):
    """Paginated, filterable list of recent events with detail pane."""

    def __init__(self, eng: Engine):
        super().__init__()
        self.eng = eng
        self.events: list = []
        self.filter_text = ""

    def compose(self) -> ComposeResult:
        with Horizontal(classes="filter-bar"):
            yield Label("filter:", classes="filter-label")
            yield Input(
                placeholder="type / event_type / content substring…",
                id="memory-filter",
            )
        with Horizontal(classes="memory-body"):
            yield DataTable(id="memory-table", zebra_stripes=True)
            with VerticalScroll(id="memory-detail"):
                yield Static(
                    "[dim]Select a row to see details.[/]",
                    id="memory-detail-content",
                )

    def on_mount(self) -> None:
        table = self.query_one("#memory-table", DataTable)
        table.add_columns("Time", "Type", "Importance", "Content")
        table.cursor_type = "row"
        self.refresh_events()

    def refresh_events(self) -> None:
        try:
            self.events = self.eng.events.list_active(
                self.eng.workspace.id, limit=200,
            )
        except Exception:
            self.events = []
        self._render_table()

    def _render_table(self) -> None:
        table = self.query_one("#memory-table", DataTable)
        table.clear()
        f = self.filter_text.lower().strip()
        for ev in self.events:
            content = (ev.content or "")[:80]
            if f and (f not in (ev.event_type or "").lower()
                      and f not in content.lower()):
                continue
            t = time.strftime("%H:%M:%S", time.localtime(ev.timestamp))
            imp = f"{ev.importance:.2f}" if ev.importance is not None else "-"
            table.add_row(t, ev.event_type, imp, content, key=ev.ulid)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "memory-filter":
            self.filter_text = event.value
            self._render_table()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        ulid = event.row_key.value
        if not ulid:
            return
        ev = next((e for e in self.events if e.ulid == ulid), None)
        if not ev:
            return
        detail = self.query_one("#memory-detail-content", Static)
        meta_str = ""
        if ev.metadata:
            import json
            try:
                meta_str = json.dumps(ev.metadata, ensure_ascii=False, indent=2)
            except Exception:
                meta_str = str(ev.metadata)
        detail.update(
            f"[bold cyan]{ev.event_type}[/] · ulid=[dim]{ev.ulid}[/]\n"
            f"[yellow]importance:[/] {ev.importance}\n"
            f"[yellow]tier:[/] {ev.tier}\n"
            f"[yellow]timestamp:[/] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ev.timestamp))}\n"
            f"[yellow]access_count:[/] {ev.access_count}\n\n"
            f"[bold]Content:[/]\n{ev.content}\n\n"
            f"[bold]Metadata:[/]\n{meta_str or '(none)'}\n"
        )


# ────────────────────────────────────────────────────────────────────────
# Tab 2: Recall playground
# ────────────────────────────────────────────────────────────────────────

class RecallTab(Vertical):
    """Interactive recall: type query → see top-K + scoring breakdown."""

    def __init__(self, eng: Engine):
        super().__init__()
        self.eng = eng

    def compose(self) -> ComposeResult:
        with Horizontal(classes="recall-input"):
            yield Input(
                placeholder="type a query and press Enter…",
                id="recall-query",
            )
        yield Static(
            "[dim]Press Enter to run recall. Results show top 5 with score breakdown.[/]",
            id="recall-status",
        )
        yield DataTable(id="recall-results", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one("#recall-results", DataTable)
        table.add_columns("Score", "BM25", "Vector", "Content")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "recall-query":
            return
        query = event.value.strip()
        if not query:
            return
        status = self.query_one("#recall-status", Static)
        table = self.query_one("#recall-results", DataTable)
        status.update("[yellow]running...[/]")
        table.clear()
        t0 = time.perf_counter()
        try:
            pack = self.eng.recall(query, top_k=10)
        except Exception as e:
            status.update(f"[red]error: {e}[/]")
            return
        dt = (time.perf_counter() - t0) * 1000
        if not pack.results:
            status.update(f"[dim]no results in {dt:.0f}ms[/]")
            return
        status.update(
            f"[green]{len(pack.results)} results[/] · "
            f"confidence [yellow]{pack.confidence:.2f}[/] · "
            f"[cyan]{dt:.0f}ms[/]"
        )
        for r in pack.results:
            content = (r.content or "")[:100]
            table.add_row(
                f"{r.score:.3f}",
                f"{r.bm25_score:.2f}" if hasattr(r, "bm25_score") else "-",
                f"{r.vec_score:.2f}" if hasattr(r, "vec_score") else "-",
                content,
            )


# ────────────────────────────────────────────────────────────────────────
# Tab 3: Stats (live MCP perf)
# ────────────────────────────────────────────────────────────────────────

class StatsTab(Vertical):
    """Live MCP performance metrics - auto-refresh every 2s."""

    def __init__(self, eng: Engine):
        super().__init__()
        self.eng = eng
        self._refresh_timer = None

    def compose(self) -> ComposeResult:
        yield Static("", id="stats-summary")
        yield Static("[bold]Per-tool stats (last 24h)[/]", classes="section-title")
        yield DataTable(id="stats-tools", zebra_stripes=True)
        yield Static("[bold]Recent calls (last 20)[/]", classes="section-title")
        yield DataTable(id="stats-recent", zebra_stripes=True)

    def on_mount(self) -> None:
        tools = self.query_one("#stats-tools", DataTable)
        tools.add_columns("Tool", "Calls", "p50", "p95", "p99", "max", "avg", "errors")
        recent = self.query_one("#stats-recent", DataTable)
        recent.add_columns("Time", "Tool", "Duration", "Status")
        self.refresh_stats()
        self._refresh_timer = self.set_interval(2.0, self.refresh_stats)

    def refresh_stats(self) -> None:
        try:
            from pmb.mcp.perf import get_perf_stats
            data = get_perf_stats(
                db_path=self.eng.workspace.db_path,
                workspace_id=self.eng.workspace.id,
                hours=24,
            )
        except Exception as e:
            self.query_one("#stats-summary", Static).update(
                f"[red]error: {e}[/]"
            )
            return

        summary = self.query_one("#stats-summary", Static)
        summary.update(
            f"[bold]MCP perf[/] (last 24h) · "
            f"[cyan]{data.get('total_calls', 0)}[/] calls · "
            f"[{'red' if data.get('error_count', 0) > 0 else 'green'}]"
            f"{data.get('error_count', 0)} errors[/] · "
            f"refresh: [dim]every 2s[/]"
        )

        tools = self.query_one("#stats-tools", DataTable)
        tools.clear()
        for t in data.get("by_tool", []):
            err = str(t.get("errors", 0))
            tools.add_row(
                t.get("name", "?"),
                str(t.get("count", 0)),
                f"{t.get('p50', 0):.1f}ms",
                f"{t.get('p95', 0):.1f}ms",
                f"{t.get('p99', 0):.1f}ms",
                f"{t.get('max', 0):.1f}ms",
                f"{t.get('avg', 0):.1f}ms",
                f"[red]{err}[/]" if t.get("errors", 0) > 0 else err,
            )

        recent = self.query_one("#stats-recent", DataTable)
        recent.clear()
        for r in data.get("recent", [])[:20]:
            t = time.strftime("%H:%M:%S", time.localtime(r.get("ts", 0)))
            recent.add_row(
                t,
                r.get("tool", "?"),
                f"{r.get('ms', 0):.1f}ms",
                "[green]ok[/]" if r.get("success") else "[red]err[/]",
            )


# ────────────────────────────────────────────────────────────────────────
# Tab 4: Dedup review
# ────────────────────────────────────────────────────────────────────────

class DedupTab(Vertical):
    """Borderline duplicate pairs awaiting decision."""

    def __init__(self, eng: Engine):
        super().__init__()
        self.eng = eng

    def compose(self) -> ComposeResult:
        yield Static("", id="dedup-summary")
        with Horizontal(classes="dedup-actions"):
            yield Label("[dim]Tab to navigate, Enter to merge, K to keep both[/]")
        yield DataTable(id="dedup-table", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one("#dedup-table", DataTable)
        table.add_columns("Similarity", "Type", "Existing", "New")
        table.cursor_type = "row"
        self.refresh_dedup()

    def refresh_dedup(self) -> None:
        try:
            pairs = self.eng.dedupe_list_pending(limit=200)
        except Exception:
            pairs = []
        self.query_one("#dedup-summary", Static).update(
            f"[bold]Pending borderline pairs:[/] [yellow]{len(pairs)}[/]"
        )
        table = self.query_one("#dedup-table", DataTable)
        table.clear()
        for p in pairs:
            table.add_row(
                f"{p['similarity']:.3f}",
                p.get("event_type", "?"),
                (p.get("candidate_content") or "")[:60],
                (p.get("new_content") or "")[:60],
                key=str(p["id"]),
            )


# ────────────────────────────────────────────────────────────────────────
# Tab 5: Tune (reuses tui_config approach)
# ────────────────────────────────────────────────────────────────────────

class TuneTab(Vertical):
    """All 67 settings with current value / source / type."""

    def __init__(self, eng: Engine):
        super().__init__()
        self.eng = eng
        self.cfg = eng.config

    def compose(self) -> ComposeResult:
        with Horizontal(classes="filter-bar"):
            yield Label("filter:")
            yield Input(placeholder="key substring…", id="tune-filter")
        yield DataTable(id="tune-table", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one("#tune-table", DataTable)
        table.add_columns("Key", "Value", "Source", "Type", "Default")
        table.cursor_type = "row"
        self.filter_text = ""
        self._refresh_table()

    def _refresh_table(self) -> None:
        from pmb.config import SCHEMA
        table = self.query_one("#tune-table", DataTable)
        table.clear()
        f = self.filter_text.lower()
        for key in sorted(SCHEMA):
            if f and f not in key.lower():
                continue
            s = SCHEMA[key]
            value = self.cfg.get(key)
            source = self.cfg.source_of(key)
            color = {"workspace": "green", "global": "yellow",
                     "override": "magenta", "default": "dim"}.get(source, "white")
            table.add_row(
                key, str(value),
                f"[{color}]{source}[/]",
                s.type.__name__,
                str(s.default),
                key=key,
            )

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "tune-filter":
            self.filter_text = event.value
            self._refresh_table()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        from pmb.cli.tui_config import EditScreen
        from pmb.config import SCHEMA
        key = event.row_key.value
        if not key or key not in SCHEMA:
            return
        setting = SCHEMA[key]
        current = self.cfg.get(key)

        def _after(new_value):
            if new_value is None or new_value == str(current):
                return
            try:
                self.cfg.set_workspace(key, new_value)
                self._refresh_table()
                self.app.notify(f"saved {key} = {new_value}", severity="information")
            except Exception as e:
                self.app.notify(f"validation failed: {e}", severity="error")

        self.app.push_screen(EditScreen(key, current, setting), _after)


# ────────────────────────────────────────────────────────────────────────
# Main app
# ────────────────────────────────────────────────────────────────────────

class PMBWorkspaceApp(App):
    CSS = """
    Screen { layout: vertical; }

    #status-bar {
        height: 1;
        background: $accent;
        color: $text;
        padding: 0 1;
    }

    TabbedContent { height: 1fr; }
    TabPane { padding: 1; }

    .filter-bar {
        height: 3;
        margin-bottom: 1;
    }

    .filter-label, .filter-bar Label {
        width: 8;
        content-align: left middle;
    }

    .filter-bar Input { width: 1fr; }

    .memory-body { height: 1fr; }

    #memory-table { width: 60%; height: 1fr; }
    #memory-detail {
        width: 40%;
        border-left: solid $primary;
        padding: 0 2;
    }

    #recall-results, #stats-tools, #stats-recent, #dedup-table, #tune-table {
        height: 1fr;
    }

    #recall-status, #stats-summary, #dedup-summary {
        margin-bottom: 1;
    }

    .section-title {
        color: $accent;
        text-style: bold;
        margin-top: 1;
    }

    .recall-input {
        height: 3;
        margin-bottom: 1;
    }

    .dedup-actions {
        height: 1;
        margin-bottom: 1;
    }

    #edit-dialog {
        width: 80;
        height: auto;
        padding: 2 3;
        background: $panel;
        border: thick $accent;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "reload", "Reload"),
        Binding("1", "show_tab('memory')", "Memory"),
        Binding("2", "show_tab('recall')", "Recall"),
        Binding("3", "show_tab('stats')", "Stats"),
        Binding("4", "show_tab('dedup')", "Dedup"),
        Binding("5", "show_tab('tune')", "Tune"),
    ]

    def __init__(self, eng: Engine):
        super().__init__()
        self.eng = eng

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(self._status_line(), id="status-bar")
        with TabbedContent(initial="memory"):
            with TabPane("[1] Memory", id="memory"):
                yield MemoryTab(self.eng)
            with TabPane("[2] Recall", id="recall"):
                yield RecallTab(self.eng)
            with TabPane("[3] Stats", id="stats"):
                yield StatsTab(self.eng)
            with TabPane("[4] Dedup", id="dedup"):
                yield DedupTab(self.eng)
            with TabPane("[5] Tune", id="tune"):
                yield TuneTab(self.eng)
        yield Footer()

    def _status_line(self) -> str:
        try:
            s = self.eng.stats()
            ev = s.get("events", {})
            return (
                f"[bold]workspace:[/] {self.eng.workspace.name} "
                f"([dim]{self.eng.workspace.id[:8]}[/]) │ "
                f"[cyan]{ev.get('total', 0)}[/] events ([dim]"
                f"{ev.get('active', 0)} active[/]) │ "
                f"[yellow]Loaded {time.strftime('%H:%M:%S')}[/]"
            )
        except Exception:
            return "[red]could not load workspace stats[/]"

    def action_show_tab(self, tab_id: str) -> None:
        tabs = self.query_one(TabbedContent)
        tabs.active = tab_id

    def action_reload(self) -> None:
        # Best-effort reload of all tabs' data
        for tab_class, tab_name in (
            (MemoryTab, "refresh_events"),
            (StatsTab, "refresh_stats"),
            (DedupTab, "refresh_dedup"),
        ):
            try:
                inst = self.query_one(tab_class)
                getattr(inst, tab_name)()
            except Exception:
                pass
        self.query_one("#status-bar", Static).update(self._status_line())
        self.notify("reloaded", severity="information")


def run_workspace_tui(eng: Engine) -> None:
    PMBWorkspaceApp(eng).run()
