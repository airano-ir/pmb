"""Status dashboard shown by bare `pmb` (no subcommand).

Read-only and fast: opens the engine WITHOUT loading the embedding model, so
running `pmb` never pays the ~15-20s cold model load. Every section is
defensive — a missing or unreadable piece renders as "-" rather than raising.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

# Human-readable explanation of how the active workspace was resolved.
_SOURCE_HELP = {
    "env": "PMB_WORKSPACE env var",
    "explicit": "explicit --workspace flag",
    "config": "project .pmb/workspace.yaml",
    "default": "saved default (pmb workspace use)",
    "git": "git remote / repo root",
    "cwd": "current directory",
    "auto": "auto-detected",
}


def _human_size(n: Optional[int]) -> str:
    if n is None:
        return "-"
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.1f} {unit}"
        f /= 1024
    return "-"


def _path_size(p: Path) -> Optional[int]:
    try:
        if not p.exists():
            return None
        if p.is_dir():
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        return p.stat().st_size
    except Exception:
        return None


def _pinned_count(eng) -> Optional[int]:
    """Count active pinned events. `pmb pin` pins by setting importance to
    1.0 (and clearing archived_at), so importance>=0.99 is the pin signal.
    Best-effort: returns None if the query fails."""
    try:
        import sqlite3
        with sqlite3.connect(str(eng.workspace.db_path)) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM events "
                "WHERE archived_at IS NULL AND importance >= 0.99"
            ).fetchone()
        return int(row[0] or 0)
    except Exception:
        return None


def render_status(console) -> None:
    """Print the PMB status panel to `console`."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table

    import pmb

    version = getattr(pmb, "__version__", "?")

    try:
        from pmb.core.engine import Engine
        eng = Engine()
        s = eng.stats()
    except Exception as e:  # pragma: no cover - defensive
        console.print(Panel.fit(
            f"[red]Could not open a workspace:[/] {e}\n"
            f"[dim]Run `pmb init` in a project, or `pmb --help`.[/]",
            title=f"PMB v{version}",
        ))
        return

    ws = s.get("workspace", {})
    ev = s.get("events", {})
    src = ws.get("source", "auto")
    src_desc = _SOURCE_HELP.get(src, src)

    # ── header: workspace + how it resolved ──
    header = Table.grid(padding=(0, 1))
    header.add_column(justify="right", style="dim")
    header.add_column()
    header.add_row("workspace", f"[bold cyan]{ws.get('name', '?')}[/] "
                                f"[dim]({str(ws.get('id', ''))[:12]})[/]")
    header.add_row("resolved via", f"{src} [dim]— {src_desc}[/]")
    header.add_row("storage", str(getattr(eng.workspace, "storage_dir", "-")))
    db_sz = _path_size(eng.workspace.db_path)
    vec_sz = _path_size(eng.workspace.vector_path)
    header.add_row("size", f"events.sqlite {_human_size(db_sz)} · "
                           f"vectors.lance {_human_size(vec_sz)}")
    warm = False
    try:
        warm = bool(eng.is_warm())
    except Exception:
        warm = False
    header.add_row("embeddings", "[green]warm[/]" if warm
                   else "[yellow]cold[/] [dim](loads on first recall)[/]")

    # ── memory counts ──
    counts = Table(show_header=True, header_style="bold magenta", box=None)
    counts.add_column("events", justify="right")
    counts.add_column("active", justify="right")
    counts.add_column("archived", justify="right")
    pinned = _pinned_count(eng)
    counts.add_column("pinned", justify="right")
    counts.add_row(
        str(ev.get("total", "-")),
        f"[green]{ev.get('active', '-')}[/]",
        f"[dim]{ev.get('archived', '-')}[/]",
        "-" if pinned is None else str(pinned),
    )

    by_type = ev.get("by_type") or {}
    type_line = "  ".join(
        f"{t}:[cyan]{n}[/]"
        for t, n in sorted(by_type.items(), key=lambda x: -x[1])[:8]
    ) or "[dim]no events yet[/]"

    # ── running MCP servers ──
    server_line = "[dim]none running[/]"
    try:
        from pmb.mcp.registry import list_servers
        servers = [sv for sv in list_servers(prune=True) if sv.get("alive")]
        if servers:
            parts = []
            for sv in servers[:6]:
                tp = sv.get("transport", "?")
                loc = (f"{sv.get('host')}:{sv.get('port')}"
                       if tp != "stdio" else "stdio")
                rss = sv.get("rss_mb")
                rss_s = f" {rss:.0f}MB" if isinstance(rss, (int, float)) else ""
                parts.append(f"pid {sv.get('pid')} [{tp} {loc}]{rss_s}")
            server_line = "\n".join(parts)
    except Exception:
        pass

    body = Group(
        header,
        "",
        counts,
        f"[dim]by type:[/] {type_line}",
        "",
        f"[bold]MCP servers[/]\n{server_line}",
        "",
        "[dim]`pmb --help` for commands · `pmb workspace use <name>` to switch[/]",
    )
    console.print(Panel(body, title=f"PMB v{version}", border_style="cyan"))

    try:
        eng.close()
    except Exception:
        pass
