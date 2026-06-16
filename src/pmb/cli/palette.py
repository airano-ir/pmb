"""pmb command palette - find any command by intent, not by memorising it.

`pmb` with no command (in a terminal) opens this: type what you want, the list
filters live, arrows pick, Enter runs. PMB has many commands; the palette turns
that from a burden (read --help) into a strength (everything is one fuzzy search
away).

Matching is INSTANT and model-free: command name/help + a curated synonym map,
so "forget old stuff" surfaces `declutter` / `prune` / `decay` even without the
word "forget". When the daemon is warm the results can be re-ranked semantically
through PMB's own embedder (optional, never blocking) - the memory tool whose CLI
understands what you mean.
"""
from __future__ import annotations

import re

# Curated intent -> command-name fragments. Lets a query match by MEANING, not
# just substring, with zero latency (no model load). Keep small + high-signal.
_SYNONYMS: dict[str, list[str]] = {
    "forget": ["declutter", "forget", "prune", "decay", "ttl"],
    "delete": ["forget", "declutter", "prune"],
    "remove": ["forget", "declutter", "prune"],
    "clean": ["declutter", "prune", "compact", "dedupe"],
    "junk": ["declutter", "prune"],
    "old": ["decay", "prune", "compact"],
    "find": ["recall", "overview", "audit", "list", "why"],
    "search": ["recall", "overview"],
    "ask": ["recall", "overview", "why"],
    "remember": ["fact", "remember", "note", "learn"],
    "save": ["fact", "remember", "note"],
    "store": ["fact", "remember", "note"],
    "teach": ["learn", "lessons"],
    "setup": ["setup", "connect", "doctor"],
    "install": ["setup", "connect"],
    "wire": ["connect", "setup"],
    "connect": ["connect", "setup"],
    "config": ["config", "tune"],
    "settings": ["config", "tune"],
    "tune": ["tune", "config"],
    "fast": ["warmup", "daemon"],
    "slow": ["warmup", "daemon"],
    "speed": ["warmup", "daemon"],
    "warm": ["warmup", "daemon"],
    "health": ["doctor", "health", "stats"],
    "status": ["stats", "doctor", "daemon"],
    "import": ["import"],
    "export": ["export", "snapshot"],
    "backup": ["snapshot", "export", "workspace"],
    "stats": ["stats", "insights"],
    "analytics": ["insights", "stats"],
    "lessons": ["lessons", "learn"],
    "rules": ["lessons"],
    "goals": ["goals", "reminders"],
    "todo": ["goals", "reminders"],
    "ui": ["tui", "dashboard"],
    "screen": ["tui", "dashboard"],
    "web": ["dashboard"],
    "graph": ["graph", "regraph", "prune-graph"],
    "index": ["index"],
    "pdf": ["index"],
    "code": ["index"],
    "history": ["timeline", "history", "list"],
    "timeline": ["timeline"],
    "tag": ["tag", "tagged", "tags", "untag"],
    "workspace": ["workspace", "workspaces", "init"],
    "switch": ["workspace", "workspaces"],
}


def command_catalog(app) -> list[dict]:
    """Build {name, help, group} for every top-level command + sub-group on the
    Typer app. Defensive: a malformed entry is skipped, never raises."""
    out: list[dict] = []
    for info in getattr(app, "registered_commands", []) or []:
        try:
            cb = getattr(info, "callback", None)
            name = info.name or (cb.__name__.replace("_", "-") if cb else None)
            if not name:
                continue
            help_ = (info.help or (cb.__doc__ if cb else "") or "")
            out.append({
                "name": name,
                "help": help_.strip().split("\n")[0].strip(),
                "group": getattr(info, "rich_help_panel", None) or "",
            })
        except Exception:
            continue
    for g in getattr(app, "registered_groups", []) or []:
        try:
            name = getattr(g, "name", None)
            if not name:
                continue
            ti = getattr(g, "typer_instance", None)
            tinfo = getattr(ti, "info", None)
            help_ = (getattr(tinfo, "help", None) or "")
            out.append({
                "name": name,
                "help": help_.strip().split("\n")[0].strip() or f"{name} commands",
                "group": "", "is_group": True,
            })
        except Exception:
            continue
    # stable order: alpha by name (the palette sorts by score when querying)
    out.sort(key=lambda c: c["name"])
    return out


def _is_subsequence(needle: str, hay: str) -> bool:
    it = iter(hay)
    return all(ch in it for ch in needle)


def score(query: str, cmd: dict) -> float:
    """Instant, model-free relevance of `cmd` to `query`."""
    q = query.strip().lower()
    if not q:
        return 0.0
    name = cmd["name"].lower()
    help_ = cmd.get("help", "").lower()
    s = 0.0
    if name == q:
        s += 100
    elif name.startswith(q):
        s += 80
    elif q in name:
        s += 55
    for tok in (t for t in re.split(r"\W+", q) if t):
        if tok in name:
            s += 22
        elif tok in help_:
            s += 8
        for frag in _SYNONYMS.get(tok, ()):  # intent -> command match
            if frag in name:
                s += 16
    if len(q) >= 2 and _is_subsequence(q.replace(" ", ""), name):
        s += 10  # "dcl" -> declutter
    return s


def rank(query: str, catalog: list[dict], limit: int = 8) -> list[dict]:
    """Top-`limit` commands for `query`. Empty query -> first `limit` (alpha)."""
    if not query.strip():
        return catalog[:limit]
    scored = [(c, score(query, c)) for c in catalog]
    scored = [(c, sc) for c, sc in scored if sc > 0]
    scored.sort(key=lambda x: (-x[1], x[0]["name"]))
    return [c for c, _ in scored[:limit]]


def render_palette(query: str, results: list[dict], selected: int = 0):
    """A static Rich renderable of the palette for `query` (used by the live UI
    and for previews/tests)."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from pmb.cli._common import wordmark

    rows = Table.grid(padding=(0, 2))
    rows.add_column(width=1)
    rows.add_column(min_width=16)
    rows.add_column(style="dim")
    for i, c in enumerate(results):
        on = i == selected
        marker = "[magenta]◆[/]" if on else " "
        nm = f"[bold bright_white]{c['name']}[/]" if on else f"[white]{c['name']}[/]"
        rows.add_row(marker, nm, c.get("help", ""))
    if not results:
        rows.add_row(" ", "[dim]no match[/]", "")

    body = Group(
        Text.from_markup(f"[dim]search[/]  {query or ''}[magenta]▍[/]"),
        "",
        rows,
    )
    return Panel(
        body,
        title=Text.from_markup(wordmark()),
        subtitle=Text("type to find · ↑↓ pick · ⏎ run · esc status", style="dim"),
        border_style="magenta", padding=(1, 2),
    )


def _read_key() -> str:
    """Block for one keypress; return a normalized name ('up'/'down'/'enter'/
    'esc'/'backspace'/'ctrl-c') or the raw character. Cross-platform, no deps."""
    import os
    if os.name == "nt":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):          # arrow / function-key prefix
            ch2 = msvcrt.getwch()
            return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(ch2, "")
        return {"\r": "enter", "\n": "enter", "\x1b": "esc",
                "\x08": "backspace", "\x03": "ctrl-c"}.get(ch, ch)
    # POSIX: put the terminal in raw mode for one read
    import sys
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":                    # escape sequence (arrows) or bare esc
            seq = sys.stdin.read(2)
            return {"[A": "up", "[B": "down",
                    "[C": "right", "[D": "left"}.get(seq, "esc")
        return {"\r": "enter", "\n": "enter", "\x7f": "backspace",
                "\x08": "backspace", "\x03": "ctrl-c"}.get(ch, ch)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def live_select(console, *, title, subtitle, headers, rows, default=0):
    """Arrow-key menu that redraws IN PLACE (Rich Live + single-key read).

    `rows` is a list of cell-tuples (markup ok); the FIRST cell is the row label
    and gets bold-highlighted on the active row. ↑/↓ move, Enter confirms, a
    digit 1-9 quick-picks, esc / ctrl-c cancels. Returns the chosen index, or
    None on cancel. Requires an interactive TTY (the caller guards with
    `console.is_terminal` and keeps a static-prompt fallback for non-TTY / CI)."""
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from pmb.cli._common import wordmark

    n = len(rows)
    if n == 0:
        return None
    selected = max(0, min(int(default), n - 1))

    def _frame(sel: int):
        t = Table(box=None, padding=(0, 1), show_header=bool(headers))
        t.add_column(" ", width=1)
        t.add_column("#", justify="right", width=2)
        for h in headers:
            t.add_column(h)
        for i, row in enumerate(rows):
            on = i == sel
            marker = "[magenta]◆[/]" if on else " "
            num = f"[bold magenta]{i + 1}[/]" if on else f"[dim]{i + 1}[/]"
            label = f"[bold bright_white]{row[0]}[/]" if on else row[0]
            t.add_row(marker, num, label, *row[1:])
        return Panel(t, title=Text.from_markup(wordmark(title)),
                     subtitle=Text(subtitle, style="dim"),
                     border_style="magenta", padding=(1, 2))

    with Live(_frame(selected), console=console, auto_refresh=False,
              screen=False, transient=False) as live:
        live.refresh()  # paint the first frame before we block on a keypress
        while True:
            try:
                key = _read_key()
            except Exception:
                return selected  # can't read keys → take the highlighted default
            if key in ("esc", "ctrl-c"):
                return None
            if key == "enter":
                return selected
            if key == "up":
                selected = max(0, selected - 1)
            elif key == "down":
                selected = min(n - 1, selected + 1)
            elif key and key.isdigit() and 1 <= int(key) <= n:
                return int(key) - 1   # number = pick + confirm
            else:
                continue
            live.update(_frame(selected), refresh=True)


def run_palette(console, catalog: list[dict]) -> str | None:
    """Interactive command palette: type to filter, up/down to pick, Enter to
    choose. Returns the chosen command name, or None on Esc / Ctrl-C. Inline
    (does not take over the screen). Requires an interactive TTY."""
    from rich.live import Live

    query, selected = "", 0
    results = rank(query, catalog)
    with Live(render_palette(query, results, selected), console=console,
              auto_refresh=False, screen=False, transient=True) as live:
        live.refresh()  # paint the first frame before we block on a keypress
        while True:
            try:
                key = _read_key()
            except Exception:
                return None
            if key in ("esc", "ctrl-c"):
                return None
            if key == "enter":
                return results[selected]["name"] if results else None
            if key == "up":
                selected = max(0, selected - 1)
            elif key == "down":
                selected = min(max(0, len(results) - 1), selected + 1)
            elif key == "backspace":
                query, selected = query[:-1], 0
            elif key and len(key) == 1 and key.isprintable():
                query, selected = query + key, 0
            else:
                continue
            results = rank(query, catalog)
            selected = min(selected, max(0, len(results) - 1))
            live.update(render_palette(query, results, selected), refresh=True)
