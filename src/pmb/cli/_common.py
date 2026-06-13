"""Shared CLI surface: the root Typer `app`, the `console`, and small helpers.

This lives in its own module (not in pmb.cli.main) so that the per-area command
modules under `pmb.cli.commands.*` can register onto the SAME `app` and reuse
`console` without importing `pmb.cli.main` - main imports those command modules
to trigger their `@app.command` registrations, so the command modules importing
main back would be a cycle. They import from here instead.
"""

from __future__ import annotations

import sys
import time

import typer
from rich.console import Console

from pmb.core.workspace import detect_workspace

# UTF-8 on Windows. Guard against a non-stdlib stdout (Textual, pytest capture).
try:
    _enc = getattr(sys.stdout, "encoding", None)
    if _enc and _enc.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# no_args_is_help=False: a bare `pmb` runs the root callback (in pmb.cli.main),
# which prints the status dashboard. `pmb --help` still shows the command list.
app = typer.Typer(
    no_args_is_help=False,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
console = Console()


def loading(message: str):
    """Transient spinner for a slow CLI operation, e.g.

        with loading("loading embedding model (first run only)…"):
            ...

    Rich auto-disables the spinner when stdout is not a TTY (CI, pytest
    capture, pipes), so it never pollutes captured/piped output."""
    return console.status(f"[cyan]{message}[/]", spinner="dots")


def _humanize_time(ts: float | None) -> str:
    if ts is None:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts))


def _open_config():
    """Construct a Config bound to the current workspace + global home."""
    from pmb.config import SCHEMA, Config  # noqa: F401
    ws = detect_workspace()
    ws.ensure_dirs()
    return Config(workspace_dir=ws.storage_dir, pmb_home=ws.pmb_home), ws


def _agent_toggles_from_config() -> dict | None:
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


# Duration parsing + TTL stamping - shared by capture (note/learn/fact --ttl)
# and the `ttl` / `prune-expired` management commands.
_DUR_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400,
              "w": 604800, "mo": 2592000, "y": 31536000}


def _parse_duration(text: str | None) -> float | None:
    """'30d' / '12h' / '2w' / '3mo' / '1y' (or a bare integer = days) -> seconds.
    Returns None if unparseable."""
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
