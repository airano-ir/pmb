"""`pmb config ...` — inspect and tune PMB knobs from the console."""

from __future__ import annotations

import os

import typer
from rich.panel import Panel
from rich.table import Table

from pmb.cli._common import _open_config, console

config_app = typer.Typer(help="Inspect and tune PMB knobs from the console.")


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
        console.print("[dim]Try: pmb config list[/]")
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
    key: str | None = typer.Argument(None, help="Key to reset (omit to reset ALL workspace overrides)"),
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
