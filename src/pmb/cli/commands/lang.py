"""`pmb lang` - manage language packs.

A language pack extends PMB's EN/RU/UK lexical defaults to another language.
Packs are file-based and opt-in: enabling one copies a template into
``$PMB_HOME/lang/<code>.yaml``. `detect` samples the workspace and SUGGESTS
packs from the corpus - it never enables anything silently (auto-activation by
script would pollute, since e.g. German and English share the Latin script).
"""
from __future__ import annotations

import re
import sqlite3

import typer

from pmb import lang as _lang
from pmb.cli._common import console

lang_app = typer.Typer(help="Manage language packs (extend lexical defaults).")


def _enabled_codes() -> set[str]:
    return set(_lang.active_codes())


@lang_app.command("list")
def list_packs():
    """List built-in language-pack templates and which are enabled."""
    from rich.table import Table
    bt = _lang.builtin_templates()
    enabled = _enabled_codes()
    if not bt and not enabled:
        console.print("[yellow]No language packs available.[/]")
        return
    table = Table(show_header=True, header_style="bold magenta",
                  title="Language packs")
    table.add_column("Code")
    table.add_column("Name")
    table.add_column("Built-in")
    table.add_column("Enabled")
    seen = set()
    for code, path in sorted(bt.items()):
        data = _lang._load_yaml(path)
        seen.add(code)
        table.add_row(code, str(data.get("name") or code), "yes",
                      "[green]yes[/]" if code in enabled else "no")
    for code in sorted(enabled - seen):  # user packs with no built-in template
        table.add_row(code, "(user pack)", "no", "[green]yes[/]")
    console.print(table)
    console.print("[dim]Enable: pmb lang enable <code>   ·   "
                  "EN/RU/UK are always on (built into the core).[/]")


@lang_app.command()
def enable(code: str = typer.Argument(..., help="Language code, e.g. de, es")):
    """Enable a language pack (copies the template into $PMB_HOME/lang/)."""
    dest_dir = _lang.user_dir()
    dest = dest_dir / f"{code}.yaml"
    if dest.exists():
        console.print(f"[yellow]'{code}' is already enabled[/] ({dest}).")
        return
    bt = _lang.builtin_templates()
    dest_dir.mkdir(parents=True, exist_ok=True)
    if code in bt:
        dest.write_text(bt[code].read_text(encoding="utf-8"), encoding="utf-8")
        src = "built-in template"
    else:
        # No template - scaffold an empty pack the user can fill in.
        dest.write_text(
            f"code: {code}\nname: {code}\nstopwords: []\nnot_proper: []\n"
            f"first_person: []\nverb_synonyms: {{}}\nattribute_aliases: {{}}\n",
            encoding="utf-8")
        src = "new empty pack (no built-in template - edit it)"
    _lang.clear_cache()
    console.print(
        f"[green]✓ Enabled '{code}'[/] ({src}) → {dest}\n"
        f"[dim]Takes effect on the next PMB process. Restart the daemon "
        f"(`pmb daemon restart`) and consider `pmb reindex` so the index "
        f"matches the extended tokenizer.[/]")


@lang_app.command()
def disable(code: str = typer.Argument(...)):
    """Disable a language pack (removes $PMB_HOME/lang/<code>.yaml)."""
    dest = _lang.user_dir() / f"{code}.yaml"
    if not dest.exists():
        console.print(f"[yellow]'{code}' is not enabled.[/]")
        return
    dest.unlink()
    _lang.clear_cache()
    console.print(f"[green]✓ Disabled '{code}'.[/] [dim]Effective next process.[/]")


@lang_app.command()
def detect(
    sample: int = typer.Option(300, "--sample", help="Recent events to sample."),
):
    """Sample the workspace and SUGGEST language packs (never auto-enables)."""
    from pmb.core.engine import Engine
    eng = Engine()
    try:
        with sqlite3.connect(str(eng.workspace.db_path)) as conn:
            rows = conn.execute(
                "SELECT content FROM events WHERE workspace_id=? "
                "AND archived_at IS NULL ORDER BY timestamp DESC LIMIT ?",
                (eng.workspace.id, int(sample))).fetchall()
    except Exception as e:
        console.print(f"[red]Could not read workspace:[/] {e}")
        raise typer.Exit(1)
    if not rows:
        console.print("[yellow]No events to sample yet.[/]")
        return

    tokens: list[str] = []
    for (content,) in rows:
        tokens.extend(re.findall(r"[^\W_]+", (content or "").lower(),
                                 flags=re.UNICODE))
    total = len(tokens) or 1
    tokset = set(tokens)

    bt = _lang.builtin_templates()
    enabled = _enabled_codes()
    hits: list[tuple[str, float, int]] = []
    for code, path in bt.items():
        if code in enabled:
            continue
        sw = {str(x).lower() for x in (_lang._load_yaml(path).get("stopwords") or [])}
        if not sw:
            continue
        overlap = sw & tokset
        # fraction of sampled tokens that are this pack's stopwords
        frac = sum(1 for t in tokens if t in sw) / total
        hits.append((code, frac, len(overlap)))

    hits.sort(key=lambda h: h[1], reverse=True)
    suggested = [h for h in hits if h[1] >= 0.02 and h[2] >= 3]
    if not suggested:
        console.print(
            "[green]No additional language packs suggested[/] - the corpus "
            "looks covered by the EN/RU/UK core"
            + (f" (enabled: {', '.join(sorted(enabled))})" if enabled else "")
            + ".")
        return
    console.print("[bold]Suggested language packs[/] (based on the corpus):")
    for code, frac, n in suggested:
        name = str(_lang._load_yaml(bt[code]).get("name") or code)
        console.print(f"  • [cyan]{code}[/] ({name}) - {frac*100:.1f}% of "
                      f"sampled tokens, {n} distinct stopwords matched")
    console.print(f"\n[dim]Enable with: pmb lang enable "
                  f"{suggested[0][0]}   (opt-in - nothing changed yet)[/]")
