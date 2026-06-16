"""`pmb` manage root commands - extracted from cli/main.py (no behavior change).

cli/main.py imports this module so these @app.command registrations run."""

from __future__ import annotations

import json
import time
from pathlib import Path

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
from pmb.core.workspace import detect_workspace, list_workspaces


def _ensure_daemon_started(pmb_home: Path | None, tool_profile: str | None) -> None:
    """S6: best-effort detached daemon spawn so the HTTP MCP entry is reachable
    the moment it's written - no window where it points at a not-yet-running
    daemon. No-op if one is already live, under pytest, or on any error (the
    lifecycle hooks autostart it otherwise). Inherits the configured tool
    profile so the shared daemon serves the same lean surface."""
    import os
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    try:
        from pmb.mcp.registry import find_live_daemon
        if find_live_daemon():
            return
        import subprocess
        import sys as _sys
        env = dict(os.environ)
        if pmb_home:
            env["PMB_HOME"] = str(pmb_home)
        if tool_profile:
            env["PMB_TOOL_PROFILE"] = tool_profile
        kwargs: dict = {}
        if os.name == "nt":
            kwargs["creationflags"] = 0x08000000 | 0x00000200  # CREATE_NO_WINDOW|NEW_GROUP
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen(
            [_sys.executable, "-m", "pmb.cli", "daemon", "run"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, env=env, **kwargs)
        console.print("  [dim]Starting the shared daemon in the background…[/]")
    except Exception:
        pass


@app.command()
def connect(
    agent: str | None = typer.Argument(
        None,
        help="claude-code | cursor | codex | windsurf | gemini | vscode | "
             "zed | opencode | continue",
    ),
    scope: str = typer.Option("project", "--scope",
                              help="project | global (where the agent supports both)"),
    remote: str | None = typer.Option(
        None, "--remote",
        help="Connect to a remote PMB server. Two forms:\n"
             "  • SSH: user@host:/abs/path/to/repo (stdio over SSH tunnel)\n"
             "  • HTTP: http://host:8765/mcp (team-shared streamable-http)",
    ),
    bearer_token: str | None = typer.Option(
        None, "--bearer-token", "--token",
        envvar="PMB_MCP_BEARER_TOKEN",
        help="Shared secret for the remote HTTP server (if it was started "
             "with `pmb mcp serve --bearer-token <secret>`).",
    ),
    name: str | None = typer.Option(
        None, "--name", help="MCP entry name (default: pmb or pmb-remote)",
    ),
    workspace: str | None = typer.Option(
        None, "--workspace", "-w",
        help="Force a SPECIFIC workspace id (override cwd-based detection). "
             "Use this to SHARE one memory across multiple AI clients - e.g. "
             "Claude Code + Cursor both pointing at 'personal' workspace.",
    ),
    pmb_home: Path | None = typer.Option(
        None, "--pmb-home",
        help="Override PMB_HOME (where workspaces live on disk). Useful for "
             "multi-user shared memory on a NAS / Dropbox path.",
    ),
    config_path: str | None = typer.Option(
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
    hooks: bool = typer.Option(
        True, "--hooks/--no-hooks",
        help="For hook-capable hosts (claude-code, codex): also install the "
             "lifecycle hooks (auto-recall + ambient + session-restore + "
             "follow-through). On claude-code this also trims the MCP surface "
             "to a 'lean' deliberate-only profile, since the hooks cover the "
             "rest. --no-hooks wires MCP only (keeps the full tool surface).",
    ),
    rules_only: bool = typer.Option(
        False, "--rules-only", "--update",
        help="Only refresh the AGENTS.md / CLAUDE.md rules block - don't touch "
             "the MCP config file. Use after a PMB update to bring the agent "
             "instructions in sync with the latest READ-FIRST guidance, without "
             "duplicating or re-adding the MCP server entry.",
    ),
    probe: bool = typer.Option(False, "--probe", help="Spawn pmb-mcp briefly to verify it starts"),
    daemon: bool | None = typer.Option(
        None, "--daemon/--stdio",
        help="S6: point the host at the ONE shared warm daemon over HTTP "
             "instead of spawning a stdio server (Engine + ~400 MB model) per "
             "client. JSON hosts (claude-code, cursor) only. Default follows "
             "`connect.default_daemon` (on); choosing it pins daemon.idle_exit_min"
             "=0 so the connection stays warm. --stdio forces the old per-client "
             "stdio entry.",
    ),
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
        JSON_AGENT_SPECS,
        probe_mcp,
        supported_agents,
    )
    from pmb.cli.connect import (
        connect as do_connect,
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
                console.print(f"[red]Failed: {p}[/] - {err}")
            else:
                colour = "green" if act in ("updated", "added") else "yellow"
                console.print(
                    f"[{colour}]Rules {act}:[/] {p}\n"
                    "  → MCP config untouched. Restart your agent to pick up "
                    "the new instructions."
                )
        return

    # S6: --daemon/--stdio overrides per-invocation; otherwise follow the
    # `connect.default_daemon` config (default on).
    if daemon is None:
        try:
            from pmb.config import Config
            _home = Path(pmb_home) if pmb_home else (Path.home() / ".pmb")
            effective_daemon = bool(Config(pmb_home=_home).get("connect.default_daemon"))
        except Exception:
            effective_daemon = True
    else:
        effective_daemon = daemon
    try:
        result = do_connect(
            agent, cwd=Path.cwd(), scope=scope, remote=remote, name_override=name,
            workspace_id=workspace, pmb_home=pmb_home, config_path=config_path,
            active=effective_active,
            active_toggles=(_toggles if effective_active else None),
            bearer_token=bearer_token,
            install_hooks=hooks,
            use_daemon=effective_daemon,
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

    # S6: when we pointed the host at the shared HTTP daemon (now the default for
    # JSON hosts), explain the win + how to opt out; when --daemon was explicitly
    # asked for but the host can't take an HTTP entry, say so.
    if result.get("daemon_http"):
        console.print(
            "[green]Shared warm daemon (HTTP) - the default now.[/] This host "
            "reuses the ONE warm process instead of spawning its own Engine + "
            "~400 MB model, so N connected clients cost ~400 MB total, not ×N. "
            "The daemon is pinned to never idle-exit so the connection stays warm "
            "(the hooks autostart it on the next message).\n"
            "  Start it now if you don't want to wait for the first hook:\n"
            "    [cyan]pmb daemon start[/]\n"
            "  Prefer the old per-client stdio server? [cyan]pmb connect "
            f"{result.get('agent', 'claude-code')} --stdio[/]  "
            "(or set [cyan]connect.default_daemon false[/] globally)."
        )
        # S6 (Caveat-S6b): start it now so the entry is reachable immediately.
        _ensure_daemon_started(pmb_home, result.get("tool_profile"))
    elif result.get("daemon_proxy"):
        console.print(
            "[green]Shared warm daemon via stdio proxy - the default for codex "
            "now.[/] codex can't take an HTTP entry, so it launches the "
            "lightweight [cyan]pmb mcp proxy[/] bridge: stdio in, forwarded to the "
            "ONE warm daemon over HTTP. No per-session model load - that's the "
            "codex cold-start lag gone; the proxy autostarts the daemon if it's "
            "not already up.\n"
            "  Start it now so the first codex session is instant:\n"
            "    [cyan]pmb daemon start[/]\n"
            "  Prefer the old heavy per-session server? [cyan]pmb connect codex "
            "--stdio[/]."
        )
        _ensure_daemon_started(pmb_home, None)
    elif result.get("daemon_http_unavailable"):
        console.print(
            "[yellow]Stdio entry (shared daemon not applicable here):[/] this host "
            "takes a command-shaped stdio entry (editor extensions / --remote), "
            "not one we can point at the shared daemon - wrote a per-client stdio "
            "server. The shared daemon is used for claude-code / cursor / codex."
        )

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

    # Hooks + lean profile (the effective default for hook-capable hosts).
    hk = result.get("hooks")
    prof = result.get("tool_profile")
    if hk is not None:
        if isinstance(hk, dict) and hk.get("error"):
            console.print(f"[yellow]Hooks: not installed[/] - {hk['error']}")
        elif (hk or {}).get("events"):
            # claude-code: the full hook set (per-turn inject + ambient + more).
            console.print(f"[green]Hooks installed:[/] {', '.join(hk['events'])}")
            console.print(
                "  → auto-recall, ambient auto-write, session-restore & "
                "follow-through now run automatically (zero agent cost)."
            )
        elif agent == "codex":
            # codex: notify (post-turn) only - NO per-turn / session-start hook.
            console.print("[green]Codex ambient wired:[/] notify → pmb codex-notify")
            console.print(
                "  → ambient auto-write runs after each turn. Codex has no "
                "per-turn hook, so read-first/auto-recall is driven by the "
                "AGENTS.md rules (the agent calls prepare()/recall() itself)."
            )
        else:
            console.print("[green]Hooks installed:[/] "
                          f"{(hk or {}).get('action', 'installed')}")
        if prof:
            console.print(
                f"[green]MCP profile:[/] {prof} - trimmed to deliberate-only "
                "tools (the hooks cover the rest; smaller tool list = faster)."
            )
    elif agent in ("claude-code", "codex") and not hooks:
        console.print("[dim]Hooks skipped (--no-hooks); full MCP tool surface kept.[/]")

    if probe:
        ok, msg = probe_mcp()
        verdict_color = "green" if ok else "red"
        console.print(f"[{verdict_color}]Probe:[/] {msg}")
        if not ok:
            raise typer.Exit(code=1)

    console.print(
        "\n[bold]Next steps:[/]"
        "\n  1. [cyan]pmb warmup[/]   - load the embedding model now so the "
        "first recall is fast (skip it and the first query pays a ~30-60s "
        "one-time load)"
        "\n  2. Restart your agent so it picks up the new MCP entry"
        "\n  3. [cyan]pmb doctor[/]   - verify the full local state"
    )


# Curated embedder profiles offered at setup. All run on the default
# sentence-transformers backend (only the model id changes), so picking one is a
# single config write + a reindex - no backend/prefix gotchas. Plain-language
# plus/minus so a human can choose by RAM vs quality.
_EMBEDDER_PROFILES = [
    {
        "key": "light",
        "label": "Light & fast",
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "ram": "~0.5 GB",
        "rec": "any machine, <=8 GB RAM",
        "pros": "tiny, fast, works everywhere",
        "cons": "weaker cross-language & rare-script (CJK) recall",
    },
    {
        "key": "balanced",
        "label": "Balanced",
        "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "ram": "~1.1 GB",
        "rec": "12 GB+ RAM",
        "pros": "noticeably more accurate, same family (no extra setup)",
        "cons": "~2x the RAM, a bit slower than Light",
    },
    {
        "key": "best",
        "label": "Best quality",
        "model": "BAAI/bge-m3",
        "ram": "~2 GB",
        "rec": "16 GB+ RAM",
        "pros": "best cross-language + CJK, sharpest matching",
        "cons": "heavy RAM, slower on CPU, longer first load",
    },
]


def _choose_embedder(console, current_model: str, yes: bool) -> str:
    """Interactive 'which memory model?' pick during setup. Returns a model id.
    Non-interactive (`--yes`) or any error keeps `current_model` - NEVER blocks
    setup. On a real terminal it's an in-place arrow menu (Rich Live); off a TTY
    (CI / piped) it degrades to a static table + numeric prompt."""
    if yes:
        return current_model
    # 0-based index of the current model -> the menu's default selection.
    default_idx = next((i for i, p in enumerate(_EMBEDDER_PROFILES)
                        if p["model"] == current_model), 0)

    # Live, in-place arrow menu when we have a real terminal.
    if console.is_terminal:
        try:
            from pmb.cli.palette import live_select
            rows = [
                (p["label"]
                 + ("  [green](current)[/]" if p["model"] == current_model else ""),
                 p["ram"], p["rec"],
                 f"[green]+[/] {p['pros']}\n[red]-[/] {p['cons']}")
                for p in _EMBEDDER_PROFILES
            ]
            idx = live_select(
                console, title="memory model",
                subtitle="↑↓ pick · ⏎ choose · esc keep current · "
                         "changing re-embeds memory",
                headers=["Option", "RAM", "Good for", "Plus / minus"],
                rows=rows, default=default_idx)
            return current_model if idx is None else _EMBEDDER_PROFILES[idx]["model"]
        except Exception:
            pass  # fall through to the static prompt

    # Fallback: static table + numeric prompt (non-TTY / CI / any error).
    try:
        from rich.table import Table
        t = Table(
            title="Memory model (embedder) - how PMB understands & matches text",
            border_style="magenta", show_lines=True,
        )
        t.add_column("#", justify="right", style="cyan")
        t.add_column("Option")
        t.add_column("RAM", style="dim")
        t.add_column("Good for", style="dim")
        t.add_column("Plus / minus")
        for i, p in enumerate(_EMBEDDER_PROFILES, 1):
            is_cur = p["model"] == current_model
            t.add_row(
                str(i),
                p["label"] + ("  [green](current)[/]" if is_cur else ""),
                p["ram"], p["rec"],
                f"[green]+[/] {p['pros']}\n[red]-[/] {p['cons']}",
            )
        console.print(t)
        console.print(
            "[dim]Tip: changing this re-embeds existing memory "
            "([cyan]pmb reindex[/]). Press Enter to keep the current one.[/]"
        )
        choice = typer.prompt("Pick a memory model (number)",
                              default=str(default_idx + 1))
        idx = int(str(choice).strip())
        if 1 <= idx <= len(_EMBEDDER_PROFILES):
            return _EMBEDDER_PROFILES[idx - 1]["model"]
    except Exception:
        pass
    return current_model


# The OFFLINE LLM tier (consolidate.backend) - writes lessons/summaries during
# `pmb consolidate`/sleep. BACKGROUND only: never on the per-message hook, so it
# does NOT affect injection speed. Only the backends resolve_llm_client actually
# supports (auto/claude/anthropic/ollama) - no codex here, no fake 'none' (the
# tier simply doesn't run unless you invoke it / enable auto_trigger).
_LLM_TIER_PROFILES = [
    {
        "key": "auto", "label": "Auto (recommended)", "backend": "auto",
        "needs": "nothing if you use Claude Code",
        "pros": "zero setup - Claude CLI if present, else API key, else Ollama",
        "cons": "uses whatever is available (less explicit)",
    },
    {
        "key": "claude", "label": "Claude CLI", "backend": "claude",
        "needs": "`claude` in PATH",
        "pros": "no API key - reuses your Claude Code auth",
        "cons": "requires Claude Code installed",
    },
    {
        "key": "ollama", "label": "Ollama (local)", "backend": "ollama",
        "needs": "`ollama serve` + a pulled model",
        "pros": "fully local & private, no cloud",
        "cons": "you run the server; quality depends on the local model",
    },
    {
        "key": "anthropic", "label": "Anthropic API", "backend": "anthropic",
        "needs": "ANTHROPIC_API_KEY",
        "pros": "fast, nothing to install locally",
        "cons": "needs a paid API key",
    },
]


def _choose_llm_tier(console, current_backend: str, yes: bool) -> str:
    """Interactive 'offline brain' pick during setup. Returns a backend id.
    Non-interactive (`--yes`) or any error keeps `current_backend` - NEVER
    blocks. This tier is BACKGROUND only (no effect on injection speed). On a
    real terminal it's an in-place arrow menu; off a TTY it degrades to a
    static table + numeric prompt."""
    if yes:
        return current_backend
    default_idx = next((i for i, p in enumerate(_LLM_TIER_PROFILES)
                        if p["backend"] == current_backend), 0)

    if console.is_terminal:
        try:
            from pmb.cli.palette import live_select
            rows = [
                (p["label"]
                 + ("  [green](current)[/]" if p["backend"] == current_backend else ""),
                 p["needs"], f"[green]+[/] {p['pros']}\n[red]-[/] {p['cons']}")
                for p in _LLM_TIER_PROFILES
            ]
            idx = live_select(
                console, title="offline brain",
                subtitle="↑↓ pick · ⏎ choose · esc keep current · "
                         "background only, never slows injection",
                headers=["Option", "Needs", "Plus / minus"],
                rows=rows, default=default_idx)
            return current_backend if idx is None else _LLM_TIER_PROFILES[idx]["backend"]
        except Exception:
            pass

    # Fallback: static table + numeric prompt (non-TTY / CI / any error).
    try:
        from rich.table import Table
        t = Table(
            title="Offline brain (LLM tier) - writes lessons/summaries in the background",
            border_style="magenta", show_lines=True,
        )
        t.add_column("#", justify="right", style="cyan")
        t.add_column("Option")
        t.add_column("Needs", style="dim")
        t.add_column("Plus / minus")
        for i, p in enumerate(_LLM_TIER_PROFILES, 1):
            is_cur = p["backend"] == current_backend
            t.add_row(
                str(i),
                p["label"] + ("  [green](current)[/]" if is_cur else ""),
                p["needs"],
                f"[green]+[/] {p['pros']}\n[red]-[/] {p['cons']}",
            )
        console.print(t)
        console.print(
            "[dim]Background only: runs during [cyan]pmb consolidate[/]/sleep, "
            "NEVER on the per-message hook - it does NOT slow injection. "
            "Press Enter to keep the current one.[/]"
        )
        choice = typer.prompt("Pick an offline brain (number)",
                              default=str(default_idx + 1))
        idx = int(str(choice).strip())
        if 1 <= idx <= len(_LLM_TIER_PROFILES):
            return _LLM_TIER_PROFILES[idx - 1]["backend"]
    except Exception:
        pass
    return current_backend


def _resolve_model_target(name: str | None, current: str) -> str:
    """Map a profile key (light/balanced/best) or a raw embedder id to a model
    id. Empty -> keep current. An unknown string is treated as a raw
    sentence-transformers id (so power users aren't boxed into the presets)."""
    if not name or not name.strip():
        return current
    n = name.strip()
    for p in _EMBEDDER_PROFILES:
        if p["key"] == n.lower() or p["model"] == n:
            return p["model"]
    return n


@app.command()
def model(
    name: str | None = typer.Argument(
        None, help="Profile (light/balanced/best) or any embedder id. "
                   "Omit for the interactive menu."),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="No prompts; re-embed + restart automatically."),
    no_reindex: bool = typer.Option(
        False, "--no-reindex",
        help="Switch the model but DON'T re-embed existing memory "
             "(run `pmb reindex` yourself later)."),
):
    """Switch the embedding model and load it - download + re-embed memory +
    restart the daemon, in ONE step.

      pmb model              # interactive menu (light / balanced / best)
      pmb model best         # switch to bge-m3 (best quality, needs RAM)
      pmb model BAAI/bge-m3  # any sentence-transformers id
    """
    import os
    import subprocess
    import sys

    from pmb.cli._common import loading
    from pmb.config import Config

    home = Path(os.environ.get("PMB_HOME") or (Path.home() / ".pmb"))
    cfg = Config(pmb_home=home)
    current = cfg.get("embedding.model")

    if name:
        target = _resolve_model_target(name, current)
    else:
        console.print(f"[dim]current model:[/] {current}")
        target = _choose_embedder(console, current, yes=False)

    if not target or target == current:
        console.print(f"[dim]Keeping current model:[/] {current}")
        return

    if not yes and not typer.confirm(
            f"Switch to {target}?  Downloads it and re-embeds ALL memory.",
            default=True):
        console.print("[dim]Cancelled - model unchanged.[/]")
        return

    cfg.set_global("embedding.model", target)
    console.print(f"[green]✓[/] embedding.model → [cyan]{target}[/] "
                  f"[dim]({home / 'config.yaml'})[/]")

    if no_reindex:
        console.print("[yellow]Skipped reindex[/] - run [cyan]pmb reindex[/] "
                      "before the new model can recall existing memory.")
    else:
        try:
            eng = Engine(cwd=Path.cwd())
            with loading(f"downloading + loading {target}, re-embedding memory…"):
                eng.warmup()
                res = eng.reindex_embeddings()
            console.print(
                f"[cyan]Re-embedded[/] {res.get('n_events', 0)} events in "
                f"[yellow]{res.get('elapsed_seconds', 0)}s[/] with the new model")
        except Exception as e:
            console.print(f"[red]Load/reindex failed:[/] {e}")
            console.print("[dim]The model is set; free RAM / check network, "
                          "then run [cyan]pmb reindex[/].[/]")
            raise typer.Exit(1)

    # Restart the daemon so the live hook path uses the new model. Via subprocess
    # (the CLI resolves the typer command cleanly); the daemon itself spawns
    # windowless from daemon.py.
    try:
        subprocess.run([sys.executable, "-m", "pmb.cli", "daemon", "restart"],
                       timeout=180)
        console.print("[green]✓[/] daemon restarted on the new model")
    except Exception as e:
        console.print(f"[dim]Restart the daemon yourself for live use "
                      f"({e}): pmb daemon restart[/]")


def _setup_done_card(console, wired: list, daemon_live: bool, eff_active: bool) -> None:
    """The persistent 'you're all set' card. `wired` is a list of
    (agent, hooks_ok, shared_daemon) tuples - one entry for `pmb setup`, several
    for `pmb setup --all`. Pure display: builds + prints the panel, nothing else."""
    from rich.console import Group as _Group
    from rich.text import Text

    from pmb.cli._common import wordmark

    names = ", ".join(a for a, _, _ in wired) or "no agent"
    any_hooks = any(h for _, h, _ in wired)

    # One-line "did it actually work?" confirmation - the persistent TL;DR even
    # after the live panel scrolls away.
    status = Text.from_markup(
        f"[green]✓[/] wired into [bold]{names}[/]"
        + ("  ·  hooks on" if any_hooks else "")
        + ("  ·  daemon live" if daemon_live else ""))

    do = Table.grid(padding=(0, 2))
    do.add_column(justify="center", width=3)
    do.add_column()
    do.add_row("[magenta]1[/]", "[bold]restart your agent[/]  "
               "[dim]- so it picks up pmb[/]")
    do.add_row("[magenta]2[/]", "[bold]just talk to it[/]  "
               "[dim]- pmb remembers & recalls automatically, no commands[/]")

    opt = Table.grid(padding=(0, 2))
    opt.add_column(justify="right", style="dim")
    opt.add_column()
    opt.add_row("seed memory", "[cyan]pmb import chatgpt <export.json>[/]")
    opt.add_row("change model", "[cyan]pmb model[/]  [dim](light / balanced / best)[/]")
    if not daemon_live:
        opt.add_row("instant recall", "[cyan]pmb daemon start[/]")
    if not eff_active:
        opt.add_row("self-logging", "[cyan]pmb setup --active[/]")

    # Management + the escape hatch: kill-all clears stray/duplicate processes
    # that pile up on a small box (each warm one holds the model in RAM).
    manage = Text.from_markup(
        "[dim]check[/] [cyan]pmb doctor[/]   [dim]·[/]   "
        "[dim]daemon[/] [cyan]pmb daemon status[/]   [dim]·[/]   "
        "[dim]reset stray procs[/] [cyan]pmb daemon kill-all[/]")

    body = _Group(status, "", do, "",
                  Text.from_markup("[dim]optional power-ups[/]"), opt,
                  "", manage)
    console.print()
    console.print(Panel(body, title=wordmark("you're all set"),
                        border_style="#a78bfa", padding=(1, 2)))


@app.command()
def setup(
    agent: str | None = typer.Argument(
        None, help="Agent to wire (claude-code / codex / cursor / ...). Omit to auto-detect."),
    active: bool = typer.Option(
        False, "--active",
        help="Install proactive-logging rules (agent records its own decisions/lessons)."),
    all_agents: bool = typer.Option(
        False, "--all",
        help="Wire EVERY detected agent at once (they all share the one warm "
             "daemon). Great when you use Claude Code + Codex + Cursor together."),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Non-interactive: take the recommended defaults, no prompts."),
):
    """Guided first-time setup - detect your agent and wire PMB in one go.

      pmb setup                       # detect your agent, ask a couple questions
      pmb setup --all                 # wire every detected agent (shared daemon)
      pmb setup codex --active --yes  # non-interactive

    Defaults are already the effective ones (ablation-tuned) - you only need
    `pmb tune` if you want to change them. Full command list: `docs/COMMANDS.md`.
    """
    from pmb.cli._common import _PMB_SPINNER, wordmark
    from pmb.cli.connect import (
        connect as do_connect,
    )
    from pmb.cli.connect import (
        detect_installed_agents,
        supported_agents,
    )
    from pmb.cli.intro import mark_welcomed, play_boot

    play_boot(console, settle=False)  # ✦ ignition, then the live setup panel
    detected = detect_installed_agents()

    # ── interactive choices happen BEFORE the live panel (you can't prompt
    #    inside a Rich Live region) ──────────────────────────────────────────
    chosen = None
    targets: list[str] = []
    if all_agents:
        targets = [a for a in detected if a in supported_agents()]
        if not targets:
            console.print(wordmark("setup"))
            console.print("[yellow]No agent configs detected yet.[/] Run an agent "
                          "once (Claude Code / Codex / Cursor / …), then "
                          "[cyan]pmb setup --all[/].")
            return
        console.print(f"[dim]wiring all detected:[/] {', '.join(targets)}")
    else:
        chosen = agent or (detected[0] if detected else None)
        if not chosen:
            console.print(wordmark("setup"))
            if yes:
                console.print("[yellow]No agent config found yet.[/] Run your agent once, "
                              "then [cyan]pmb connect <agent>[/].")
                return
            console.print(f"[dim]detected:[/] {', '.join(detected) or 'none yet'}")
            chosen = typer.prompt("Which agent to connect?", default="claude-code")
        if chosen not in supported_agents():
            console.print(f"[red]Unknown agent:[/] {chosen}. "
                          f"Options: {', '.join(supported_agents())}")
            raise typer.Exit(2)
    # Proactive logging stays opt-in (--active) - one fewer question, simpler
    # setup. The next-step card points to it for anyone who wants it.
    _toggles = _agent_toggles_from_config()
    eff_active = active or bool((_toggles or {}).get("active_mode"))

    # ── memory-model (embedder) choice - BEFORE the live panel (can't prompt
    #    inside a Rich Live region). Persists to the GLOBAL config so the daemon
    #    and every workspace use it; warm step below then loads the chosen one.
    picked_ram = "~0.5 GB"
    try:
        import os as _os

        from pmb.config import Config as _Config
        _home = Path(_os.environ.get("PMB_HOME") or (Path.home() / ".pmb"))
        _cfg = _Config(pmb_home=_home)
        _current = _cfg.get("embedding.model")
        _picked = _choose_embedder(console, _current, yes)
        picked_ram = next((p["ram"] for p in _EMBEDDER_PROFILES
                           if p["model"] == _picked), picked_ram)
        if _picked and _picked != _current:
            _cfg.set_global("embedding.model", _picked)
            console.print(f"[green]✓[/] memory model → [cyan]{_picked}[/] "
                          f"[dim]({_home / 'config.yaml'})[/]")
            console.print("[yellow]If you already have memory, run "
                          "[cyan]pmb reindex[/] to re-embed it with the new model.[/]")
    except Exception as e:
        console.print(f"[dim]embedder choice skipped: {e}[/]")

    # ── offline-brain (LLM tier) choice - BACKGROUND work only, no injection
    #    cost. Persisted to the global config like the embedder.
    try:
        import os as _os2

        from pmb.config import Config as _Config2
        _home2 = Path(_os2.environ.get("PMB_HOME") or (Path.home() / ".pmb"))
        _cfg2 = _Config2(pmb_home=_home2)
        _cur_llm = _cfg2.get("consolidate.backend")
        _picked_llm = _choose_llm_tier(console, _cur_llm, yes)
        if _picked_llm and _picked_llm != _cur_llm:
            _cfg2.set_global("consolidate.backend", _picked_llm)
            console.print(f"[green]✓[/] offline brain → [cyan]{_picked_llm}[/] "
                          f"[dim]({_home2 / 'config.yaml'})[/]")
    except Exception as e:
        console.print(f"[dim]offline-brain choice skipped: {e}[/]")

    # ── shared-daemon default: wire the host to the ONE warm daemon (HTTP for
    #    JSON hosts, stdio proxy for codex) instead of a per-client Engine+model.
    #    Follows `connect.default_daemon` (on by default); the daemon is started
    #    right after wiring so the first message is instant.
    try:
        import os as _osd

        from pmb.config import Config as _CfgD
        _homed = Path(_osd.environ.get("PMB_HOME") or (Path.home() / ".pmb"))
        effective_daemon = bool(_CfgD(pmb_home=_homed).get("connect.default_daemon"))
    except Exception:
        effective_daemon = True

    # ── `--all`: wire every detected agent against the ONE shared daemon, warm
    #    the engine once, then one combined card. (No per-agent live panel - a
    #    simple line per agent reads better for a batch.) ───────────────────────
    if all_agents:
        wired: list = []
        for ag in targets:
            try:
                r = do_connect(ag, cwd=Path.cwd(), active=eff_active,
                               active_toggles=(_toggles if eff_active else None),
                               install_hooks=True, use_daemon=effective_daemon)
            except Exception as e:
                console.print(f"  [red]✗[/] {ag}: {str(e)[:80]}")
                continue
            hk_ok = (bool(r.get("hooks"))
                     and not (r.get("hooks") or {}).get("error"))
            shared = bool(r.get("daemon_http") or r.get("daemon_proxy"))
            tags = ("  +hooks" if hk_ok else "") + ("  +shared-daemon" if shared else "")
            console.print(f"  [green]✓[/] {ag}{tags}  "
                          f"[dim]{r.get('config_path', '')}[/]")
            wired.append((ag, hk_ok, shared))
        if not wired:
            console.print("[red]Nothing wired.[/]")
            raise typer.Exit(1)

        # Warm the shared engine ONCE (every host reuses it), with a quick self-test.
        try:
            from pmb.core.engine import Engine
            try:
                from transformers.utils import logging as _hf_log
                _hf_log.disable_progress_bar()
            except Exception:
                pass
            with console.status(
                    f"waking the memory engine (first run pulls {picked_ram})…"):
                eng = Engine(cwd=Path.cwd())
                eng.warmup(with_first_query=True)
            console.print("  [green]✓[/] memory engine ready")
            try:
                eng.close()
            except Exception:
                pass
        except Exception as e:
            console.print("  [yellow]○[/] engine not warmed (run `pmb warmup` "
                          f"later)  [dim]{str(e)[:40]}[/]")

        # One shared daemon for all of them.
        if any(s for _, _, s in wired):
            _ensure_daemon_started(None, None)
        try:
            from pmb.mcp.registry import find_live_daemon
            _daemon_live = bool(find_live_daemon())
        except Exception:
            _daemon_live = False

        _setup_done_card(console, wired, _daemon_live, eff_active)
        mark_welcomed()
        return

    # ── ONE live screen: scan -> wire -> ✦ wake -> prove ──────────────────────
    from rich.spinner import Spinner
    from rich.table import Table
    from rich.text import Text

    rows: list[list[str]] = []          # each: [kind, text, detail]
    state = {"done": False}
    spark = Spinner(_PMB_SPINNER, style="magenta")
    _GLYPH = {"ok": "[green]✓[/]", "warn": "[yellow]○[/]", "err": "[red]✗[/]"}

    def _panel():
        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=2, justify="center")
        grid.add_column()
        grid.add_column(style="dim")
        for kind, txt, detail in rows:
            grid.add_row(spark if kind == "active" else _GLYPH.get(kind, " "),
                         txt, detail)
        return Panel(
            grid, title=Text.from_markup(wordmark()),
            subtitle=Text("ready" if state["done"] else "setting up", style="dim"),
            border_style="magenta", padding=(1, 2),
        )

    def _run(emit):
        emit("ok", f"found {', '.join(detected) or 'no agent'}", "")
        h = emit("active", f"wiring into {chosen}", "")
        try:
            result = do_connect(
                chosen, cwd=Path.cwd(), active=eff_active,
                active_toggles=(_toggles if eff_active else None),
                install_hooks=True, use_daemon=effective_daemon,
            )
        except ValueError as e:
            emit("err", f"wiring failed: {e}", "", replace=h)
            raise typer.Exit(2)
        hooks_ok = (bool(result.get("hooks"))
                    and not (result.get("hooks") or {}).get("error"))
        state["hooks_ok"] = hooks_ok  # surfaced in the final "you're all set" card
        # remember whether this wiring shares the daemon, so we start it below
        state["daemon_http"] = bool(result.get("daemon_http"))
        state["daemon_proxy"] = bool(result.get("daemon_proxy"))
        state["tool_profile"] = result.get("tool_profile")
        shared = state["daemon_http"] or state["daemon_proxy"]
        emit("ok", f"wired into {chosen}"
             + ("  +hooks" if hooks_ok else "")
             + ("  +shared-daemon" if shared else ""),
             str(result.get("config_path", "")), replace=h)

        eng = None
        h = emit("active", "waking the memory engine", f"first run pulls {picked_ram}")
        try:
            from pmb.core.engine import Engine
            try:  # keep the panel clean - no stray HF progress bar
                from transformers.utils import logging as _hf_log
                _hf_log.disable_progress_bar()
            except Exception:
                pass
            eng = Engine(cwd=Path.cwd())
            eng.warmup(with_first_query=True)
            emit("ok", "memory engine ready", "", replace=h)
        except Exception as e:
            emit("warn", "engine not warmed (run `pmb warmup` later)",
                 str(e)[:40], replace=h)

        if eng is not None:
            h = emit("active", "proving it works", "")
            try:
                import time as _t
                from datetime import datetime
                stamp = datetime.now().strftime("%Y-%m-%d")
                t0 = _t.time()
                ulid = eng.record_fact(f"PMB activated on {stamp}")
                pack = eng.recall("when was pmb activated", top_k=3)
                ms = (_t.time() - t0) * 1000.0
                hit = any(getattr(r, "ulid", None) == ulid
                          for r in getattr(pack, "results", []))
                emit("ok", f"remembered & recalled in {ms:.0f} ms" if hit
                     else f"memory write + read ok ({ms:.0f} ms)", "", replace=h)
            except Exception as e:
                emit("warn", "self-test skipped", str(e)[:40], replace=h)
            try:
                eng.close()
            except Exception:
                pass

    def _emit_live(live):
        def emit(kind, text, detail="", replace=None):
            if replace is None:
                rows.append([kind, text, detail])
                idx = len(rows) - 1
            else:
                rows[replace] = [kind, text, detail]
                idx = replace
            live.update(_panel())
            return idx
        return emit

    def _emit_plain(kind, text, detail="", replace=None):
        if kind == "active":
            return None
        sym = {"ok": "✓", "warn": "○", "err": "✗"}.get(kind, "·")
        console.print(f"  {sym} {text}" + (f"  [dim]{detail}[/]" if detail else ""))
        return None

    live_cm, emit = None, _emit_plain
    if console.is_terminal:
        try:
            from rich.live import Live
            live_cm = Live(_panel(), console=console, refresh_per_second=12)
            live_cm.__enter__()
            emit = _emit_live(live_cm)
        except Exception:
            live_cm, emit = None, _emit_plain
    try:
        _run(emit)
        state["done"] = True
        if live_cm is not None:
            live_cm.update(_panel())
    finally:
        if live_cm is not None:
            try:
                live_cm.__exit__(None, None, None)
            except Exception:
                pass

    # Start the shared daemon now (if this wiring uses it) so the very first
    # message is instant - no waiting for the first hook to autostart it.
    if state.get("daemon_http") or state.get("daemon_proxy"):
        _ensure_daemon_started(None, state.get("tool_profile"))

    # ── the persistent, actionable "you're all set" card ──
    # Is a shared warm daemon already up? If so, recall is instant and we should
    # NOT tell the user to start one; if not, offer it as a power-up.
    try:
        from pmb.mcp.registry import find_live_daemon
        _daemon_live = bool(find_live_daemon())
    except Exception:
        _daemon_live = False

    shared = bool(state.get("daemon_http") or state.get("daemon_proxy"))
    _setup_done_card(console,
                     [(chosen, bool(state.get("hooks_ok")), shared)],
                     _daemon_live, eff_active)
    mark_welcomed()  # onboarded - bare `pmb` now opens the palette, not the welcome






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

_ALL_EVENTS = 10_000_000  # effectively "no limit" for analytics/export scans


def _day_key(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


def _kind_marker(meta: dict | None) -> str:
    kind = (meta or {}).get("kind", "")
    if kind == "failure":
        return "[red]⚠[/] "
    if kind == "lesson":
        return "[magenta]★[/] "
    return ""


@app.command()
def timeline(
    limit: int = typer.Option(60, "-n", "--limit", help="Max events to show"),
    event_type: str | None = typer.Option(None, "--type", help="Filter by event type"),
    days: float | None = typer.Option(None, "--days", help="Only events from the last N days"),
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
    days: float | None = typer.Option(None, "--days", help="Override: last N days"),
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
    out: str | None = typer.Option(None, "--out", "-o", help="Write to file (default: stdout)"),
    event_type: str | None = typer.Option(None, "--type", help="Only this event type"),
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
    tags: list[str] = typer.Argument(..., help="One or more tags to add"),
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
    tags: list[str] = typer.Argument(...),
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



@app.command()
def workspaces():
    """List all known workspaces."""
    items = list_workspaces()
    if not items:
        console.print("[yellow]No workspaces yet. Run `pmb init` in a project.[/]")
        return

    try:
        active_id = detect_workspace().id
    except Exception:
        active_id = None

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("", width=1)  # active marker
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Root")
    table.add_column("Source")
    table.add_column("Created")
    for w in items:
        mark = "[green]*[/]" if w.id == active_id else ""
        table.add_row(mark, w.id[:12], w.name, str(w.root), w.source,
                      w.created_at[:10])
    console.print(table)
    console.print("[dim]* = active · switch with "
                  "`pmb workspace use <name>`[/]")


@app.command()
def goals(
    action: str = typer.Argument(
        "list", help="list (default) | done | reconcile"),
    ulid: str | None = typer.Argument(
        None, help="Goal ULID (required for `done`)."),
    all_: bool = typer.Option(
        False, "--all", help="Include done/cancelled goals, not just open ones."),
    apply: bool = typer.Option(
        False, "--apply",
        help="With `reconcile`, close unambiguously completed goals."),
):
    """List open goals, mark one done, or reconcile completed work.

    Open goals are how PMB remembers what to do NEXT - the agent records them
    from "remember we'll do X next" / "the plan is …". Examples:
      pmb goals                 # open goals (pending + in_progress)
      pmb goals --all           # everything, including done/cancelled
      pmb goals done <ulid>     # mark a goal complete
      pmb goals reconcile       # preview completed-work matches
      pmb goals reconcile --apply
    """
    eng = Engine()
    if action == "done":
        if not ulid:
            console.print("[yellow]Usage:[/] pmb goals done <ulid>")
            raise typer.Exit(2)
        res = eng.update_goal(goal_ulid=ulid, status="done", progress=100)
        if res.get("error"):
            console.print(f"[red]{res['error']}[/]")
            raise typer.Exit(1)
        console.print(f"[green]✓ goal marked done:[/] {ulid}")
        return
    if action == "reconcile":
        res = eng.reconcile_goals(dry_run=not apply)
        if not res["matches"]:
            console.print("[green]No unambiguous completed-goal matches found.[/]")
            return
        table = Table(show_header=True, header_style="bold magenta",
                      title="goal reconciliation")
        table.add_column("Score", justify="right")
        table.add_column("Open goal")
        table.add_column("Completed activity")
        for match in res["matches"]:
            table.add_row(
                f"{match['score']:.2f}",
                esc((match["goal_title"] or "")[:70]),
                esc((match["completed_summary"] or "")[:70]),
            )
        console.print(table)
        if apply:
            console.print(f"[green]Closed {res['n_applied']} goal(s).[/]")
        else:
            console.print("[yellow]Dry-run only.[/] Re-run with --apply.")
        return
    if action != "list":
        console.print(
            f"[yellow]Unknown action[/] {action!r}. Use list | done | reconcile."
        )
        raise typer.Exit(2)

    items = eng.list_goals(limit=200)
    if not all_:
        items = [g for g in items
                 if (g.get("status") or "pending") in ("pending", "in_progress")]
    if not items:
        console.print("[yellow]No open goals.[/] The agent records goals from "
                      "\"remember we'll do X next\" / \"the plan is …\".")
        return
    table = Table(show_header=True, header_style="bold magenta",
                  title="goals" if all_ else "open goals")
    table.add_column("Status"); table.add_column("Prog", justify="right")
    table.add_column("Title"); table.add_column("Due")
    table.add_column("ULID", style="dim")
    for g in items:
        due = g.get("due_at")
        table.add_row(
            str(g.get("status") or "pending"),
            f"{g.get('progress', 0)}%",
            esc((g.get("title") or "")[:70]),
            _humanize_time(due) if due else "-",
            g["ulid"][:12],
        )
    console.print(table)
    console.print("[dim]Mark done: `pmb goals done <ulid>`[/]")


# ═══════════════════════════════════════════════════════════════════════
