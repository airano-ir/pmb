"""Agent hooks - force-feed PMB into the agent at the protocol level.

Soft instructions in CLAUDE.md get skipped; hooks don't. The agent's host
runs a shell command at well-defined lifecycle points and folds the output
into the model's context. PMB wires three of them:

  • UserPromptSubmit → `pmb prepare-context`     auto-recall: classify the
        message and inject matching lessons / decisions / recall / project
        context BEFORE the model thinks. (the big one)
  • SessionStart     → `pmb session-restore`     after a compaction / resume,
        rebuild "where you left off" from what THIS session recorded.
  • Stop             → `pmb lesson-followcheck`   when the turn ends, infer
        which surfaced lessons were actually followed (no self-report needed)
        so the adherence dashboard reflects reality.

Targets:
  - claude-code : settings.json hooks {UserPromptSubmit, SessionStart,
                  PreToolUse, PostToolUse, Stop} - the full set.
  - codex       : ~/.codex/config.toml `notify` → `pmb codex-notify` (ambient
                  auto-write on agent-turn-complete). Codex has NO per-turn /
                  session-start shell hook, so read-first / auto-recall is
                  driven by the AGENTS.md rules `pmb connect codex` writes (the
                  agent calls prepare()/recall() itself), not by an injected
                  hook. (Older PMB wrote a ~/.codex/hooks/pmb-session-start.sh
                  that Codex never executed; install now cleans it up.)
  - cursor      : not supported (no generic user-prompt shell hook)

Public API (kept stable for tests / CLI):

    install_hook(agent) -> dict
    uninstall_hook(agent) -> dict
    list_installed() -> list[dict]
    hook_command_for(agent) -> str        # the UserPromptSubmit line
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HOOK_SCRIPT_NAME = "pmb-session-start"

# Substrings that identify a PMB-installed hook command, so install is
# idempotent and uninstall can find every one we added.
_PMB_MARKERS = (
    "prepare-context", "session-restore", "lesson-followcheck",
    "track-action", "autowrite", "pretool",
)


def _pmb_entry() -> str:
    """Absolute path to the pmb CLI for the hook line. Hooks run with a
    minimal PATH, so prefer the venv-internal binary we're running from."""
    py = Path(sys.executable)
    for candidate in (py.parent / "pmb.exe", py.parent / "pmb"):
        if candidate.exists():
            return str(candidate)
    return "pmb"


def _pmb_hook_entry() -> str:
    """Absolute path to the stdlib-only `pmb-hook` fast lane (S2). Falls back to
    the full `pmb` binary if pmb-hook isn't installed (older wheels), so an
    upgrade is seamless either way."""
    py = Path(sys.executable)
    for candidate in (py.parent / "pmb-hook.exe", py.parent / "pmb-hook"):
        if candidate.exists():
            return str(candidate)
    return "pmb-hook"


def _claude_hook_specs() -> list[dict]:
    """The hooks we install for claude-code, as (event, command) specs.
    `event` is the Claude Code hook event name.

    All five route through `pmb-hook` (S2) - the stdlib-only fast lane that
    talks to the warm daemon (≈10-50 ms) and falls back to the full CLI cold
    path only when the daemon is absent. The old `pmb <sub>` lines keep working
    and are upgraded in place on the next `pmb hooks install` (markers match
    both)."""
    h = _pmb_hook_entry()
    return [
        {
            "event": "UserPromptSubmit",
            "command": f'"{h}" prepare-context --max-chars 4000 --quiet',
        },
        {
            "event": "SessionStart",
            "command": f'"{h}" session-restore --max-chars 3000 --quiet',
        },
        # PreToolUse: R11 lesson guard - fire a matching rule ("use pnpm, never
        # npm") at tool-call time, even if the agent never called memory.
        # Daemon-served + advisory (never blocks); no-op without a daemon.
        {
            "event": "PreToolUse",
            "matcher": "Bash|Edit|Write|NotebookEdit",
            "command": f'"{h}" pretool --quiet',
        },
        # PostToolUse: ambient observer - log the agent's action (instant).
        {
            "event": "PostToolUse",
            "command": f'"{h}" track-action --quiet',
        },
        {
            "event": "Stop",
            "command": f'"{h}" lesson-followcheck --window 30 --quiet',
        },
        # Stop: ambient auto-write - journal the turn if the agent didn't.
        # No-op unless `autowrite.enabled` is true in config, so installing
        # the hook is safe; it stays silent until the user opts in.
        {
            "event": "Stop",
            "command": f'"{h}" autowrite --window 30 --quiet',
        },
    ]


def hook_command_for(agent: str) -> str:
    """Return the UserPromptSubmit command for an agent (back-compat).

    Older callers / tests use this to get "the hook line". It now returns
    specifically the prepare-context (auto-recall) command, which is the
    per-turn context injector - via the `pmb-hook` fast lane (S2).
    """
    h = _pmb_hook_entry()
    if agent in ("claude-code", "codex"):
        return f'"{h}" prepare-context --max-chars 4000 --quiet'
    raise ValueError(f"no hook support for agent {agent!r}")


# ── claude-code ────────────────────────────────────────────────────

def _claude_settings_path() -> Path:
    return Path.home() / ".claude" / "settings.json"


def _load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {}


def _install_claude_hook() -> dict:
    p = _claude_settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg = _load_json(p)
    hooks = cfg.setdefault("hooks", {})

    actions: list[dict] = []
    for spec in _claude_hook_specs():
        event = spec["event"]
        cmd = spec["command"]
        entries = hooks.setdefault(event, [])
        # Find any existing PMB-tagged hook under this event and update it
        # in place; identify by the subcommand marker so we don't duplicate.
        marker = next((m for m in _PMB_MARKERS if m in cmd), None)
        existing = None
        for entry in entries:
            for h in entry.get("hooks", []):
                hc = h.get("command") or ""
                if marker and marker in hc:
                    existing = h
                    break
            if existing:
                break
        if existing:
            existing["command"] = cmd
            actions.append({"event": event, "action": "updated"})
        else:
            entries.append({
                "matcher": spec.get("matcher", "*"),
                "hooks": [{"type": "command", "command": cmd}],
            })
            actions.append({"event": event, "action": "created"})

    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"agent": "claude-code", "path": str(p), "actions": actions,
            "events": [s["event"] for s in _claude_hook_specs()]}


def _uninstall_claude_hook() -> dict:
    p = _claude_settings_path()
    if not p.exists():
        return {"agent": "claude-code", "action": "not_installed"}
    cfg = _load_json(p)
    hooks = cfg.get("hooks", {})
    removed = 0
    for event, entries in list(hooks.items()):
        new_entries = []
        for entry in entries:
            kept = []
            for h in entry.get("hooks", []):
                hc = h.get("command") or ""
                if any(m in hc for m in _PMB_MARKERS):
                    removed += 1
                    continue
                kept.append(h)
            if kept:
                ne = dict(entry)
                ne["hooks"] = kept
                new_entries.append(ne)
        if new_entries:
            hooks[event] = new_entries
        else:
            hooks.pop(event, None)
    if removed == 0:
        return {"agent": "claude-code", "action": "not_installed", "path": str(p)}
    cfg["hooks"] = hooks
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"agent": "claude-code", "path": str(p), "action": "removed",
            "n_removed": removed}


# ── codex (OpenAI Codex CLI) ───────────────────────────────────────

def _codex_hooks_dir() -> Path:
    return Path.home() / ".codex" / "hooks"


def _codex_config_path() -> Path:
    return Path.home() / ".codex" / "config.toml"


def _install_codex_notify() -> dict:
    """Wire `notify` in ~/.codex/config.toml to `pmb codex-notify`, which is
    the ambient observer + auto-write for Codex (parses the rollout log on
    each agent-turn-complete). Best-effort: needs tomllib (read) + a TOML
    writer (write); if either is missing we skip notify but keep the
    session-start script."""
    p = _codex_config_path()
    pmb = _pmb_entry()
    notify_value = [pmb, "codex-notify"]
    try:
        try:
            import tomllib as _toml_r  # py3.11+
        except Exception:
            import tomli as _toml_r  # type: ignore
        import tomli_w as _toml_w  # type: ignore
    except Exception:
        return {"notify": "skipped", "reason": "no TOML reader/writer"}
    data = {}
    if p.exists():
        try:
            data = _toml_r.loads(p.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    existing = data.get("notify")
    # Only overwrite if it's absent or already a pmb notify (don't clobber a
    # user's own notify program).
    if existing and not (isinstance(existing, list)
                         and any("codex-notify" in str(x) for x in existing)):
        return {"notify": "skipped",
                "reason": f"existing non-pmb notify: {existing}"}
    data["notify"] = notify_value
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_toml_w.dumps(data), encoding="utf-8")
    except Exception as e:
        return {"notify": "failed", "reason": str(e)}
    return {"notify": "installed", "value": notify_value, "path": str(p)}


def _install_codex_hook() -> dict:
    """Codex's ONLY extension point is `notify` (fired on agent-turn-complete),
    so that's the real integration - wire it to `pmb codex-notify` (the ambient
    observer + auto-write that reads the rollout log). Codex has no per-turn /
    session-start shell hook, so read-first / auto-recall is driven by the
    AGENTS.md rules `pmb connect codex` writes, not by an injected hook.

    Older PMB versions also wrote a ~/.codex/hooks/pmb-session-start.sh here.
    Codex never executed it - it only made `pmb hooks list` look like codex had
    a session-start hook when it didn't. Clean it up on (re)install."""
    legacy = _codex_hooks_dir() / f"{_HOOK_SCRIPT_NAME}.sh"
    legacy_removed = False
    try:
        if legacy.exists():
            legacy.unlink()
            legacy_removed = True
    except Exception:
        pass
    notify = _install_codex_notify()
    return {"agent": "codex", "action": "installed",
            "mechanism": "rollout-notify", "notify": notify,
            "path": str(_codex_config_path()),
            "legacy_script_removed": legacy_removed}


def _uninstall_codex_hook() -> dict:
    d = _codex_hooks_dir()
    script = d / f"{_HOOK_SCRIPT_NAME}.sh"
    removed_script = False
    if script.exists():
        script.unlink()
        removed_script = True
    # Remove our notify from config.toml (only if it's ours).
    notify_removed = False
    p = _codex_config_path()
    if p.exists():
        try:
            try:
                import tomllib as _toml_r
            except Exception:
                import tomli as _toml_r  # type: ignore
            import tomli_w as _toml_w  # type: ignore
            data = _toml_r.loads(p.read_text(encoding="utf-8"))
            nv = data.get("notify")
            if isinstance(nv, list) and any("codex-notify" in str(x) for x in nv):
                data.pop("notify", None)
                p.write_text(_toml_w.dumps(data), encoding="utf-8")
                notify_removed = True
        except Exception:
            pass
    if not removed_script and not notify_removed:
        return {"agent": "codex", "action": "not_installed", "path": str(script)}
    return {"agent": "codex", "path": str(script), "action": "removed",
            "notify_removed": notify_removed}


# ── capability registry ────────────────────────────────────────────
# Ambient memory needs to OBSERVE the agent's actions. How (or whether) we
# can depends entirely on what the host exposes:
#
#   "hooks"    - rich lifecycle hooks (PostToolUse + Stop + SessionStart).
#                Full ambient: auto-recall, session-restore, follow-through,
#                ambient auto-write. (Claude Code.)
#   "rollout"  - no per-tool hook, but the host writes an action log we can
#                parse + a turn-complete notify. Auto-recall + ambient
#                auto-write. (Codex.)
#   "mcp-only" - MCP works (recall/record via tools + CLAUDE.md/AGENTS.md
#                rules), but there's no way to observe file edits / shell
#                commands. Auto-recall works; ambient auto-write does NOT
#                (nothing to observe). (Cursor, Windsurf, VS Code, Zed,
#                Gemini, opencode, continue.)
_AGENT_CAP: dict[str, str] = {
    "claude": "hooks",
    "claude-code": "hooks",
    "codex": "rollout",
    "cursor": "mcp-only",
    "windsurf": "mcp-only",
    "vscode": "mcp-only",
    "zed": "mcp-only",
    "gemini": "mcp-only",
    "opencode": "mcp-only",
    "continue": "mcp-only",
}


def ambient_capability(agent: str) -> str:
    """What ambient mechanism is available for `agent`:
    'hooks' | 'rollout' | 'mcp-only' | 'unknown'."""
    return _AGENT_CAP.get(agent, "unknown")


def capability_report() -> list[dict]:
    """Per-agent ambient capability + what works, for `pmb hooks capabilities`."""
    feature = {
        "hooks": ("auto-recall + session-restore + follow-through + "
                  "ambient auto-write"),
        "rollout": "auto-recall + ambient auto-write (via rollout log + notify)",
        "mcp-only": ("auto-recall via rules + ambient via the project "
                     "observer (`pmb ambient-watch .`) - watches git for "
                     "file changes since the host gives no hooks"),
        "unknown": "unknown agent",
    }
    out = []
    for ag in ("claude-code", "codex", "cursor", "windsurf", "vscode",
               "zed", "gemini", "opencode", "continue"):
        cap = ambient_capability(ag)
        # Ambient is available everywhere now: directly via hooks/rollout, or
        # via the project observer for mcp-only hosts.
        mech = {"hooks": "hooks", "rollout": "rollout",
                "mcp-only": "project-observer"}.get(cap, "none")
        out.append({
            "agent": ag,
            "capability": cap,
            "ambient": cap in ("hooks", "rollout", "mcp-only"),
            "ambient_mechanism": mech,
            "details": feature[cap],
        })
    return out


# ── public ─────────────────────────────────────────────────────────

def install_hook(agent: str) -> dict:
    """Install the best available ambient mechanism for `agent`.

    Dispatches on capability: rich hooks (Claude Code), rollout+notify
    (Codex), or - for MCP-only hosts - reports that ambient observation
    isn't available there (auto-recall still works via `pmb connect`).
    """
    cap = ambient_capability(agent)
    if cap == "hooks":
        return _install_claude_hook()
    if cap == "rollout":
        return _install_codex_hook()
    if cap == "mcp-only":
        return {
            "agent": agent,
            "action": "mcp_only",
            "capability": "mcp-only",
            "reason": (
                f"{agent} exposes MCP but no hooks / action log, so we can't "
                f"observe the agent directly. Two steps for full PMB:\n"
                f"  1. `pmb connect {agent}`   - MCP + auto-recall rules\n"
                f"  2. `pmb ambient-watch .`   - ambient auto-write by "
                f"watching the project's git changes (run it next to your "
                f"editor). Coordination still holds: silent if the agent "
                f"journaled via MCP."
            ),
        }
    raise ValueError(f"unknown agent {agent!r}")


def uninstall_hook(agent: str) -> dict:
    if agent in ("claude", "claude-code"):
        return _uninstall_claude_hook()
    if agent == "codex":
        return _uninstall_codex_hook()
    if ambient_capability(agent) == "mcp-only":
        return {"agent": agent, "action": "not_installed",
                "reason": "mcp-only agent - nothing was hook-installed"}
    raise ValueError(f"unknown agent {agent!r}")


def list_installed() -> list[dict]:
    """Report which hooks are installed, per agent + event."""
    out: list[dict] = []
    # claude-code: report each of the three events.
    p = _claude_settings_path()
    cfg = _load_json(p)
    hooks = cfg.get("hooks", {})
    for spec in _claude_hook_specs():
        event = spec["event"]
        present = False
        for entry in hooks.get(event, []):
            for h in entry.get("hooks", []):
                hc = h.get("command") or ""
                if any(m in hc for m in _PMB_MARKERS):
                    present = True
                    break
            if present:
                break
        out.append({
            "agent": "claude-code", "event": event,
            "installed": present, "path": str(p),
        })
    # codex: the REAL mechanism is `notify` in config.toml (post-turn ambient
    # observer). Codex has no session-start / per-turn shell hook, so that's what
    # we report - not the legacy pmb-session-start.sh (which Codex never ran).
    notify_on = False
    cp = _codex_config_path()
    if cp.exists():
        try:
            try:
                import tomllib as _tr
            except Exception:
                import tomli as _tr  # type: ignore
            nv = _tr.loads(cp.read_text(encoding="utf-8")).get("notify")
            notify_on = (isinstance(nv, list)
                         and any("codex-notify" in str(x) for x in nv))
        except Exception:
            notify_on = False
    out.append({"agent": "codex", "event": "notify (ambient auto-write)",
                "installed": notify_on, "path": str(cp)})
    return out
