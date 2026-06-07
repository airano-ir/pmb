"""Agent hooks — force-feed PMB into the agent at the protocol level.

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
  - claude-code : settings.json hooks {UserPromptSubmit, SessionStart, Stop}
  - codex       : ~/.codex/hooks/pmb-session-start.sh (UserPromptSubmit-equiv
                  only; Codex lacks generic stop/session lifecycle hooks)
  - cursor      : not supported (no generic user-prompt shell hook)

Public API (kept stable for tests / CLI):

    install_hook(agent) -> dict
    uninstall_hook(agent) -> dict
    list_installed() -> list[dict]
    hook_command_for(agent) -> str        # the UserPromptSubmit line
"""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Optional


_HOOK_SCRIPT_NAME = "pmb-session-start"

# Substrings that identify a PMB-installed hook command, so install is
# idempotent and uninstall can find every one we added.
_PMB_MARKERS = ("prepare-context", "session-restore", "lesson-followcheck")


def _pmb_entry() -> str:
    """Absolute path to the pmb CLI for the hook line. Hooks run with a
    minimal PATH, so prefer the venv-internal binary we're running from."""
    py = Path(sys.executable)
    for candidate in (py.parent / "pmb.exe", py.parent / "pmb"):
        if candidate.exists():
            return str(candidate)
    return "pmb"


def _claude_hook_specs() -> list[dict]:
    """The three hooks we install for claude-code, as
    (event, command) specs. `event` is the Claude Code hook event name."""
    pmb = _pmb_entry()
    return [
        {
            "event": "UserPromptSubmit",
            "command": f'"{pmb}" prepare-context --stdin --max-chars 4000 --quiet',
        },
        {
            "event": "SessionStart",
            "command": f'"{pmb}" session-restore --max-chars 3000 --quiet',
        },
        {
            "event": "Stop",
            "command": f'"{pmb}" lesson-followcheck --window 30 --quiet',
        },
    ]


def hook_command_for(agent: str) -> str:
    """Return the UserPromptSubmit command for an agent (back-compat).

    Older callers / tests use this to get "the hook line". It now returns
    specifically the prepare-context (auto-recall) command, which is the
    per-turn context injector.
    """
    pmb = _pmb_entry()
    if agent in ("claude-code", "codex"):
        return f'"{pmb}" prepare-context --stdin --max-chars 4000 --quiet'
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
                "matcher": "*",
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


def _install_codex_hook() -> dict:
    d = _codex_hooks_dir()
    d.mkdir(parents=True, exist_ok=True)
    cmd = hook_command_for("codex")
    script = d / f"{_HOOK_SCRIPT_NAME}.sh"
    body = (
        "#!/bin/sh\n"
        "# Installed by `pmb hooks install codex`.\n"
        "# Reads the user message on stdin, prints PMB context on stdout.\n"
        f"{cmd}\n"
    )
    script.write_text(body, encoding="utf-8")
    try:
        os.chmod(script, 0o755)
    except Exception:
        pass
    return {"agent": "codex", "path": str(script), "action": "installed",
            "command": cmd}


def _uninstall_codex_hook() -> dict:
    d = _codex_hooks_dir()
    script = d / f"{_HOOK_SCRIPT_NAME}.sh"
    if not script.exists():
        return {"agent": "codex", "action": "not_installed", "path": str(script)}
    script.unlink()
    return {"agent": "codex", "path": str(script), "action": "removed"}


# ── public ─────────────────────────────────────────────────────────

def install_hook(agent: str) -> dict:
    """Install the lifecycle hooks for one agent."""
    if agent in ("claude", "claude-code"):
        return _install_claude_hook()
    if agent == "codex":
        return _install_codex_hook()
    if agent == "cursor":
        return {
            "agent": "cursor",
            "action": "not_supported",
            "reason": "Cursor lacks a generic user-prompt shell hook. "
                      "PMB context is still injected via CLAUDE.md rules "
                      "installed by `pmb connect cursor`.",
        }
    raise ValueError(f"unknown agent {agent!r}")


def uninstall_hook(agent: str) -> dict:
    if agent in ("claude", "claude-code"):
        return _uninstall_claude_hook()
    if agent == "codex":
        return _uninstall_codex_hook()
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
    # codex
    cs = _codex_hooks_dir() / f"{_HOOK_SCRIPT_NAME}.sh"
    out.append({"agent": "codex", "event": "session-start",
                "installed": cs.exists(), "path": str(cs)})
    return out
