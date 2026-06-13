"""Codex rollout parser - the ambient observer for OpenAI Codex CLI.

Claude Code gives PMB a PostToolUse hook to observe each action. Codex has
no per-tool hook, but it writes a full session log (`~/.codex/sessions/
YYYY/MM/DD/rollout-*.jsonl`) containing every `function_call` the agent
made. So we observe Codex the same way auto-recall observes Claude - by
reading what's already there.

A Codex `notify` program (fired on `agent-turn-complete`) calls
`pmb codex-notify`, which uses this module to:

  • find the active rollout,
  • parse the NEW function_calls since the last turn (offset-tracked),
  • map them to (tool, target, status) - shell_command → Bash,
    apply_patch → Edit - and detect record_* calls (the agent journaled
    itself → coordination gate),

then ambient auto-write runs exactly as on Claude Code.

Pure stdlib (json + pathlib) - no heavy imports, this is a hot-ish path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Codex tool name → PMB (tool, is_record). Anything not here is non-significant.
_SHELL_NAMES = {"shell_command", "shell", "local_shell", "exec_command", "bash"}
_PATCH_NAMES = {"apply_patch", "edit_file", "write_file", "create_file", "str_replace"}
_RECORD_PREFIXES = ("record_", "remember", "index_")  # PMB write tools


def _codex_sessions_dir() -> Path:
    return Path.home() / ".codex" / "sessions"


def find_latest_rollout(sessions_dir: Path | None = None) -> Path | None:
    """The most recently modified rollout-*.jsonl, or None."""
    base = sessions_dir or _codex_sessions_dir()
    if not base.exists():
        return None
    newest = None
    newest_mtime = -1.0
    for p in base.rglob("rollout-*.jsonl"):
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m > newest_mtime:
            newest_mtime = m
            newest = p
    return newest


def _extract_call(payload: dict) -> dict | None:
    """Map one function_call payload to a PMB action dict, or None if it's a
    trivial/UI tool (render_chart, update_plan, web_search, …)."""
    name = (payload.get("name") or "").strip()
    if not name:
        return None
    # Agent journaled itself? (record_batch / record_fact / remember / index_*)
    low = name.lower()
    if low.startswith(_RECORD_PREFIXES):
        return {"_record": True, "name": name}

    args_raw = payload.get("arguments") or "{}"
    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
    except Exception:
        args = {}

    if low in _SHELL_NAMES:
        cmd = args.get("command") or args.get("cmd") or ""
        if isinstance(cmd, list):
            cmd = " ".join(str(c) for c in cmd)
        return {"tool": "Bash", "target": str(cmd)[:400],
                "command": str(cmd), "call_id": payload.get("call_id")}
    if low in _PATCH_NAMES:
        # apply_patch arguments vary; try common shapes for the file path.
        target = (args.get("path") or args.get("file_path")
                  or args.get("filename") or "")
        if not target and isinstance(args.get("input"), str):
            # apply_patch free-form: pull the first "*** ... File: path" hint
            m = re.search(r"(?:\+\+\+|Update File:|Add File:|\*\*\*)\s*(\S+)",
                          args["input"])
            target = m.group(1) if m else ""
        return {"tool": "Edit", "target": str(target)[:400],
                "command": "", "call_id": payload.get("call_id")}
    # Everything else (js, render_chart, render_table, update_plan, web_search)
    # is not a journaling-worthy action.
    return None


@dataclass
class RolloutScan:
    actions: list[dict] = field(default_factory=list)   # {tool,target,command,status,call_id}
    agent_recorded: bool = False                        # saw a record_* call
    new_offset: int = 0                                 # lines consumed
    rollout_path: str | None = None


def parse_rollout_actions(
    rollout_path: Path,
    since_offset: int = 0,
) -> RolloutScan:
    """Parse function_calls from `rollout_path` starting at line `since_offset`.

    Returns the significant actions, whether the agent called a record_* tool
    (coordination), and the new line offset to persist. Pairs each call with
    its function_call_output to fill in a coarse status (ok / non-zero exit).
    """
    scan = RolloutScan(rollout_path=str(rollout_path))
    pending: dict[str, dict] = {}  # call_id → action awaiting its output
    line_no = 0
    try:
        with open(rollout_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if line_no <= since_offset:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                p = o.get("payload") if isinstance(o.get("payload"), dict) else o
                pt = p.get("type")
                if pt == "function_call":
                    rec = _extract_call(p)
                    if rec is None:
                        continue
                    if rec.get("_record"):
                        scan.agent_recorded = True
                        continue
                    rec["status"] = ""
                    cid = rec.get("call_id")
                    if cid:
                        pending[cid] = rec
                    scan.actions.append(rec)
                elif pt == "function_call_output":
                    cid = p.get("call_id")
                    out = p.get("output") or ""
                    act = pending.get(cid)
                    if act is not None:
                        m = re.search(r"Exit code:\s*(-?\d+)", str(out))
                        if m:
                            act["status"] = "ok" if m.group(1) == "0" else m.group(1)
                        elif out:
                            act["status"] = "ok"
    except FileNotFoundError:
        pass
    scan.new_offset = line_no
    return scan
