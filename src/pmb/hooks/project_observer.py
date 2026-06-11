"""Project observer — ambient memory for MCP-only agents (Cursor, VS Code,
Zed, Gemini, Windsurf, …).

Those hosts give PMB no hooks and no parseable action log, so we can't watch
the *agent*. But the agent's work lands as **file changes in the project**,
and those we CAN watch — independently of the host. The observer polls git
for changed files (cross-platform, no extra deps, no watchdog), records each
as an agent action, and — after the project goes idle for a bit (the
"turn-ended" signal we don't get from a Stop hook) — runs the same ambient
auto-write as Claude Code / Codex.

Tradeoffs vs hook-based ambient (honest):
  • Sees file edits, not shell commands (no Bash/pytest/commit unless they
    change tracked files — though commits show up via git log separately).
  • "Turn end" is approximated by an idle timer, not an exact Stop event.
  • Attributes any change in the working tree, including ones you made by
    hand — it's "what happened in the project", not strictly "what the agent
    did". That's usually fine for a journal.

Pure stdlib + git subprocess. Used by `pmb ambient-watch`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _run_git(repo: Path, *args: str, timeout: float = 10.0) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        if r.returncode != 0:
            return None
        return r.stdout
    except Exception:
        return None


def is_git_repo(repo: Path) -> bool:
    out = _run_git(repo, "rev-parse", "--is-inside-work-tree")
    return bool(out and out.strip() == "true")


def _dirty_files(repo: Path) -> list[str]:
    """Paths with uncommitted changes (modified + added + untracked), via
    `git status --porcelain`. Excludes deletions (nothing to journal)."""
    out = _run_git(repo, "status", "--porcelain", "--untracked-files=all")
    if not out:
        return []
    files: list[str] = []
    for line in out.splitlines():
        if len(line) < 4:
            continue
        code, path = line[:2], line[3:].strip()
        if code.strip() in ("D", "DD") or code == " D":
            continue  # deletion — skip
        # Renames look like "old -> new"; take the new path.
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.strip().strip('"')
        if path:
            files.append(path)
    return files


def snapshot_changes(
    repo: Path,
    prev_state: dict[str, float],
) -> tuple[list[dict], dict[str, float]]:
    """Diff the working tree against the previous snapshot.

    `prev_state` maps path → mtime from the last scan. Returns
    (new_actions, new_state): a path is a new action if it's newly dirty or
    its mtime changed since last scan. new_state is the current snapshot to
    persist for the next call.
    """
    actions: list[dict] = []
    new_state: dict[str, float] = {}
    for path in _dirty_files(repo):
        try:
            mt = (repo / path).stat().st_mtime
        except OSError:
            mt = 0.0
        new_state[path] = mt
        if path not in prev_state or abs(prev_state[path] - mt) > 1e-6:
            actions.append({
                "tool": "Edit", "target": path, "status": "ok", "command": "",
            })
    return actions, new_state
