"""
Git-backed workspace sync - push/pull a PMB workspace to any git remote.

Why this exists
---------------
A PMB workspace is just a directory of files under
`~/.pmb/workspaces/<id>/`:

    events.sqlite        - the event log (SQLite)
    vectors.lance/        - vector index (LanceDB)
    bm25_index.pkl        - lexical index (derived; regenerable)
    vocab_bridges.json    - mined synonym bridges (derived)
    meta.yaml             - workspace metadata

Because it's just files, git can version, back up, and SYNC it - with
zero servers and zero cloud. This unlocks three things no cloud-free
memory tool offers out of the box:

  • Cross-device sync     - `git pull` your memory onto a second laptop.
  • Team shared memory    - a private repo that several people pull/push.
  • Backup + time-travel  - every push is a restore point; `git log` is
                            an audit trail of how memory evolved.

Design choices
--------------
  • Best-effort WAL checkpoint before commit so the committed
    `events.sqlite` is self-contained (no orphaned -wal/-shm).
  • Derived caches (bm25, vocab bridges) ARE synced by default so a fresh
    `clone` is immediately queryable without a rebuild pass. Pass
    `include_cache=False` to keep the repo lean and rebuild on first use.
  • All git work is plain `subprocess` - no GitPython dependency, matching
    the style already used in `workspace.py`.
  • Binary-merge safety: SQLite/LanceDB don't line-merge. We treat the
    remote as source of truth on pull conflicts unless `--ours` is passed,
    and surface the conflict rather than silently corrupting an index.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# Files that make up a workspace's portable state.
_CORE_FILES = ["events.sqlite", "meta.yaml"]
_WAL_FILES = ["events.sqlite-wal", "events.sqlite-shm"]
_CACHE_FILES = ["bm25_index.pkl", "vocab_bridges.json"]
_TRACKED_DIRS = ["vectors.lance"]

_GITIGNORE_LEAN = """\
# PMB workspace - lean mode: derived caches are rebuilt locally on first use.
bm25_index.pkl
vocab_bridges.json
# transient SQLite journals (folded into events.sqlite before each commit)
events.sqlite-wal
events.sqlite-shm
"""

_GITIGNORE_FULL = """\
# PMB workspace - full mode: only transient SQLite journals are ignored.
events.sqlite-wal
events.sqlite-shm
"""


@dataclass
class GitResult:
    ok: bool
    action: str
    detail: str = ""
    extra: dict | None = None

    def as_dict(self) -> dict:
        d = {"ok": self.ok, "action": self.action, "detail": self.detail}
        if self.extra:
            d.update(self.extra)
        return d


class WorkspaceGitSync:
    """Git operations scoped to a single workspace storage directory."""

    def __init__(self, storage_dir: Path):
        self.dir = Path(storage_dir)

    # -- low-level -----------------------------------------------------

    def _git(
        self, *args: str, check: bool = True, timeout: float = 60.0,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.dir), *args],
            capture_output=True, text=True, timeout=timeout, check=check,
        )

    @staticmethod
    def _git_available() -> bool:
        return shutil.which("git") is not None

    def is_git(self) -> bool:
        if not (self.dir / ".git").exists():
            return False
        try:
            r = self._git("rev-parse", "--is-inside-work-tree", check=False)
            return r.returncode == 0
        except Exception:
            return False

    # -- WAL safety ----------------------------------------------------

    def _checkpoint_sqlite(self) -> None:
        """Fold the write-ahead log back into events.sqlite so the committed
        file is complete. Best-effort: a lock (engine mid-write) just skips."""
        db = self.dir / "events.sqlite"
        if not db.exists():
            return
        try:
            con = sqlite3.connect(str(db), timeout=2.0)
            try:
                con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                con.close()
        except Exception:
            pass  # locked or non-WAL - commit the file as-is

    # -- staging -------------------------------------------------------

    def _write_gitignore(self, include_cache: bool) -> None:
        content = _GITIGNORE_FULL if include_cache else _GITIGNORE_LEAN
        (self.dir / ".gitignore").write_text(content, encoding="utf-8")

    def _stage_all(self) -> None:
        # Stage tracked files + the LanceDB dir; .gitignore handles exclusions.
        self._git("add", "-A")

    def _has_staged_changes(self) -> bool:
        r = self._git("status", "--porcelain", check=False)
        return bool((r.stdout or "").strip())

    # -- public API ----------------------------------------------------

    def init(
        self,
        remote: str | None = None,
        branch: str = "main",
        include_cache: bool = True,
    ) -> GitResult:
        """Initialize a git repo in the workspace dir, optional remote."""
        if not self._git_available():
            return GitResult(False, "init", "git not found in PATH")
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.is_git():
            # already a repo - just (re)point the remote if given
            if remote:
                self._set_remote(remote)
            self._write_gitignore(include_cache)
            return GitResult(True, "init", "already a git repo", {"remote": remote})

        self._git("init", check=True)
        # set default branch name deterministically
        self._git("symbolic-ref", "HEAD", f"refs/heads/{branch}", check=False)
        self._write_gitignore(include_cache)
        if remote:
            self._set_remote(remote)
        return GitResult(True, "init", f"initialized git repo on {branch}",
                         {"remote": remote, "branch": branch})

    def _set_remote(self, remote: str) -> None:
        existing = self._git("remote", check=False).stdout.split()
        if "origin" in existing:
            self._git("remote", "set-url", "origin", remote, check=False)
        else:
            self._git("remote", "add", "origin", remote, check=False)

    def status(self) -> GitResult:
        if not self.is_git():
            return GitResult(False, "status", "workspace is not git-initialized "
                             "(run `pmb workspace init`)")
        porcelain = self._git("status", "--porcelain", check=False).stdout
        branch = self._git("rev-parse", "--abbrev-ref", "HEAD", check=False).stdout.strip()
        remote = self._git("remote", "get-url", "origin", check=False)
        remote_url = remote.stdout.strip() if remote.returncode == 0 else None
        n_changed = len([ln for ln in porcelain.splitlines() if ln.strip()])
        try:
            last = self._git("log", "-1", "--format=%h %ci %s", check=False).stdout.strip()
        except Exception:
            last = ""
        return GitResult(
            True, "status",
            f"{n_changed} uncommitted change(s)" if n_changed else "clean",
            {
                "branch": branch or None,
                "remote": remote_url,
                "dirty": n_changed > 0,
                "n_changed": n_changed,
                "last_commit": last or None,
            },
        )

    def push(
        self,
        remote: str = "origin",
        branch: str = "main",
        message: str | None = None,
        include_cache: bool = True,
    ) -> GitResult:
        if not self._git_available():
            return GitResult(False, "push", "git not found in PATH")
        if not self.is_git():
            init = self.init(branch=branch, include_cache=include_cache)
            if not init.ok:
                return GitResult(False, "push", f"auto-init failed: {init.detail}")

        self._checkpoint_sqlite()
        self._write_gitignore(include_cache)
        self._stage_all()

        committed = False
        if self._has_staged_changes():
            msg = message or f"pmb sync {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%SZ')}"
            r = self._git("commit", "-m", msg, check=False)
            if r.returncode != 0:
                return GitResult(False, "push", f"commit failed: {r.stderr.strip()[:300]}")
            committed = True

        # Is there a remote to push to?
        has_remote = self._git("remote", "get-url", remote, check=False).returncode == 0
        if not has_remote:
            return GitResult(
                True, "push",
                "committed locally (no remote configured - set one with "
                "`pmb workspace init --remote <url>`)" if committed else "nothing to commit",
                {"committed": committed, "pushed": False},
            )

        pr = self._git("push", "-u", remote, branch, check=False)
        if pr.returncode != 0:
            return GitResult(
                False, "push",
                f"committed={committed} but push failed: {pr.stderr.strip()[:400]}",
                {"committed": committed, "pushed": False},
            )
        return GitResult(True, "push",
                         f"pushed to {remote}/{branch}" + (" (new commit)" if committed else " (up to date)"),
                         {"committed": committed, "pushed": True})

    def pull(
        self,
        remote: str = "origin",
        branch: str = "main",
        strategy: str = "theirs",
    ) -> GitResult:
        """Pull workspace state from the remote.

        `strategy` decides binary-conflict resolution: 'theirs' (remote wins -
        the safe default for a synced personal memory) or 'ours' (keep local).
        SQLite/LanceDB can't line-merge, so we resolve at the file level.
        """
        if not self._git_available():
            return GitResult(False, "pull", "git not found in PATH")
        if not self.is_git():
            return GitResult(False, "pull", "workspace not git-initialized "
                             "(run `pmb workspace init --remote <url>` first)")
        has_remote = self._git("remote", "get-url", remote, check=False).returncode == 0
        if not has_remote:
            return GitResult(False, "pull", f"no remote {remote!r} configured")

        # Stash any local changes so pull is clean; remote is source of truth.
        self._checkpoint_sqlite()
        x_theirs = "-X" + ("theirs" if strategy == "theirs" else "ours")
        pr = self._git("pull", "--no-edit", x_theirs, remote, branch, check=False)
        if pr.returncode != 0:
            return GitResult(False, "pull",
                             f"pull failed: {pr.stderr.strip()[:400]}",
                             {"hint": "resolve manually in " + str(self.dir)})
        return GitResult(True, "pull", f"pulled {remote}/{branch} ({strategy} on conflict)",
                         {"strategy": strategy})


def clone_workspace(url: str, name: str, pmb_home: Path) -> GitResult:
    """Clone a remote workspace repo into ~/.pmb/workspaces/<name>.

    `name` becomes the local workspace id. Use it afterwards via
    `PMB_WORKSPACE=<name>` or `pmb connect <agent> --workspace <name>`.
    """
    if shutil.which("git") is None:
        return GitResult(False, "clone", "git not found in PATH")
    dest = Path(pmb_home) / "workspaces" / name
    if dest.exists() and any(dest.iterdir()):
        return GitResult(False, "clone",
                         f"destination already exists and is not empty: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["git", "clone", url, str(dest)],
        capture_output=True, text=True, timeout=300, check=False,
    )
    if r.returncode != 0:
        return GitResult(False, "clone", f"clone failed: {r.stderr.strip()[:400]}")

    # Robustness: if the remote's HEAD points at a branch that doesn't match
    # what was actually pushed (e.g. a self-hosted bare repo defaulting to
    # `master` while we push `main`), `git clone` leaves an empty working
    # tree. Detect that and check out the branch that actually has content.
    has_files = any(p.name != ".git" for p in dest.iterdir())
    if not has_files:
        br = subprocess.run(
            ["git", "-C", str(dest), "branch", "-r"],
            capture_output=True, text=True, check=False,
        ).stdout
        remote_branches = [
            ln.strip().split("/", 1)[-1]
            for ln in br.splitlines()
            if "->" not in ln and ln.strip()
        ]
        target = "main" if "main" in remote_branches else (
            "master" if "master" in remote_branches else
            (remote_branches[0] if remote_branches else None)
        )
        if target:
            subprocess.run(
                ["git", "-C", str(dest), "checkout", target],
                capture_output=True, text=True, check=False,
            )

    return GitResult(True, "clone", f"cloned into workspace {name!r}",
                     {"workspace": name, "path": str(dest)})
