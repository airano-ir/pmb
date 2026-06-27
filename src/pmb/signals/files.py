"""
File correlation tracking.

Idea: files that are often modified together → they are logically related.
When the user later looks at file X, we can show events related to files
that are frequently modified together with X.

Approach: out-of-process observation (no hooks inside Claude Code).
For now it works via git: we look at which files are in the same commit
and count their co-occurrence.

Possible future additions:
- Inotify/FSEvents observation
- A hook from Claude Code (if an API becomes available)
"""

from __future__ import annotations

import time
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmb.core.engine import Engine


def _files_of(ev) -> list:
    """The changed-files list from a git-change memory, tolerating both
    schemas: `pmb track changes` writes 'files', the legacy git-signal path
    wrote 'files_changed'."""
    md = ev.metadata or {}
    return md.get("files") or md.get("files_changed") or []


class FileCorrelation:
    """Analysis of file co-occurrence in git commits."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def _get_git_events(self) -> list:
        """Git-change memories. `pmb track changes` stores these as facts with
        metadata.source == 'git-change' (files under 'files'); the older
        git-signal path used event_type='git' (files under 'files_changed').
        Return both so file history / correlation see commits tracked either
        way - the user-facing `pmb track changes` writes the fact form, so
        filtering on event_type='git' alone found nothing."""
        ws = self.engine.workspace.id
        facts = self.engine.events.list_active(ws, limit=10000, event_type="fact")
        out = [e for e in facts if (e.metadata or {}).get("source") == "git-change"]
        out += self.engine.events.list_active(ws, limit=10000, event_type="git")
        return out

    def correlations(self, file_path: str, top_k: int = 10) -> list[tuple[str, int]]:
        """
        Find the files most frequently modified together with file_path.

        Returns list of (file_path, count) sorted descending.
        """
        target = file_path.replace("\\", "/").strip()
        events = self._get_git_events()

        co_occur: Counter = Counter()
        for ev in events:
            files = _files_of(ev)
            files_norm = [f.replace("\\", "/").strip() for f in files]
            if target in files_norm:
                for other in files_norm:
                    if other != target:
                        co_occur[other] += 1

        return co_occur.most_common(top_k)

    def all_files_touched(self, since_days: int = 30) -> list[tuple[str, int]]:
        """All files mentioned in git events over the last N days + count."""
        events = self._get_git_events()
        cutoff = time.time() - since_days * 86400

        counter: Counter = Counter()
        for ev in events:
            if ev.timestamp < cutoff:
                continue
            for f in _files_of(ev):
                counter[f.replace("\\", "/")] += 1

        return counter.most_common()

    def file_history(self, file_path: str) -> list[dict]:
        """Commit history for a single file."""
        target = file_path.replace("\\", "/").strip()
        events = self._get_git_events()

        history = []
        for ev in events:
            files = [f.replace("\\", "/").strip() for f in _files_of(ev)]
            if target in files:
                history.append({
                    "ulid": ev.ulid,
                    "sha": ev.metadata.get("commit_short") or ev.metadata.get("short_sha"),
                    "subject": ev.metadata.get("subject", ev.content[:80]),
                    "timestamp": ev.timestamp,
                    "author": ev.metadata.get("author"),
                })
        history.sort(key=lambda x: -x["timestamp"])
        return history
