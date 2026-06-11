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


class FileCorrelation:
    """Analysis of file co-occurrence in git commits."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def _get_git_events(self) -> list:
        return self.engine.events.list_active(
            self.engine.workspace.id, limit=10000, event_type="git",
        )

    def correlations(self, file_path: str, top_k: int = 10) -> list[tuple[str, int]]:
        """
        Find the files most frequently modified together with file_path.

        Returns list of (file_path, count) sorted descending.
        """
        target = file_path.replace("\\", "/").strip()
        events = self._get_git_events()

        co_occur: Counter = Counter()
        for ev in events:
            files = ev.metadata.get("files_changed", [])
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
            for f in ev.metadata.get("files_changed", []):
                counter[f.replace("\\", "/")] += 1

        return counter.most_common()

    def file_history(self, file_path: str) -> list[dict]:
        """Commit history for a single file."""
        target = file_path.replace("\\", "/").strip()
        events = self._get_git_events()

        history = []
        for ev in events:
            files = [f.replace("\\", "/").strip()
                     for f in ev.metadata.get("files_changed", [])]
            if target in files:
                history.append({
                    "ulid": ev.ulid,
                    "sha": ev.metadata.get("short_sha"),
                    "subject": ev.metadata.get("subject", ev.content[:80]),
                    "timestamp": ev.timestamp,
                    "author": ev.metadata.get("author"),
                })
        history.sort(key=lambda x: -x["timestamp"])
        return history
