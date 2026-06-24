"""Exploration memo cache - memoize the agent's research conclusions, keyed to
the code state they were derived from, and replay them with a hash-gated
freshness check instead of re-deriving.

The idea (incremental cognition): the dominant token cost of an agent is not the
size of any one snapshot - prompt caching makes repeated context cheap - it is
the EXPLORATION LOOP repeated cold every new session (re-grep, re-read N files
to rebuild a model it already had). This memoizes the OUTPUT of that loop:

  record_exploration(intent, conclusion, files)
      -> stores intent + conclusion + each source file's content sha1.

  recall_exploration(intent)
      -> finds matching memos and, for each, re-hashes the source files NOW:
         all unchanged  -> fresh=True, trust the conclusion (zero re-reading);
         some changed   -> fresh=False + stale_files, re-check only those.

The hash gate is the safety boundary: a memo is only trusted while the files it
was derived from are byte-identical. Think incremental build (ccache/Bazel:
cache the artifact, invalidate on input change) applied to agent reasoning.

This is NOT a structure snapshot (index_project) nor fact memory - it caches
grounded conclusions the agent actually derived, so it cannot drift into a
plausible guess, and it saves exploration STEPS, which is where the cost is.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmb.core.engine import Engine

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> set[str]:
    """Lowercase word tokens >= 3 chars - cheap lexical match on intent."""
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) >= 3}


def _file_sha1(path: Path) -> str | None:
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _resolve(project_path: str, f: str) -> Path:
    p = Path(f)
    return p if p.is_absolute() else Path(project_path) / f


def _git_head(root: str) -> str | None:
    """Best-effort short HEAD sha of the repo at `root`, or None if not a repo."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
            encoding="utf-8", errors="replace",
        )
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except Exception:
        return None


def record_exploration(
    engine: Engine,
    intent: str,
    conclusion: str,
    files: list[str] | None = None,
    project_path: str | None = None,
) -> dict:
    """Memoize a conclusion reached by exploring `files`.

    Each source file's CURRENT content is hashed now, so recall can later detect
    exactly what changed. `files` may be absolute or relative to project_path
    (default: cwd).
    """
    if not (intent or "").strip() or not (conclusion or "").strip():
        return {"recorded": False, "error": "intent and conclusion are required"}
    root = str(Path(project_path).resolve()) if project_path else str(Path.cwd())

    sources: list[dict] = []
    for f in files or []:
        sha = _file_sha1(_resolve(root, f))
        sources.append({"file": str(f), "sha1": sha})

    files_line = ", ".join(s["file"] for s in sources) or "(none)"
    content = (
        f"Exploration memo: {intent}\n"
        f"Conclusion: {conclusion}\n"
        f"Sources: {files_line}"
    )
    engine.record_batch_async(items=[{
        "type": "fact",
        "content": content,
        "importance": 0.7,
        "metadata": {
            "source": "exploration-memo",
            "intent": intent,
            "conclusion": conclusion,
            "project_path": root,
            "sources": sources,
            "recorded_at_head": _git_head(root),
        },
    }])
    return {"recorded": True, "intent": intent, "n_sources": len(sources)}


def recall_exploration(
    engine: Engine,
    intent: str,
    project_path: str | None = None,
    top_k: int = 3,
) -> dict:
    """Find memoized conclusions matching `intent`, each with a freshness check.

    Returns {"intent", "n", "matches": [{intent, conclusion, fresh, stale_files,
    sources:[{file, fresh}]}]}. `fresh` is True only when every source file is
    byte-identical to when the memo was recorded.
    """
    import sqlite3

    root = str(Path(project_path).resolve()) if project_path else str(Path.cwd())
    cur_head = _git_head(root)
    q = _tokens(intent)
    if not q:
        return {"intent": intent, "n": 0, "matches": []}

    scored: list[tuple[int, dict]] = []
    try:
        with sqlite3.connect(engine.workspace.db_path) as c:
            cur = c.execute(
                "SELECT metadata_json FROM events "
                "WHERE workspace_id=? AND archived_at IS NULL "
                "AND json_extract(metadata_json,'$.source')='exploration-memo' "
                "AND json_extract(metadata_json,'$.project_path')=? "
                "ORDER BY rowid DESC",
                (engine.workspace.id, root),
            )
            for (mj,) in cur.fetchall():
                try:
                    md = json.loads(mj) if mj else {}
                except Exception:
                    continue
                score = len(q & _tokens(md.get("intent", "")))
                if score:
                    scored.append((score, md))
    except Exception as e:
        log.warning("recall_exploration read failed: %s", e)
        return {"intent": intent, "n": 0, "matches": []}

    scored.sort(key=lambda x: -x[0])

    matches: list[dict] = []
    for _score, md in scored[:top_k]:
        checked: list[dict] = []
        stale: list[str] = []
        for s in md.get("sources") or []:
            cur_sha = _file_sha1(_resolve(root, s.get("file", "")))
            ok = cur_sha is not None and cur_sha == s.get("sha1")
            checked.append({"file": s.get("file"), "fresh": ok})
            if not ok:
                stale.append(s.get("file"))
        recorded_head = md.get("recorded_at_head")
        matches.append({
            "intent": md.get("intent"),
            "conclusion": md.get("conclusion"),
            "fresh": not stale,
            "stale_files": stale,
            "sources": checked,
            # The hash gate covers only LISTED sources; this flags that the repo
            # moved since the memo, so a conclusion may depend on files that
            # changed but were not listed. Verify when True even if fresh.
            "repo_changed_since": bool(
                recorded_head and cur_head and recorded_head != cur_head
            ),
        })

    return {"intent": intent, "n": len(matches), "matches": matches}
