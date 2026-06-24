"""Git-tied semantic change tracking + per-module purpose summaries.

Two layers that sit ON TOP of `index_project` (which captures only the
deterministic structure - symbols + imports):

  1. summarize_modules() - ask a small LLM (Haiku by default) for a one-line
     "what does this module DO" per indexed file. The structural index says
     WHICH symbols exist; this says WHY the file exists. Stored as a linked
     `fact` so it surfaces in recall right next to the file.

  2. track_changes() - read NEW git commits since the last cursor, ask the LLM
     to summarise the INTENT of each change (the "why", which a raw diff does
     not carry), and store it as a `git` activity linked to the files it
     touched. We layer ON git: we keep the intent, not the diff.

Both reuse existing machinery instead of growing a parallel stack:

  * the LLM client abstraction in ``pmb.health.consolidate``
    (``resolve_llm_client``), which already supports Claude CLI / Anthropic /
    Ollama and runs the model with NO tools - safe for feeding it untrusted
    diffs and code.
  * ``engine.record_batch_async`` for writes (durable outbox, embeddings,
    graph linking happen downstream).
  * the same sqlite read pattern as ``pmb.ingest.project``.

Cost: one LLM call per commit (changes) or per file (modules), on Haiku
(~$0.001/call) or fully local via Ollama. A per-run cap bounds it, and both
layers are idempotent - a commit already tracked (via a per-repo cursor) or a
file already summarised is skipped.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmb.core.engine import Engine
    from pmb.health.consolidate import LLMClient

log = logging.getLogger(__name__)

# Keep token cost predictable: cap the slice of diff/file we feed the LLM.
_DIFF_CHAR_CAP = 4000
_FILE_HEAD_CAP = 1200
_SUBJECT_CAP = 200
_BODY_CAP = 1000


# ----------------------------------------------------------------------
# git helpers (we shell out, like core/git_sync.py does)
# ----------------------------------------------------------------------

def _git(root: Path | str, *args: str, timeout: float = 30.0) -> str:
    """Run ``git -C <root> <args>`` and return stdout. Raises on non-zero."""
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({proc.returncode}): "
            f"{(proc.stderr or '').strip()[:300]}"
        )
    return proc.stdout


def _repo_root(path: Path | str) -> str | None:
    """Absolute toplevel of the git repo that contains `path`, or None.

    Canonicalised through Path.resolve() so it matches the project_path that
    index_project stores. git prints a forward-slash path even on Windows
    (`C:/x/y`) while Path.resolve() yields the native form (`C:\\x\\y`); without
    this the two would differ as strings and track-change events would not join
    the file index in project_structure.
    """
    try:
        out = _git(path, "rev-parse", "--show-toplevel").strip()
        return str(Path(out).resolve()) if out else None
    except Exception:
        return None


def _new_commits(root: str, since: str | None, max_commits: int) -> list[str]:
    """Return commit SHAs to process, OLDEST first.

    With a cursor: everything in ``since..HEAD`` (capped to the newest
    `max_commits` so a stale cursor cannot trigger a giant backfill). Without:
    the last `max_commits` commits.
    """
    if since:
        out = _git(root, "rev-list", "--reverse", f"{since}..HEAD")
        shas = [s for s in out.splitlines() if s.strip()]
        if len(shas) > max_commits:
            shas = shas[-max_commits:]
        return shas
    out = _git(root, "rev-list", "--reverse", "--max-count", str(max_commits), "HEAD")
    return [s for s in out.splitlines() if s.strip()]


def _commit_info(root: str, sha: str) -> dict:
    """Gather subject/body, changed files, and a truncated diff for one commit."""
    meta = _git(root, "show", "-s", "--format=%H%n%an%n%aI%n%s%n%b", sha)
    lines = meta.splitlines()
    full_sha = lines[0].strip() if lines else sha
    author = lines[1] if len(lines) > 1 else ""
    date = lines[2] if len(lines) > 2 else ""
    subject = lines[3] if len(lines) > 3 else ""
    body = "\n".join(lines[4:]).strip() if len(lines) > 4 else ""

    # --first-parent makes merge commits report their changed files + diff (a
    # plain `git show` on a merge shows nothing by default), so the LLM gets
    # real content instead of guessing intent from the subject alone.
    # --root so the very first commit (no parent) still lists its files;
    # --first-parent so merges report their first-parent changes.
    files_out = _git(root, "diff-tree", "--no-commit-id", "--name-only",
                     "-r", "--first-parent", "--root", sha)
    files = [f for f in files_out.splitlines() if f.strip()]

    diff = _git(root, "show", "--first-parent", "--format=", "--unified=2", sha)
    diff_truncated = len(diff) > _DIFF_CHAR_CAP

    return {
        "sha": full_sha,
        "short": full_sha[:8],
        "author": author,
        "date": date,
        "subject": subject[:_SUBJECT_CAP],
        "body": body[:_BODY_CAP],
        "files": files,
        "diff": diff[:_DIFF_CHAR_CAP],
        "diff_truncated": diff_truncated,
    }


# ----------------------------------------------------------------------
# per-repo cursor (operational state, kept OUT of the event store so it
# does not pollute memory; lives next to the workspace DB)
# ----------------------------------------------------------------------

def _cursor_file(engine: Engine) -> Path:
    root = Path(engine.workspace.root)
    root.mkdir(parents=True, exist_ok=True)
    return root / "track_cursors.json"


def _read_cursor(engine: Engine, repo_root: str) -> str | None:
    try:
        data = json.loads(_cursor_file(engine).read_text(encoding="utf-8"))
        return data.get(repo_root) if isinstance(data, dict) else None
    except Exception:
        return None


def _write_cursor(engine: Engine, repo_root: str, sha: str) -> None:
    f = _cursor_file(engine)
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    data[repo_root] = sha
    try:
        f.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as e:
        log.warning("could not persist track cursor: %s", e)


# ----------------------------------------------------------------------
# LLM prompts
# ----------------------------------------------------------------------

_INTENT_PROMPT = """\
You are summarising a single git commit for a developer's project memory.
Capture the INTENT - WHY this change was made and what it accomplishes - not a
line-by-line restatement of the diff. Be concrete and concise. Plain text only,
no markdown, in exactly this shape:

<one or two sentences on the intent / why>
- <file>: <short note on what changed there>

(one bullet per important file, at most 6 bullets)

Commit subject: {subject}
Commit body:
{body}
Files changed:
{files}
Diff (may be truncated):
{diff}
"""

_MODULE_PROMPT = """\
In ONE sentence, state what this source file is responsible for in the project.
Be concrete about the domain (e.g. "parses .gitignore and walks a repo's source
files"), not generic ("defines functions"). No markdown, no preamble, just the
sentence.

{header}
"""


def _summarize_commit(llm: LLMClient, info: dict) -> str:
    prompt = _INTENT_PROMPT.format(
        subject=info["subject"] or "(none)",
        body=info["body"] or "(none)",
        files="\n".join(info["files"][:30]) or "(none)",
        diff=info["diff"] or "(empty)",
    )
    try:
        return (llm.complete(prompt, max_tokens=240) or "").strip()
    except Exception as e:
        log.warning("LLM commit summary failed for %s: %s", info["short"], e)
        return ""


# ----------------------------------------------------------------------
# (1) git change tracking
# ----------------------------------------------------------------------

def track_changes(
    engine: Engine,
    path: Path | str = ".",
    since: str | None = None,
    backend: str = "auto",
    model: str | None = None,
    max_commits: int = 20,
    llm: LLMClient | None = None,
) -> dict:
    """Summarise the intent of new git commits and store them as linked memory.

    Idempotent via a per-repo cursor: each run processes only commits after the
    last one tracked, then advances the cursor.

    Returns a dict with project_name/path, n_commits, the tracked commits, the
    new cursor, and duration_ms - or {"error": ...} on failure.
    """
    root = _repo_root(path)
    if not root:
        return {"error": f"not a git repository: {Path(path).resolve()}"}

    project_name = Path(root).name
    cursor = since or _read_cursor(engine, root)
    try:
        shas = _new_commits(root, cursor, max_commits)
    except Exception as e:
        return {"error": f"could not read git history: {e}"}

    if not shas:
        return {
            "project_name": project_name, "project_path": root,
            "n_commits": 0, "tracked": [],
            "message": "nothing new since the last track",
        }

    if llm is None:
        from pmb.health.consolidate import resolve_llm_client
        try:
            llm = resolve_llm_client(backend=backend, model=model)
        except RuntimeError as e:
            return {"error": str(e)}

    t0 = time.time()
    items: list[dict] = []
    tracked: list[dict] = []
    for sha in shas:
        try:
            info = _commit_info(root, sha)
        except Exception as e:
            log.warning("skipping commit %s: %s", sha[:8], e)
            continue
        summary = _summarize_commit(llm, info)
        if not summary:
            continue
        files_line = ", ".join(info["files"][:12])
        content = (
            f"Change in {project_name} ({info['short']}): {info['subject']}\n"
            f"Why: {summary}\n"
            f"Files: {files_line}"
        )
        items.append({
            # NB: a `fact`, not an `activity` - record_batch keeps a fact's
            # custom metadata (source/project_path/commit/files), but replaces
            # an activity's metadata with {actor, activity_kind}, which would
            # drop the fields project_structure joins on.
            "type": "fact",
            "content": content,
            "importance": 0.6,
            "metadata": {
                "source": "git-change",
                "project_path": root,
                "project_name": project_name,
                "commit": info["sha"],
                "commit_short": info["short"],
                "author": info["author"],
                "commit_date": info["date"],
                "files": info["files"][:40],
                "diff_truncated": info["diff_truncated"],
            },
        })
        tracked.append({"commit": info["short"], "subject": info["subject"]})

    if items:
        engine.record_batch_async(items=items)
    # Advance the cursor even if some commits produced no summary, so we do not
    # re-fetch them next run; the newest SHA is the high-water mark.
    _write_cursor(engine, root, shas[-1])

    return {
        "project_name": project_name,
        "project_path": root,
        "n_commits": len(tracked),
        "tracked": tracked,
        "cursor": shas[-1][:8],
        "duration_ms": round((time.time() - t0) * 1000),
        "queued": bool(items),
    }


# ----------------------------------------------------------------------
# (2) per-module purpose summaries
# ----------------------------------------------------------------------

def _indexed_files(engine: Engine, project_path: str) -> list[dict]:
    """Project file-index rows for `project_path`, newest first."""
    import sqlite3
    out: list[dict] = []
    try:
        with sqlite3.connect(engine.workspace.db_path) as c:
            cur = c.execute(
                "SELECT content, metadata_json FROM events "
                "WHERE workspace_id=? AND archived_at IS NULL "
                "AND json_extract(metadata_json,'$.source')='project' "
                "AND json_extract(metadata_json,'$.project_path')=? "
                "AND json_extract(metadata_json,'$.index_artifact')=1 "
                "ORDER BY rowid DESC",
                (engine.workspace.id, project_path),
            )
            for content, mj in cur.fetchall():
                try:
                    md = json.loads(mj) if mj else {}
                except Exception:
                    md = {}
                fp = md.get("file_path")
                if fp:
                    out.append({
                        "file_path": fp,
                        "content": content or "",
                        "language": md.get("language", ""),
                    })
    except Exception as e:
        log.warning("could not read indexed files: %s", e)
    return out


def _files_with_summary(engine: Engine, project_path: str) -> set[str]:
    import sqlite3
    seen: set[str] = set()
    try:
        with sqlite3.connect(engine.workspace.db_path) as c:
            cur = c.execute(
                "SELECT json_extract(metadata_json,'$.file_path') FROM events "
                "WHERE workspace_id=? AND archived_at IS NULL "
                "AND json_extract(metadata_json,'$.source')='module-summary' "
                "AND json_extract(metadata_json,'$.project_path')=?",
                (engine.workspace.id, project_path),
            )
            for (fp,) in cur.fetchall():
                if fp:
                    seen.add(fp)
    except Exception as e:
        log.warning("could not read existing summaries: %s", e)
    return seen


def summarize_modules(
    engine: Engine,
    path: Path | str = ".",
    backend: str = "auto",
    model: str | None = None,
    limit: int = 200,
    force: bool = False,
    llm: LLMClient | None = None,
) -> dict:
    """Attach a one-line 'what this module does' summary to each indexed file.

    Requires `index_project` to have run first. Idempotent: a file that already
    has a summary is skipped unless `force`. `limit` caps LLM calls per run.
    """
    project_path = str(Path(path).resolve())
    files = _indexed_files(engine, project_path)
    if not files:
        return {"error": (
            f"no indexed files for {project_path}. "
            f"Run `pmb index project` first."
        )}

    # Dedupe by file_path (newest row wins - rows are newest-first).
    seen: set[str] = set()
    uniq: list[dict] = []
    for f in files:
        if f["file_path"] in seen:
            continue
        seen.add(f["file_path"])
        uniq.append(f)
    files = uniq

    if not force:
        done = _files_with_summary(engine, project_path)
        files = [f for f in files if f["file_path"] not in done]

    if not files:
        return {
            "project_path": project_path, "n_summarized": 0,
            "message": "all indexed files already summarised (use --force to redo)",
        }

    capped = files[:limit]

    if llm is None:
        from pmb.health.consolidate import resolve_llm_client
        try:
            llm = resolve_llm_client(backend=backend, model=model)
        except RuntimeError as e:
            return {"error": str(e)}

    t0 = time.time()
    items: list[dict] = []
    for f in capped:
        header = f["content"][:_FILE_HEAD_CAP]
        try:
            text = llm.complete(_MODULE_PROMPT.format(header=header), max_tokens=90)
        except Exception as e:
            log.warning("module summary failed for %s: %s", f["file_path"], e)
            continue
        summary = (text or "").strip()
        if not summary:
            continue
        summary = summary.splitlines()[0].strip()  # one line only
        items.append({
            "type": "fact",
            "content": f"Module {f['file_path']}: {summary}",
            "importance": 0.5,
            "metadata": {
                "source": "module-summary",
                "project_path": project_path,
                "file_path": f["file_path"],
                "language": f["language"],
            },
        })

    if items:
        engine.record_batch_async(items=items)

    return {
        "project_path": project_path,
        "n_indexed_files": len(files),
        "n_summarized": len(items),
        "n_over_cap": max(0, len(files) - len(capped)),
        "duration_ms": round((time.time() - t0) * 1000),
        "queued": bool(items),
    }
