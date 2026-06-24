"""Auto-capture exploration memos from the Claude Code transcript at turn end.

The Stop hook does NOT receive the conversation, but it DOES receive a
`transcript_path` (a JSONL of the session). This module parses the LAST turn
from that transcript and, if the agent did expensive exploration (read >= N
distinct source files) and produced an answer, records an exploration memo via
`pmb.memo.exploration.record_exploration` - so a future session can reuse the
conclusion instead of re-deriving it.

Stdlib-only (this runs in the hook path). The heavy engine import is lazy and
happens only when there is something to record.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmb.core.engine import Engine

log = logging.getLogger(__name__)

_READ_TOOLS = frozenset({"Read", "Grep", "Glob"})


def _iter_jsonl(path: str):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    except OSError:
        return


def _text_of(content) -> str:
    """Join the text blocks of a message `content` (str or list-of-blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def extract_exploration_from_transcript(
    transcript_path: str, min_files: int = 3,
) -> dict | None:
    """Parse the last turn of a Claude Code transcript.

    Returns {"intent", "conclusion", "files"} when the last turn read >=
    `min_files` distinct files and ended with an assistant answer; else None.
    """
    entries = list(_iter_jsonl(transcript_path))
    if not entries:
        return None

    # The intent is the last REAL user message (text), not a tool_result turn.
    last_user_idx = None
    intent = ""
    for i, e in enumerate(entries):
        if e.get("type") == "user":
            txt = _text_of((e.get("message") or {}).get("content"))
            if txt.strip():
                last_user_idx = i
                intent = txt.strip()
    if last_user_idx is None:
        return None

    files: list[str] = []
    concl_parts: list[str] = []
    for e in entries[last_user_idx + 1:]:
        if e.get("type") != "assistant":
            continue
        for b in (e.get("message") or {}).get("content") or []:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text" and b.get("text", "").strip():
                concl_parts.append(b["text"].strip())
            elif t == "tool_use" and b.get("name") in _READ_TOOLS:
                inp = b.get("input") or {}
                f = inp.get("file_path") or inp.get("path")
                if f:
                    files.append(f)

    files = list(dict.fromkeys(files))
    conclusion = "\n\n".join(concl_parts).strip()
    if len(files) < min_files or not conclusion:
        return None

    return {"intent": intent[:500], "conclusion": conclusion, "files": files}


def run_autocapture(
    engine: Engine,
    transcript_path: str,
    min_files: int = 3,
    apply: bool = True,
) -> dict:
    """Capture an exploration memo from `transcript_path` if the last turn
    qualifies. `apply=False` is a dry run (no write)."""
    if not transcript_path or not Path(transcript_path).exists():
        return {"captured": False, "reason": "no transcript"}
    info = extract_exploration_from_transcript(transcript_path, min_files=min_files)
    if not info:
        return {"captured": False, "reason": "below threshold or no Q/A"}
    if not apply:
        return {"captured": False, "dry_run": True,
                "intent": info["intent"], "files": info["files"]}
    try:
        from pmb.memo.exploration import record_exploration
        res = record_exploration(engine, info["intent"], info["conclusion"], info["files"])
    except Exception as e:
        log.warning("autocapture record failed: %s", e)
        return {"captured": False, "reason": f"record failed: {e}"}
    return {"captured": True, "n_sources": res.get("n_sources"), "intent": info["intent"]}
