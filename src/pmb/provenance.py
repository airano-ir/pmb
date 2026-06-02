"""
Source attribution - turn an event's metadata into a short human label of
WHERE the memory came from.

Used by `pmb recall` (show provenance per hit), `pmb audit` (group by source),
and `pmb why`. Trust feature: a memory system you can't trace is scary; this
lets the user see "this fact came from my ChatGPT import" or "I typed this".

Pure function, no I/O - trivially testable.
"""

from __future__ import annotations

from typing import Optional


def describe_source(metadata: Optional[dict]) -> str:
    """Return a short label like 'chatgpt · Project planning', 'note (cli)',
    'markdown · notes.md', 'agent', or '-' when origin is unknown."""
    meta = metadata or {}
    src = (meta.get("source") or "").strip().lower()

    # Imported conversations carry the conversation title.
    if src in ("chatgpt", "claude"):
        conv = meta.get("conversation")
        role = meta.get("role")
        label = src
        if conv and conv != "untitled":
            label += f" · {conv}"
        if role:
            label += f" ({role})"
        return label

    if src == "mem0":
        return "import:mem0"

    if src == "markdown":
        f = meta.get("file")
        return f"markdown · {f}" if f else "markdown"

    if src in ("cli-note", "note"):
        return "note (cli)"

    if src == "lesson":
        # distinguish a recorded failure ("don't do this") from a lesson
        if (meta.get("kind") or "").lower() == "failure":
            return "failure"
        return "lesson"

    if src == "watch":
        f = meta.get("file")
        return f"watch · {f}" if f else "watch"

    if src == "git":
        return "git commit"

    if src:
        return src

    # No explicit source - infer a coarse origin from other fields.
    if meta.get("agent_id") or meta.get("actor") == "agent":
        return "agent"
    if meta.get("conversation"):
        return f"conversation · {meta['conversation']}"
    return "-"


def source_key(metadata: Optional[dict]) -> str:
    """Coarse bucket key for grouping in `pmb audit` (collapses the detail
    that describe_source adds)."""
    meta = metadata or {}
    src = (meta.get("source") or "").strip().lower()
    if src:
        return src
    if meta.get("agent_id") or meta.get("actor") == "agent":
        return "agent"
    return "unknown"
