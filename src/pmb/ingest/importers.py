"""
Importers: bring your existing memory into PMB from other tools.

The #1 adoption barrier for a memory system is the cold start - a brand new
workspace knows nothing about you, so it feels useless on day one. But you
already have years of context sitting in ChatGPT exports, Claude history,
a mem0 instance, or a folder of Markdown notes. This module parses those
into PMB `record_batch_bulk` items so the first recall already knows you.

Supported sources
-----------------
  chatgpt   - OpenAI data export `conversations.json`
  claude    - Anthropic data export `conversations.json` (chat_messages schema)
  mem0      - a mem0 memories export (list of {memory|text, ...})
  markdown  - a single .md file or a directory tree of them (Obsidian, notes)

Each parser is defensive: a malformed record is skipped and counted, never
fatal. Output is a list of `{"type": "fact", "content", "importance",
"metadata"}` dicts ready for `Engine.record_batch_bulk`.

Design notes
------------
  • Original timestamps are preserved in `metadata.original_ts` (the event's
    own timestamp is the import time; we don't rewrite history silently).
  • Role/source is tagged in metadata so you can later filter or dedupe.
  • Long messages are capped (4000 chars) to match the batch guard.
  • `roles` filter (default: user-only) keeps signal high - your own words
    are usually what's worth remembering, not the assistant's boilerplate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

_MAX_CHARS = 4000
_MIN_CHARS = 12  # drop trivial "ok", "thanks", etc.


@dataclass
class ImportResult:
    source: str
    items: list[dict] = field(default_factory=list)
    skipped: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def n_parsed(self) -> int:
        return len(self.items)


def _mk_item(content: str, importance: float, meta: dict) -> Optional[dict]:
    content = (content or "").strip()
    if len(content) < _MIN_CHARS:
        return None
    if len(content) > _MAX_CHARS:
        content = content[:_MAX_CHARS]
    return {"type": "fact", "content": content,
            "importance": importance, "metadata": meta}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# ChatGPT export (conversations.json)
# ----------------------------------------------------------------------

def parse_chatgpt(path: Path, roles: set[str]) -> ImportResult:
    res = ImportResult(source="chatgpt")
    p = Path(path)
    target = p / "conversations.json" if p.is_dir() else p
    try:
        data = _read_json(target)
    except Exception as e:  # noqa: BLE001
        res.notes.append(f"could not read {target}: {e}")
        return res

    convs = data if isinstance(data, list) else data.get("conversations", [])
    for conv in convs:
        if not isinstance(conv, dict):
            res.skipped += 1
            continue
        title = conv.get("title") or "untitled"
        mapping = conv.get("mapping") or {}
        for node in mapping.values():
            msg = (node or {}).get("message") if isinstance(node, dict) else None
            if not isinstance(msg, dict):
                continue
            role = ((msg.get("author") or {}).get("role") or "").lower()
            if role not in roles:
                if role:
                    res.skipped += 1
                continue
            content = msg.get("content") or {}
            parts = content.get("parts") if isinstance(content, dict) else None
            text = ""
            if isinstance(parts, list):
                text = "\n".join(str(x) for x in parts if isinstance(x, str))
            item = _mk_item(
                text,
                importance=0.6 if role == "user" else 0.4,
                meta={"source": "chatgpt", "role": role, "conversation": title,
                      "original_ts": msg.get("create_time")},
            )
            if item:
                res.items.append(item)
            else:
                res.skipped += 1
    return res


# ----------------------------------------------------------------------
# Claude / Anthropic export (conversations.json - chat_messages schema)
# ----------------------------------------------------------------------

def parse_claude(path: Path, roles: set[str]) -> ImportResult:
    res = ImportResult(source="claude")
    p = Path(path)
    target = p / "conversations.json" if p.is_dir() else p
    try:
        data = _read_json(target)
    except Exception as e:  # noqa: BLE001
        res.notes.append(f"could not read {target}: {e}")
        return res

    convs = data if isinstance(data, list) else [data]
    # map Anthropic sender → role vocabulary used by `roles` filter
    sender_to_role = {"human": "user", "assistant": "assistant"}
    for conv in convs:
        if not isinstance(conv, dict):
            res.skipped += 1
            continue
        title = conv.get("name") or conv.get("title") or "untitled"
        msgs = conv.get("chat_messages") or conv.get("messages") or []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            sender = (m.get("sender") or m.get("role") or "").lower()
            role = sender_to_role.get(sender, sender)
            if role not in roles:
                if role:
                    res.skipped += 1
                continue
            text = m.get("text")
            if not text and isinstance(m.get("content"), list):
                # newer schema: content is a list of blocks
                text = "\n".join(
                    b.get("text", "") for b in m["content"]
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            item = _mk_item(
                text or "",
                importance=0.6 if role == "user" else 0.4,
                meta={"source": "claude", "role": role, "conversation": title,
                      "original_ts": m.get("created_at")},
            )
            if item:
                res.items.append(item)
            else:
                res.skipped += 1
    return res


# ----------------------------------------------------------------------
# mem0 export (list of distilled memories)
# ----------------------------------------------------------------------

def parse_mem0(path: Path, roles: set[str]) -> ImportResult:
    res = ImportResult(source="mem0")
    try:
        data = _read_json(Path(path))
    except Exception as e:  # noqa: BLE001
        res.notes.append(f"could not read {path}: {e}")
        return res

    rows = data if isinstance(data, list) else (
        data.get("memories") or data.get("results") or data.get("data") or []
    )
    for r in rows:
        if isinstance(r, str):
            text = r
            meta_extra = {}
        elif isinstance(r, dict):
            text = r.get("memory") or r.get("text") or r.get("content") or ""
            meta_extra = {"original_ts": r.get("created_at") or r.get("updated_at")}
        else:
            res.skipped += 1
            continue
        item = _mk_item(
            text,
            importance=0.7,  # mem0 entries are already distilled facts
            meta={"source": "mem0", **meta_extra},
        )
        if item:
            res.items.append(item)
        else:
            res.skipped += 1
    return res


# ----------------------------------------------------------------------
# Markdown - a file or a directory tree (Obsidian, plain notes)
# ----------------------------------------------------------------------

def parse_markdown(path: Path, roles: set[str]) -> ImportResult:
    res = ImportResult(source="markdown")
    p = Path(path)
    files = [p] if p.is_file() else sorted(p.rglob("*.md"))
    if not files:
        res.notes.append(f"no .md files found under {p}")
        return res

    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            res.skipped += 1
            continue
        for chunk in _split_markdown(text):
            item = _mk_item(
                chunk,
                importance=0.55,
                meta={"source": "markdown", "file": f.name},
            )
            if item:
                res.items.append(item)
            else:
                res.skipped += 1
    return res


def _split_markdown(text: str) -> list[str]:
    """Split a markdown doc into section-sized chunks on `#`/`##` headers.

    Small docs come back as a single chunk; large docs split per heading so
    each becomes an independently-recallable fact instead of one diffuse blob.
    """
    lines = text.splitlines()
    chunks: list[str] = []
    cur: list[str] = []
    for ln in lines:
        if ln.lstrip().startswith("#") and cur:
            chunks.append("\n".join(cur).strip())
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        chunks.append("\n".join(cur).strip())
    # if the whole doc was small / headerless, keep it as one chunk
    return [c for c in chunks if c.strip()] or [text.strip()]


PARSERS: dict[str, Callable[[Path, set[str]], ImportResult]] = {
    "chatgpt": parse_chatgpt,
    "claude": parse_claude,
    "mem0": parse_mem0,
    "markdown": parse_markdown,
}


def parse_source(source: str, path: Path, roles: Optional[set[str]] = None) -> ImportResult:
    """Dispatch to the right parser. `roles` filters chat sources
    (default: {'user'} - your own words carry the most signal)."""
    source = source.lower().strip()
    if source not in PARSERS:
        raise ValueError(
            f"unknown source {source!r}. Supported: {', '.join(sorted(PARSERS))}"
        )
    return PARSERS[source](Path(path), roles or {"user"})
