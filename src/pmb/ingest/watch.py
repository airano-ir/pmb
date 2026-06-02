"""
Watch-folder capture: turn an existing notes file/folder into PMB memory
automatically. Point PMB at `~/journal.md` (or a directory of `.md`); each
time new content appears it gets ingested as facts - zero extra effort.

The scan is a pure function over (path, already-seen hashes) so it's testable
without the CLI loop or the Engine. The CLI `pmb watch` command:
  - loads seen-hashes state from the workspace,
  - calls `scan_new_chunks` on an interval (or once),
  - records each new chunk via `record_fact(..., source="watch")`,
  - persists the updated state.

Dedup is content-hash based, so editing earlier text doesn't re-ingest it and
appending new text ingests only the new paragraphs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_MIN_LEN = 20          # ignore trivially short paragraphs
_MAX_LEN = 4000        # cap to match the batch guard


def _chunk_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines into paragraph-sized chunks."""
    out: list[str] = []
    cur: list[str] = []
    for line in text.splitlines():
        if line.strip() == "":
            if cur:
                out.append("\n".join(cur).strip())
                cur = []
        else:
            cur.append(line)
    if cur:
        out.append("\n".join(cur).strip())
    return [c for c in out if c.strip()]


def _files_for(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.md"))
    return []


def scan_new_chunks(path: Path, seen: set[str]) -> tuple[list[dict], set[str]]:
    """Return (new_items, updated_seen).

    new_items: [{"content", "file", "hash"}] for paragraphs not in `seen`.
    updated_seen: `seen` plus the hashes of the new items.
    Pure - no recording, no I/O beyond reading the files.
    """
    path = Path(path)
    updated = set(seen)
    new_items: list[dict] = []
    for f in _files_for(path):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for chunk in _split_paragraphs(text):
            if len(chunk) < _MIN_LEN:
                continue
            if len(chunk) > _MAX_LEN:
                chunk = chunk[:_MAX_LEN]
            h = _chunk_hash(chunk)
            if h in updated:
                continue
            updated.add(h)
            new_items.append({"content": chunk, "file": f.name, "hash": h})
    return new_items, updated


# -- state persistence (CLI uses these; kept here for one home) -------------

def load_state(state_path: Path) -> set[str]:
    try:
        data = json.loads(Path(state_path).read_text(encoding="utf-8"))
        return set(data.get("seen", []))
    except Exception:
        return set()


def save_state(state_path: Path, seen: set[str]) -> None:
    p = Path(state_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"seen": sorted(seen)}), encoding="utf-8")
