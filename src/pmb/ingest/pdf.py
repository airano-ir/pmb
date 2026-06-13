"""PDF ingestion - extract text + metadata, chunk by section, persist as
PMB events that the agent can recall like any other fact.

Design notes:

  * pypdf is the default backend - pure-Python, MIT-licensed, ships
    inside the wheel. No system PDF dependency.
  * One PDF → one PARENT fact_tree per file ("PDF: <title>") plus N
    child events, one per chunk. Each chunk has:
        metadata.source         = "pdf"
        metadata.source_path    = "<absolute path>"
        metadata.source_hash    = sha1 of file (for idempotent re-ingest)
        metadata.pdf_page       = 1-based page number
        metadata.pdf_chunk      = chunk index within the doc
        metadata.pdf_section    = heuristic section title if detected
  * Re-ingesting the same file is a no-op - the source_hash + page check
    drops every duplicate chunk via the engine's L1 exact-text dedup
    AND a fast pre-check that bails out if any event already has the
    same hash.
  * Chunk target: ~1500 characters. Big enough that semantic search has
    context, small enough that surface area is decent.

Public API:
    ingest_pdf(engine, path)                  -> dict (n_chunks, n_pages, ...)
    ingest_pdfs(engine, paths_or_dir, recurse) -> dict (n_files, n_chunks, ...)
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from pmb import lang as _lang

if TYPE_CHECKING:
    from pmb.core.engine import Engine

log = logging.getLogger(__name__)

CHUNK_TARGET_CHARS = 1500
CHUNK_OVERLAP = 150     # carry tail into next chunk to keep context
MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB guard - anything larger is suspicious


# ----------------------------------------------------------------------
# Backend selection
# ----------------------------------------------------------------------

def _import_backend():
    """Try pypdf first (pure-Python), then pdfplumber as a richer fallback.

    Returns: (backend_name, extract_fn) where extract_fn(path) → list[(page_no, text)].
    """
    try:
        from pypdf import PdfReader  # type: ignore
        def _pypdf_extract(path: Path):
            reader = PdfReader(str(path))
            pages = []
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                pages.append((i + 1, text))
            meta = {}
            try:
                m = reader.metadata or {}
                meta = {
                    "title":   str(m.get("/Title") or "").strip() or None,
                    "author":  str(m.get("/Author") or "").strip() or None,
                    "subject": str(m.get("/Subject") or "").strip() or None,
                    "n_pages": len(reader.pages),
                }
            except Exception:
                pass
            return pages, meta
        return "pypdf", _pypdf_extract
    except ImportError:
        pass

    try:
        import pdfplumber  # type: ignore
        def _plumber_extract(path: Path):
            pages = []
            meta = {}
            with pdfplumber.open(str(path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text() or ""
                    except Exception:
                        text = ""
                    pages.append((i + 1, text))
                meta = {"n_pages": len(pdf.pages)}
            return pages, meta
        return "pdfplumber", _plumber_extract
    except ImportError:
        pass

    raise RuntimeError(
        "No PDF backend available. Install one of:\n"
        "  pip install pypdf            # default, pure-Python, MIT\n"
        "  pip install pdfplumber       # better for table-heavy PDFs"
    )


# ----------------------------------------------------------------------
# Chunking - keep paragraphs together, respect heading boundaries
# ----------------------------------------------------------------------

# Heading words: EN inline; localized equivalents (RU/UK/DE "Chapter"/"Section")
# come from the packs (pdf_heading_words). E2: the ALL-CAPS heading test is now
# str.isupper() in Python - Unicode-correct for every script - so this module no
# longer needs the packs' `sentence_uppercase` char-class ranges.
_HEAD_WORDS = "|".join(["Chapter", "Section", "Part", "Appendix"]
                       + [str(w) for w in _lang.merged_list("pdf_heading_words")])
_HEAD_WORD_RE = re.compile(
    r"^\s*(?:" + _HEAD_WORDS + r")\s+[\dIVXLCM.]+\b", re.IGNORECASE)
_NUMBERED_HEAD_RE = re.compile(r"^\s*\d{1,2}(?:\.\d{1,2}){0,3}\s+(\S[^\n]{2,120})$")


def _looks_allcaps_heading(line: str) -> bool:
    """A short line whose letters are ALL uppercase, in ANY script (str.isupper()
    handles Cyrillic / Greek / accented Latin). Digits + punctuation ignored."""
    s = line.strip()
    if not (5 <= len(s) <= 80):
        return False
    letters = [c for c in s if c.isalpha()]
    return len(letters) >= 3 and "".join(letters).isupper()


def _split_into_chunks(text: str, target: int = CHUNK_TARGET_CHARS) -> list[str]:
    """Greedy paragraph-respecting chunker.

    Splits on double-newline (paragraph boundary). When the running buffer
    exceeds `target`, flushes a chunk and starts the next with a small
    overlap from the previous tail so cross-chunk context survives.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > target:
            chunks.append(cur.strip())
            # Carry the last `CHUNK_OVERLAP` chars to preserve context.
            tail = cur[-CHUNK_OVERLAP:] if CHUNK_OVERLAP and len(cur) > CHUNK_OVERLAP else ""
            cur = (tail + "\n\n" + p).strip() if tail else p
        else:
            cur = (cur + "\n\n" + p) if cur else p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def _detect_section(chunk: str) -> str | None:
    """Best-effort heuristic: first heading-looking line in the chunk -
    a 'Chapter N …' / 'Section N …' marker, a numbered 'N.M Title', or an
    ALL-CAPS line (any script via str.isupper(), E2)."""
    for raw in chunk[:400].splitlines():
        ls = raw.strip()
        if not ls:
            continue
        if _HEAD_WORD_RE.match(ls):
            return ls[:120]
        m = _NUMBERED_HEAD_RE.match(ls)
        if m and m.group(1)[:1].isupper():
            return ls[:120]
        if _looks_allcaps_heading(ls):
            return ls[:120]
    return None


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def _file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _is_already_indexed(engine: Engine, source_hash: str) -> bool:
    """Check if a PDF with this exact content hash was already ingested."""
    import sqlite3
    try:
        with sqlite3.connect(engine.workspace.db_path) as c:
            row = c.execute(
                "SELECT COUNT(*) FROM events WHERE workspace_id=? "
                "AND archived_at IS NULL "
                "AND metadata_json LIKE ? LIMIT 1",
                (engine.workspace.id, f'%"source_hash": "{source_hash}"%'),
            ).fetchone()
            return (row and row[0] > 0)
    except Exception:
        return False


def ingest_pdf(
    engine: Engine,
    path: Path | str,
    importance: float = 0.6,
    force: bool = False,
) -> dict:
    """Extract text from one PDF, chunk it, persist as PMB events.

    Args:
        engine: a live PMB Engine
        path: PDF file path
        importance: per-chunk importance (default 0.6 - high-signal but not pinned)
        force: re-ingest even if the file's content hash is already in PMB

    Returns:
        {file, n_pages, n_chunks, n_skipped, duration_ms, source_hash,
         empty: bool}
    """
    p = Path(path).resolve()
    if not p.exists():
        return {"error": f"file not found: {p}"}
    if p.suffix.lower() != ".pdf":
        return {"error": f"not a PDF: {p}"}
    size = p.stat().st_size
    if size > MAX_PDF_BYTES:
        return {"error": f"PDF too large ({size / 1024 / 1024:.1f}MB > {MAX_PDF_BYTES // 1024 // 1024}MB)"}

    backend_name, extract = _import_backend()
    t0 = time.time()
    src_hash = _file_hash(p)

    if not force and _is_already_indexed(engine, src_hash):
        return {
            "file": str(p), "source_hash": src_hash,
            "n_chunks": 0, "n_skipped": 1, "n_pages": 0,
            "duration_ms": round((time.time() - t0) * 1000),
            "backend": backend_name,
            "skipped": True,
            "reason": "already indexed (use force=True to re-ingest)",
        }

    pages, meta = extract(p)
    if not pages:
        return {"file": str(p), "empty": True, "n_chunks": 0}

    title = (meta.get("title") or p.stem).strip()
    n_pages = meta.get("n_pages") or len(pages)

    # ── Build chunks ──────────────────────────────────────────────────
    items: list[dict] = []
    chunk_idx = 0
    for (page_no, text) in pages:
        if not (text or "").strip():
            continue
        for chunk in _split_into_chunks(text):
            section = _detect_section(chunk)
            md = {
                "source":          "pdf",
                "source_path":     str(p),
                "source_filename": p.name,
                "source_hash":     src_hash,
                "pdf_title":       title,
                "pdf_page":        page_no,
                "pdf_chunk":       chunk_idx,
                "pdf_section":     section,
                "pdf_backend":     backend_name,
            }
            items.append({
                "type": "fact",
                "content": chunk,
                "importance": importance,
                "metadata": md,
            })
            chunk_idx += 1

    # Parent "document index" event so the agent can pull the whole PDF
    # via a single fact_tree if it wants. Linked through a shared key.
    head_item = {
        "type": "fact",
        "content": f"PDF: {title} ({n_pages} pages, {chunk_idx} chunks). Path: {p}",
        "importance": 0.7,  # higher: it's the "entrypoint" to the doc
        "metadata": {
            "source":          "pdf",
            "source_path":     str(p),
            "source_filename": p.name,
            "source_hash":     src_hash,
            "pdf_title":       title,
            "pdf_n_pages":     n_pages,
            "pdf_n_chunks":    chunk_idx,
            "pdf_entrypoint":  True,
        },
    }

    all_items = [head_item] + items
    engine.record_batch_async(items=all_items)
    duration_ms = round((time.time() - t0) * 1000)

    return {
        "file": str(p),
        "source_hash": src_hash,
        "backend": backend_name,
        "n_pages": n_pages,
        "n_chunks": chunk_idx,
        "n_skipped": 0,
        "duration_ms": duration_ms,
        "queued": True,
    }


def ingest_pdfs(
    engine: Engine,
    paths_or_dir: Iterable[Path | str] | Path | str,
    recurse: bool = False,
    importance: float = 0.6,
    force: bool = False,
) -> dict:
    """Bulk ingest - accept a list of paths, or a directory (with optional
    recursion). Returns aggregated stats. Errors are reported per-file."""
    files: list[Path] = []
    seen = set()

    def _add(p: Path):
        rp = p.resolve()
        if rp in seen:
            return
        seen.add(rp)
        if rp.is_file() and rp.suffix.lower() == ".pdf":
            files.append(rp)
        elif rp.is_dir():
            pattern = "**/*.pdf" if recurse else "*.pdf"
            for f in rp.glob(pattern):
                if f.is_file():
                    files.append(f.resolve())

    if isinstance(paths_or_dir, (str, Path)):
        _add(Path(paths_or_dir))
    else:
        for p in paths_or_dir:
            _add(Path(p))

    results = []
    n_chunks = 0
    n_skipped = 0
    for f in files:
        r = ingest_pdf(engine, f, importance=importance, force=force)
        results.append(r)
        n_chunks += r.get("n_chunks", 0)
        n_skipped += r.get("n_skipped", 0)

    return {
        "n_files":   len(files),
        "n_chunks":  n_chunks,
        "n_skipped": n_skipped,
        "files":     results,
    }
