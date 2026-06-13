"""Project structure ingestion - scan a code repo, extract per-file
symbols + imports, save as PMB events that the agent can later recall
("where is the auth flow", "what files use LanceDB").

Design:

  * One ROOT fact_tree per project ("Project: <name>") with subfacts
    summarising stack + layout.
  * One per-file event:
      type = fact
      metadata = {
          source: "project",
          project_path: "<abs path>",
          file_path: "<rel path>",
          language: "python" | "typescript" | "rust" | ...,
          loc: int,
          sha1: str,
          symbols: ["def foo", "class Bar", ...],
          imports: ["fastmcp", "from .core import Engine"],
      }
  * Imports go straight into the entity-graph: each import target
    becomes a `tech` entity linked to the file. Cross-file dependencies
    surface as co-occurrence edges over time.
  * Idempotent: per-file sha1 is checked before write. Re-running on a
    clean repo writes nothing.

This module never spawns a heavy parser per file - Python uses the
stdlib `ast` module, other languages use cheap regex. For deep AST
parsing of TS/Rust we'd need tree-sitter; deferred (60% of the value
for 5% of the work).
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmb.core.engine import Engine

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# File walking - respect .gitignore + common ignore directories
# ----------------------------------------------------------------------

# Directories we never descend into. Covers most ecosystems.
_DEFAULT_IGNORE_DIRS = {
    ".git", ".hg", ".svn",
    "node_modules", "bower_components",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env", ".tox",
    "target", "dist", "build", "out", ".next", ".nuxt",
    ".idea", ".vscode", ".vs",
    "vendor", "third_party",
    ".cache", "tmp", "temp",
    "coverage", ".nyc_output",
}

# Extensions we don't try to read as source.
_SKIP_EXT = {
    ".pyc", ".pyo", ".so", ".o", ".a", ".dll", ".dylib", ".exe",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".svg",
    ".mp4", ".mp3", ".wav", ".pdf",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".lock", ".min.js", ".map",
}

# Best-effort language detection by extension.
_LANG_BY_EXT = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java", ".kt": "kotlin", ".scala": "scala",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".sql": "sql",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".html": "html", ".css": "css", ".scss": "css",
    ".md": "markdown", ".rst": "rst",
    ".toml": "toml", ".yaml": "yaml", ".yml": "yaml",
    ".json": "json", ".xml": "xml",
}


def _load_gitignore(root: Path) -> list[re.Pattern]:
    """Crude .gitignore parser. Returns a list of compiled regex patterns
    matching POSIX paths relative to `root`. Misses negation rules and
    advanced glob syntax but is good enough for the 95% case."""
    patterns: list[re.Pattern] = []
    gi = root / ".gitignore"
    if not gi.exists():
        return patterns
    try:
        text = gi.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return patterns
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("!"):
            continue
        glob_to_re = (
            s.replace(".", r"\.")
             .replace("**/", "(?:.*/)?")
             .replace("*", "[^/]*")
             .replace("?", ".")
        )
        if not s.endswith("/"):
            glob_to_re += "(/|$)"
        try:
            patterns.append(re.compile(glob_to_re))
        except re.error:
            continue
    return patterns


def _is_ignored(rel: str, gitignore: list[re.Pattern]) -> bool:
    for pat in gitignore:
        if pat.search(rel):
            return True
    return False


def _walk_files(root: Path) -> Iterable[Path]:
    """Yield every source-looking file under `root`. Respects .gitignore
    + the default ignore-dir set."""
    gitignore = _load_gitignore(root)
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(root).parts
        if any(part in _DEFAULT_IGNORE_DIRS for part in rel_parts):
            continue
        if p.suffix.lower() in _SKIP_EXT:
            continue
        if _is_ignored("/".join(rel_parts), gitignore):
            continue
        # Skip files >5 MB - they're likely data, not source.
        try:
            if p.stat().st_size > 5 * 1024 * 1024:
                continue
        except OSError:
            continue
        yield p


# ----------------------------------------------------------------------
# Per-language extraction
# ----------------------------------------------------------------------

def _extract_python(text: str) -> tuple[list[str], list[str]]:
    """Returns (symbols, imports). Uses stdlib ast - no syntax errors leak."""
    try:
        from pmb.reasoning.code_ast import extract_python_symbols
        syms = extract_python_symbols(text)
        symbol_strs = []
        imports: list[str] = []
        for s in syms:
            if s.kind == "import":
                imports.append(s.name)
            else:
                prefix = ""
                if s.kind == "class":
                    prefix = "class "
                elif s.kind == "function":
                    prefix = "def "
                elif s.kind == "method":
                    prefix = f"{s.parent}.{s.name} (method)"
                    symbol_strs.append(prefix)
                    continue
                symbol_strs.append(f"{prefix}{s.name}")
        return symbol_strs, imports
    except Exception:
        return [], []


_TS_IMPORT_RE = re.compile(
    r"""^\s*(?:import\s+(?:.+?\s+from\s+)?|export\s+(?:\*\s+from\s+|\{[^}]+\}\s+from\s+))
        ['\"]([^'\"]+)['\"]""",
    re.MULTILINE | re.VERBOSE,
)
_TS_SYMBOL_RE = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|interface|type|enum|const|let|var)\s+([A-Za-z_$][\w$]*)",
    re.MULTILINE,
)


def _extract_ts(text: str) -> tuple[list[str], list[str]]:
    imports = list(dict.fromkeys(_TS_IMPORT_RE.findall(text)))
    syms = list(dict.fromkeys(_TS_SYMBOL_RE.findall(text)))
    return syms[:40], imports[:30]


_RUST_USE_RE = re.compile(r"^\s*(?:pub\s+)?use\s+([\w:]+(?:::[\w*\{\}]+)?)\s*;", re.MULTILINE)
_RUST_SYM_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn|struct|enum|trait|impl|type)\s+([A-Za-z_][\w]*)", re.MULTILINE)


def _extract_rust(text: str) -> tuple[list[str], list[str]]:
    imports = list(dict.fromkeys(_RUST_USE_RE.findall(text)))
    syms = list(dict.fromkeys(_RUST_SYM_RE.findall(text)))
    return syms[:40], imports[:30]


_GO_IMPORT_RE = re.compile(r'^\s*import\s+\(\s*\n([^)]*)\)|^\s*import\s+"([^"]+)"', re.MULTILINE)
_GO_SYM_RE = re.compile(r"^\s*func\s+(?:\([^)]+\)\s+)?([A-Z][\w]*)|^\s*type\s+([A-Z][\w]*)", re.MULTILINE)


def _extract_go(text: str) -> tuple[list[str], list[str]]:
    imports: list[str] = []
    for blk, single in _GO_IMPORT_RE.findall(text):
        if single:
            imports.append(single)
        if blk:
            for line in blk.splitlines():
                m = re.search(r'"([^"]+)"', line)
                if m:
                    imports.append(m.group(1))
    syms: list[str] = []
    for fn, ty in _GO_SYM_RE.findall(text):
        if fn: syms.append(fn)
        if ty: syms.append(ty)
    return syms[:40], list(dict.fromkeys(imports))[:30]


def _extract_for_lang(lang: str, text: str) -> tuple[list[str], list[str]]:
    if lang == "python":           return _extract_python(text)
    if lang in ("typescript", "javascript"): return _extract_ts(text)
    if lang == "rust":             return _extract_rust(text)
    if lang == "go":               return _extract_go(text)
    return [], []


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def _hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="replace")).hexdigest()


def _is_file_indexed(engine: Engine, project_path: str, file_path: str, sha1: str) -> bool:
    """Idempotency check - has this file with this exact content already been
    indexed for this project?"""
    import sqlite3
    try:
        with sqlite3.connect(engine.workspace.db_path) as c:
            row = c.execute(
                "SELECT COUNT(*) FROM events WHERE workspace_id=? "
                "AND archived_at IS NULL "
                "AND metadata_json LIKE ? "
                "AND metadata_json LIKE ? LIMIT 1",
                (
                    engine.workspace.id,
                    f'%"project_path": "{project_path}"%',
                    f'%"sha1": "{sha1}"%',
                ),
            ).fetchone()
            return (row and row[0] > 0)
    except Exception:
        return False


def _is_project_root_indexed(engine: Engine, project_path: str) -> bool:
    """Return True when the project already has an active root index row."""
    import sqlite3
    try:
        with sqlite3.connect(engine.workspace.db_path) as c:
            row = c.execute(
                "SELECT 1 FROM events WHERE workspace_id=? "
                "AND archived_at IS NULL "
                "AND json_extract(metadata_json, '$.source') = 'project' "
                "AND json_extract(metadata_json, '$.project_root') = 1 "
                "AND json_extract(metadata_json, '$.project_path') = ? "
                "LIMIT 1",
                (engine.workspace.id, project_path),
            ).fetchone()
            return row is not None
    except Exception:
        return False


def index_project(
    engine: Engine,
    path: Path | str,
    importance: float = 0.55,
    force: bool = False,
    max_files: int = 5000,
) -> dict:
    """Walk a project directory, persist per-file structure as PMB events.

    Args:
        engine: live PMB Engine
        path: project root
        importance: per-file event importance (default 0.55 - high signal)
        force: re-index files even if their sha1 already exists
        max_files: safety cap so a giant repo doesn't lock the workspace

    Returns:
        {project, n_files_seen, n_indexed, n_skipped, by_language: {...},
         duration_ms}
    """
    root = Path(path).resolve()
    if not root.exists() or not root.is_dir():
        return {"error": f"not a directory: {root}"}

    t0 = time.time()
    project_name = root.name
    project_path = str(root)

    files_seen = 0
    files_indexed = 0
    files_skipped = 0
    low_signal_skipped = 0
    by_lang: dict[str, int] = {}
    sample_files: list[str] = []

    items: list[dict] = []
    # Root entrypoint event so the agent can reach the whole project
    # via a single recall. Keep it idempotent: the old timestamped root row
    # created a fresh duplicate every time the same project was indexed.
    root_item: dict | None = None
    if not _is_project_root_indexed(engine, project_path):
        root_item = {
            "type": "fact",
            "content": (
                f"Project: {project_name}\n"
                f"Path: {project_path}"
            ),
            "importance": 0.75,
            "metadata": {
                "source":        "project",
                "project_path":  project_path,
                "project_name":  project_name,
                "project_root":  True,
            },
        }
        items.append(root_item)

    for p in _walk_files(root):
        files_seen += 1
        if files_seen > max_files:
            break

        rel = p.relative_to(root)
        lang = _LANG_BY_EXT.get(p.suffix.lower(), "other")

        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        sha1 = _hash_text(text)

        if not force and _is_file_indexed(engine, project_path, str(rel), sha1):
            files_skipped += 1
            continue

        symbols, imports = _extract_for_lang(lang, text)
        loc = sum(1 for _ in text.splitlines())
        by_lang[lang] = by_lang.get(lang, 0) + 1

        # A structural index row with neither symbols nor imports contains
        # only a filename and LOC count. It is cheap to regenerate but costly
        # in memory quality, so do not write it in the first place.
        if not symbols and not imports:
            low_signal_skipped += 1
            continue

        if len(sample_files) < 30:
            sample_files.append(str(rel))

        # Compose the content - content drives BM25 + vector recall, so
        # we put the most informative things first.
        head = f"File: {rel} ({lang}, {loc} lines)"
        symbols_str = ", ".join(symbols[:20]) if symbols else "(no symbols)"
        imports_str = ", ".join(imports[:15]) if imports else "(no imports)"
        body = f"\nSymbols: {symbols_str}\nImports: {imports_str}"

        items.append({
            "type": "fact",
            "content": head + body,
            "importance": importance,
            "metadata": {
                "source":          "project",
                "project_path":    project_path,
                "project_name":    project_name,
                "file_path":       str(rel),
                "file_path_posix": rel.as_posix(),
                "language":        lang,
                "loc":             loc,
                "sha1":            sha1,
                "symbols":         symbols[:40],
                "imports":         imports[:30],
                "index_artifact":  True,
            },
        })
        files_indexed += 1

    # Top-up the entrypoint event with the summary we just gathered.
    if root_item is not None:
        root_item["content"] += (
            f"\nFiles seen: {files_seen} (indexed {files_indexed}, "
            f"skipped {files_skipped} unchanged, "
            f"{low_signal_skipped} low-signal)\n"
            f"By language: " + ", ".join(f"{k}={v}" for k, v in by_lang.items())
            + ("\nSample files: " + ", ".join(sample_files[:10]) if sample_files else "")
        )

    if items:
        engine.record_batch_async(items=items)

    return {
        "project_name":   project_name,
        "project_path":   project_path,
        "n_files_seen":   files_seen,
        "n_indexed":      files_indexed,
        "n_skipped":      files_skipped,
        "n_low_signal_skipped": low_signal_skipped,
        "by_language":    by_lang,
        "duration_ms":    round((time.time() - t0) * 1000),
        "queued":         True,
        "sample_files":   sample_files[:10],
    }
