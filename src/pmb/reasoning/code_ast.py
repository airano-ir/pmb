"""
Code AST extraction (Improvement J — code half).

Parses code content (Python first; extensible) and emits structured
entities: functions, classes, imports, decorators. These become graph
nodes alongside the raw text, so "find code that uses X" lights up
correctly via entity graph, not just BM25 on raw source.

Stack: pure stdlib for Python (`ast` module).
Other languages would need tree-sitter — we leave that as a future
extension. When content doesn't parse as Python, we fall back to a regex
scanner that still recovers def/class/import NAMES (no signature/docstring),
so half-written code from a streaming agent still yields entities. Caller
still indexes the raw text.

Cost: <5ms per file (Python's ast is fast).

Why this matters:
  Devs ask things like "which function calls X" or "find the auth check".
  Without AST, we just lexically match — misses cases where the call uses
  different variable names. With AST entities ('function', 'class'), the
  graph layer connects code by structure, not just words.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class CodeSymbol:
    kind: str           # function, class, import, decorator
    name: str
    signature: str = ""
    docstring: str = ""
    line: int = 0
    parent: str | None = None  # for methods: parent class name


def looks_like_python(text: str) -> bool:
    """Cheap signal that we should bother parsing as Python."""
    if not text:
        return False
    if len(text) < 10:
        return False
    triggers = ("def ", "class ", "import ", "from ", "@", "    ")
    return any(t in text for t in triggers)


def extract_python_symbols(code: str) -> list[CodeSymbol]:
    """Parse Python code and return structured symbols.

    Resilient to broken/incomplete code: if `ast.parse` fails (Cursor /
    Claude Code constantly stream half-written code), we fall back to a
    regex-based scanner that still extracts function/class/import names.
    Worse precision than AST (no signature/docstring), but never crashes.
    """
    if not code or not looks_like_python(code):
        return []
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return _extract_via_regex(code)

    out: list[CodeSymbol] = []

    def walk(node, parent_class: str | None = None):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            args = []
            for a in node.args.args:
                args.append(a.arg)
            sig = f"{node.name}({', '.join(args)})"
            out.append(CodeSymbol(
                kind="function" if parent_class is None else "method",
                name=node.name,
                signature=sig,
                docstring=(ast.get_docstring(node) or "")[:300],
                line=node.lineno,
                parent=parent_class,
            ))
            # Don't walk into function bodies (we collect top-level defs only,
            # to avoid catching nested closures as 'methods')
            return
        if isinstance(node, ast.ClassDef):
            bases = [ast.unparse(b) if hasattr(ast, "unparse") else "" for b in node.bases]
            out.append(CodeSymbol(
                kind="class",
                name=node.name,
                signature=f"class {node.name}({', '.join(bases)})",
                docstring=(ast.get_docstring(node) or "")[:300],
                line=node.lineno,
            ))
            # Walk children to collect methods
            for child in node.body:
                walk(child, parent_class=node.name)
            return
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(CodeSymbol(
                    kind="import",
                    name=alias.name,
                    signature=f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""),
                    line=node.lineno,
                ))
            return
        if isinstance(node, ast.ImportFrom):
            mod = node.module or "."
            for alias in node.names:
                out.append(CodeSymbol(
                    kind="import",
                    name=f"{mod}.{alias.name}",
                    signature=f"from {mod} import {alias.name}",
                    line=node.lineno,
                ))
            return
        # Recurse into module / class bodies
        for child in ast.iter_child_nodes(node):
            walk(child, parent_class=parent_class)

    for child in ast.iter_child_nodes(tree):
        walk(child)

    return out


# Regex fallback for code that won't parse — half-written, mid-edit, etc.
# We only catch the cheap stuff: top-level def / class / import names.
# All patterns end at \n to avoid swallowing subsequent lines.
_RE_DEF = re.compile(r"^[ \t]*(?:async[ \t]+)?def[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(", re.MULTILINE)
_RE_CLASS = re.compile(r"^[ \t]*class[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*[\(:]", re.MULTILINE)
_RE_IMPORT_PLAIN = re.compile(r"^[ \t]*import[ \t]+([A-Za-z_][A-Za-z0-9_., \t]*)$", re.MULTILINE)
_RE_IMPORT_FROM = re.compile(
    r"^[ \t]*from[ \t]+([A-Za-z_][A-Za-z0-9_.]*)[ \t]+import[ \t]+([A-Za-z_*][A-Za-z0-9_,*. \t]*)$",
    re.MULTILINE,
)


def _extract_via_regex(code: str) -> list[CodeSymbol]:
    """Fallback symbol extraction for code that won't parse as valid Python.

    Catches the common case where the user (or AI) is in the middle of
    writing — incomplete function bodies, dangling colons, half-edited
    classes. We grab names only; no signature, no docstring, no nesting.
    """
    out: list[CodeSymbol] = []
    code_head = code[:20000]  # cap — broken code doesn't deserve full scan

    for m in _RE_DEF.finditer(code_head):
        name = m.group(1)
        out.append(CodeSymbol(
            kind="function", name=name, signature=f"{name}(...)",
            line=code_head.count("\n", 0, m.start()) + 1,
        ))
    for m in _RE_CLASS.finditer(code_head):
        name = m.group(1)
        out.append(CodeSymbol(
            kind="class", name=name, signature=f"class {name}",
            line=code_head.count("\n", 0, m.start()) + 1,
        ))
    for m in _RE_IMPORT_PLAIN.finditer(code_head):
        names = m.group(1)
        for nm in names.split(","):
            nm = nm.strip().split(" as ")[0].strip()
            if nm:
                out.append(CodeSymbol(
                    kind="import", name=nm, signature=f"import {nm}",
                    line=code_head.count("\n", 0, m.start()) + 1,
                ))
    for m in _RE_IMPORT_FROM.finditer(code_head):
        mod, names = m.group(1), m.group(2)
        for nm in names.split(","):
            nm = nm.strip().split(" as ")[0].strip()
            if nm and nm != "*":
                out.append(CodeSymbol(
                    kind="import", name=f"{mod}.{nm}",
                    signature=f"from {mod} import {nm}",
                    line=code_head.count("\n", 0, m.start()) + 1,
                ))
    return out


def symbols_to_entity_names(symbols: list[CodeSymbol]) -> list[tuple[str, str]]:
    """Convert symbols into (kind, name) pairs for the graph layer.

    Mapping:
      function/method → kind='function', name=<func or class.method>
      class           → kind='class', name=<class>
      import          → kind='import', name=<module>
    """
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for s in symbols:
        if s.kind == "function":
            key = ("function", s.name.lower())
        elif s.kind == "method":
            full = f"{s.parent or '?'}.{s.name}"
            key = ("function", full.lower())
        elif s.kind == "class":
            key = ("class", s.name.lower())
        elif s.kind == "import":
            key = ("import", s.name.lower())
        else:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out
