"""Module-level constant extraction in the Python symbol indexer."""
from __future__ import annotations

from pmb.ingest.project import _extract_python
from pmb.reasoning.code_ast import extract_python_symbols

SAMPLE = '''
import os

MAGIC_CONSTANT = 42
API_BASE_URL = "https://x"
_lowercase_var = 1          # not a constant
table = "users"             # not a constant

def alpha(a, b):
    LOCAL_NOT_CONST = 5     # inside a function - must NOT be captured
    return a + b

class Widget:
    CLASS_ATTR = 7          # class attribute - module-only, must NOT be captured
    def render(self):
        return MAGIC_CONSTANT
'''


def test_extract_python_symbols_captures_module_constants():
    syms = extract_python_symbols(SAMPLE)
    consts = {s.name for s in syms if s.kind == "constant"}
    assert "MAGIC_CONSTANT" in consts
    assert "API_BASE_URL" in consts
    # lowercase module vars are not constants
    assert "_lowercase_var" not in consts
    assert "table" not in consts
    # function locals and class attributes are not module-level constants
    assert "LOCAL_NOT_CONST" not in consts
    assert "CLASS_ATTR" not in consts


def test_extract_python_renders_const_in_symbol_list():
    symbols, imports = _extract_python(SAMPLE)
    assert "MAGIC_CONSTANT (const)" in symbols
    assert "API_BASE_URL (const)" in symbols
    # existing kinds still render
    assert "def alpha" in symbols
    assert "class Widget" in symbols
    assert "Widget.render (method)" in symbols
    assert "os" in imports
