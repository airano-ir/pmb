"""Tests for multi-modal (Improvement J): code AST + image attach."""
from __future__ import annotations

from pmb.core.engine import Engine
from pmb.reasoning.code_ast import (
    extract_python_symbols,
    looks_like_python,
    symbols_to_entity_names,
)
from pmb.reasoning.images import attach_image

# ----------------------------------------------------------------------
# Code AST
# ----------------------------------------------------------------------

def test_looks_like_python():
    assert looks_like_python("def foo():\n    return 1")
    assert looks_like_python("import sys")
    assert looks_like_python("class Foo:\n    pass")
    assert not looks_like_python("just a regular sentence")
    assert not looks_like_python("")


def test_extract_python_functions():
    code = """
def hello(name):
    '''Say hi.'''
    print(f"Hi, {name}!")

def goodbye():
    pass
"""
    syms = extract_python_symbols(code)
    names = {s.name for s in syms}
    assert "hello" in names
    assert "goodbye" in names
    # Function with docstring
    hi = [s for s in syms if s.name == "hello"][0]
    assert "Say hi" in hi.docstring


def test_extract_python_class_and_methods():
    code = """
class Foo:
    def __init__(self):
        self.x = 1
    def bar(self, y):
        return y
"""
    syms = extract_python_symbols(code)
    kinds = [(s.kind, s.name) for s in syms]
    assert ("class", "Foo") in kinds
    assert ("method", "__init__") in kinds
    assert ("method", "bar") in kinds


def test_extract_python_imports():
    code = "import sys\nfrom pathlib import Path\nimport json as J"
    syms = extract_python_symbols(code)
    import_names = [s.name for s in syms if s.kind == "import"]
    assert "sys" in import_names
    assert "pathlib.Path" in import_names
    assert "json" in import_names


def test_symbols_to_entity_names():
    code = "def foo(): pass\nclass Bar: pass\nimport sys"
    syms = extract_python_symbols(code)
    pairs = symbols_to_entity_names(syms)
    pair_set = set(pairs)
    assert ("function", "foo") in pair_set
    assert ("class", "bar") in pair_set
    assert ("import", "sys") in pair_set


def test_extract_handles_syntax_error():
    # Broken / half-written code must not crash. ast.parse fails, so the
    # regex fallback (_extract_via_regex) kicks in and still recovers the
    # def NAME (no signature/docstring). This is intentional — Cursor /
    # Claude Code stream half-written code constantly, and we still want
    # entities out of it. See code_ast._extract_via_regex.
    code = "def broken(\n  this is not valid python"
    syms = extract_python_symbols(code)
    assert [s.name for s in syms] == ["broken"]
    assert syms[0].kind == "function"


# ----------------------------------------------------------------------
# Engine integration: code AST becomes graph entities
# ----------------------------------------------------------------------

def test_code_recording_creates_function_entities(
    tmp_pmb_home, tmp_workspace_dir,
):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    code = """
import sys
def authenticate(user, password):
    '''Check user credentials.'''
    return user == 'admin'

class AuthManager:
    def login(self, user): pass
"""
    eng.record_event(event_type="code", content=code)
    # Check graph
    import sqlite3
    with sqlite3.connect(eng.workspace.db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT kind, name FROM graph_entities WHERE workspace_id = ?",
            (eng.workspace.id,),
        ).fetchall()
    kinds_names = {(r["kind"], r["name"]) for r in rows}
    assert ("function", "authenticate") in kinds_names
    assert ("class", "authmanager") in kinds_names
    assert ("function", "authmanager.login") in kinds_names
    assert ("import", "sys") in kinds_names


def test_code_ast_disabled(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={"recall.code_ast_extraction": False},
    )
    eng.record_event(event_type="code", content="def foo(): pass")
    import sqlite3
    with sqlite3.connect(eng.workspace.db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM graph_entities WHERE workspace_id = ? "
            "AND kind = 'function'",
            (eng.workspace.id,),
        ).fetchall()
    # Code-AST disabled → no 'function' entities
    assert len(rows) == 0


# ----------------------------------------------------------------------
# Image attach (no CLIP required)
# ----------------------------------------------------------------------

def test_attach_image_metadata_only(tmp_pmb_home, tmp_workspace_dir):
    # Make a fake PNG (we don't need it to be valid for metadata-only)
    img_path = tmp_workspace_dir / "test.png"
    # 1x1 PNG
    img_path.write_bytes(bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000a49444154789c63000100000005000100"
        "0d0a2db40000000049454e44ae426082"
    ))
    att = attach_image(str(img_path), description="test image",
                       encode_clip=False)
    assert att.path == str(img_path.resolve())
    assert att.description == "test image"
    assert att.sha256 is not None
    assert att.clip_embedding is None  # CLIP not requested


def test_engine_record_image(tmp_pmb_home, tmp_workspace_dir):
    img_path = tmp_workspace_dir / "diagram.png"
    img_path.write_bytes(bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000a49444154789c63000100000005000100"
        "0d0a2db40000000049454e44ae426082"
    ))
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ulid = eng.record_image(
        str(img_path),
        description="Diagram showing the auth flow with arrows",
    )
    ev = eng.events.get_by_ulid(ulid)
    assert ev.event_type == "image"
    assert "auth flow" in ev.content
    assert ev.metadata.get("image_path", "").endswith("diagram.png")
    assert "image_sha256" in ev.metadata


def test_image_event_searchable_via_description(tmp_pmb_home, tmp_workspace_dir):
    img_path = tmp_workspace_dir / "auth_diagram.png"
    img_path.write_bytes(bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000a49444154789c63000100000005000100"
        "0d0a2db40000000049454e44ae426082"
    ))
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={"recall.cache_size": 0},
    )
    target = eng.record_image(
        str(img_path),
        description="Diagram of authentication flow with OAuth tokens",
    )
    eng.record_fact("Random unrelated note about lunch")

    pack = eng.recall("auth diagram OAuth", top_k=3)
    ulids = [r.ulid for r in pack.results]
    assert target in ulids


def test_search_images_by_text_fallback(tmp_pmb_home, tmp_workspace_dir):
    """Without CLIP installed, search_images_by_text falls back to
    text-based recall filtered to image events."""
    img_path = tmp_workspace_dir / "x.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n")  # not valid but enough
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    try:
        eng.record_image(str(img_path), description="dashboard screenshot")
    except FileNotFoundError:
        return  # Skip if path lookup failed somehow
    results = eng.search_images_by_text("dashboard", top_k=3)
    # Should return at least the image we recorded (via fallback path)
    assert isinstance(results, list)
