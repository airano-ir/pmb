"""Tests for Engine.project_structure - the memory-sourced project map."""
from __future__ import annotations


def _seed_project(eng, pp: str = "/tmp/myproj") -> None:
    """Insert the kind of rows index_project + track modules would write."""
    eng.record_batch([
        {
            "type": "fact",
            "content": "File: src/a.py (python, 10 lines)\nSymbols: def f\nImports: os",
            "importance": 0.55,
            "metadata": {
                "source": "project", "project_path": pp, "project_name": "myproj",
                "file_path": "src/a.py", "language": "python", "loc": 10,
                "symbols": ["def f", "class B"], "imports": ["os"],
                "index_artifact": True,
            },
        },
        {
            "type": "fact",
            "content": "File: src/b.py (python, 4 lines)\nSymbols: def g\nImports: sys",
            "importance": 0.55,
            "metadata": {
                "source": "project", "project_path": pp, "project_name": "myproj",
                "file_path": "src/b.py", "language": "python", "loc": 4,
                "symbols": ["def g"], "imports": ["sys"], "index_artifact": True,
            },
        },
        {
            "type": "fact",
            "content": "Module src/a.py: parses the config and walks the tree",
            "importance": 0.5,
            "metadata": {
                "source": "module-summary", "project_path": pp,
                "file_path": "src/a.py", "language": "python",
            },
        },
    ])
    eng.wait_for_writes(timeout=120)


def test_project_structure_assembles_from_memory(isolated_engine):
    eng = isolated_engine
    _seed_project(eng)

    st = eng.project_structure("myproj")
    assert st.get("empty") is not True
    assert st["n_files"] == 2
    assert st["languages"].get("python") == 2
    assert "src" in st["tree"]
    assert {f["file"] for f in st["tree"]["src"]} == {"src/a.py", "src/b.py"}

    # The module purpose from track modules is surfaced on the file.
    by_file = {m["file"]: m for m in st["key_modules"]}
    assert by_file["src/a.py"]["purpose"] == "parses the config and walks the tree"
    assert st["n_with_purpose"] == 1


def test_project_structure_matches_by_path_substring(isolated_engine):
    eng = isolated_engine
    _seed_project(eng, pp="/home/me/code/widgets")
    st = eng.project_structure("widgets")
    assert st.get("empty") is not True
    assert st["path"] == "/home/me/code/widgets"
    assert st["n_files"] == 2


def test_project_structure_empty_when_not_indexed(isolated_engine):
    st = isolated_engine.project_structure("does-not-exist")
    assert st.get("empty") is True
    assert "hint" in st


def test_project_structure_empty_name(isolated_engine):
    st = isolated_engine.project_structure("")
    assert st.get("empty") is True
