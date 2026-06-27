"""Entity-extractor stopword coverage for code-index scaffolding + dates."""
from __future__ import annotations

from pmb.graph.entities import EntityExtractor


def test_extractor_skips_index_scaffolding_and_months():
    """Regression: `index project` facts ("File: x.py ... Symbols: ...
    Imports: ...") and dated content ("On June 24, 2026") were leaking
    scaffolding words (file / symbols / imports / lines / june) into the entity
    graph as nodes - some even miscategorised as people."""
    ex = EntityExtractor()
    text = (
        "File: target_module.py (python, 26 lines)\n"
        "Symbols: def alpha, def beta, class GammaWidget\n"
        "Imports: os, json, pathlib.Path\n"
        "On June 24, 2026 this module was added."
    )
    names = {n.lower() for _, n in ex.extract(text).all_named()}
    for noise in (
        "file", "files", "symbol", "symbols", "import", "imports",
        "line", "lines", "def", "june",
    ):
        assert noise not in names, f"{noise!r} should be filtered from entities"


def test_extractor_still_keeps_real_filenames():
    """The 'file' stopword must filter the WORD 'file', not real filenames."""
    ex = EntityExtractor()
    got = ex.extract("File: target_module.py (python, 26 lines)")
    # the actual filename is still captured on the file layer
    assert any("target_module.py" in f for f in got.files)
