"""The file-path entity regex must match real paths AND stay linear (no ReDoS).

CodeQL flagged `_FILE_RE` (graph/entities.py): the old dir prefix
`(?:[A-Za-z0-9_\\-./]+/)*` put '/' inside the inner '+' under a '*', which
backtracks catastrophically on long slashless input.
"""
from __future__ import annotations

import time

from pmb.graph.entities import _FILE_RE


def test_file_regex_still_matches_paths():
    assert _FILE_RE.search("see src/pmb/core/engine.py here").group(0) == \
        "src/pmb/core/engine.py"
    assert _FILE_RE.search("just foo.py").group(0) == "foo.py"
    assert _FILE_RE.search("a/b/c/d.tsx").group(0) == "a/b/c/d.tsx"


def test_file_regex_is_linear_on_pathological_input():
    # A long run with no closing `.ext` made the old regex backtrack for
    # seconds. Linear matching finishes in well under a tenth of a second.
    evil = "/" + ("a" * 20000)
    t0 = time.perf_counter()
    _FILE_RE.search(evil)
    assert (time.perf_counter() - t0) < 0.1
