"""L1 parity gate — relocating the RU/UK lexical floor into the built-in
``pmb/lang/packs/{ru,uk}.yaml`` packs (active by default) must keep every merged
lexical structure BYTE-IDENTICAL to the pre-refactor snapshot.

The snapshot (`_lang_parity_snapshot.json`) was captured from the hardcoded
EN+RU+UK floor BEFORE any data moved to packs. If a relocation changes a merged
set/group — e.g. cross-contaminating a shared category — this test fails loudly
instead of silently degrading Russian/Ukrainian recall.
"""
from __future__ import annotations

import json
from pathlib import Path

_SNAP = json.loads(
    (Path(__file__).parent / "_lang_parity_snapshot.json").read_text(encoding="utf-8")
)


def _live() -> dict:
    import pmb.core.text_match as TM
    import pmb.reasoning.attributes as A
    import pmb.reasoning.pamvr as P
    return {
        "pamvr._STOP": sorted(P._STOP),
        "pamvr.VERB_SYNS": {k: sorted(v) for k, v in P.VERB_SYNS.items()},
        "pamvr._NOT_PROPER": sorted(P._NOT_PROPER),
        "pamvr._FIRST_PERSON": sorted(P._FIRST_PERSON),
        "pamvr._RELATION_MARKERS": sorted(getattr(P, "_RELATION_MARKERS", set())),
        "attributes._ALIAS_GROUPS": {k: sorted(v) for k, v in A._ALIAS_GROUPS.items()},
        "text_match.STOPWORDS": sorted(TM.STOPWORDS),
    }


def test_merged_lexical_floor_is_byte_identical():
    live = _live()
    snap = _SNAP
    assert set(live) == set(snap), "structure set changed"
    diffs = {}
    for k in snap:
        if live[k] != snap[k]:
            diffs[k] = {
                "missing": _delta(snap[k], live[k]),
                "extra": _delta(live[k], snap[k]),
            }
    assert not diffs, f"lexical floor drifted after pack relocation: {diffs}"


def _delta(a, b):
    """Items in a not in b (handles list or dict-of-lists)."""
    if isinstance(a, dict):
        return {k: sorted(set(a.get(k, [])) - set(b.get(k, []))) for k in a
                if set(a.get(k, [])) - set(b.get(k, []))}
    return sorted(set(a) - set(b))
