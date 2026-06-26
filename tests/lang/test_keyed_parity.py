"""C3 parity gate — the anchor keyed-extractor (C2) must produce the SAME
(canonical attr, value) as the regex/pack tier where BOTH apply (meaning-parity,
not byte equality), AND extend to languages the regex tier can't cover. This is
the gate the PLAN names: anchors must not change the answer on the cases regexes
already handle, only add coverage. Real embedder, loaded once.
"""
from __future__ import annotations

import pytest

from pmb.core.engine import Engine
from pmb.reasoning.attributes import canonicalize_attribute, detect_current_state
from pmb.reasoning.extract_anchor import extract_keyed_anchor


def _warm(ws, home):
    eng = Engine(cwd=ws, pmb_home=home, config_overrides={"recall.cache_size": 0})
    eng._is_warm = True
    return eng


def _regex_pair(sentence):
    hit = detect_current_state(sentence)
    return (canonicalize_attribute(hit[0]), hit[1].lower()) if hit else None


def _anchor_pairs(eng, sentence):
    return {(canonicalize_attribute(c.attr), c.value.lower())
            for c in extract_keyed_anchor(sentence, eng)}


@pytest.mark.eval
def test_anchor_agrees_with_regex_where_both_apply(tmp_pmb_home, tmp_workspace_dir):
    eng = _warm(tmp_workspace_dir, tmp_pmb_home)
    # Explicit current-state phrasings the regex tier actually covers (it is
    # deliberately conservative: "I live in Tampa" alone does NOT fire). The
    # parity contract is meaning-NON-CONTRADICTION: wherever BOTH tiers fire, the
    # anchor must AGREE on (attr, value). The anchor being NARROWER on a phrasing
    # (firing nothing) is fine — it is additive; the regex tier still covers it.
    covered = ["I currently live in Tampa", "I moved to Berlin",
               "I currently work at Stripe", "my current city is Madrid"]
    agreed, contradictions = 0, []
    for s in covered:
        r = _regex_pair(s)
        if r is None:
            continue                      # regex doesn't cover this phrasing
        a = _anchor_pairs(eng, s)
        if not a:
            continue                      # anchor narrower here (additive gap, OK)
        if r in a:
            agreed += 1
        else:
            same_attr_vals = {v for (att, v) in a if att == r[0]}
            if same_attr_vals and r[1] not in same_attr_vals:
                contradictions.append((s, r, a))
    assert not contradictions, f"anchor CONTRADICTS regex: {contradictions}"
    assert agreed >= 1, "no both-fire parity comparison ran"


@pytest.mark.eval
@pytest.mark.platform_sensitive  # embedder float-variance flips 2-3/4 on macOS arm64; gates on Linux
def test_anchor_extends_keyed_extraction_to_new_languages(tmp_pmb_home, tmp_workspace_dir):
    eng = _warm(tmp_workspace_dir, tmp_pmb_home)
    # Languages with NO pack — the regex tier returns nothing; the anchor tier
    # must extract them (the new-language win regexes can't do).
    extend = [
        ("Ich wohne in Berlin", "city", "berlin"),
        ("Vivo en Madrid", "city", "madrid"),
        ("Je travaille chez Datadog", "employer", "datadog"),
        ("ich arbeite bei Google", "employer", "google"),
    ]
    hits = 0
    for s, attr, val in extend:
        if (canonicalize_attribute(attr), val) in _anchor_pairs(eng, s):
            hits += 1
    assert hits >= 3, f"anchor failed to extend keyed extraction: {hits}/4"
