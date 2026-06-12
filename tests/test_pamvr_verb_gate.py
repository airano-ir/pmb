"""B4 — the lexical verb-synonym boost is dropped when lang.mode=anchors
(verb_match=False). Default (hybrid) keeps it, so recall is unchanged by default.
"""
from __future__ import annotations

from pmb.reasoning.pamvr import prepare_query_features


def test_verb_match_on_by_default():
    f = prepare_query_features("where does Alice work")
    assert f.query_verb == "work"
    assert f.verb_stems, "default (hybrid) mode keeps the verb-synonym stems"
    assert f.use_verb_match is True


def test_verb_match_dropped_in_anchors_mode():
    f = prepare_query_features("where does Alice work", verb_match=False)
    assert f.use_verb_match is False
    assert f.verb_stems == set(), "anchors mode drops the lexical verb boost"
    # topic features are independent of the verb boost and must still compute
    assert f.topic_tokens
