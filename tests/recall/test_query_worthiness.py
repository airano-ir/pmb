"""Query-worthiness classifier (SAE pattern, new axis): margin between
conversational and knowledge exemplars. Pure math, tested with a stub embedder
(no model) so it's deterministic; the real cross-lingual separation is measured
in the eval/benches."""
from __future__ import annotations

from pmb.reasoning.query_worthiness import (
    CONVERSATIONAL,
    KNOWLEDGE,
    QueryWorthiness,
)

_CONV = set(CONVERSATIONAL)
_KNOW = set(KNOWLEDGE)


def _stub(texts):
    """conversational exemplars (and the literal 'CONV') -> [1,0];
    knowledge exemplars (and 'KNOW') -> [0,1]; so margin is +1 / -1."""
    out = []
    for t in texts:
        if t in _CONV or t == "CONV":
            out.append([1.0, 0.0])
        elif t in _KNOW or t == "KNOW":
            out.append([0.0, 1.0])
        else:
            out.append([0.5, 0.5])
    return out


def test_margin_separates_conversational_from_knowledge():
    qw = QueryWorthiness(_stub)
    assert qw.conversational_margin("CONV") > 0.5
    assert qw.conversational_margin("KNOW") < 0.0
    assert qw.is_conversational("CONV", tau=0.05) is True
    assert qw.is_conversational("KNOW", tau=0.05) is False


def test_tau_is_respected():
    qw = QueryWorthiness(_stub)
    # a neutral message ([0.5,0.5]) has margin ~0 -> not conversational at tau 0.05
    assert qw.is_conversational("neutral text", tau=0.05) is False


def test_empty_message_is_not_conversational():
    qw = QueryWorthiness(_stub)
    assert qw.conversational_margin("") == 0.0
    assert qw.is_conversational("", tau=0.05) is False
