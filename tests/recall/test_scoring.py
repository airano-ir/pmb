"""X1/X2/X3 — unified base scoring, adaptive channel weights, calibrated
confidence. The headline guarantee is PARITY: with the default identity weights
the extracted functions reproduce recall's old inline math exactly, so default
behaviour is byte-identical (the recall/tier/eval suites are the end-to-end
proof; these pin the math directly)."""
from __future__ import annotations

import pytest

from pmb.reasoning import scoring as S


@pytest.mark.parametrize("hit,imp,rec,rmul", [
    (1.0, 0.0, 0.0, 1.0),
    (2.5, 1.0, 1.0, 1.0),
    (0.7, 0.4, 0.8, 0.5),
    (3.3, 0.9, 0.2, 1.0),
    (0.0, 0.5, 0.5, 1.0),
])
def test_combine_base_score_identity_matches_old_formula(hit, imp, rec, rmul):
    old = hit * (0.5 + 0.5 * imp) * (1.0 + 0.2 * rec * rmul)
    impf = S.importance_factor(imp)                      # identity → 0.5+0.5*imp
    new = S.combine_base_score(hit, impf, rec, rec_mul=rmul)
    assert new == pytest.approx(old)
    assert S.combine_base_score(hit, impf, rec, rec_mul=rmul,
                                weights={"hit": 1.0, "importance": 1.0,
                                         "recency": 1.0}) == pytest.approx(old)


def test_importance_factor_carries_the_weight():
    assert S.importance_factor(0.5) == pytest.approx(0.75)              # identity
    assert S.importance_factor(0.5, {"importance": 0.0}) == pytest.approx(0.5)
    assert S.importance_factor(1.0, {"importance": 1.0}) == pytest.approx(1.0)
    # the SAME factor is what recall feeds to the boosts → consistent weighting
    assert S.importance_factor(0.4, {"importance": 2.0}) == pytest.approx(0.5 + 0.4)


def test_channel_weights_scale_each_channel():
    impf = S.importance_factor(0.5)
    base = S.combine_base_score(2.0, impf, 0.5)
    # a >1 hit weight raises the score; <1 lowers it
    assert S.combine_base_score(2.0, impf, 0.5, weights={"hit": 1.5}) > base
    assert S.combine_base_score(2.0, impf, 0.5, weights={"hit": 0.5}) < base
    # recency weight 0 removes the recency reward → score == hit*impf
    flat = S.combine_base_score(2.0, impf, 0.5, weights={"recency": 0.0})
    assert flat == pytest.approx(2.0 * impf)


def test_calibrated_confidence_matches_old_and_is_bounded():
    # single result
    assert S.calibrated_confidence(0.8) == pytest.approx(min(1.0, 0.8 * 0.7 + 0.1))
    # with a gap
    assert S.calibrated_confidence(0.9, 0.4) == pytest.approx(
        min(1.0, 0.9 * 0.7 + 0.5 * 0.3 + 0.1))
    # bounds + clamping
    for t1, t2 in [(-5, None), (5, None), (1.0, 0.0), (0.0, 0.0)]:
        c = S.calibrated_confidence(t1, t2)
        assert 0.0 <= c <= 1.0


def test_calibrated_confidence_monotonic_in_top1():
    prev = -1.0
    for t1 in [0.0, 0.25, 0.5, 0.75, 1.0]:
        c = S.calibrated_confidence(t1, 0.0)
        assert c >= prev
        prev = c


def test_channel_weights_parse_and_default(monkeypatch):
    class _Cfg:
        def __init__(self, v): self._v = v
        def get(self, k): return self._v
    assert S.is_identity(S.channel_weights(_Cfg("")))          # empty → identity
    assert S.is_identity(S.channel_weights(_Cfg("not json")))  # malformed → identity
    w = S.channel_weights(_Cfg('{"hit": 1.2, "recency": 0.8, "bogus": 9}'))
    assert w["hit"] == 1.2 and w["recency"] == 0.8
    assert w["importance"] == 1.0          # missing channel defaults to identity
    assert "bogus" not in w                # unknown key ignored


def test_propose_channel_weights_nudges_and_clamps():
    # 'hit' high on useful recalls, low on useless → weight nudged > 1
    samples = [({"hit": 1.0, "importance": 0.5, "recency": 0.5}, True),
               ({"hit": 0.0, "importance": 0.5, "recency": 0.5}, False)]
    w = S.propose_channel_weights(samples, lr=0.5)
    assert w["hit"] > 1.0
    assert all(0.5 <= w[c] <= 1.5 for c in S.CHANNELS)   # clamped
    assert S.is_identity(S.propose_channel_weights([]))  # no data → identity
