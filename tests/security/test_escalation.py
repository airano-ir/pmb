"""Tests for Multi-Modal Recall escalation (Improvement G)."""
from __future__ import annotations

from pmb.core.engine import Engine, RecallPack, RecallResult

# ----------------------------------------------------------------------
# Confidence math
# ----------------------------------------------------------------------

def test_confidence_empty_pack_is_zero():
    p = RecallPack(query="", workspace_name="w", workspace_id="i",
                   results=[], n_total_in_workspace=0, elapsed_ms=0)
    assert p.confidence == 0.0


def _mk_result(ulid: str, score: float) -> RecallResult:
    return RecallResult(
        ulid=ulid, content="c", score=score,
        bm25_score=0.0, vec_score=0.0, importance=0.5, recency_score=0.0,
        timestamp=0.0, event_type="qa", metadata={},
    )


def test_confidence_high_top1_and_gap():
    p = RecallPack(query="", workspace_name="w", workspace_id="i",
                   results=[_mk_result("a", 0.9), _mk_result("b", 0.2)],
                   n_total_in_workspace=2, elapsed_ms=0)
    # top1=0.9, gap=0.7 → conf = 0.9*0.7 + 0.7*0.3 + 0.1 = 0.94
    assert p.confidence > 0.8


def test_confidence_low_when_tied():
    p = RecallPack(query="", workspace_name="w", workspace_id="i",
                   results=[_mk_result("a", 0.4), _mk_result("b", 0.39)],
                   n_total_in_workspace=2, elapsed_ms=0)
    # top1=0.4, gap=0.01 → conf ≈ 0.4*0.7 + 0.01*0.3 + 0.1 = 0.38
    assert p.confidence < 0.5


def test_confidence_single_result():
    p = RecallPack(query="", workspace_name="w", workspace_id="i",
                   results=[_mk_result("a", 0.7)],
                   n_total_in_workspace=1, elapsed_ms=0)
    # conf = 0.7*0.7 + 0.1 = 0.59
    assert 0.5 < p.confidence < 0.7


# ----------------------------------------------------------------------
# Engine recall_smart
# ----------------------------------------------------------------------

def test_smart_recall_returns_immediately_on_high_confidence(
    tmp_pmb_home, tmp_workspace_dir,
):
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={"recall.cache_size": 0},
    )
    eng.record_fact("Postgres uses port 5433 on the api service")
    eng.record_fact("Random unrelated note about lunch")
    pack = eng.recall_smart("Postgres port 5433", top_k=3,
                            confidence_threshold=0.1)  # low threshold for test
    assert pack is not None
    assert len(pack.results) >= 1


def test_smart_recall_does_not_crash_on_low_signal(
    tmp_pmb_home, tmp_workspace_dir,
):
    """If everything fails / no results, recall_smart should return a pack
    (possibly empty) not crash."""
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={"recall.cache_size": 0},
    )
    pack = eng.recall_smart("nothing here", top_k=3, confidence_threshold=0.9)
    assert pack is not None
    assert hasattr(pack, "results")


def test_smart_recall_picks_higher_confidence_across_stages(
    tmp_pmb_home, tmp_workspace_dir,
):
    """Sanity: recall_smart returns at least as good a result as recall."""
    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        config_overrides={"recall.cache_size": 0},
    )
    eng.record_fact("Caroline researched adoption agencies")
    eng.record_fact("Melanie does pottery and painting")
    base = eng.recall("adoption", top_k=3)
    smart = eng.recall_smart("adoption", top_k=3,
                             confidence_threshold=0.99)
    # Tolerance: confidence folds in a time.time()-based recency term, and the
    # two calls run a few ms apart, so the SAME result scores ~1e-10 lower on
    # the later (smart) call. The contract is "smart is not WORSE than base",
    # not bit-exact equality - compare with a float epsilon so CI is not 50/50.
    assert smart.confidence >= base.confidence - 1e-6
