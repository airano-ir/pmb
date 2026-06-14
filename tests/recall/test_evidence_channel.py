"""R3: absolute evidence channel.

recall results carry `signals.raw_cosine` — the UN-normalized vector similarity
1/(1+dist) in [0,1]. Unlike `score` (min-max normalized over the candidate set,
so the top hit is ≈1.0 even for an irrelevant corpus), raw_cosine is an absolute
signal a gate can trust. `auto_recall.evidence_min_cosine` (default 0 = off)
uses it to stop GENERIC_FACTUAL from surfacing when the workspace's best vector
match is weak — but only matters when the engine is WARM (cold/BM25-only recall
has no vectors, so raw_cosine is 0 and the gate is a no-op, as intended).
"""
from __future__ import annotations

import pytest

from pmb.core.engine import Engine
from pmb.core.engine.types import RecallResult

# ── fast unit: the channel is wired through to the dict ──────────────────────

def test_recallresult_exposes_raw_cosine():
    r = RecallResult(
        ulid="x", event_type="fact", content="c", metadata={}, timestamp=0.0,
        score=0.9, bm25_score=0.1, vec_score=0.2, importance=0.5,
        recency_score=0.3, raw_vec=0.42)
    sig = r.to_dict()["signals"]
    assert sig["raw_cosine"] == 0.42
    # default is 0.0 when not provided (cold / graph-only hits)
    r2 = RecallResult(ulid="y", event_type="fact", content="c", metadata={},
                      timestamp=0.0, score=0.5, bm25_score=0.0, vec_score=0.0,
                      importance=0.5, recency_score=0.0)
    assert r2.to_dict()["signals"]["raw_cosine"] == 0.0


# ── warm integration: raw_cosine is real, and the gate bites ─────────────────

@pytest.fixture
def warm_engine(tmp_pmb_home, tmp_workspace_dir):
    # crosslingual_bm25_damp pinned OFF (1.0): these tests isolate the
    # evidence/specificity gates, which gate on ABSOLUTE raw_cosine. The OOV
    # cross-lingual damper is an orthogonal RANKING feature - on this 2-fact toy
    # corpus it re-orders which low-relevance hit lands at rank 0, perturbing the
    # gate assertions (on a real corpus raw_cosine stays low, so the floor still
    # suppresses - verified separately). Pin it off so the gate is tested alone.
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
                 config_overrides={"recall.cache_size": 0,
                                   "auto_recall.enabled": True,
                                   "recall.crosslingual_bm25_damp": 1.0,
                                   # isolate the evidence/specificity gates from
                                   # the orthogonal conversational query-worthiness
                                   # gate (tested separately).
                                   "auto_recall.conversational_gap_max": 0.0})
    eng.warmup()  # load the embedder FIRST so the facts get real vectors
    eng.record_fact("The API runs Postgres 17 on port 5432")
    eng.record_fact("We deploy with Kubernetes on GKE")
    eng.wait_for_writes()
    return eng


def test_raw_cosine_is_positive_when_warm(warm_engine):
    d = warm_engine.recall("what database port does the api use", top_k=5).to_dict()
    assert d["results"]
    rc = float((d["results"][0].get("signals") or {}).get("raw_cosine") or 0.0)
    assert rc > 0.0, "a warm hybrid hit must carry a real absolute cosine"


def _top_raw_cosine(eng, q):
    d = eng.recall(q, top_k=5).to_dict()
    if not d["results"]:
        return 0.0
    return float((d["results"][0].get("signals") or {}).get("raw_cosine") or 0.0)


def test_absolute_cosine_discriminates_relevant_from_nonsense(warm_engine):
    # The whole point of R3: even though BOTH queries get a min-maxed `score`
    # near 1.0 at rank 0, the ABSOLUTE signal is higher for a real topical match
    # than for gibberish — which is exactly what evidence_min_cosine gates on.
    # (The 1/(1+L2) scale is compressed, so we assert the ORDERING, not a
    # magnitude.)
    rel = _top_raw_cosine(warm_engine, "postgres database port number")
    nons = _top_raw_cosine(warm_engine, "xyzzy plugh frobnicate quux blorp")
    assert rel > nons


def test_evidence_floor_gates_when_above_the_top_hit(warm_engine):
    # A floor above EVERY hit's absolute cosine makes the gated recall path
    # surface nothing — proving evidence_min_cosine is actually consulted.
    from pmb.hooks import run_auto_context
    from pmb.hooks.auto_recall import Intent
    q = "what rate limiter should we use here"  # statement-ish, no project word
    warm_engine.config._overrides["auto_recall.evidence_min_cosine"] = 1.0
    res = run_auto_context(warm_engine, q)
    # if it routed to the gated recall path, the 1.0 floor must have emptied it
    if Intent.GENERIC_FACTUAL in res.intents and Intent.PAST_QUERY not in res.intents:
        assert not res.recall_hits


def test_default_floor_suppresses_no_answer_generic_factual(warm_engine):
    # The shipped default (auto_recall.evidence_min_cosine = 0.045, calibrated on
    # the real corpus) must drop a GENERIC_FACTUAL question the workspace knows
    # nothing about - the observed false-positive channel where the min-max top
    # hit (score ≈ 1.0) surfaced unrelated lessons/facts.
    from pmb.hooks import run_auto_context
    from pmb.hooks.auto_recall import Intent
    res = run_auto_context(warm_engine, "xyzzy plugh frobnicate quux blorp?")
    if Intent.GENERIC_FACTUAL in res.intents and Intent.PAST_QUERY not in res.intents:
        assert not res.recall_hits, \
            "a no-answer GENERIC_FACTUAL query must surface nothing under the default floor"


def test_default_floor_keeps_real_generic_factual_match(warm_engine):
    # The flip side: the default floor must NOT over-block - a genuine topical
    # match (raw_cosine well above 0.045) still surfaces.
    from pmb.hooks import run_auto_context
    from pmb.hooks.auto_recall import Intent
    res = run_auto_context(warm_engine, "what database port does the api use?")
    if Intent.GENERIC_FACTUAL in res.intents and Intent.PAST_QUERY not in res.intents:
        assert res.recall_hits, \
            "a real topical GENERIC_FACTUAL match must still clear the default floor"
