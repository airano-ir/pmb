"""E1 — corpus-derived stopwords (document frequency), language-free."""
from __future__ import annotations

from pmb.core.engine import Engine
from pmb.core.text_match import apply_corpus_stopwords, distinctive_tokens
from pmb.maintenance.corpus_stopwords import (
    compute_corpus_stopwords,
    load_corpus_stopwords,
    refresh_corpus_stopwords,
)


def _eng(ws, home):
    # dedup.enable=False: these fixtures are templated near-duplicates (differ
    # only by an integer) so they SHARE the high-document-frequency tokens the
    # test asserts on. With L2 semantic dedup on, a WARM embedder (full-suite
    # ordering) collapses them into one ulid and n_docs falls under min_docs —
    # an order-dependent failure. This unit covers corpus df-statistics, not the
    # dedup feature, so every injected document must land deterministically.
    return Engine(cwd=ws, pmb_home=home,
                  config_overrides={"recall.cache_size": 0, "dedup.enable": False})


def test_high_document_frequency_token_is_stopword(tmp_pmb_home, tmp_workspace_dir):
    eng = _eng(tmp_workspace_dir, tmp_pmb_home)
    for i in range(12):
        c = f"shared marker note entry {i} payload"   # 'shared/marker/...' in all
        if i < 5:
            c += " widget"                            # 5/12 = 42% > 25%
        if i == 0:
            c += " rareword"                          # 1/12 = 8% < 25%
        eng.record_fact(c)
    words = compute_corpus_stopwords(eng, min_docs=10)
    assert "widget" in words           # 42% document frequency
    assert "shared" in words           # 100%
    assert "rareword" not in words     # below threshold


def test_too_small_corpus_returns_empty(tmp_pmb_home, tmp_workspace_dir):
    eng = _eng(tmp_workspace_dir, tmp_pmb_home)
    eng.record_fact("only one document with widget in it here")
    assert compute_corpus_stopwords(eng, min_docs=50) == set()


def test_apply_excludes_from_distinctive_tokens():
    try:
        apply_corpus_stopwords({"widget"})
        assert "widget" not in distinctive_tokens("the widget sits here")
        assert "gadget" in distinctive_tokens("the gadget sits here")
    finally:
        apply_corpus_stopwords(set())   # reset — never leak into other tests


def test_refresh_caches_and_loads(tmp_pmb_home, tmp_workspace_dir):
    eng = _eng(tmp_workspace_dir, tmp_pmb_home)
    for i in range(52):   # past the default min_docs=50 small-corpus guard
        eng.record_fact(f"shared marker entry {i} widget payload data")
    try:
        out = refresh_corpus_stopwords(eng)
        assert out["n"] >= 1
        assert "widget" in load_corpus_stopwords(eng)
    finally:
        apply_corpus_stopwords(set())
