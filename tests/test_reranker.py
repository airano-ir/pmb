"""Tests for the cross-encoder reranker — uses mocked CrossEncoder, no model load."""
from __future__ import annotations

from pmb.core.engine import Engine


class _StubCrossEncoder:
    """Predicts 1.0 for the pair whose document text contains the query
    substring, 0.0 otherwise. Lets us deterministically check reordering."""

    def __init__(self, *a, **kw):
        self.calls = 0

    def predict(self, pairs, show_progress_bar=False):
        self.calls += 1
        out = []
        for q, doc in pairs:
            out.append(1.0 if q.lower() in doc.lower() else 0.0)
        return out


def test_reranker_disabled_by_default(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    assert eng.search.reranker is None
    # rerank=True with no rerank_model wired -> still no reranker available
    eng.remember("Q", "Postgres on port 5432")
    pack = eng.recall("postgres", top_k=3, rerank=True)
    # Result still returns (no crash) even if reranker isn't configured
    assert len(pack.results) >= 1


def test_reranker_changes_top_when_enabled(
    tmp_pmb_home, tmp_workspace_dir, monkeypatch,
):
    """The stub reranker should reorder so the doc containing the literal
    query string surfaces above its peers."""
    from pmb.core import search as S
    monkeypatch.setattr(S, "_CrossEncoder", lambda: _StubCrossEncoder)

    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        rerank_model="cross-encoder/test-stub",
    )

    # Seed three events. The "alpha" string only appears in event B.
    u_a = eng.remember("Q1", "Postgres in production, alphabet soup")  # noise
    u_b = eng.remember("Q2", "Look here: alpha is the marker string")  # target
    u_c = eng.remember("Q3", "Tailwind CSS rules and alpha channel")   # tangent

    # Without rerank, hybrid scoring may already find B but not guaranteed
    # to be top — that's what the reranker is for. We just need to assert
    # that WITH rerank=True the stub's logic surfaces u_b in top-1.
    pack = eng.recall("alpha is the marker", top_k=3, rerank=True, graph_boost=0.0)
    assert pack.results, "expected at least one hit"
    assert pack.results[0].ulid == u_b


def test_reranker_falls_through_on_exception(
    tmp_pmb_home, tmp_workspace_dir, monkeypatch,
):
    """If the cross-encoder blows up, recall must still return the
    pre-rerank ordering rather than crashing the caller."""

    class _BadEncoder:
        def __init__(self, *a, **kw): pass
        def predict(self, pairs, show_progress_bar=False):
            raise RuntimeError("boom")

    from pmb.core import search as S
    monkeypatch.setattr(S, "_CrossEncoder", lambda: _BadEncoder)

    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        rerank_model="cross-encoder/broken",
    )
    eng.remember("Q1", "Postgres on port 5432")
    eng.remember("Q2", "Redis for rate limiting")
    pack = eng.recall("postgres", top_k=3, rerank=True)
    # No crash, results came back
    assert len(pack.results) >= 1


def test_reranker_single_hit_passthrough(
    tmp_pmb_home, tmp_workspace_dir, monkeypatch,
):
    """One hit -> nothing to reorder, reranker is not even called."""
    stub = _StubCrossEncoder()

    from pmb.core import search as S
    monkeypatch.setattr(S, "_CrossEncoder", lambda: (lambda *a, **kw: stub))

    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        rerank_model="cross-encoder/test-stub",
    )
    eng.remember("Q", "only one fact here")
    pack = eng.recall("only one fact", top_k=3, rerank=True, graph_boost=0.0)
    assert len(pack.results) == 1
    # Stub wasn't asked because there's nothing to reorder among 1 item
    assert stub.calls == 0


def test_rerank_flag_off_skips_model_load(
    tmp_pmb_home, tmp_workspace_dir, monkeypatch,
):
    """rerank=False must NOT trigger a CrossEncoder() construction."""
    loaded = {"n": 0}

    def _bad_factory():
        loaded["n"] += 1
        raise AssertionError("CrossEncoder should not have been imported")

    from pmb.core import search as S
    monkeypatch.setattr(S, "_CrossEncoder", _bad_factory)

    eng = Engine(
        cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
        rerank_model="cross-encoder/test",
    )
    eng.remember("Q1", "Postgres")
    eng.remember("Q2", "Redis")
    pack = eng.recall("postgres", top_k=3, rerank=False)
    assert len(pack.results) >= 1
    assert loaded["n"] == 0
