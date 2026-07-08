"""Tests for pluggable embedder backends + the dimension-mismatch guard.

Uses a fake in-memory embedder (no network, no model download) injected via
_ModelCache so we can exercise arbitrary dimensions deterministically.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pmb.core import search as search_mod
from pmb.core.search import HybridSearch, _model_dim, _OllamaEmbedAdapter, _OpenAIEmbedAdapter


class FakeEmbedder:
    """Deterministic hash-based embedder of a chosen dimension."""

    def __init__(self, dim: int):
        self.dim = dim

    def encode(self, texts, show_progress_bar=False, batch_size: int = 32):
        single = isinstance(texts, str)
        inputs = [texts] if single else list(texts)
        out = np.zeros((len(inputs), self.dim), dtype=np.float32)
        for i, t in enumerate(inputs):
            for j, ch in enumerate(t.encode("utf-8")[: self.dim]):
                out[i, j] = (ch % 17) / 17.0
        return out[0] if single else out

    def get_sentence_embedding_dimension(self):
        return self.dim


class FakeEvent:
    def __init__(self, ulid: str, text: str):
        self.ulid = ulid
        self._text = text

    def to_text(self):
        return self._text


@pytest.fixture
def fresh(monkeypatch):
    """A HybridSearch on a temp dir wired to a fake embedder of a given dim."""
    def _make(dim: int):
        tmp = tempfile.mkdtemp()
        # force the model cache to hand back our fake embedder
        monkeypatch.setattr(
            search_mod._ModelCache, "get",
            classmethod(lambda cls, *a, **k: FakeEmbedder(dim)),
        )
        hs = HybridSearch(vector_path=Path(tmp) / "vectors.lance")
        return hs
    return _make


# ----------------------------------------------------------------------
# _model_dim helper
# ----------------------------------------------------------------------

def test_model_dim_from_method():
    assert _model_dim(FakeEmbedder(768)) == 768


def test_model_dim_probe_fallback():
    class NoMethod:
        def encode(self, texts, show_progress_bar=False, batch_size=32):
            arr = np.zeros((len(list(texts)), 512), dtype=np.float32)
            return arr
    assert _model_dim(NoMethod()) == 512


# ----------------------------------------------------------------------
# Dynamic table dim - fresh workspace honours the embedder's dim
# ----------------------------------------------------------------------

def test_fresh_workspace_uses_embedder_dim(fresh):
    hs = fresh(768)
    hs.add("u1", "the backend runs on Postgres")
    # table created with 768-dim because that's what the embedder produces
    assert hs._table_dim == 768


def test_roundtrip_nonstandard_dim(fresh):
    hs = fresh(256)
    hs.add("u1", "Alice leads the frontend team")
    hs.add("u2", "Bob owns the deploy pipeline")
    hs.build_bm25() if hasattr(hs, "build_bm25") else None
    hits = hs.search("who leads frontend", top_k=2)
    assert len(hits) >= 1  # recall works end-to-end at 256-dim


# ----------------------------------------------------------------------
# Dimension-mismatch guard - the corruption-prevention safety net
# ----------------------------------------------------------------------

def test_dim_mismatch_is_rejected(fresh, monkeypatch):
    hs = fresh(384)
    hs.add("u1", "first event at 384 dimensions")
    assert hs._table_dim == 384

    # now pretend the embedder switched to 768-dim mid-workspace
    monkeypatch.setattr(
        search_mod._ModelCache, "get",
        classmethod(lambda cls, *a, **k: FakeEmbedder(768)),
    )
    hs._model = None  # force re-fetch of the (now 768-dim) embedder
    with pytest.raises(RuntimeError) as exc:
        hs.add("u2", "this should be refused")
    assert "dimension mismatch" in str(exc.value).lower()


def test_reindex_recreates_table_when_embedder_dim_changes(fresh, monkeypatch):
    hs = fresh(384)
    hs.add("u1", "first event at 384 dimensions")
    assert hs._table_dim == 384

    monkeypatch.setattr(
        search_mod._ModelCache, "get",
        classmethod(lambda cls, *a, **k: FakeEmbedder(768)),
    )
    hs._model = None

    assert hs.reindex_all([FakeEvent("u1", "same event re-embedded")]) == 1
    assert hs._table_dim == 768

    hs.add("u2", "new event after reindex")


# ----------------------------------------------------------------------
# Adapter construction (no network calls - just shape/contract)
# ----------------------------------------------------------------------

def test_ollama_adapter_constructs():
    a = _OllamaEmbedAdapter("nomic-embed-text", "http://localhost:11434")
    assert a.model_name == "nomic-embed-text"
    assert a.base_url == "http://localhost:11434"
    assert a.get_sentence_embedding_dimension() is None  # not probed yet


def test_ollama_adapter_strips_trailing_slash():
    a = _OllamaEmbedAdapter("m", "http://host:11434/")
    assert a.base_url == "http://host:11434"


def test_openai_adapter_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    a = _OpenAIEmbedAdapter("text-embedding-3-small")
    with pytest.raises(RuntimeError) as exc:
        a.encode(["hello world"])
    assert "OPENAI_API_KEY" in str(exc.value)


def test_openai_adapter_empty_input_no_key_ok(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    a = _OpenAIEmbedAdapter("text-embedding-3-small")
    out = a.encode([])  # empty → no API call, no key needed
    assert out.shape[0] == 0
