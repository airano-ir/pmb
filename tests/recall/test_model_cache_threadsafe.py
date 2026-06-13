"""Global model caches must load each heavy model only once under concurrency."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from pmb.core import search


def _exercise_cache(monkeypatch, cache, loader_name):
    created: list[object] = []
    gate = threading.Barrier(8)

    class _FakeModel:
        pass

    def _loader():
        def _build(_name):
            time.sleep(0.03)
            model = _FakeModel()
            created.append(model)
            return model
        return _build

    monkeypatch.setattr(search, loader_name, _loader)
    cache._model = None
    cache._name = None
    if hasattr(cache, "_backend"):
        cache._backend = None
        cache._base_url = None

    def _get():
        gate.wait()
        return cache.get("one-model")

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            models = list(pool.map(lambda _: _get(), range(8)))
        assert len(created) == 1
        assert all(model is models[0] for model in models)
    finally:
        cache._model = None
        cache._name = None
        if hasattr(cache, "_backend"):
            cache._backend = None
            cache._base_url = None


def test_embedding_model_cache_loads_once(monkeypatch):
    _exercise_cache(monkeypatch, search._ModelCache, "_SentenceTransformer")


def test_cross_encoder_cache_loads_once(monkeypatch):
    _exercise_cache(monkeypatch, search._CrossEncoderCache, "_CrossEncoder")
