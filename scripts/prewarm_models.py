#!/usr/bin/env python
"""Pre-download the HuggingFace models PMB's test suite loads.

CI runs the suite with ``HF_HUB_OFFLINE=1`` (``conftest.py`` forces it, so model
loads are deterministic and never hit the network mid-test). That relies on
``actions/cache`` providing the models. A cache MISS — most commonly a Dependabot
PR, which can't restore the default branch's cache — then makes every
model-dependent test fail with::

    OSError: We couldn't connect to 'https://huggingface.co' ... not found in cache

Running this first, with networking ON, downloads the models into the HF cache so
the offline suite finds them even on a cold cache. On a cache HIT it's a no-op
(the loads read straight from disk).

Keep the ids in sync with ``src/pmb/core/search.py`` and
``src/pmb/reasoning/images.py``.
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder, SentenceTransformer

# (id, source-of-truth) — comments point at the module that owns the default.
SENTENCE_TRANSFORMERS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # search.DEFAULT_MODEL
    "clip-ViT-B-32",                                                # reasoning/images.py
]
CROSS_ENCODERS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",                         # search.DEFAULT_RERANK_MODEL
]


def _warm(loader, name: str) -> None:
    """Load one model into the HF cache. Best-effort: a model that won't load on
    the installed transformers (e.g. clip-ViT-B-32 against a newer transformers
    that no longer recognizes its image-processor config) must NOT fail the whole
    CI step. The runtime path (reasoning/images.py) already degrades CLIP to None,
    and the multimodal tests are written to tolerate it being unavailable."""
    try:
        loader(name)
        print(f"prewarm: ok   {name}", flush=True)
    except Exception as e:
        print(f"prewarm: SKIP {name} ({type(e).__name__}: {e})", flush=True)


def main() -> None:
    for name in SENTENCE_TRANSFORMERS:
        _warm(SentenceTransformer, name)
    for name in CROSS_ENCODERS:
        _warm(CrossEncoder, name)
    print("prewarm: done", flush=True)


if __name__ == "__main__":
    main()
