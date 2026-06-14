#!/usr/bin/env python
"""Pre-download the HuggingFace models PMB's test suite loads.

CI runs the suite with ``HF_HUB_OFFLINE=1`` (``conftest.py`` forces it, so model
loads are deterministic and never hit the network mid-test). That relies on the
HF cache holding the models. A cache MISS — most commonly after a ``pyproject``
version bump rotates the cache key, or a Dependabot PR that can't restore the
default branch's cache — makes this step DOWNLOAD them. If that download hits a
transient HF blip and we swallow it, every model-dependent test then fails with::

    OSError: We couldn't connect to 'https://huggingface.co' ... not found in cache

…i.e. 80+ cryptic offline errors 15 minutes later instead of one clear signal.

So: retry each download a few times (transient 5xx / timeouts), and if the ONE
ESSENTIAL model (the text embedder the whole suite needs) still can't be cached,
FAIL THIS STEP LOUDLY (exit 1) — a fast, obvious "HF was unreachable, re-run"
instead of a confusing mass test failure. CLIP / the cross-encoder stay
best-effort: the runtime degrades them to None and their tests tolerate it.

On a cache HIT this is a no-op (loads read straight from disk).

Keep the ids in sync with ``src/pmb/core/search.py`` and
``src/pmb/reasoning/images.py``.
"""
from __future__ import annotations

import sys
import time

from sentence_transformers import CrossEncoder, SentenceTransformer

# The one model the whole suite depends on (search.DEFAULT_MODEL). If this can't
# be cached, the offline suite cannot run meaningfully — fail fast.
ESSENTIAL_EMBEDDER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# (id, source-of-truth) — comments point at the module that owns the default.
SENTENCE_TRANSFORMERS = [
    ESSENTIAL_EMBEDDER,                                             # search.DEFAULT_MODEL
    "clip-ViT-B-32",                                                # reasoning/images.py (optional)
]
CROSS_ENCODERS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",                         # search.DEFAULT_RERANK_MODEL (optional)
]

_ATTEMPTS = 3


def _warm(loader, name: str, *, essential: bool) -> bool:
    """Load one model into the HF cache, retrying transient network failures.

    Returns True if the model is cached (or is non-essential and may be skipped),
    False only when an ESSENTIAL model could not be cached after all retries.
    """
    last: Exception | None = None
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            loader(name)
            print(f"prewarm: ok   {name}", flush=True)
            return True
        except Exception as e:  # noqa: BLE001 - want to retry ANY load failure
            last = e
            print(f"prewarm: try {attempt}/{_ATTEMPTS} failed {name} "
                  f"({type(e).__name__}: {e})", flush=True)
            if attempt < _ATTEMPTS:
                time.sleep(min(2 ** attempt, 10))  # 2s, 4s, … capped
    if essential:
        print(f"prewarm: FATAL essential model could not be cached: {name} "
              f"({type(last).__name__}: {last})", flush=True)
        return False
    # Non-essential: the runtime path degrades it to None and its tests tolerate
    # that, so a miss must NOT fail CI.
    print(f"prewarm: SKIP {name} (non-essential; last "
          f"{type(last).__name__ if last else 'n/a'})", flush=True)
    return True


def main() -> None:
    ok = True
    for name in SENTENCE_TRANSFORMERS:
        ok = _warm(SentenceTransformer, name,
                   essential=(name == ESSENTIAL_EMBEDDER)) and ok
    for name in CROSS_ENCODERS:
        _warm(CrossEncoder, name, essential=False)
    print("prewarm: done", flush=True)
    if not ok:
        # Fail fast & clear instead of 80+ cryptic offline OSErrors downstream.
        sys.exit(1)


if __name__ == "__main__":
    main()
