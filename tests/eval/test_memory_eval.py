"""V1 — deterministic memory-quality mini-eval (recall regression gate).

Records a small FROZEN corpus of EN / RU / UK facts, then runs labelled
paraphrase queries through the FULL recall pipeline (BM25 + vector + PAMVR) and
asserts top-1 / top-3 hit-rate FLOORS per language bucket — including the
cross-lingual bucket (English question -> Russian fact) that the RU-recall path
exists to serve.

Why this gate matters: the real LoCoMo/PAMVR harnesses live in
`scripts/benchmarks/*` and never run in CI, so any normalize()/PAMVR/lang-pack
change could silently regress top-1 with green CI. This runs in the normal
suite. The embedder is real but DETERMINISTIC (same text -> same vector, no
randomness), and CI already loads it for other tests, so no new model download.

Floors are calibrated a few points BELOW the measured rates: a real regression
(e.g. a botched lang-pack relocation breaking RU recall) drops a bucket below
its floor and fails loudly, while normal embedder noise does not.
"""
from __future__ import annotations

import pytest

from pmb.core.engine import Engine

# ── frozen corpus: (lang, text) — index is the stable id ────────────────────
CORPUS: list[tuple[str, str]] = [
    ("en", "We use PostgreSQL as the primary production database."),      # 0
    ("en", "The staging API listens on port 8080."),                      # 1
    ("en", "Alice is the tech lead at Stripe."),                          # 2
    ("en", "We deploy the backend to AWS Fargate."),                      # 3
    ("en", "My birthday is on the 14th of March."),                       # 4
    ("ru", "Я живу в Киеве."),                                            # 5
    ("ru", "Мой любимый язык программирования — Rust."),                  # 6
    ("ru", "Мы используем Redis для кеширования."),                       # 7
    ("uk", "Мене звати Олег."),                                           # 8
    ("uk", "Я пишу тести за допомогою pytest."),                          # 9
]

# ── labelled queries: (bucket, query, expected_corpus_index) ────────────────
QUERIES: list[tuple[str, str, int]] = [
    # EN -> EN
    ("en", "what database do we run in production", 0),
    ("en", "which port is the staging api on", 1),
    ("en", "who leads the team at Stripe", 2),
    ("en", "where do we deploy the backend", 3),
    ("en", "when is my birthday", 4),
    # RU -> RU
    ("ru", "в каком городе я живу", 5),
    ("ru", "какой язык программирования мне нравится", 6),
    ("ru", "что мы используем для кеша", 7),
    # UK -> UK
    ("uk", "як мене звати", 8),
    ("uk", "чим я пишу тести", 9),
    # cross-lingual: EN question -> RU fact (the RU-recall bridge)
    ("xl", "where do I live", 5),
    ("xl", "what do we use for caching", 7),
    ("xl", "what is my favourite programming language", 6),
]

# Calibrated floors. Measured on 2026-06-10 (real deterministic embedder):
#   en top1/3 = 1.00/1.00 · ru = 1.00/1.00 · uk = 1.00/1.00 · xl = 0.33/1.00
#   overall top1/3 = 0.85/1.00
# Floors sit a margin BELOW measured so embedder noise doesn't flake CI but a
# real regression (e.g. a lang-pack relocation breaking RU recall would tank ru
# to ~0) fails loudly. Cross-lingual top-1 is inherently noisy (an in-language
# paraphrase can outrank the exact foreign fact), so it is NOT gated on top-1 —
# only that the foreign fact stays in top-3 (the RU-recall guarantee).
TOP1_FLOOR = {"en": 0.80, "ru": 0.66, "uk": 0.50, "xl": 0.00}
TOP3_FLOOR = {"en": 0.80, "ru": 0.66, "uk": 0.50, "xl": 0.66}
OVERALL_TOP1_FLOOR = 0.62
OVERALL_TOP3_FLOOR = 0.85


def _rank_of(eng, query: str, expected_ulid: str, top_k: int = 5):
    pack = eng.recall(query=query, top_k=top_k)
    for i, r in enumerate(pack.results):
        if getattr(r, "ulid", None) == expected_ulid:
            return i + 1            # 1-based rank
    return None


@pytest.mark.eval
def test_memory_quality_floors(tmp_pmb_home, tmp_workspace_dir, capsys):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    try:
        eng.warmup()
    except Exception:
        pass
    ulids = [eng.record_fact(text, metadata={"kind": "fact", "lang": lang})
             for lang, text in CORPUS]
    # Make sure every fact's vector is in LanceDB before we query (recall's
    # vector channel is empty otherwise and the eval would measure BM25 only).
    for drain in ("wait_for_embed_queue", "_drain_embed_queue"):
        try:
            getattr(eng, drain)()
        except Exception:
            pass

    buckets: dict[str, list[int | None]] = {"en": [], "ru": [], "uk": [], "xl": []}
    for bucket, query, idx in QUERIES:
        buckets[bucket].append(_rank_of(eng, query, ulids[idx]))

    lines, all_ranks = [], []
    for b, ranks in buckets.items():
        all_ranks += ranks
        n = len(ranks)
        t1 = sum(1 for r in ranks if r == 1) / n
        t3 = sum(1 for r in ranks if r and r <= 3) / n
        lines.append(f"  {b}: top1={t1:.2f} top3={t3:.2f} (n={n}) ranks={ranks}")
    o1 = sum(1 for r in all_ranks if r == 1) / len(all_ranks)
    o3 = sum(1 for r in all_ranks if r and r <= 3) / len(all_ranks)
    report = "memory-quality eval\n" + "\n".join(lines) + \
        f"\n  OVERALL: top1={o1:.2f} top3={o3:.2f} (n={len(all_ranks)})"
    with capsys.disabled():
        print("\n" + report)

    # Per-bucket + overall floors. Failure here = a real recall regression.
    for b, ranks in buckets.items():
        n = len(ranks)
        t1 = sum(1 for r in ranks if r == 1) / n
        t3 = sum(1 for r in ranks if r and r <= 3) / n
        assert t1 >= TOP1_FLOOR[b], f"{b} top1 {t1:.2f} < floor {TOP1_FLOOR[b]}\n{report}"
        assert t3 >= TOP3_FLOOR[b], f"{b} top3 {t3:.2f} < floor {TOP3_FLOOR[b]}\n{report}"
    assert o1 >= OVERALL_TOP1_FLOOR, f"overall top1 {o1:.2f} < {OVERALL_TOP1_FLOOR}\n{report}"
    assert o3 >= OVERALL_TOP3_FLOOR, f"overall top3 {o3:.2f} < {OVERALL_TOP3_FLOOR}\n{report}"
