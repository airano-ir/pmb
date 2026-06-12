"""E1 — corpus-derived stopwords (document frequency), not a hand-written list.

A token that appears in more than `df_threshold` of THIS workspace's documents
carries almost no discriminative signal — that is exactly what BM25 encodes
implicitly via IDF. The explicit consumers (`distinctive_tokens`, lesson
relevance) never used it. This computes the high-document-frequency tokens once
in the maintenance tick, caches them per workspace, and `text_match` unions them
into its stopword set (gated, additive — the English floor stays inline as the
bootstrap for empty/new workspaces). No language data: it is pure statistics
over the user's own corpus, so it works in any language.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter

# 4+ letter Unicode word tokens (no digits/underscore) — same shape the
# distinctive tokenizer keeps, so the stopword set lines up with its vocabulary.
_WORD = re.compile(r"[^\W\d_]{4,}", re.UNICODE)


def compute_corpus_stopwords(engine, *, min_docs: int = 50,
                             df_threshold: float = 0.25,
                             cap: int = 200) -> set[str]:
    """Tokens whose document frequency exceeds `df_threshold` across the
    workspace's fact/qa/lesson events. Returns an empty set for a corpus too
    small (< min_docs) to estimate frequencies reliably."""
    docfreq: Counter = Counter()
    n_docs = 0
    try:
        with sqlite3.connect(str(engine.workspace.db_path)) as c:
            rows = c.execute(
                "SELECT content FROM events WHERE workspace_id = ? "
                "AND archived_at IS NULL AND event_type IN ('fact','qa','lesson')",
                (engine.workspace.id,)).fetchall()
    except Exception:
        return set()
    for (content,) in rows:
        n_docs += 1
        for t in {w.lower() for w in _WORD.findall(content or "")}:
            docfreq[t] += 1
    if n_docs < min_docs:
        return set()
    out = {t for t, df in docfreq.items() if df / n_docs > df_threshold}
    if len(out) > cap:   # keep only the most frequent if pathologically large
        out = {t for t, _ in docfreq.most_common(cap)
               if docfreq[t] / n_docs > df_threshold}
    return out


def _cache_path(engine):
    return engine.workspace.storage_dir / "corpus_stopwords.json"


def write_corpus_stopwords(engine, words) -> None:
    try:
        _cache_path(engine).write_text(json.dumps(sorted(words)), encoding="utf-8")
    except Exception:
        pass


def load_corpus_stopwords(engine) -> set[str]:
    try:
        return set(json.loads(_cache_path(engine).read_text(encoding="utf-8")))
    except Exception:
        return set()


def refresh_corpus_stopwords(engine) -> dict:
    """Recompute + cache; apply to the live tokenizer. Tick step (gated)."""
    words = compute_corpus_stopwords(engine)
    write_corpus_stopwords(engine, words)
    try:
        from pmb.core.text_match import apply_corpus_stopwords
        apply_corpus_stopwords(words)
    except Exception:
        pass
    return {"n": len(words)}
