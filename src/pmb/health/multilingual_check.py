"""
P1-2: detect English-only embedding model on workspace with non-Latin
content. Surfaces as a warning in `pmb doctor` and `pmb stats`.

Real-world failure mode (Alternix integrator report 2026-05-27):

    embedding.backend = fastembed
    embedding.fastembed_model = sentence-transformers/all-MiniLM-L6-v2
    workspace contents = ~90% Russian + Ukrainian

→ 8/10 correct top-1 on RU/UK queries with `all-MiniLM-L6-v2` (English-only).
→ 10/10 with `paraphrase-multilingual-MiniLM-L12-v2` (default in PMB).

The integrator overrode the default to "make startup fast" without
realising they were silently degrading retrieval quality for their
language. PMB ships the right default, but doesn't currently complain
when a worse override is in place AND the data is multilingual.

This module fixes that: cheap analysis (sample 200 recent events, count
non-ASCII letters), compare against the active model name.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

# Models that are explicitly English-only (or English-heavy).
# `all-MiniLM-L6-v2` is the canonical "fast English baseline" — covered.
_EN_ONLY_MODELS = {
    "all-minilm-l6-v2",
    "all-minilm-l12-v2",
    "all-mpnet-base-v2",
    "all-distilroberta-v1",
    "sentence-transformers/all-minilm-l6-v2",
    "sentence-transformers/all-minilm-l12-v2",
    "sentence-transformers/all-mpnet-base-v2",
}

# Models known to handle multiple languages well.
_MULTILINGUAL_MODELS = {
    "paraphrase-multilingual-minilm-l12-v2",
    "paraphrase-multilingual-mpnet-base-v2",
    "distiluse-base-multilingual-cased-v2",
    "sentence-transformers/paraphrase-multilingual-minilm-l12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
}


# Latin = U+0000-U+024F (basic latin + extended). Everything outside is
# treated as "non-Latin" and contributes to the multilingual signal.
_NON_LATIN_LETTER_RE = re.compile(r"[^\x00-ɏ\W\d_]")
_ANY_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def model_is_english_only(model_name: str | None) -> bool:
    if not model_name:
        return False
    return model_name.strip().lower() in _EN_ONLY_MODELS


def model_is_multilingual(model_name: str | None) -> bool:
    if not model_name:
        return False
    return model_name.strip().lower() in _MULTILINGUAL_MODELS


def sample_non_latin_ratio(
    db_path: Path,
    sample_size: int = 200,
) -> dict:
    """Return ratio of non-Latin letters across a sample of event contents.

    Returns:
        {
            "n_sampled": int,
            "n_non_latin_letters": int,
            "n_total_letters": int,
            "non_latin_ratio": float,   # 0.0 — 1.0
            "looks_multilingual": bool, # ratio >= 0.05
        }
    """
    n_sampled = 0
    n_non_latin = 0
    n_total = 0
    if not db_path.exists():
        return {"n_sampled": 0, "n_non_latin_letters": 0,
                "n_total_letters": 0, "non_latin_ratio": 0.0,
                "looks_multilingual": False}
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                "SELECT content FROM events "
                "WHERE archived_at IS NULL "
                "ORDER BY ulid DESC LIMIT ?",
                (sample_size,),
            ).fetchall()
        for (content,) in rows:
            if not content:
                continue
            n_sampled += 1
            n_non_latin += len(_NON_LATIN_LETTER_RE.findall(content))
            n_total += len(_ANY_LETTER_RE.findall(content))
    except Exception:
        pass
    ratio = (n_non_latin / n_total) if n_total else 0.0
    return {
        "n_sampled": n_sampled,
        "n_non_latin_letters": n_non_latin,
        "n_total_letters": n_total,
        "non_latin_ratio": round(ratio, 3),
        "looks_multilingual": ratio >= 0.05,
    }


def evaluate(
    db_path: Path,
    model_name: str | None,
    sample_size: int = 200,
) -> dict:
    """Combined check: sample data + compare against active model.

    Returns:
        {
            "warning": Optional[str],     # human-readable, None if all good
            "severity": "ok" | "warn" | "error",
            "model": str,
            "non_latin_ratio": float,
            "n_sampled": int,
            "looks_multilingual": bool,
            "model_is_english_only": bool,
            "model_is_multilingual": bool,
            "recommendation": Optional[str],
        }
    """
    stats = sample_non_latin_ratio(db_path, sample_size=sample_size)
    en_only = model_is_english_only(model_name)
    multi = model_is_multilingual(model_name)
    warning: str | None = None
    severity = "ok"
    recommendation: str | None = None

    if stats["looks_multilingual"] and en_only:
        severity = "warn"
        pct = int(stats["non_latin_ratio"] * 100)
        warning = (
            f"This workspace has ~{pct}% non-Latin characters but uses "
            f"`{model_name}` which is an English-only embedding model. "
            f"Retrieval quality on multilingual content will suffer."
        )
        recommendation = (
            "Switch to a multilingual model:\n"
            "  pmb config set embedding.model "
            "paraphrase-multilingual-MiniLM-L12-v2\n"
            "  pmb reindex   # re-embed events under the new model"
        )
    elif stats["looks_multilingual"] and not multi and not en_only:
        # Unknown model — can't say for sure.
        severity = "warn"
        warning = (
            f"This workspace has non-Latin content but the embedding model "
            f"`{model_name}` is not in the known-multilingual list. "
            f"Verify it actually handles your languages."
        )

    return {
        "warning": warning,
        "severity": severity,
        "model": model_name or "(unset)",
        "non_latin_ratio": stats["non_latin_ratio"],
        "n_sampled": stats["n_sampled"],
        "looks_multilingual": stats["looks_multilingual"],
        "model_is_english_only": en_only,
        "model_is_multilingual": multi,
        "recommendation": recommendation,
    }
