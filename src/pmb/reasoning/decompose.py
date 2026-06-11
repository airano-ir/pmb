"""
Adaptive Query Decomposition for multi-hop questions.

Inspired by RAG-Fusion (arXiv 2402.03367) and IRCoT (arXiv 2212.10509):
when a question requires combining 2+ facts, break it into sub-questions,
retrieve each independently, then fuse the results via Reciprocal Rank
Fusion (RRF). This is the proven multi-hop unlock in the literature.

Where this fits in PMB:
  read-time only (NOT at write). Triggered selectively by `_looks_multihop()`
  in recall(). For simple lookups the original fast path runs unchanged
  (no LLM at query) — only multi-hop queries pay the decomposition cost.

Cost model:
  +1 LLM call per multi-hop query (~500ms - 3s depending on backend) +
  N additional hybrid searches (cheap, ~50-200ms each). Total budget:
  under 1s with Anthropic API, 2-5s with Claude CLI, ~500ms with a small
  local Ollama model.

Caching:
  Decompositions are cached on disk (queries are deterministic — same
  query → same decomposition). Makes repeat queries free.

Failure mode:
  If LLM fails / parsing fails, recall falls back to the original single
  query. Never blocks user.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------

DECOMPOSE_PROMPT = """\
A user wants the memory system to answer this question. The question
combines multiple facts (multi-hop). Break it into 2-3 atomic sub-questions
that, when answered, together reveal the original answer.

Rules:
- Each sub-question should be lookup-style (single fact, no chaining).
- Sub-questions should cover DIFFERENT facets — not paraphrases of each other.
- Output VALID JSON: a list of 2-3 short strings. Nothing else.
- If the question is actually single-hop and doesn't need decomposition,
  output a list with just one item: the original question.

Original question:
\"\"\"
{query}
\"\"\"

JSON list:"""


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

@dataclass
class Decomposition:
    original: str
    sub_queries: list[str] = field(default_factory=list)
    from_cache: bool = False


# ----------------------------------------------------------------------
# Decomposer
# ----------------------------------------------------------------------

class QueryDecomposer:
    """LLM-driven query splitter with on-disk cache."""

    def __init__(self, llm_client, cache_dir: Path | None = None):
        self.llm = llm_client
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_path = cache_dir / "query_decompositions.jsonl"
        else:
            self.cache_path = None
        self._cache: dict[str, list[str]] = {}
        self._load_cache()

    def decompose(self, query: str, max_sub: int = 3) -> Decomposition:
        if not query or not query.strip():
            return Decomposition(original=query, sub_queries=[query])
        key = self._cache_key(query)
        cached = self._cache.get(key)
        if cached is not None:
            return Decomposition(
                original=query, sub_queries=cached, from_cache=True,
            )

        try:
            text = self._call_llm(DECOMPOSE_PROMPT.format(query=query))
        except Exception as e:
            log.warning("Decomposer LLM call failed: %s", e)
            return Decomposition(original=query, sub_queries=[query])

        try:
            subs = self._parse_array(text)
            subs = [s.strip()[:300] for s in subs if isinstance(s, str) and s.strip()]
            subs = subs[:max_sub] or [query]
        except Exception as e:
            log.warning("Decomposer parse failed: %s | %s", e, text[:200])
            subs = [query]

        # Cache to memory + disk
        self._cache[key] = subs
        self._persist_cache_entry(key, query, subs)
        return Decomposition(original=query, sub_queries=subs)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cache_key(self, query: str) -> str:
        return hashlib.sha1(query.strip().lower().encode("utf-8")).hexdigest()[:16]

    def _load_cache(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        if row.get("subs"):
                            self._cache[row["key"]] = row["subs"]
                    except Exception:
                        continue
        except Exception as e:
            log.debug("Decomposer cache load failed: %s", e)

    def _persist_cache_entry(self, key: str, query: str, subs: list[str]) -> None:
        if not self.cache_path:
            return
        try:
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "key": key, "query": query, "subs": subs,
                    "t": time.time(),
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _call_llm(self, prompt: str) -> str:
        if hasattr(self.llm, "complete"):
            return self.llm.complete(prompt) or ""
        if callable(self.llm):
            return self.llm(prompt) or ""
        raise RuntimeError("Decomposer: llm must expose .complete() or be callable")

    def _parse_array(self, text: str) -> list:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rsplit("```", 1)[0].strip()
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("no JSON array in output")
        return json.loads(text[start : end + 1])


# ----------------------------------------------------------------------
# Result fusion via Reciprocal Rank Fusion (RRF)
# ----------------------------------------------------------------------

def reciprocal_rank_fuse(
    rankings: list[list[str]], k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. For each list of ranked ulids, compute
    score = 1 / (k + rank). Sum across lists. Return [(ulid, score), ...]
    sorted descending.

    k=60 is the standard default from Cormack et al. 2009.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, ulid in enumerate(ranking):
            scores[ulid] = scores.get(ulid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])
