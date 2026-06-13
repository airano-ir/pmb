"""
Optional LLM-as-judge reranker for top-K candidates.

When ON, after the hybrid pipeline produces top-N, this module asks a
local LLM (Ollama-backed by default, small model like qwen2.5:0.5b /
phi3:mini / llama3.2:1b) to pick the single most-relevant candidate for
the given query.

Trade-offs:
  - +100-300ms per query (model load amortised, generation small).
  - +5-15pp top-1 on hard queries (cross-encoder ceiling).
  - 100% local - no API key, no cost.
  - Off by default - opt-in flag `recall.llm_rerank` per workspace.

Design choices:
  - Show LLM at most 10 candidates (typical context for small models).
  - Trim each candidate to 200 chars to fit a tight prompt.
  - Use deterministic, short prompt asking for an index integer.
  - On any error (model down, parse failure), return hits UNCHANGED.
    The pipeline degrades gracefully.

This is intentionally separate from the cross-encoder reranker so they
can be A/B compared and stacked (CE -> LLM is a valid order).
"""

from __future__ import annotations

import re
from typing import Any

# Recommended models for this task (good cost/quality):
#   - qwen2.5:0.5b      (~400MB)
#   - llama3.2:1b       (~1.3GB)
#   - phi3:mini         (~2.3GB)
#   - qwen2.5:1.5b      (~900MB)  <- default
DEFAULT_LLM_RERANK_MODEL = "qwen2.5:1.5b"


_PROMPT_TEMPLATE = """You pick the single most relevant memory for a question.

QUESTION: {query}

CANDIDATES:
{candidates}

Reply with ONLY the number of the most relevant candidate (1-{n}).
If multiple candidates seem relevant, pick the most specific one.
If none are relevant, reply with 0.

ANSWER:"""


_NUM_RE = re.compile(r"\b(\d+)\b")


def _trim(text: str, n: int = 200) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def llm_rerank_top_k(
    query: str,
    candidates: list[Any],            # list of items with `.content` attr (or dict 'content')
    llm_client,                       # any obj with .complete(prompt) -> str
    max_candidates: int = 10,
    max_candidate_chars: int = 200,
) -> int | None:
    """Ask the LLM to pick the best candidate. Returns the index of the
    new top-1 (within `candidates`), or None if the LLM didn't return a
    usable answer. Callers should treat None as "keep current order".
    """
    if not candidates or llm_client is None:
        return None
    cands = candidates[:max_candidates]

    def _content_of(c):
        if isinstance(c, dict):
            return c.get("content", "")
        return getattr(c, "content", "") or ""

    listing = "\n".join(
        f"{i+1}. {_trim(_content_of(c), max_candidate_chars)}"
        for i, c in enumerate(cands)
    )
    prompt = _PROMPT_TEMPLATE.format(
        query=query.strip(), candidates=listing, n=len(cands),
    )
    try:
        raw = llm_client.complete(prompt, max_tokens=8)
    except Exception:
        return None
    if not raw:
        return None
    m = _NUM_RE.search(raw)
    if not m:
        return None
    idx = int(m.group(1))
    if idx <= 0 or idx > len(cands):
        return None
    return idx - 1  # convert to 0-based
