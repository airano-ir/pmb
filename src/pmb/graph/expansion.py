"""
LLM-based query expansion for graph traversal.

The graph helps when a query *mentions* the entity it's looking for. Abstract
queries like "describe our backend stack" never reach the graph because the
EntityExtractor only finds generic concepts ("backend", "stack"). This module
asks an LLM to rewrite the query into 1-5 concrete entity hints
("postgres", "redis", "pgbouncer") which the graph then traverses normally.

Why this is opt-in:
  - Adds ~one LLM call per recall when enabled. With `claude -p` that's
    25-50s - unusable for hot recall paths.
  - Quality not yet validated against real user queries.
  - For most explicit-entity queries the existing extractor is already
    enough and this would just add cost.

Caching:
  - We cache rewrites in <workspace>/query_expansion.jsonl keyed by query
    string. Repeat queries pay zero. The cache is invalidated only if the
    user deletes the file - there's no version key because the LLM prompt
    rarely changes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmb.health.consolidate import LLMClient


_PROMPT = """\
Extract 1-5 CONCRETE technology, file, or component names from this query.
Only include names that look like real things you would find in a codebase
(e.g. "postgres", "redis", "src/auth.py", "argon2", "kubernetes", "jwt").
Do NOT include generic words like "backend", "stack", "system", "service".

Output strict JSON only:
{
  "entities": ["name1", "name2", ...]
}

If the query has no concrete names, output {"entities": []}.

Query: """


def _cache_path(workspace_storage_dir: Path) -> Path:
    return workspace_storage_dir / "query_expansion.jsonl"


def _load_cache(workspace_storage_dir: Path) -> dict[str, list[str]]:
    p = _cache_path(workspace_storage_dir)
    if not p.exists():
        return {}
    out: dict[str, list[str]] = {}
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    out[rec["query"]] = list(rec.get("entities") or [])
                except Exception:
                    continue
    except Exception:
        return {}
    return out


def _append_cache(
    workspace_storage_dir: Path, query: str, entities: list[str]
) -> None:
    p = _cache_path(workspace_storage_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"query": query, "entities": entities, "ts": time.time()}
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def expand_query(
    query: str,
    workspace_storage_dir: Path,
    llm: LLMClient | None = None,
) -> list[str]:
    """
    Return a list of concrete-entity hints for `query`. Empty list means
    "no useful expansion found".

    The LLM is responsible for being conservative: it should output `[]`
    when the query has no concrete reference. This module just wraps the
    call and caches the result.
    """
    # Cache lookup
    cache = _load_cache(workspace_storage_dir)
    if query in cache:
        return cache[query]

    if llm is None:
        return []

    try:
        # LLMClient.consolidate(events_text) returns
        # {consolidate, summary, confidence, reasoning}. We don't use that
        # shape here - we want raw JSON. Send the prompt as a single
        # "events_text" item and parse the response ourselves.
        out = llm.consolidate([_PROMPT + query])
        # The shared parser already strips fences and extracts JSON, so
        # `summary` and `reasoning` are populated even when the model
        # returned our schema. Read the raw fields it might have stored:
        # we recover entities from `summary` if it looks JSON, else from
        # the consolidate prompt's primary parse path. As a fallback we
        # try to find a JSON object in `reasoning`.
        entities = _entities_from_consolidate_output(out)
    except Exception:
        entities = []

    # Cache even empty results - re-asking the LLM is expensive
    _append_cache(workspace_storage_dir, query, entities)
    return entities


def _entities_from_consolidate_output(out: dict) -> list[str]:
    """
    The shared LLM wrapper returns {consolidate, summary, confidence, reasoning}.
    Some backends will put our `entities` JSON in `summary`, others in the
    raw response text. Try a few places, return [] on failure.
    """
    candidates: list[str] = []
    for key in ("summary", "reasoning"):
        raw = out.get(key) if isinstance(out, dict) else None
        if not raw:
            continue
        # Look for "entities": [...] anywhere in the string
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                parsed = json.loads(raw[start : end + 1])
                ents = parsed.get("entities")
                if isinstance(ents, list):
                    for e in ents:
                        if isinstance(e, str) and e.strip():
                            candidates.append(e.strip().lower())
                    if candidates:
                        return candidates
        except Exception:
            continue
    return candidates
