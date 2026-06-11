"""
LRU cache for recall results — the "working memory" buffer in front of the
hybrid search. Two purposes:

  1. Speed. If the agent issues the same recall twice within the TTL we
     skip the full embed + LanceDB + BM25 + graph + rerank pipeline.
  2. Stability. Agents often repeat the same query verbatim across turns;
     returning the exact same result keeps the conversation coherent.

Cache invalidation:
  - Time-based: entries expire after `ttl_seconds`.
  - Write-based: every event write bumps a workspace `generation` counter;
    entries born under an older generation are stale and dropped on the
    next get(). This is cheap and correct: any change to the corpus
    invalidates everything (we don't try to be clever about which queries
    are affected).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class _Entry:
    value: Any
    born_at: float
    generation: int


class RecallCache:
    """Tiny LRU cache. Guarded by a lock (S9): the memory daemon serves recalls
    from a worker-thread pool and bumps the generation on writes, so get/put/
    bump_generation can interleave across threads — an unguarded OrderedDict
    raises `RuntimeError: OrderedDict mutated during iteration` or drops counts
    under that race. The lock is uncontended in the single-process CLI path."""

    def __init__(self, max_size: int = 128, ttl_seconds: float = 300.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._data: OrderedDict[str, _Entry] = OrderedDict()
        self._generation = 0
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self.max_size > 0

    def bump_generation(self) -> None:
        """Mark every existing entry stale (called after writes)."""
        with self._lock:
            self._generation += 1

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        with self._lock:
            ent = self._data.get(key)
            if ent is None:
                self.misses += 1
                return None
            if ent.generation != self._generation:
                del self._data[key]
                self.misses += 1
                return None
            if self.ttl_seconds > 0 and (time.time() - ent.born_at) > self.ttl_seconds:
                del self._data[key]
                self.misses += 1
                return None
            # LRU touch
            self._data.move_to_end(key)
            self.hits += 1
            return ent.value

    def put(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = _Entry(
                value=value, born_at=time.time(), generation=self._generation,
            )
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._data),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "generation": self._generation,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": (self.hits / total) if total else 0.0,
            }

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0


def make_recall_cache_key(
    query: str, top_k: int, recency_half_life_days: float,
    graph_boost: float, rerank: bool, rerank_top_n: int,
) -> str:
    """Stable key. We deliberately exclude time-of-day — TTL handles that."""
    return f"{query}|{top_k}|{recency_half_life_days}|{graph_boost}|{rerank}|{rerank_top_n}"
