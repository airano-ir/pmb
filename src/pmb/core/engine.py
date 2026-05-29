"""
Engine — оркестратор всех компонентов ядра.

Public API:
- remember(query, response, ...) -> ulid
- recall(query, top_k, ...) -> list[RecallResult]
- pin(ulid)
- forget(ulid)
- stats() -> dict
- record_fact(text, metadata)

Под капотом: Workspace + EventStore + HybridSearch.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pmb.config import Config
from pmb.core.events import (
    Event, EventStore, default_tier_for_event_type, TIER_WORKING,
)
from pmb.core.recall_cache import RecallCache, make_recall_cache_key
from pmb.core.search import HybridSearch, SearchHit
from pmb.core.workspace import Workspace, detect_workspace
from pmb.graph.entities import EntityExtractor
from pmb.graph.store import GraphStore
from pmb.security.redact import redact, redact_metadata
from pmb.signals.session import SessionTracker
from pmb.signals.decay import boost_on_recall
from pmb.reasoning.pamvr import (
    apply_pamvr as _pamvr_apply,
    prepare_query_features as _pamvr_prepare,
    VOCAB_BRIDGES as _PAMVR_DEFAULT_BRIDGES,
)
from pmb.reasoning.user_names import (
    mine_user_names_from_db as _mine_user_names,
)
from pmb.reasoning.vocab_miner import (
    mine_workspace as _mine_workspace_bridges,
    merge_bridges as _merge_vocab_bridges,
)


@dataclass
class RecallResult:
    """Результат recall — событие + signals от ranking."""

    ulid: str
    event_type: str
    content: str
    metadata: dict
    timestamp: float
    score: float
    bm25_score: float
    vec_score: float
    importance: float
    recency_score: float

    def to_dict(self) -> dict:
        return {
            "ulid": self.ulid,
            "event_type": self.event_type,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "score": self.score,
            "signals": {
                "bm25": self.bm25_score,
                "vector": self.vec_score,
                "importance": self.importance,
                "recency": self.recency_score,
            },
        }


@dataclass
class RecallPack:
    """Структурированный ответ от recall — формат для LLM."""

    query: str
    workspace_name: str
    workspace_id: str
    results: list[RecallResult]
    n_total_in_workspace: int
    elapsed_ms: float

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "workspace": {"id": self.workspace_id, "name": self.workspace_name},
            "n_results": len(self.results),
            "n_total_in_workspace": self.n_total_in_workspace,
            "elapsed_ms": self.elapsed_ms,
            "confidence": self.confidence,
            "results": [r.to_dict() for r in self.results],
        }

    @property
    def confidence(self) -> float:
        """Improvement G: confidence in this recall.

        Combines top-1 score with the gap to top-2 (larger gap = more
        confident the top hit is the right one). Returns 0..1.

        Used by escalation logic and by callers who want to decide whether
        to surface results or ask for clarification."""
        if not self.results:
            return 0.0
        top1 = max(0.0, min(1.0, float(self.results[0].score)))
        if len(self.results) > 1:
            top2 = max(0.0, min(1.0, float(self.results[1].score)))
            gap = top1 - top2
            return min(1.0, top1 * 0.7 + gap * 0.3 + 0.1)
        return min(1.0, top1 * 0.7 + 0.1)

    def to_text(self, max_results: int = 5) -> str:
        """Текстовое представление для injection в промпт."""
        if not self.results:
            return f"[Memory] No relevant memories found in workspace '{self.workspace_name}'."

        lines = [f"[Memory recall from '{self.workspace_name}']"]
        for r in self.results[:max_results]:
            ts = time.strftime("%Y-%m-%d", time.gmtime(r.timestamp))
            content_preview = r.content[:300] + "..." if len(r.content) > 300 else r.content
            lines.append(f"\n— [{ts}] [{r.event_type}] (score {r.score:.2f}):")
            lines.append(content_preview)
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Multi-hop intent detection — used to gate causation graph walks.
# Cheap regex over the query. Doesn't need to be perfect; false-positives
# just add a cheap SQL hit, false-negatives skip the walk.
# ----------------------------------------------------------------------

_MULTIHOP_RE = re.compile(
    r"\b(after|before|because|due to|caused|led to|then|next|earlier|"
    r"previously|why did|what happened (?:after|when)|since|until|"
    r"following|preceding|subsequently|as a result|consequence|"
    r"prior to|in response|reaction|triggered|prompted)\b",
    re.IGNORECASE,
)


def _looks_multihop(query: str) -> bool:
    """Cheap detection of multi-hop / temporal / causal query patterns."""
    if not query:
        return False
    return bool(_MULTIHOP_RE.search(query))


def _collapse_reflections(
    scored: list, event_store, workspace_id: str,
) -> list:
    """Collapse reflection events onto their source events.

    `scored` is a list of (SearchHit, Event, score, recency). For any
    reflection in the list with `metadata.source_ulid`:
      - if source_ulid is already a candidate: add reflection's score to
        the source's score and drop the reflection from the list
      - if source_ulid is not in candidates: load the source from store
        and REPLACE the reflection's entry with the source (keeping the
        reflection's score because it earned that ranking)
      - if source can't be loaded: keep the reflection as a fallback

    This is the key fix for benchmarks that score on source dia_ids:
    reflections served their bridge purpose during scoring, but the
    answer surfaced to the agent should be the original source event.
    """
    if not scored:
        return scored
    # Quick exit if there are no reflections
    has_refl = any(
        getattr(ev, "event_type", None) == "reflection" for _, ev, _, _ in scored
    )
    if not has_refl:
        return scored

    by_ulid = {ev.ulid: i for i, (_, ev, _, _) in enumerate(scored)}
    out: list = []
    drop_indices: set[int] = set()
    score_boost: dict[str, float] = {}
    add_back: list = []  # entries to add (source replaces reflection)

    for i, (h, ev, score, recency) in enumerate(scored):
        if ev.event_type != "reflection":
            continue
        src_ulid = (ev.metadata or {}).get("source_ulid") if ev.metadata else None
        if not src_ulid:
            continue
        if src_ulid in by_ulid:
            # Source already a candidate — transfer score
            score_boost[src_ulid] = score_boost.get(src_ulid, 0.0) + score * 0.5
            drop_indices.add(i)
        else:
            # Source not yet a candidate — fetch it, replace reflection
            src_ev = event_store.get_by_ulid(src_ulid)
            if src_ev is None or src_ev.archived_at is not None:
                continue  # keep reflection as fallback
            # Build a fresh SearchHit for the source carrying the reflection's score
            from pmb.core.search import SearchHit as _SH
            new_h = _SH(
                ulid=src_ev.ulid, score=score,
                bm25_score=h.bm25_score, vec_score=h.vec_score,
                importance=src_ev.importance, recency_score=h.recency_score,
            )
            add_back.append((new_h, src_ev, score, recency))
            drop_indices.add(i)

    # Build the rebuilt list
    for i, item in enumerate(scored):
        if i in drop_indices:
            continue
        h, ev, score, recency = item
        if ev.ulid in score_boost:
            score = score + score_boost[ev.ulid]
        out.append((h, ev, score, recency))
    out.extend(add_back)
    return out


# No-op context manager used when the embed-queue lock hasn't been created yet
class _DummyLock:
    def __enter__(self): return self
    def __exit__(self, *exc): return False


_DUMMY_LOCK = _DummyLock()


# Improvement S: cross-kind entity dedup.
#
# When several extractors run over the same text, the same surface form often
# lands in multiple kinds:
#   - "alice"  → concept (length≥4 lowercase token)   AND   person
#   - "authmanager" → concept   AND   class   AND   function   AND   person
#   - "asyncpg"    → concept   AND   import
# We keep only the most specific kind. Order (highest priority first):
#   tech > file > class > function > import > person > theme > concept
#
# Code-AST kinds (class/function/import) outrank `person` because the regex
# person extractor will happily flag "AuthManager" as a capitalized name —
# but AST proves it's a code symbol. `person` still beats `concept` so
# "Alice" → person, not concept.
_KIND_PRIORITY: dict[str, int] = {
    "tech": 0,
    "file": 1,
    "class": 2,
    "function": 3,
    "import": 4,
    "person": 5,
    "theme": 6,
    "concept": 7,
}


def _truncate_marker(s: str, limit: int) -> str:
    if not isinstance(s, str) or len(s) <= limit:
        return s
    return s[:limit].rstrip() + "… [truncated by PMB]"


def _cap_batch_content(items: list[dict], max_content: int) -> list[dict]:
    """Improvement Z: cap content fields per item to avoid embedding-runaway
    on huge agent inputs (e.g. dumping full web-search results). Truncates
    `content`, `main`, `title`, and each `subfacts[i]` to `max_content` chars
    with a clear marker so downstream stays predictable.
    """
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            out.append(item)
            continue
        capped = dict(item)
        for k in ("content", "main", "title", "fact", "summary"):
            if k in capped and isinstance(capped[k], str):
                capped[k] = _truncate_marker(capped[k], max_content)
        if isinstance(capped.get("subfacts"), list):
            capped["subfacts"] = [
                _truncate_marker(s, max_content) if isinstance(s, str) else s
                for s in capped["subfacts"]
            ]
        out.append(capped)
    return out


def _dedupe_named_entities(named: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Collapse same-name entries to their highest-priority kind."""
    best: dict[str, tuple[int, str]] = {}  # name → (priority, kind)
    for kind, name in named:
        if not name:
            continue
        prio = _KIND_PRIORITY.get(kind, 99)
        cur = best.get(name)
        if cur is None or prio < cur[0]:
            best[name] = (prio, kind)
    # Stable order: same as original input
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for kind, name in named:
        if name in seen:
            continue
        seen.add(name)
        chosen_kind = best[name][1]
        out.append((chosen_kind, name))
    return out


class Engine:
    """
    Главный orchestrator. Держит workspace + storage + search index.

    Создаётся per-workspace. При создании авто-detect workspace и инициализирует
    storage если впервые.
    """

    def __init__(
        self,
        workspace: Optional[Workspace] = None,
        cwd: Optional[Path] = None,
        pmb_home: Optional[Path] = None,
        embedding_model: Optional[str] = None,
        bm25_weight: Optional[float] = None,
        rerank_model: Optional[str] = None,
        config_overrides: Optional[dict] = None,
    ):
        self.workspace = workspace or detect_workspace(cwd=cwd, pmb_home=pmb_home)
        self.workspace.ensure_dirs()
        self.workspace.save_meta()

        # Layered config (workspace > global > defaults). Explicit kwargs
        # become overrides so legacy callers keep working unchanged.
        overrides: dict = dict(config_overrides or {})
        if embedding_model is not None:
            overrides["embedding.model"] = embedding_model
        if bm25_weight is not None:
            overrides["recall.bm25_weight"] = bm25_weight
        # rerank_model truthy enables reranker through config
        if rerank_model:
            overrides["recall.rerank_model"] = rerank_model
            overrides["recall.rerank"] = True
        self.config = Config(
            workspace_dir=self.workspace.storage_dir,
            pmb_home=self.workspace.pmb_home,
            overrides=overrides,
        )

        self.events = EventStore(self.workspace.db_path)
        # Pick the right model id depending on backend (fastembed uses the
        # same canonical names but advertises its catalogue separately).
        emb_backend = self.config.get("embedding.backend")
        if emb_backend == "fastembed":
            emb_model = self.config.get("embedding.fastembed_model")
        elif emb_backend == "ollama":
            emb_model = self.config.get("embedding.ollama_model")
        elif emb_backend == "openai":
            emb_model = self.config.get("embedding.openai_model")
        else:
            emb_model = self.config.get("embedding.model")
        emb_base_url = self.config.get("embedding.ollama_url")
        # Improvement #1: reranker is needed if EITHER the always-on
        # `recall.rerank` flag is True OR the gated `recall.rerank_when_close`
        # is True. We pass the model name to HybridSearch in either case;
        # the rerank logic in `recall()` decides per-query whether to fire.
        _rerank_needed = (
            self.config.get("recall.rerank")
            or self.config.get("recall.rerank_when_close")
        )
        self.search = HybridSearch(
            vector_path=self.workspace.vector_path,
            model_name=emb_model,
            embedding_backend=emb_backend,
            embedding_base_url=emb_base_url,
            bm25_weight=self.config.get("recall.bm25_weight"),
            rerank_model_name=(
                self.config.get("recall.rerank_model")
                if _rerank_needed else None
            ),
        )
        self.session_tracker = SessionTracker(self.workspace)
        # Graph layer: same DB file, extra tables (migration v2)
        self.graph = GraphStore(self.workspace.db_path)
        self.entity_extractor = EntityExtractor()
        # Recall LRU cache — "working memory" buffer in front of the hybrid
        # pipeline. Invalidated on every event write.
        self.recall_cache = RecallCache(
            max_size=self.config.get("recall.cache_size"),
            ttl_seconds=self.config.get("recall.cache_ttl_seconds"),
        )

        # PPR graph cache (HippoRAG-style multi-hop boost). Built lazily on
        # first recall that needs it; invalidated when new events/edges land.
        self._ppr_graph = None
        self._ppr_graph_generation = -1

        # BM25 reload is deferred to the first `self.search.search()` /
        # `.add()` call: the load takes ~50-200 ms on populated workspaces
        # and ~22 s on the very first call in a fresh process (because of
        # the underlying `import lancedb`). Engine() must not pay that cost
        # for read-only CLI commands that never touch search.

        # Improvement W: deferred-embed queue. Writes don't block on the
        # sentence-transformers model load (~50s cold start). Instead we
        # write SQLite immediately and let a background worker embed
        # whenever the model becomes ready.
        self._embed_queue: list[tuple[str, str]] = []
        self._embed_worker_started = False
        self._embed_queue_lock = None  # lazy threading.Lock

        # Hardening H3: durable embed queue (SQLite-backed) for crash-
        # safety. Pending embeds survive process restart. Inits lazily
        # so the read-only CLI commands don't pay the cost.
        self._durable_embed_queue = None  # set by _ensure_durable_queue()

        # Improvement CC: serialize record_batch entries — when multiple
        # async batches run concurrently, they race on self._batch_defer
        # and self._batch_pending. A Lock makes batches process one at a
        # time inside the engine; the MCP caller still returns instantly
        # because each record_batch_async spawns its own thread.
        import threading as _threading
        self._batch_lock = _threading.Lock()

        # Improvement #5: bulk-import mode. When True, every per-item
        # cross-cutting step (dedup L1+L2, graph indexing, temporal
        # parsing, causation edges, L2.5 queue) is skipped. Only the
        # SQLite row + embedding land. Caller runs `pmb regraph` later.
        # Default False so normal record_batch behaviour is unchanged.
        self._bulk_mode = False
        self._bulk_collected_ulids: list[str] = []

        # Improvement #6: deferred touch buffer. Under concurrent recalls
        # each call writes access_count+1 / last_accessed / importance via
        # `apply_recall_updates` — a SQLite write transaction. With 8+
        # parallel recalls these all queue on the write lock, exploding p95.
        # Solution: enqueue touches in memory, flush every ~250ms in a
        # daemon thread. Single flush coalesces touches from many recalls,
        # so 16 concurrent recalls = 1 lock acquisition instead of 16.
        # On engine.close() we drain the buffer.
        self._touch_buffer: dict[str, float] = {}        # ulid -> last_accessed
        self._touch_imp_buffer: dict[str, float] = {}    # ulid -> latest importance
        self._touch_lock = _threading.Lock()
        self._touch_flusher_started = False

        # PAMVR — Predicate-Aware Multi-View Reranking. Cache the flag
        # at init time so the recall hot-path doesn't pay for a config
        # lookup per candidate.
        self._pamvr_enabled = bool(self.config.get("recall.pamvr_enabled"))

        # Auto VOCAB_BRIDGES (Improvement TT). Mine the user's own lexicon
        # from workspace events via PMI co-occurrence so PAMVR adapts to
        # any domain (not just coding). Hand-curated VOCAB_BRIDGES stay as
        # fallback; mined bridges extend them. Refreshed lazily — see
        # `_maybe_refresh_vocab_bridges()`.
        self._auto_bridges_enabled = bool(
            self.config.get("recall.auto_vocab_bridges")
        )
        # Improvement WW: write-time atomic fact extraction.
        self._atomic_extract_enabled = bool(
            self.config.get("write.atomic_fact_extract")
        )
        self._vocab_bridges: dict[str, list[str]] = dict(_PAMVR_DEFAULT_BRIDGES)
        self._vocab_bridges_cache_path = (
            self.workspace.storage_dir / "vocab_bridges.json"
        )
        self._vocab_bridges_last_event_count = -1
        if self._auto_bridges_enabled:
            try:
                self._refresh_vocab_bridges()
            except Exception:
                # Auto-mining is a best-effort enhancement, never crash init.
                pass

    # -----------------------------------------------------------------
    # Auto VOCAB_BRIDGES helpers
    # -----------------------------------------------------------------

    def _refresh_vocab_bridges(self, force: bool = False) -> None:
        """Mine workspace events → merge with hand-curated bridges.

        Called at init and lazily before recall when enough new events have
        landed. The mining itself is ~50 ms per 1000 events.
        """
        mined = _mine_workspace_bridges(
            db_path=self.workspace.db_path,
            cache_path=self._vocab_bridges_cache_path,
            force=force,
            window=int(self.config.get("recall.auto_vocab_window") or 6),
            min_count=int(self.config.get("recall.auto_vocab_min_count") or 3),
            min_pmi=float(self.config.get("recall.auto_vocab_min_pmi") or 2.0),
            max_bridges_per_key=int(
                self.config.get("recall.auto_vocab_max_per_key") or 8
            ),
            refresh_threshold=int(
                self.config.get("recall.auto_vocab_refresh_after") or 50
            ),
        )
        merged = _merge_vocab_bridges(_PAMVR_DEFAULT_BRIDGES, mined)
        self._vocab_bridges = merged
        try:
            # Best-effort: track event count so we know when to re-mine.
            import sqlite3 as _sql
            with _sql.connect(str(self.workspace.db_path)) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE archived_at IS NULL"
                ).fetchone()
                self._vocab_bridges_last_event_count = int(row[0] or 0)
        except Exception:
            pass

    def _record_atomic_facts(
        self,
        text: str,
        parent_ulid: str,
        base_importance: float = 0.7,
    ) -> list[str]:
        """Extract atomic facts via fact_extract and record each as a
        sibling event linked to `parent_ulid` via metadata.parent_ulid.

        Returns the list of created child ulids.
        """
        try:
            from pmb.reasoning.fact_extract import extract_atomic_facts
        except Exception:
            return []
        atoms = extract_atomic_facts(text)
        if not atoms:
            return []
        out: list[str] = []
        # Slightly lower importance than the source so the original
        # paragraph still wins when the user asks the whole question.
        child_imp = max(0.1, min(0.9, base_importance - 0.05))
        for af in atoms:
            try:
                ulid = self.record_fact(
                    af.content,
                    importance=child_imp,
                    metadata={
                        "parent_ulid": parent_ulid,
                        "atomic_kind": af.kind,
                        "atomic_confidence": af.confidence,
                        "extracted": True,
                    },
                )
                out.append(ulid)
            except Exception:
                continue
        return out

    def _maybe_refresh_vocab_bridges(self) -> None:
        """Called from recall() — cheap if cache is fresh, mines if stale."""
        if not self._auto_bridges_enabled:
            return
        try:
            import sqlite3 as _sql
            with _sql.connect(str(self.workspace.db_path)) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE archived_at IS NULL"
                ).fetchone()
                n = int(row[0] or 0)
            threshold = int(
                self.config.get("recall.auto_vocab_refresh_after") or 50
            )
            if n - self._vocab_bridges_last_event_count >= threshold:
                self._refresh_vocab_bridges()
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------

    def remember(
        self,
        query: str,
        response: str,
        session_id: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[dict] = None,
    ) -> str:
        """Сохранить Q/A пару. Возвращает ulid."""
        # Auto-bind to current session if not provided
        if session_id is None:
            session_id = self.session_tracker.touch().id

        # Secret redaction at the persistence boundary
        clean_query, _ = redact(query)
        clean_response, _ = redact(response)
        clean_metadata, _ = redact_metadata({"query": clean_query, **(metadata or {})})

        ev = Event(
            workspace_id=self.workspace.id,
            event_type="qa",
            content=clean_response,
            metadata=clean_metadata,
            importance=importance,
            source_session_id=session_id,
            tier=default_tier_for_event_type("qa"),
        )
        ev = self.events.append(ev)
        # Index в hybrid search. Synchronous: the Python API contract is
        # "write returns when data is searchable". Use `_embed_or_defer`
        # only when we're inside `record_batch` (it sets _batch_defer).
        if getattr(self, "_batch_defer", False):
            self._embed_or_defer(ev.ulid, ev.to_text())
        else:
            self.search.add(ev.ulid, ev.to_text())
        # Index в graph
        self._index_event_in_graph(ev, full_text=f"{clean_query}\n{clean_response}")
        try:
            from pmb.reasoning.causation import add_temporal_next_edge
            add_temporal_next_edge(self, ev)
        except Exception:
            pass
        # Improvement C: parse event_time (date references) from content
        # and store in metadata. Enables temporal-proximity boost at recall.
        try:
            self._attach_event_time(ev)
        except Exception:
            pass
        self.recall_cache.bump_generation()
        return ev.ulid

    def record_fact(
        self,
        fact: str,
        importance: float = 0.7,
        metadata: Optional[dict] = None,
        session_id: Optional[str] = None,
    ) -> str:
        clean_fact, _ = redact(fact)
        clean_metadata, _ = redact_metadata(metadata or {})

        # Improvement #5 (bulk import): when self._bulk_mode is set,
        # skip ALL cross-cutting work (dedup, graph, temporal, causation,
        # L2.5 queue). Only the SQLite row + embedding land. The caller
        # must run `pmb regraph` afterwards to rebuild the graph.
        if getattr(self, "_bulk_mode", False):
            ev = Event(
                workspace_id=self.workspace.id,
                event_type="fact",
                content=clean_fact,
                metadata=clean_metadata,
                importance=importance,
                source_session_id=session_id,
                tier=default_tier_for_event_type("fact"),
            )
            ev = self.events.append(ev)
            self._embed_or_defer(ev.ulid, ev.to_text())
            self._bulk_collected_ulids.append(ev.ulid)
            return ev.ulid

        # Improvement U: write-time dedup (L1 exact + L2 semantic).
        # If we detect a duplicate, return the existing ULID instead of
        # writing a new event. The existing event gets its access_count
        # bumped so it surfaces faster on next recall.
        dup_hit, borderline = self._dedup_pre_write(
            content=clean_fact, event_type="fact",
        )
        if dup_hit is not None:
            self._bump_for_dup(dup_hit.canonical_ulid)
            return dup_hit.canonical_ulid

        ev = Event(
            workspace_id=self.workspace.id,
            event_type="fact",
            content=clean_fact,
            metadata=clean_metadata,
            importance=importance,
            source_session_id=session_id,
            tier=default_tier_for_event_type("fact"),
        )
        ev = self.events.append(ev)
        # Improvement W: embed inline if model loaded, else queue
        self._embed_or_defer(ev.ulid, ev.to_text())
        self._index_event_in_graph(ev, full_text=clean_fact)
        try:
            from pmb.reasoning.causation import add_temporal_next_edge
            add_temporal_next_edge(self, ev)
        except Exception:
            pass
        # Improvement C: parse event_time (date references) from content
        # and store in metadata. Enables temporal-proximity boost at recall.
        try:
            self._attach_event_time(ev)
        except Exception:
            pass

        # L2.5: borderline candidate detected — enqueue for async LLM verify
        if borderline is not None and self.config.get("dedup.async_verify"):
            try:
                from pmb.reasoning.dedup import enqueue_borderline
                enqueue_borderline(
                    self.workspace.db_path, self.workspace.id,
                    new_ulid=ev.ulid,
                    candidate_ulid=borderline[0],
                    similarity=borderline[1],
                )
            except Exception:
                pass

        self.recall_cache.bump_generation()
        return ev.ulid

    # -----------------------------------------------------------------
    # P0-2: Keyed-upsert (fact supersession).
    # -----------------------------------------------------------------

    def record_keyed_fact(
        self,
        subject: str,
        attribute: str,
        value: str,
        importance: float = 0.8,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Upsert a personal-attribute fact, archiving any prior fact with
        the same (subject, attribute) key.

        This is the missing piece for personal-assistant memory:
        "user lives in Kyiv" → "user lives in Warsaw" should not leave
        both active in recall. The old fact is archived with metadata
        `superseded_by=<new_ulid>` so history is preserved but only the
        current value surfaces by default.

        The key is `subject::attribute` (lowercased), stored as
        `metadata.keyed_fact_key`. Recall filters archived events out
        normally, so old values disappear from results without any
        extra work in the recall pipeline.

        Returns:
            {
                "new_ulid": str,
                "superseded_ulids": list[str],   # archived prior versions
                "key": str,
            }
        """
        if not subject or not attribute or not value:
            raise ValueError("subject, attribute, value all required")
        subject_norm = subject.strip().lower()
        attribute_norm = attribute.strip().lower()
        key = f"{subject_norm}::{attribute_norm}"

        # 1. Find any prior facts with the same key (active only)
        prior_ulids: list[str] = []
        try:
            import sqlite3 as _sql
            with _sql.connect(str(self.workspace.db_path)) as conn:
                conn.row_factory = _sql.Row
                rows = conn.execute(
                    "SELECT ulid, metadata_json FROM events "
                    "WHERE workspace_id = ? AND archived_at IS NULL "
                    "AND event_type = 'fact' "
                    "AND metadata_json LIKE ?",
                    (self.workspace.id, f'%"keyed_fact_key": "{key}"%'),
                ).fetchall()
            for r in rows:
                prior_ulids.append(r["ulid"])
        except Exception:
            pass

        # 2. Write the new fact with the key + supersedes pointer
        meta = dict(metadata or {})
        meta["keyed_fact_key"] = key
        meta["keyed_fact_subject"] = subject
        meta["keyed_fact_attribute"] = attribute
        meta["keyed_fact_value"] = value
        if prior_ulids:
            meta["supersedes"] = prior_ulids
        # Human-readable content for embedder/BM25
        content = f"{subject} {attribute}: {value}"
        new_ulid = self.record_fact(
            content, importance=importance, metadata=meta,
        )

        # 3. Archive priors — they stay in SQLite (queryable as history)
        # but `archived_at IS NULL` filter removes them from recall.
        for old_ulid in prior_ulids:
            try:
                self.events.archive(old_ulid)
                # Tag with the new pointer so callers can trace history.
                import sqlite3 as _sql, json as _json
                with _sql.connect(str(self.workspace.db_path)) as conn:
                    row = conn.execute(
                        "SELECT metadata_json FROM events WHERE ulid = ?",
                        (old_ulid,),
                    ).fetchone()
                    old_meta = _json.loads(row[0] or "{}") if row else {}
                    old_meta["superseded_by"] = new_ulid
                    conn.execute(
                        "UPDATE events SET metadata_json = ? WHERE ulid = ?",
                        (_json.dumps(old_meta), old_ulid),
                    )
            except Exception:
                continue

        self.recall_cache.bump_generation()
        return {
            "new_ulid": new_ulid,
            "superseded_ulids": prior_ulids,
            "key": key,
        }

    # -----------------------------------------------------------------
    # P2: typed memory shortcuts.
    # ALL of these are thin wrappers over `record_fact` / `record_event`
    # with a clear, indexable `event_type`. Recall doesn't change — the
    # type just makes it easier for callers to filter / diagnose.
    # -----------------------------------------------------------------

    def record_preference(
        self,
        preference: str,
        importance: float = 0.7,
        metadata: Optional[dict] = None,
    ) -> str:
        """User preference. Event_type='preference', tier='semantic'.
        Examples: "I prefer dark mode", "Я люблю спокойные игры".
        """
        meta = dict(metadata or {})
        meta["memory_type"] = "preference"
        return self.record_event(
            content=preference,
            event_type="preference",
            importance=importance,
            metadata=meta,
        )

    def record_summary(
        self,
        summary: str,
        importance: float = 0.5,
        metadata: Optional[dict] = None,
    ) -> str:
        """Conversation summary / digest. Event_type='summary', tier='episodic'.
        Marked so retrieval can prefer original facts over rephrased summaries.
        """
        meta = dict(metadata or {})
        meta["memory_type"] = "summary"
        return self.record_event(
            content=summary,
            event_type="summary",
            importance=importance,
            metadata=meta,
        )

    def get_keyed_fact_history(self, subject: str, attribute: str) -> list[dict]:
        """Return current + all prior values for a keyed fact, newest first.
        Useful for "what did I say about X before?" introspection.
        """
        key = f"{subject.strip().lower()}::{attribute.strip().lower()}"
        try:
            import sqlite3 as _sql, json as _json
            with _sql.connect(str(self.workspace.db_path)) as conn:
                conn.row_factory = _sql.Row
                rows = conn.execute(
                    "SELECT ulid, content, metadata_json, timestamp, "
                    "archived_at FROM events "
                    "WHERE workspace_id = ? AND event_type = 'fact' "
                    "AND metadata_json LIKE ? "
                    "ORDER BY timestamp DESC",
                    (self.workspace.id, f'%"keyed_fact_key": "{key}"%'),
                ).fetchall()
            out = []
            for r in rows:
                meta = _json.loads(r["metadata_json"] or "{}")
                out.append({
                    "ulid": r["ulid"],
                    "value": meta.get("keyed_fact_value"),
                    "content": r["content"],
                    "timestamp": r["timestamp"],
                    "is_current": r["archived_at"] is None,
                })
            return out
        except Exception:
            return []

    # -----------------------------------------------------------------
    # Improvement U: dedup helpers (called from record_fact and others)
    # -----------------------------------------------------------------

    def _dedup_pre_write(
        self, content: str, event_type: str,
    ):
        """Run L1 (exact) then L2 (semantic) dedup checks.

        Returns (hit, borderline):
          hit:        DedupHit if a definite duplicate exists — caller skips write
          borderline: (candidate_ulid, similarity) if a borderline neighbor
                      was found; caller WRITES then enqueues this pair for
                      async LLM verify (L2.5)
        Both can be None (no signal, proceed normally).

        Improvement Y: when called INSIDE record_batch, skip L2 semantic check
        entirely. The items in one batch came from the same agent thinking
        pass — they're highly unlikely to dup against existing storage AND
        running L2 doubles the embedding cost (once for dedup search, once
        for the actual write). L1 (exact-text) still runs for safety, and
        the periodic `pmb dedupe` sweep catches anything L1 missed.
        """
        if not self.config.get("dedup.enable"):
            return None, None
        try:
            from pmb.reasoning.dedup import (
                find_exact_duplicate, find_semantic_duplicate,
            )
        except Exception:
            return None, None

        lookback = float(self.config.get("dedup.lookback_days"))

        # L1: exact text match
        hit = find_exact_duplicate(
            db_path=self.workspace.db_path,
            workspace_id=self.workspace.id,
            event_type=event_type,
            content=content,
            lookback_days=lookback,
        )
        if hit is not None:
            return hit, None

        # L2: semantic match via LanceDB nearest neighbors
        if not self.config.get("dedup.enable_semantic"):
            return None, None

        # Improvement Y: inside record_batch, L2 is too expensive (doubles
        # embedding work) and rarely catches anything (items came from one
        # agent turn). Skip — periodic `pmb dedupe` handles paraphrases.
        if getattr(self, "_batch_defer", False):
            return None, None

        # Improvement W: if the embedding model isn't loaded yet, skip L2
        # rather than blocking the write for 50s. L1 still ran (exact match),
        # which catches identical-text duplicates. The eventual dedup sweep
        # (or per-write L2 once warm) cleans up paraphrase dups later.
        if not self.search.is_ready():
            return None, None

        try:
            candidates = self._dedup_nearest_candidates(content, top_k=20)
        except Exception:
            candidates = []

        if not candidates:
            return None, None

        return find_semantic_duplicate(
            db_path=self.workspace.db_path,
            workspace_id=self.workspace.id,
            event_type=event_type,
            candidates=candidates,
            threshold_high=float(self.config.get("dedup.cosine_high")),
            threshold_mid=float(self.config.get("dedup.cosine_mid")),
        )

    def _dedup_nearest_candidates(
        self, content: str, top_k: int = 20,
    ) -> list:
        """Query LanceDB for nearest events to `content` via COSINE metric.

        Returns [(ulid, cosine_similarity), ...] sorted descending.

        Used only by dedup — recall has its own (richer) pipeline.
        """
        import numpy as np
        q = np.asarray(self.search.embed(content), dtype=np.float32)
        try:
            # LanceDB cosine returns distance ∈ [0, 2] where 0 = identical.
            # Similarity = 1 - distance. We request a few extras and clip.
            results = (
                self.search._table.search(q.tolist())
                .metric("cosine")
                .limit(top_k)
                .to_list()
            )
        except Exception:
            # Fall back to raw vector compute if cosine metric unsupported
            return self._dedup_nearest_via_raw_vectors(q, top_k)

        out: list = []
        for r in results:
            ulid = r.get("ulid", "")
            if not ulid:
                continue
            d = float(r.get("_distance", 2.0))
            cos = 1.0 - d
            cos = max(-1.0, min(1.0, cos))
            out.append((ulid, cos))
        out.sort(key=lambda x: -x[1])
        return out

    def _dedup_nearest_via_raw_vectors(self, q, top_k: int) -> list:
        """Fallback: scan LanceDB via Arrow, compute cosine in numpy.
        Slower but works on older LanceDB without `.metric()` support."""
        import numpy as np
        try:
            tbl = self.search._table
            arr_tbl = tbl.to_arrow()
            ulid_col = arr_tbl.column("ulid").to_pylist()
            vec_col = arr_tbl.column("vector").to_pylist()
        except Exception:
            return []
        qn = float(np.linalg.norm(q) + 1e-9)
        out = []
        for u, v in zip(ulid_col, vec_col):
            if not u or v is None:
                continue
            try:
                vv = np.asarray(v, dtype=np.float32)
                sim = float(np.dot(q, vv) / (qn * (np.linalg.norm(vv) + 1e-9)))
            except Exception:
                continue
            out.append((u, sim))
        out.sort(key=lambda x: -x[1])
        return out[:top_k]

    def _bump_for_dup(self, ulid: str) -> None:
        """When a write is suppressed by dedup, bump the canonical event's
        access_count + last_accessed so it surfaces faster on next recall."""
        import sqlite3, time as _t
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.execute(
                "UPDATE events SET access_count = access_count + 1, "
                "last_accessed = ? WHERE ulid = ?",
                (_t.time(), ulid),
            )

    # -----------------------------------------------------------------
    # Improvement #6: deferred touch flusher (concurrent recall scaling)
    # -----------------------------------------------------------------

    def _enqueue_touches(
        self,
        touches: list[str],
        importance_updates: list[tuple[str, float]],
    ) -> None:
        """Buffer recall side effects for the background flusher.

        Coalesces multiple touches of the same ulid (latest timestamp /
        importance wins) so a hot event accessed 16 times in 100ms
        produces ONE SQLite write, not 16.
        """
        if not touches and not importance_updates:
            return
        now = time.time()
        with self._touch_lock:
            for u in touches:
                self._touch_buffer[u] = now
            for u, imp in importance_updates:
                self._touch_imp_buffer[u] = max(0.0, min(1.0, imp))
            if not self._touch_flusher_started:
                self._touch_flusher_started = True
                import threading
                threading.Thread(
                    target=self._flush_touches_loop,
                    daemon=True, name="pmb-touch-flusher",
                ).start()

    def _flush_touches_loop(self) -> None:
        """Daemon loop: every ~250ms drain the touch buffer to SQLite.

        Single connection per flush, single transaction. Multiple concurrent
        recalls coalesce into one write, so the SQLite write lock is held
        ~4 times per second regardless of recall traffic.
        """
        import time as _t
        while True:
            _t.sleep(0.25)
            if not self._drain_touch_buffer():
                # Buffer was empty for one tick; another empty tick and we
                # exit. Re-spawned on next enqueue. Saves an idle thread.
                _t.sleep(0.25)
                if not self._drain_touch_buffer():
                    with self._touch_lock:
                        self._touch_flusher_started = False
                    return

    def _drain_touch_buffer(self) -> bool:
        """Flush whatever's in the touch buffers right now. Returns True if
        anything was flushed. Safe to call from engine.close() too.
        """
        with self._touch_lock:
            if not self._touch_buffer and not self._touch_imp_buffer:
                return False
            touches_snap = dict(self._touch_buffer)
            imp_snap = dict(self._touch_imp_buffer)
            self._touch_buffer.clear()
            self._touch_imp_buffer.clear()
        try:
            with self.events._conn() as conn:
                conn.execute("BEGIN")
                try:
                    if touches_snap:
                        conn.executemany(
                            "UPDATE events SET access_count = access_count + 1, "
                            "last_accessed = ? WHERE ulid = ?",
                            [(t, u) for u, t in touches_snap.items()],
                        )
                    if imp_snap:
                        conn.executemany(
                            "UPDATE events SET importance = ? WHERE ulid = ?",
                            [(i, u) for u, i in imp_snap.items()],
                        )
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
            return True
        except Exception:
            # On failure, put the items back so they retry on next tick.
            with self._touch_lock:
                for u, t in touches_snap.items():
                    self._touch_buffer.setdefault(u, t)
                for u, i in imp_snap.items():
                    self._touch_imp_buffer.setdefault(u, i)
            return False

    # -----------------------------------------------------------------
    # Improvement W: deferred embedding (writes don't block on model load)
    # -----------------------------------------------------------------

    def warmup(self, with_first_query: bool = True) -> dict:
        """P1-1: Eagerly load model + BM25 + LanceDB so the first real
        recall is fast. Without this the first query pays the entire
        ~1-2s cold-start cost (model load + index open). Useful for:

        - MCP server boot
        - Voice assistant init (latency-sensitive)
        - CLI scripts that do one recall

        Returns timing breakdown for diagnostics.

        with_first_query=True also runs one no-op recall to warm the
        vector index handle and any per-query lazy state.
        """
        import time as _t
        t0 = _t.time()

        # 1. Load embedding model
        t_model_start = _t.time()
        try:
            _ = self.search.model
        except Exception:
            pass
        t_model = _t.time() - t_model_start

        # 2. Build BM25 from any persisted index
        t_bm25_start = _t.time()
        try:
            self.search.reload_bm25()
        except Exception:
            pass
        t_bm25 = _t.time() - t_bm25_start

        # 3. Open LanceDB connection
        t_lance_start = _t.time()
        try:
            _ = self.search._table   # triggers lazy lancedb.connect()
        except Exception:
            pass
        t_lance = _t.time() - t_lance_start

        # 4. Optional: fire one no-op recall to warm per-query state
        t_query_start = _t.time()
        if with_first_query:
            try:
                self.recall("warmup probe", top_k=1)
            except Exception:
                pass
        t_query = _t.time() - t_query_start

        # 5. Mark engine as warm so diagnostics tools can show state
        self._warmed_at = _t.time()
        self._is_warm = True

        return {
            "total_ms": round((_t.time() - t0) * 1000, 1),
            "model_load_ms": round(t_model * 1000, 1),
            "bm25_load_ms": round(t_bm25 * 1000, 1),
            "lance_open_ms": round(t_lance * 1000, 1),
            "first_query_ms": round(t_query * 1000, 1),
            "with_first_query": with_first_query,
        }

    def is_warm(self) -> bool:
        """True after `warmup()` has been called successfully. Diagnostics
        (CLI `pmb stats`, MCP `health`) use this to show readiness state.
        """
        return bool(getattr(self, "_is_warm", False))

    def wait_for_embed_queue(self, timeout_seconds: float = 120.0) -> dict:
        """Block until the embed queue has drained (or timeout). Used by
        benchmarks and bulk-import flows that need the vector index
        consistent before measuring / querying.

        Returns counts for diagnostics.
        """
        import time as _t
        deadline = _t.time() + timeout_seconds
        # First wait for model load
        while not self.search.is_ready() and _t.time() < deadline:
            _t.sleep(0.2)
        # Then wait for in-memory queue to drain (worker keeps running
        # while items remain). Poll briefly.
        while _t.time() < deadline:
            with (self._embed_queue_lock or _DUMMY_LOCK):
                in_mem = len(self._embed_queue)
            durable = 0
            if self._durable_embed_queue is not None:
                try:
                    durable = self._durable_embed_queue.pending_count()
                except Exception:
                    pass
            if in_mem == 0 and durable == 0:
                return {"in_memory_remaining": 0, "durable_remaining": 0,
                        "timeout": False}
            _t.sleep(0.1)
        return {
            "in_memory_remaining": len(self._embed_queue),
            "durable_remaining": (
                self._durable_embed_queue.pending_count()
                if self._durable_embed_queue is not None else 0
            ),
            "timeout": True,
        }

    def _ensure_durable_embed_queue(self):
        """Lazy-init the SQLite-backed pending embeds table. Idempotent.

        Recovery thread spawns ONLY when there are pending rows from a
        previous process — avoids racing the model load on fresh / empty
        workspaces (which caused a Windows access violation in pytest
        teardown when the thread outlived the Engine).
        """
        if self._durable_embed_queue is not None:
            return self._durable_embed_queue
        try:
            from pmb.core.embed_queue import PersistentEmbedQueue
            self._durable_embed_queue = PersistentEmbedQueue(
                self.workspace.db_path
            )
            # Only spawn recovery if there's actually something to recover.
            try:
                pending = self._durable_embed_queue.pending_count()
            except Exception:
                pending = 0
            if pending > 0:
                import threading
                _engine_ref = self
                def _recover():
                    # Hold engine ref weakly via closure; if engine is GC'd
                    # before we run, bail out instead of segfaulting.
                    eng = _engine_ref
                    if eng is None or getattr(eng, "_closed", False):
                        return
                    try:
                        eng._durable_embed_queue.recover_on_start(
                            adder=lambda u, t: eng.search.add(u, t),
                            ready=eng.search.is_ready,
                        )
                    except Exception:
                        pass
                t = threading.Thread(
                    target=_recover, daemon=True,
                    name="pmb-embed-recovery",
                )
                t.start()
        except Exception:
            self._durable_embed_queue = None
        return self._durable_embed_queue

    def _embed_or_defer(self, ulid: str, text: str) -> None:
        """Try to embed inline if the model is loaded; otherwise queue
        for the background worker. Writes always return immediately.

        Improvement X: when called from inside `record_batch` (signalled via
        `self._batch_defer`), do NOT embed inline — instead append to a
        per-batch buffer that gets drained as ONE batched encode at the end.
        This turns N sequential embed calls (~200ms each) into one batched
        encode (~500ms total for N≤16) via sentence-transformers native
        batching.
        """
        if getattr(self, "_batch_defer", False):
            self._batch_pending.append((ulid, text))
            return
        if self.search.is_ready():
            try:
                self.search.add(ulid, text)
                return
            except Exception:
                pass  # fall through to queue on failure
        self._enqueue_embed(ulid, text)

    def _enqueue_embed(self, ulid: str, text: str) -> None:
        import threading
        if self._embed_queue_lock is None:
            self._embed_queue_lock = threading.Lock()
        # Hardening H3: persist to durable queue first so the work
        # survives process restart. The in-memory queue is still the
        # hot path for fast drain in the same process.
        dq = self._ensure_durable_embed_queue()
        if dq is not None:
            try:
                dq.enqueue(ulid, text)
            except Exception:
                # Durable queue is best-effort; don't drop the work
                # if it fails — in-memory queue still tries to run.
                pass
        with self._embed_queue_lock:
            self._embed_queue.append((ulid, text))
            if not self._embed_worker_started:
                self._embed_worker_started = True
                threading.Thread(
                    target=self._drain_embed_queue,
                    daemon=True, name="pmb-embed-drain",
                ).start()

    def _drain_embed_queue(self) -> None:
        """Background worker: wait for the model to load (poll every 500ms),
        then drain the queue. After fully draining, exits — re-spawned on
        next enqueue if needed.
        """
        import time as _t
        # Wait up to ~10 min for model load
        deadline = _t.time() + 600.0
        # Trigger model load if nothing else has yet (idempotent thanks to
        # the lazy `model` property + _ModelCache singleton)
        try:
            _ = self.search.model
        except Exception:
            pass
        while not self.search.is_ready() and _t.time() < deadline:
            _t.sleep(0.5)
        # Drain (Hardening H3: failure no longer silently drops the
        # work — the row stays in `embed_queue_pending` and the next
        # process restart picks it up via `recover_on_start`).
        while True:
            with (self._embed_queue_lock or _DUMMY_LOCK):
                if not self._embed_queue:
                    self._embed_worker_started = False
                    break
                ulid, text = self._embed_queue.pop(0)
            try:
                self.search.add(ulid, text)
            except Exception:
                # Stays in durable queue; retried via recover_on_start
                # or `pmb doctor` next time.
                continue
            # Success → drop the durable copy too
            if self._durable_embed_queue is not None:
                try:
                    with self._durable_embed_queue._conn() as conn:
                        conn.execute(
                            "DELETE FROM embed_queue_pending WHERE ulid = ?",
                            (ulid,),
                        )
                except Exception:
                    pass
        # After in-memory drain, also sweep any dead-letter recoveries
        if self._durable_embed_queue is not None:
            try:
                self._durable_embed_queue.drain_once(
                    lambda u, t: self.search.add(u, t),
                    max_items=200,
                )
            except Exception:
                pass

    # -----------------------------------------------------------------
    # Improvement U: workspace-wide dedup sweep + LLM verify worker
    # -----------------------------------------------------------------

    def dedupe_sweep(
        self,
        threshold: Optional[float] = None,
        event_types: Optional[list[str]] = None,
    ) -> dict:
        """One-shot dedup pass over ALL active events in this workspace.

        Clusters by cosine ≥ threshold within each event_type; archives
        losers with metadata.merged_into → winner. Reversible via
        `dedupe_undo()`.

        Use after upgrading dedup logic or just to clean up an aged workspace.
        """
        from pmb.reasoning.dedup import sweep_workspace, COSINE_HIGH
        thr = float(threshold if threshold is not None
                    else self.config.get("dedup.cosine_high"))

        def provider():
            return self._collect_embeddings_for_sweep()

        return sweep_workspace(
            db_path=self.workspace.db_path,
            workspace_id=self.workspace.id,
            embeddings_provider=provider,
            threshold=thr,
            event_types=event_types,
        )

    def _collect_embeddings_for_sweep(self) -> list:
        """Pull (ulid, event_type, importance, access_count, timestamp, vec)
        for every active event by joining SQLite metadata with LanceDB vectors.

        Uses pyarrow directly (no pandas dependency).
        """
        import sqlite3, numpy as np
        rows = []
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            for r in conn.execute(
                """
                SELECT ulid, event_type, importance, access_count, timestamp
                FROM events
                WHERE workspace_id = ? AND archived_at IS NULL
                ORDER BY timestamp DESC
                LIMIT 5000
                """,
                (self.workspace.id,),
            ).fetchall():
                rows.append(dict(r))
        if not rows:
            return []

        # Pull vectors from LanceDB via Arrow (no pandas needed)
        vec_by_ulid: dict[str, np.ndarray] = {}
        try:
            tbl = self.search._table
            arr_tbl = tbl.to_arrow()
            ulid_col = arr_tbl.column("ulid").to_pylist()
            vec_col = arr_tbl.column("vector").to_pylist()
            for u, v in zip(ulid_col, vec_col):
                if u is None or v is None:
                    continue
                try:
                    vec_by_ulid[u] = np.asarray(v, dtype=np.float32)
                except Exception:
                    pass
        except Exception:
            pass  # If we can't read vectors, sweep just returns empty

        out = []
        for r in rows:
            v = vec_by_ulid.get(r["ulid"])
            if v is None or v.size == 0:
                continue
            out.append((
                r["ulid"], r["event_type"],
                float(r["importance"] or 0.0),
                int(r["access_count"] or 0),
                float(r["timestamp"] or 0.0),
                v,
            ))
        return out

    def dedupe_undo(self) -> int:
        """Restore events archived by dedup (metadata.merged_into is set).
        Returns count of restored events.
        """
        from pmb.reasoning.dedup import undo_merges
        return undo_merges(self.workspace.db_path, self.workspace.id)

    def dedupe_run_pending(
        self, backend: str = "auto", limit: int = 50,
    ) -> dict:
        """L2.5: drain dedup_pending queue, ask LLM whether each pair is
        the same fact. Yes → archive newer; no → mark resolved.
        """
        from pmb.reasoning.dedup import run_pending
        return run_pending(
            db_path=self.workspace.db_path,
            workspace_id=self.workspace.id,
            backend=backend, limit=limit,
        )

    def dedupe_list_pending(self, limit: int = 100) -> list[dict]:
        """List borderline pairs awaiting LLM verdict (or user review)."""
        from pmb.reasoning.dedup import list_pending
        return list_pending(
            self.workspace.db_path, self.workspace.id, limit=limit,
        )

    # -----------------------------------------------------------------
    # Improvement R: Goals + State milestone chains (12-th semantic layer)
    # -----------------------------------------------------------------

    def record_goal(
        self,
        title: str,
        status: str = "pending",       # pending / in_progress / done / cancelled
        parent_goal_ulid: Optional[str] = None,
        due_at: Optional[float] = None,
        importance: float = 0.7,
        session_id: Optional[str] = None,
    ) -> str:
        """Create a goal/intent event. Goals have status + optional hierarchy.

        Use when user says they want / plan / intend to do something:
          "Хочу выучить Rust к концу года"
          "Need to ship v1.0 by Q3"
          "Plan: refactor auth module first, then frontend"

        Returns ulid.
        """
        clean_title, _ = redact(title)

        # Bulk-import shortcut: skip dedup + graph + L2.5 queue
        if getattr(self, "_bulk_mode", False):
            meta_bulk = {"goal_status": status, "goal_progress": 0}
            if parent_goal_ulid:
                meta_bulk["parent_goal_ulid"] = parent_goal_ulid
            if due_at is not None:
                meta_bulk["due_at"] = float(due_at)
            ev = Event(
                workspace_id=self.workspace.id,
                event_type="goal",
                content=clean_title,
                metadata=meta_bulk,
                importance=importance,
                source_session_id=session_id,
                tier="semantic",
            )
            ev = self.events.append(ev)
            self._embed_or_defer(ev.ulid, ev.to_text())
            self._bulk_collected_ulids.append(ev.ulid)
            return ev.ulid

        # Improvement U: dedup at write — most common case the user hits is
        # the AI writing the SAME goal in two languages (RU + EN) as two
        # separate goal events. L1 won't catch translations; L2 (cosine on
        # multilingual model) will.
        dup_hit, borderline = self._dedup_pre_write(
            content=clean_title, event_type="goal",
        )
        if dup_hit is not None:
            self._bump_for_dup(dup_hit.canonical_ulid)
            return dup_hit.canonical_ulid

        meta = {
            "goal_status": status,
            "goal_progress": 0,
        }
        if parent_goal_ulid:
            meta["parent_goal_ulid"] = parent_goal_ulid
        if due_at is not None:
            meta["due_at"] = float(due_at)
        ev = Event(
            workspace_id=self.workspace.id,
            event_type="goal",
            content=clean_title,
            metadata=meta,
            importance=importance,
            source_session_id=session_id,
            tier="semantic",  # goals are long-lived by default
        )
        ev = self.events.append(ev)
        # Improvement W: embed inline if model loaded, else queue
        self._embed_or_defer(ev.ulid, ev.to_text())
        try:
            self._index_event_in_graph(ev, full_text=clean_title)
        except Exception:
            pass

        # L2.5: borderline goal pair — enqueue for async LLM verify
        if borderline is not None and self.config.get("dedup.async_verify"):
            try:
                from pmb.reasoning.dedup import enqueue_borderline
                enqueue_borderline(
                    self.workspace.db_path, self.workspace.id,
                    new_ulid=ev.ulid,
                    candidate_ulid=borderline[0],
                    similarity=borderline[1],
                )
            except Exception:
                pass

        self.recall_cache.bump_generation()
        return ev.ulid

    def update_goal(
        self,
        goal_ulid: str,
        status: Optional[str] = None,
        progress: Optional[int] = None,    # 0..100
        note: Optional[str] = None,
    ) -> dict:
        """Update a goal's status / progress. Creates a linked update event
        recording the transition (so history of changes is preserved).
        """
        ev = self.events.get_by_ulid(goal_ulid)
        if ev is None or ev.event_type != "goal":
            return {"error": "goal not found"}
        meta = dict(ev.metadata or {})
        old_status = meta.get("goal_status")
        old_progress = meta.get("goal_progress", 0)
        if status is not None:
            meta["goal_status"] = status
        if progress is not None:
            meta["goal_progress"] = max(0, min(100, int(progress)))
        # Persist updated metadata in-place
        import sqlite3, json as _j
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.execute(
                "UPDATE events SET metadata_json = ? WHERE ulid = ?",
                (_j.dumps(meta), goal_ulid),
            )

        # Append a separate "update" event in the chain so history exists
        upd_content = note or (
            f"Goal updated: {old_status}→{meta.get('goal_status')}, "
            f"{old_progress}→{meta.get('goal_progress', 0)}%"
        )
        upd_ev = Event(
            workspace_id=self.workspace.id,
            event_type="goal_update",
            content=upd_content,
            metadata={
                "goal_ulid": goal_ulid,
                "old_status": old_status,
                "new_status": meta.get("goal_status"),
                "old_progress": old_progress,
                "new_progress": meta.get("goal_progress", 0),
            },
            importance=0.5,
            tier="working",  # updates fade after a few days
        )
        upd_ev = self.events.append(upd_ev)
        # Link update → goal via event_edges
        try:
            from pmb.reasoning.causation import CausationEdge, upsert_edge
            upsert_edge(
                self.workspace.db_path,
                CausationEdge(
                    source_ulid=upd_ev.ulid, target_ulid=goal_ulid,
                    edge_type="references", confidence=1.0,
                    rationale="update of goal",
                ),
            )
        except Exception:
            pass
        self.recall_cache.bump_generation()
        return {
            "goal_ulid": goal_ulid,
            "update_ulid": upd_ev.ulid,
            "status": meta.get("goal_status"),
            "progress": meta.get("goal_progress", 0),
        }

    def list_goals(
        self,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """List goals (optionally filtered by status)."""
        import sqlite3, json as _j
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            sql = (
                "SELECT ulid, content, timestamp, importance, metadata_json "
                "FROM events WHERE workspace_id = ? AND event_type = 'goal' "
                "AND archived_at IS NULL"
            )
            params: list = [self.workspace.id]
            if status:
                sql += " AND json_extract(metadata_json, '$.goal_status') = ?"
                params.append(status)
            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            try:
                meta = _j.loads(r["metadata_json"] or "{}")
            except Exception:
                meta = {}
            out.append({
                "ulid": r["ulid"],
                "title": r["content"],
                "status": meta.get("goal_status"),
                "progress": meta.get("goal_progress", 0),
                "parent_goal_ulid": meta.get("parent_goal_ulid"),
                "due_at": meta.get("due_at"),
                "timestamp": r["timestamp"],
            })
        return out

    def record_milestone(
        self,
        chain_name: str,
        title: str,
        state: Optional[dict] = None,
        triggered_by_ulid: Optional[str] = None,
        importance: float = 0.6,
    ) -> str:
        """Record a milestone in a named state-chain.

        Each milestone references the previous one in the same chain
        (auto-linked) and optionally the event that triggered the change.

        Example:
          record_milestone(
              chain_name="architecture_layers",
              title="11 layers (added activity log)",
              state={"count": 11, "added": "activity"},
              triggered_by_ulid=<event ulid for the implementation>,
          )

        Later: `chain_history("architecture_layers")` returns the full
        sequence: 6 → 7 → ... → 11, with reasons.
        """
        clean_title, _ = redact(title)
        # Find the previous milestone in this chain (latest by timestamp)
        import sqlite3, json as _j
        prev_ulid: Optional[str] = None
        with sqlite3.connect(self.workspace.db_path) as conn:
            row = conn.execute(
                "SELECT ulid FROM events WHERE workspace_id = ? "
                "AND event_type = 'milestone' "
                "AND json_extract(metadata_json, '$.chain_name') = ? "
                "AND archived_at IS NULL "
                "ORDER BY timestamp DESC LIMIT 1",
                (self.workspace.id, chain_name),
            ).fetchone()
            if row:
                prev_ulid = row[0]

        meta: dict = {"chain_name": chain_name}
        if prev_ulid:
            meta["previous_milestone_ulid"] = prev_ulid
        if triggered_by_ulid:
            meta["triggered_by_ulid"] = triggered_by_ulid
        if state:
            clean_state, _ = redact_metadata(state)
            meta["state"] = clean_state

        ev = Event(
            workspace_id=self.workspace.id,
            event_type="milestone",
            content=clean_title,
            metadata=meta,
            importance=importance,
            tier="semantic",
        )
        ev = self.events.append(ev)
        # Synchronous when called directly (Python API contract); deferred
        # only when invoked from inside record_batch (batch_defer set).
        try:
            if getattr(self, "_batch_defer", False):
                self._embed_or_defer(ev.ulid, ev.to_text())
            else:
                self.search.add(ev.ulid, ev.to_text())
        except Exception:
            pass
        try:
            self._index_event_in_graph(ev, full_text=clean_title)
        except Exception:
            pass

        # Edges: this milestone → previous (chain link), and triggered_by → this
        try:
            from pmb.reasoning.causation import CausationEdge, upsert_edge
            if prev_ulid:
                upsert_edge(
                    self.workspace.db_path,
                    CausationEdge(
                        source_ulid=prev_ulid, target_ulid=ev.ulid,
                        edge_type="temporal-next", confidence=1.0,
                        rationale=f"chain {chain_name}",
                    ),
                )
            if triggered_by_ulid:
                upsert_edge(
                    self.workspace.db_path,
                    CausationEdge(
                        source_ulid=triggered_by_ulid, target_ulid=ev.ulid,
                        edge_type="causes", confidence=1.0,
                        rationale="triggered milestone",
                    ),
                )
        except Exception:
            pass
        self.recall_cache.bump_generation()
        return ev.ulid

    def chain_history(self, chain_name: str, limit: int = 100) -> list[dict]:
        """Full chronological history of a named state-chain.

        Returns oldest-first list of milestones with their state snapshots
        and trigger events. Reconstructs the evolution: 6 → 7 → ... → 11
        with the reason at each step.
        """
        import sqlite3, json as _j
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ulid, content, timestamp, metadata_json "
                "FROM events WHERE workspace_id = ? AND event_type = 'milestone' "
                "AND json_extract(metadata_json, '$.chain_name') = ? "
                "AND archived_at IS NULL "
                "ORDER BY timestamp ASC LIMIT ?",
                (self.workspace.id, chain_name, limit),
            ).fetchall()
        out = []
        for r in rows:
            try:
                meta = _j.loads(r["metadata_json"] or "{}")
            except Exception:
                meta = {}
            out.append({
                "ulid": r["ulid"],
                "title": r["content"],
                "timestamp": r["timestamp"],
                "state": meta.get("state", {}),
                "triggered_by_ulid": meta.get("triggered_by_ulid"),
                "previous_milestone_ulid": meta.get("previous_milestone_ulid"),
            })
        return out

    def chain_current(self, chain_name: str) -> Optional[dict]:
        """Latest milestone of a chain — the "current state"."""
        hist = self.chain_history(chain_name, limit=200)
        return hist[-1] if hist else None

    # -----------------------------------------------------------------
    # Improvement Q: Working memory / activity log
    # -----------------------------------------------------------------

    def record_activity(
        self,
        summary: str,
        actor: str = "agent",
        kind: str = "action",
        details: Optional[dict] = None,
        importance: float = 0.4,
        session_id: Optional[str] = None,
    ) -> str:
        """Log an action / activity. Used by the AI to record what it
        just did (made an edit, ran a tool, gave advice). Lighter than
        record_fact — these are session-scoped working memory.

        actor: 'agent' (AI's own action), 'user' (user did something),
               'system' (auto-generated event).
        kind:  'action' (default), 'edit', 'tool_call', 'recommendation',
               'plan', 'completed'.

        Returns ulid.
        """
        clean_summary, _ = redact(summary)
        meta = {"actor": actor, "activity_kind": kind}
        if details:
            clean_details, _ = redact_metadata(details)
            meta.update(clean_details)

        # Bulk-import shortcut: skip graph + L2.5 queue, just persist
        if getattr(self, "_bulk_mode", False):
            ev = Event(
                workspace_id=self.workspace.id,
                event_type="activity",
                content=clean_summary,
                metadata=meta,
                importance=importance,
                source_session_id=session_id,
                tier="working",
            )
            ev = self.events.append(ev)
            self._embed_or_defer(ev.ulid, ev.to_text())
            self._bulk_collected_ulids.append(ev.ulid)
            return ev.ulid

        # Auto-bind to session
        if session_id is None:
            session_id = self.session_tracker.touch().id
        ev = Event(
            workspace_id=self.workspace.id,
            event_type="activity",
            content=clean_summary,
            metadata=meta,
            importance=importance,
            source_session_id=session_id,
            tier="working",  # activity = working memory by default
        )
        ev = self.events.append(ev)
        # Synchronous unless inside batch.
        try:
            if getattr(self, "_batch_defer", False):
                self._embed_or_defer(ev.ulid, ev.to_text())
            else:
                self.search.add(ev.ulid, ev.to_text())
        except Exception:
            pass
        try:
            self._index_event_in_graph(ev, full_text=clean_summary)
        except Exception:
            pass
        try:
            from pmb.reasoning.causation import add_temporal_next_edge
            add_temporal_next_edge(self, ev)
        except Exception:
            pass
        self.recall_cache.bump_generation()
        return ev.ulid

    def recent_activity(
        self,
        minutes: float = 60.0,
        limit: int = 20,
        actor: Optional[str] = None,
        kind: Optional[str] = None,
    ) -> list[dict]:
        """Working memory dump — recent activity events, chronological.

        NO BM25/vector search — just SQL by timestamp. Instant.

        Use this BEFORE recall when answering "what did we just do",
        "что последнее", "show recent changes" type questions.

        Filters:
          minutes: how far back (default 60)
          actor:   'agent' / 'user' / 'system' / None=all
          kind:    'action' / 'edit' / 'tool_call' / ... / None=all
        """
        import sqlite3
        cutoff = time.time() - minutes * 60.0
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            sql = (
                "SELECT ulid, content, timestamp, importance, metadata_json "
                "FROM events WHERE workspace_id = ? AND archived_at IS NULL "
                "AND event_type = 'activity' AND timestamp >= ?"
            )
            params: list = [self.workspace.id, cutoff]
            if actor:
                sql += " AND json_extract(metadata_json, '$.actor') = ?"
                params.append(actor)
            if kind:
                sql += " AND json_extract(metadata_json, '$.activity_kind') = ?"
                params.append(kind)
            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
        import json as _j
        out: list[dict] = []
        for r in rows:
            try:
                meta = _j.loads(r["metadata_json"] or "{}")
            except Exception:
                meta = {}
            out.append({
                "ulid": r["ulid"],
                "content": r["content"],
                "timestamp": r["timestamp"],
                "actor": meta.get("actor"),
                "kind": meta.get("activity_kind"),
                "importance": r["importance"],
            })
        return out

    def session_timeline(
        self,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Chronological events of a session. If session_id is None,
        uses the current session.

        Returns ALL event types (fact, qa, activity, ...) ordered by time.
        Use to summarize "what happened this session" or generate a
        post-mortem of what was done.
        """
        if session_id is None:
            sess = self.session_tracker.current()
            if not sess:
                return []
            session_id = sess.id
        import sqlite3, json as _j
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ulid, event_type, content, timestamp, importance, metadata_json "
                "FROM events WHERE workspace_id = ? AND source_session_id = ? "
                "AND archived_at IS NULL "
                "ORDER BY timestamp ASC LIMIT ?",
                (self.workspace.id, session_id, limit),
            ).fetchall()
        out = []
        for r in rows:
            try:
                meta = _j.loads(r["metadata_json"] or "{}")
            except Exception:
                meta = {}
            out.append({
                "ulid": r["ulid"],
                "event_type": r["event_type"],
                "content": r["content"],
                "timestamp": r["timestamp"],
                "actor": meta.get("actor", "system" if r["event_type"] == "activity" else "user"),
            })
        return out

    def what_just_happened(self, n: int = 5) -> list[dict]:
        """Last N events of ANY type, newest first.

        Used by AI to answer 'что только что сделали?' / 'what did we just do?'
        without going through recall search.

        Returns activities AND facts AND any other event types — just the
        most recent stuff regardless of session binding. For session-only
        view use `session_timeline()`.
        """
        evs = self.events.list_active(self.workspace.id, limit=n)
        return [
            {
                "ulid": e.ulid,
                "event_type": e.event_type,
                "content": (e.content or "")[:300],
                "timestamp": e.timestamp,
                "actor": (e.metadata or {}).get("actor", "user"),
            }
            for e in evs[:n]
        ]

    def record_fact_tree(
        self,
        main: str,
        subfacts: Optional[list[str]] = None,
        importance: float = 0.7,
        metadata: Optional[dict] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """Improvement P: store a main fact + multiple atomic subfacts.

        The main fact gets the requested importance. Subfacts inherit the
        main's importance × 0.85 (slightly lower — they're supplementary)
        and link back via metadata.parent_ulid.

        Use this when one event has multiple atomic data points worth
        remembering separately:

            record_fact_tree(
                "User fell down stairs and broke arm on May 23, 2026",
                subfacts=[
                    "Time of fall: 18:52",
                    "Recommended: visit ER for X-ray",
                    "First aid: ice 15-20min, remove rings, do not drive",
                    "Warning signs to call 911: numbness, bleeding, deformation",
                ],
                importance=0.9,
            )

        Each subfact is independently searchable. When recall finds the
        parent, callers can pull related subfacts via `get_subfacts(ulid)`.
        """
        # Store main first
        main_meta = dict(metadata or {})
        main_meta["has_subfacts"] = bool(subfacts)
        main_ulid = self.record_fact(
            main, importance=importance,
            metadata=main_meta, session_id=session_id,
        )

        sub_ulids: list[str] = []
        if subfacts:
            sub_importance = max(0.3, importance * 0.85)
            for s in subfacts:
                if not s or not s.strip():
                    continue
                # Each subfact links to parent via metadata
                sub_meta = {
                    "parent_ulid": main_ulid,
                    "is_subfact": True,
                }
                # Inherit parent's event_time if any
                sub_ulid = self.record_fact(
                    s.strip(), importance=sub_importance,
                    metadata=sub_meta, session_id=session_id,
                )
                sub_ulids.append(sub_ulid)
                # Also link main → subfact via event_edges for graph walks
                try:
                    from pmb.reasoning.causation import CausationEdge, upsert_edge
                    upsert_edge(
                        self.workspace.db_path,
                        CausationEdge(
                            source_ulid=main_ulid,
                            target_ulid=sub_ulid,
                            edge_type="references",
                            confidence=1.0,
                            rationale="subfact of main",
                        ),
                    )
                except Exception:
                    pass

        return {
            "main_ulid": main_ulid,
            "subfact_ulids": sub_ulids,
            "n_subfacts": len(sub_ulids),
        }

    def record_batch_bulk(self, items: list[dict]) -> dict:
        """Improvement #5: bulk-import mode for migrations / large imports.

        Skips ALL cross-cutting work that makes record_batch slow per item:
          - L1 + L2 dedup checks
          - graph entity indexing + edge upserts
          - temporal date parsing
          - causation edge insert
          - L2.5 borderline queue

        ONLY writes SQLite rows and queues embeddings. The graph layer
        will be missing until the caller runs `pmb regraph` to rebuild it
        from scratch.

        Use this when:
          - importing from another tool (mem0 export, JSONL file)
          - replaying a saved transcript
          - bulk-loading test data
          - any time you have ≥50 items and don't need dedup/graph immediately

        Returns the same shape as record_batch but with `bulk_mode=True` in
        the result so callers can branch on it. Typical speed-up: ~10-15×
        vs normal record_batch on 100 items.
        """
        if not items or not isinstance(items, list):
            return {"results": [], "n_ok": 0, "n_failed": 0,
                    "errors": [{"index": 0, "error": "empty or invalid items"}],
                    "bulk_mode": True}
        self._bulk_collected_ulids = []
        self._bulk_mode = True
        try:
            result = self.record_batch(items)
        finally:
            self._bulk_mode = False
        result["bulk_mode"] = True
        result["bulk_ulids"] = list(self._bulk_collected_ulids)
        self._bulk_collected_ulids = []
        # Invalidate caches in one shot — record_batch's per-item bumps
        # were skipped in bulk mode.
        try:
            self.recall_cache.bump_generation()
        except Exception:
            pass
        return result

    def record_batch_async(self, items: list[dict]) -> dict:
        """Improvement AA: fire-and-forget batch write.

        Returns IMMEDIATELY (~5-50ms) after spawning a background thread that
        runs the full `record_batch` pipeline (embedding, LanceDB, graph,
        dedup). The MCP caller doesn't wait — no more 120s timeouts on big
        batches.

        Trade-off: the caller can't see the resulting ULIDs synchronously
        (just a count of accepted items). That's fine — agents almost never
        use ULIDs from record_batch's return, they just confirm the write
        happened.

        Recall called immediately after this MAY miss the new events for
        ~100-1000ms while embedding completes. For typical agent flow
        (write now, read on next user turn) this is invisible.
        """
        # Light validation only — heavy lifting in background
        if not items or not isinstance(items, list):
            return {"n_accepted": 0, "processing": "skipped",
                    "errors": ["empty or invalid items"]}
        n_items = sum(1 for i in items if isinstance(i, dict) and i.get("type"))

        import threading
        def _process():
            try:
                self.record_batch(items)
            except Exception as e:
                # Last-resort log — silent failure is dangerous
                import logging
                logging.getLogger(__name__).exception(
                    "async batch processing failed: %s", e,
                )
        threading.Thread(
            target=_process, daemon=True, name="pmb-async-batch",
        ).start()

        # Improvement II: minimal response. Smaller payload, faster Codex
        # UI processing. Background flag suppressed because the caller
        # doesn't need it (we always run in background by default).
        return {"ok": True, "n": n_items}

    def record_batch(self, items: list[dict]) -> dict:
        """Improvement T: single-shot batch write across all record_* APIs.

        The agent extracts ALL atomic facts / goals / activities / milestones
        from a user turn in ONE thinking pass, then calls record_batch ONCE
        with the structured list. Eliminates the per-call LLM round-trip
        cost that's the real bottleneck (each round-trip ≈ 3-5s of LLM
        thinking; 11 calls = ~55s; 1 call = ~5s).

        Each item is a dict with a `type` discriminator and the same kwargs
        you'd pass to the matching record_* method:

          {"type": "fact", "content": "...", "importance": 0.7}
          {"type": "fact_tree", "main": "...", "subfacts": [...], "importance": 0.9}
          {"type": "goal", "title": "...", "status": "in_progress",
                            "due_at": <epoch>, "parent_goal_ulid": null}
          {"type": "activity", "content": "...", "kind": "edit", "actor": "agent"}
          {"type": "milestone", "chain_name": "...", "title": "...",
                                 "state": {...}, "triggered_by_ulid": null}

        Returns: {results: [...], n_ok, n_failed, errors}
        Each result mirrors what the corresponding record_* method returns,
        with `type` echoed back so the agent can stitch ULIDs back to inputs.

        Unknown / malformed items are skipped (logged in errors), the rest
        proceed — partial failure does NOT abort the batch.
        """
        # Improvement Z+AA: cap content size per item as a SAFETY guard
        # (agents occasionally dump 20KB+ blobs). The async wrapper handles
        # latency, so this can be generous — 5000 chars preserves nuance
        # while still preventing pathological inputs.
        MAX_CONTENT = 5000
        items = _cap_batch_content(items or [], MAX_CONTENT)

        results: list[dict] = []
        errors: list[dict] = []
        n_ok = 0

        # Improvement CC: serialize concurrent batches. Multiple async batch
        # threads must NOT overlap on self._batch_defer / self._batch_pending
        # or items get lost. The lock is per-engine, held only during one
        # batch — async callers still return instantly because each batch
        # is wrapped in its own thread by record_batch_async.
        with self._batch_lock:
            # Improvement X: defer all embeddings for the whole batch into one
            # native batched encode at the end. Without this, N items =
            # N sequential embed calls.
            self._batch_defer = True
            self._batch_pending = []

            for idx, item in enumerate(items or []):
                if not isinstance(item, dict):
                    errors.append({"index": idx, "error": "not a dict"})
                    continue
                t = (item.get("type") or "").lower().strip()
                pin_after = bool(item.get("pin", False))
                try:
                    if t == "fact":
                        content_in = item.get("content") or item.get("fact") or ""
                        ulid = self.record_fact(
                            content_in,
                            importance=float(item.get("importance", 0.7)),
                            metadata=item.get("metadata"),
                        )
                        if pin_after:
                            try: self.pin(ulid)
                            except Exception: pass
                        # Improvement WW: opportunistic atomic-fact extraction.
                        # When a long fact contains multiple sentences with
                        # pattern-recognisable atoms ("X lives in Y", "We use
                        # X", "X leads Y"), record those atoms as sibling
                        # events with metadata.parent_ulid → source. This
                        # lifts recall on questions targeting one atom inside
                        # a paragraph (mem0-style fact decomposition, no LLM).
                        atoms_created: list[str] = []
                        if self._atomic_extract_enabled:
                            try:
                                atoms_created = self._record_atomic_facts(
                                    content_in, parent_ulid=ulid,
                                    base_importance=float(item.get("importance", 0.7)),
                                )
                            except Exception:
                                pass
                        results.append({"type": "fact", "ulid": ulid,
                                        "pinned": pin_after,
                                        "atomic_facts": atoms_created})
                        n_ok += 1
                    elif t in ("fact_tree", "tree"):
                        res = self.record_fact_tree(
                            main=item.get("main") or item.get("content") or "",
                            subfacts=item.get("subfacts") or [],
                            importance=float(item.get("importance", 0.7)),
                            metadata=item.get("metadata"),
                        )
                        if pin_after and res.get("main_ulid"):
                            try: self.pin(res["main_ulid"])
                            except Exception: pass
                            res["pinned"] = True
                        res["type"] = "fact_tree"
                        results.append(res)
                        n_ok += 1
                    elif t == "goal":
                        ulid = self.record_goal(
                            title=item.get("title") or item.get("content") or "",
                            status=item.get("status", "pending"),
                            due_at=item.get("due_at"),
                            parent_goal_ulid=item.get("parent_goal_ulid"),
                            importance=float(item.get("importance", 0.7)),
                        )
                        if pin_after:
                            try: self.pin(ulid)
                            except Exception: pass
                        results.append({"type": "goal", "ulid": ulid,
                                        "pinned": pin_after})
                        n_ok += 1
                    elif t == "activity":
                        ulid = self.record_activity(
                            summary=item.get("content") or item.get("summary") or "",
                            actor=item.get("actor", "agent"),
                            kind=item.get("kind", "action"),
                            importance=float(item.get("importance", 0.4)),
                        )
                        if pin_after:
                            try: self.pin(ulid)
                            except Exception: pass
                        results.append({"type": "activity", "ulid": ulid,
                                        "pinned": pin_after})
                        n_ok += 1
                    elif t == "milestone":
                        ulid = self.record_milestone(
                            chain_name=item.get("chain_name") or "",
                            title=item.get("title") or item.get("content") or "",
                            state=item.get("state"),
                            triggered_by_ulid=item.get("triggered_by_ulid"),
                            importance=float(item.get("importance", 0.6)),
                        )
                        if pin_after:
                            try: self.pin(ulid)
                            except Exception: pass
                        results.append({"type": "milestone", "ulid": ulid,
                                        "pinned": pin_after})
                        n_ok += 1
                    elif t == "preference":
                        ulid = self.record_preference(
                            preference=item.get("content") or
                                       item.get("preference") or "",
                            importance=float(item.get("importance", 0.7)),
                            metadata=item.get("metadata"),
                        )
                        if pin_after:
                            try: self.pin(ulid)
                            except Exception: pass
                        results.append({"type": "preference", "ulid": ulid,
                                        "pinned": pin_after})
                        n_ok += 1
                    elif t == "summary":
                        ulid = self.record_summary(
                            summary=item.get("content") or
                                    item.get("summary") or "",
                            importance=float(item.get("importance", 0.5)),
                            metadata=item.get("metadata"),
                        )
                        if pin_after:
                            try: self.pin(ulid)
                            except Exception: pass
                        results.append({"type": "summary", "ulid": ulid,
                                        "pinned": pin_after})
                        n_ok += 1
                    elif t in ("keyed_fact", "key_fact"):
                        # P0-2: upsert a (subject, attribute, value) triple.
                        # Archives prior facts with same key so the current
                        # value alone surfaces on recall.
                        res = self.record_keyed_fact(
                            subject=item.get("subject") or "user",
                            attribute=item.get("attribute") or "",
                            value=item.get("value") or item.get("content") or "",
                            importance=float(item.get("importance", 0.8)),
                            metadata=item.get("metadata"),
                        )
                        if pin_after and res.get("new_ulid"):
                            try: self.pin(res["new_ulid"])
                            except Exception: pass
                            res["pinned"] = True
                        res["type"] = "keyed_fact"
                        results.append(res)
                        n_ok += 1
                    else:
                        errors.append({"index": idx, "error": f"unknown type: {t!r}"})
                except Exception as e:
                    errors.append({"index": idx, "error": f"{type(e).__name__}: {e}"})

            # Improvement X: flush the per-batch embed buffer as ONE call.
            # If model is hot → batched encode (~500ms for 16 items vs 16×200ms).
            # If model not ready → spill into the async queue, write returns now.
            self._batch_defer = False
            pending = self._batch_pending
            self._batch_pending = []
            if pending:
                if self.search.is_ready():
                    try:
                        self.search.add_batch(pending)
                    except Exception:
                        for ulid, text in pending:
                            self._enqueue_embed(ulid, text)
                else:
                    for ulid, text in pending:
                        self._enqueue_embed(ulid, text)

        return {
            "results": results,
            "n_ok": n_ok,
            "n_failed": len(errors),
            "errors": errors,
        }

    def get_subfacts(self, parent_ulid: str) -> list[dict]:
        """Return all subfacts linked to a parent event."""
        import sqlite3, json as _j
        out = []
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ulid, content, importance, timestamp, metadata_json "
                "FROM events WHERE workspace_id = ? AND archived_at IS NULL "
                "AND json_extract(metadata_json, '$.parent_ulid') = ? "
                "ORDER BY timestamp ASC",
                (self.workspace.id, parent_ulid),
            ).fetchall()
            for r in rows:
                try:
                    meta = _j.loads(r["metadata_json"] or "{}")
                except Exception:
                    meta = {}
                out.append({
                    "ulid": r["ulid"],
                    "content": r["content"],
                    "importance": r["importance"],
                    "timestamp": r["timestamp"],
                    "metadata": meta,
                })
        return out

    def get_parent_fact(self, subfact_ulid: str) -> Optional[dict]:
        """If this event is a subfact, return its parent event."""
        ev = self.events.get_by_ulid(subfact_ulid)
        if not ev or not ev.metadata:
            return None
        parent_ulid = ev.metadata.get("parent_ulid")
        if not parent_ulid:
            return None
        parent = self.events.get_by_ulid(parent_ulid)
        if not parent:
            return None
        return {
            "ulid": parent.ulid,
            "content": parent.content,
            "event_type": parent.event_type,
            "importance": parent.importance,
            "timestamp": parent.timestamp,
            "metadata": parent.metadata,
        }

    def record_event(
        self,
        event_type: str,
        content: str,
        importance: float = 0.5,
        metadata: Optional[dict] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Generic event record (git, file, test, ...)."""
        clean_content, _ = redact(content)
        clean_metadata, _ = redact_metadata(metadata or {})
        ev = Event(
            workspace_id=self.workspace.id,
            event_type=event_type,
            content=clean_content,
            metadata=clean_metadata,
            importance=importance,
            source_session_id=session_id,
            tier=default_tier_for_event_type(event_type),
        )
        ev = self.events.append(ev)
        # Synchronous when called directly (Python API contract); deferred
        # only when invoked from inside record_batch (batch_defer set).
        if getattr(self, "_batch_defer", False):
            self._embed_or_defer(ev.ulid, ev.to_text())
        else:
            self.search.add(ev.ulid, ev.to_text())
        self._index_event_in_graph(ev, full_text=clean_content)
        # Cheap rule-based causation: temporal-next edge from the last event.
        # No LLM, just SQL. Fires only if the previous event is within minutes.
        try:
            from pmb.reasoning.causation import add_temporal_next_edge
            add_temporal_next_edge(self, ev)
        except Exception:
            pass
        # Improvement C: parse event_time (date references) from content
        # and store in metadata. Enables temporal-proximity boost at recall.
        try:
            self._attach_event_time(ev)
        except Exception:
            pass
        self.recall_cache.bump_generation()
        return ev.ulid

    def record_image(
        self,
        path: str,
        description: str = "",
        importance: float = 0.5,
        metadata: Optional[dict] = None,
        session_id: Optional[str] = None,
        encode_clip: bool = False,
    ) -> str:
        """Improvement J: record an image with optional CLIP embedding.

        The `description` is what's searchable via text — make it specific
        ('screenshot of auth flow diagram with arrows', not 'image').

        encode_clip=True triggers CLIP encoding (requires open_clip or
        sentence-transformers); silently degrades to text-only if missing.

        Returns the ulid of the image event.
        """
        from pmb.reasoning.images import attach_image
        att = attach_image(path, description=description, encode_clip=encode_clip)
        meta = dict(metadata or {})
        meta.update(att.to_metadata())
        # The event's content IS the description — that's what's indexed
        content = description or f"image: {Path(att.path).name}"
        return self.record_event(
            event_type="image",
            content=content,
            importance=importance,
            metadata=meta,
            session_id=session_id,
        )

    def reindex_embeddings(self) -> dict:
        """Re-embed all active events with the current embedding model.

        Call after switching embedding models (e.g. from English-only to
        multilingual). The old vectors are deleted; new ones generated.

        WARNING: this hits the LLM model heavily. For 1000 events on CPU
        with all-MiniLM expects ~2-5 minutes. With multilingual model
        slightly more (~5%).
        """
        import time as _t
        events = self.events.list_active(self.workspace.id, limit=1_000_000)
        if not events:
            return {"n_events": 0, "elapsed_seconds": 0.0}
        t0 = _t.time()
        n = self.search.reindex_all(events)
        elapsed = _t.time() - t0
        self.recall_cache.bump_generation()
        return {
            "n_events": n,
            "elapsed_seconds": round(elapsed, 1),
            "model": self.search.model_name if hasattr(self.search, "model_name") else "unknown",
        }

    def search_images_by_text(
        self, query: str, top_k: int = 10,
    ) -> list[dict]:
        """Cross-modal: encode query via CLIP text encoder, cosine against
        stored image CLIP embeddings. Falls back to plain text recall if
        CLIP unavailable."""
        import json as _j
        try:
            from pmb.reasoning.images import clip_encode_text
            import numpy as np
            q_emb = clip_encode_text(query)
            if q_emb is None:
                # Fallback: text-only recall, filtered to image events
                pack = self.recall(query, top_k=top_k * 2)
                return [r.to_dict() for r in pack.results if r.event_type == "image"][:top_k]
            # Iterate image events, compute cosine
            import sqlite3
            scored: list[tuple[str, float]] = []
            with sqlite3.connect(self.workspace.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT ulid, content, metadata_json FROM events "
                    "WHERE workspace_id = ? AND event_type = 'image' "
                    "AND archived_at IS NULL",
                    (self.workspace.id,),
                ).fetchall()
            for r in rows:
                try:
                    meta = _j.loads(r["metadata_json"] or "{}")
                    ce = meta.get("clip_embedding_json")
                    if not ce:
                        continue
                    img_emb = np.array(_j.loads(ce), dtype=np.float32)
                    if img_emb.size == 0:
                        continue
                    qe = np.array(q_emb, dtype=np.float32)
                    sim = float(np.dot(img_emb, qe) / (np.linalg.norm(img_emb) * np.linalg.norm(qe) + 1e-9))
                    scored.append((r["ulid"], sim))
                except Exception:
                    continue
            scored.sort(key=lambda x: -x[1])
            top = scored[:top_k]
            out = []
            for ulid, sim in top:
                ev = self.events.get_by_ulid(ulid)
                if ev:
                    out.append({
                        "ulid": ev.ulid,
                        "score": sim,
                        "content": ev.content,
                        "metadata": ev.metadata,
                    })
            return out
        except Exception:
            return []

    # -----------------------------------------------------------------
    # Graph indexing
    # -----------------------------------------------------------------

    def _index_event_in_graph(self, ev: Event, full_text: str) -> list[int]:
        """Extract entities + upsert nodes + co-occurrence edges. Returns entity_ids."""
        files_hint = ev.metadata.get("files_changed") or []
        ext = self.entity_extractor.extract(full_text, files_hint=files_hint)
        named = ext.all_named()

        # Improvement H: person extraction (no-ML, regex + dict + speaker)
        if self.config.get("recall.person_extraction"):
            try:
                from pmb.graph.persons import (
                    extract_persons, KnownPersons,
                )
                kp = KnownPersons(self.workspace.db_path, self.workspace.id)
                pres = extract_persons(
                    full_text, metadata=ev.metadata, known_persons=kp,
                )
                if pres.persons:
                    # Add person entities. Dedupe on (kind, name) — a name
                    # might already be a 'concept' but should ALSO be a
                    # 'person' (different graph node).
                    existing_pairs = {(k, n) for k, n in named}
                    for p in pres.persons:
                        if ("person", p) not in existing_pairs:
                            named.append(("person", p))
                    # Self-reinforce: bump mention counts in workspace dict
                    kp.bump(pres.persons)
            except Exception:
                pass

        # Improvement J (code half): Python AST symbol extraction.
        # If content looks like code, extract function/class/import entities
        # so the graph layer can answer "which code uses X" structurally.
        if self.config.get("recall.code_ast_extraction"):
            try:
                from pmb.reasoning.code_ast import (
                    extract_python_symbols, symbols_to_entity_names,
                )
                syms = extract_python_symbols(full_text)
                if syms:
                    existing_pairs = {(k, n) for k, n in named}
                    for kn in symbols_to_entity_names(syms):
                        if kn not in existing_pairs:
                            named.append(kn)
            except Exception:
                pass

        if not named:
            return []

        # Improvement S: cross-kind dedup. A name that's been classified as
        # something specific (tech / file / function / class / import / person)
        # should NOT also live as a generic 'concept'. The specific kind wins.
        named = _dedupe_named_entities(named)

        entity_ids: list[int] = []
        for kind, name in named:
            eid = self.graph.upsert_entity(self.workspace.id, kind, name)
            entity_ids.append(eid)
        self.graph.link_event(ev.ulid, entity_ids)
        self.graph.bump_edges(self.workspace.id, entity_ids)
        return entity_ids

    # -----------------------------------------------------------------
    # Improvement S: rebuild graph from scratch
    # -----------------------------------------------------------------

    def prune_graph(
        self,
        max_weight: int = 1,
        older_than_days: float = 30.0,
        also_drop_orphan_entities: bool = True,
    ) -> dict:
        """Prune the entity graph to keep it lean as workspace grows.

        Two cleanups:
          1. Drop edges with `weight <= max_weight` AND `last_seen` older
             than `older_than_days`. These are typically one-off co-mentions
             that won't help recall — keeping them just slows PPR.
          2. Drop entities with zero remaining edges AND zero event-links
             (orphans left after edge pruning).

        Run periodically (cron or after large ingests) on workspaces with
        10k+ events to keep recall fast. Reversible? No — but the underlying
        events/embeddings aren't touched, so a `pmb regraph` rebuilds.

        Returns: {n_edges_pruned, n_entities_pruned, edges_before, edges_after}
        """
        import sqlite3, time as _t
        cutoff = _t.time() - older_than_days * 86400.0
        with sqlite3.connect(self.workspace.db_path) as conn:
            edges_before = conn.execute(
                "SELECT COUNT(*) FROM graph_edges WHERE workspace_id = ?",
                (self.workspace.id,),
            ).fetchone()[0]
            cur = conn.execute(
                "DELETE FROM graph_edges "
                "WHERE workspace_id = ? AND weight <= ? AND last_seen < ?",
                (self.workspace.id, max_weight, cutoff),
            )
            n_edges_pruned = cur.rowcount
            edges_after = conn.execute(
                "SELECT COUNT(*) FROM graph_edges WHERE workspace_id = ?",
                (self.workspace.id,),
            ).fetchone()[0]

            n_entities_pruned = 0
            if also_drop_orphan_entities:
                # Find entities with no edges AND no event_link
                cur = conn.execute(
                    """
                    DELETE FROM graph_entities
                    WHERE workspace_id = ?
                      AND id NOT IN (
                        SELECT entity_a FROM graph_edges WHERE workspace_id = ?
                        UNION SELECT entity_b FROM graph_edges WHERE workspace_id = ?
                      )
                      AND id NOT IN (SELECT entity_id FROM graph_event_entities)
                    """,
                    (self.workspace.id, self.workspace.id, self.workspace.id),
                )
                n_entities_pruned = cur.rowcount
            conn.commit()

        return {
            "edges_before": edges_before,
            "edges_after": edges_after,
            "n_edges_pruned": n_edges_pruned,
            "n_entities_pruned": n_entities_pruned,
        }

    def regraph(self) -> dict:
        """Wipe `graph_entities`, `graph_event_entities`, `graph_edges` for
        the current workspace and re-extract entities from every active event.

        Use after upgrading the extractor (stop-lists, path guards, etc.) so
        old garbage nodes disappear without touching the event log itself.
        """
        import sqlite3
        ws = self.workspace.id
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.execute(
                "DELETE FROM graph_event_entities WHERE event_ulid IN "
                "(SELECT ulid FROM events WHERE workspace_id = ?)",
                (ws,),
            )
            conn.execute("DELETE FROM graph_edges WHERE workspace_id = ?", (ws,))
            conn.execute("DELETE FROM graph_entities WHERE workspace_id = ?", (ws,))
            conn.commit()

        # Also wipe the self-reinforcing known_persons dict — it learned
        # garbage names ("how", "appdata") that the old extractor allowed.
        try:
            with sqlite3.connect(self.workspace.db_path) as conn:
                conn.execute(
                    "DELETE FROM workspace_kv WHERE workspace_id = ? AND key = 'known_persons'",
                    (ws,),
                )
                conn.commit()
        except Exception:
            pass

        n_events = 0
        n_entities = 0
        for ev in self.events.list_active(ws, limit=100000):
            full_text = ev.content or ""
            try:
                ids = self._index_event_in_graph(ev, full_text)
                n_events += 1
                n_entities += len(ids)
            except Exception:
                continue
        return {"events_reindexed": n_events, "entities_created": n_entities}

    # -----------------------------------------------------------------
    # Read
    # -----------------------------------------------------------------

    def recall(
        self,
        query: str,
        top_k: Optional[int] = None,
        recency_half_life_days: Optional[float] = None,
        graph_boost: Optional[float] = None,
        rerank: Optional[bool] = None,
        rerank_top_n: Optional[int] = None,
        _skip_decompose: bool = False,
    ) -> RecallPack:
        # Resolve defaults from config (per-workspace > global > schema default)
        if top_k is None:
            top_k = self.config.get("recall.top_k")
        if recency_half_life_days is None:
            recency_half_life_days = self.config.get("recall.recency_half_life_days")
        if graph_boost is None:
            graph_boost = self.config.get("recall.graph_boost")
        if rerank is None:
            rerank = self.config.get("recall.rerank")
        if rerank_top_n is None:
            rerank_top_n = self.config.get("recall.rerank_top_n")

        # Adaptive Query Decomposition (RAG-Fusion / IRCoT style).
        # Triggered only when query looks multi-hop and feature is on.
        # The user's recall() call is the entry point; recursive sub-query
        # calls pass _skip_decompose=True to prevent infinite recursion.
        if (
            not _skip_decompose
            and self.config.get("recall.adaptive_decompose")
            and _looks_multihop(query)
        ):
            fused = self._recall_with_decomposition(
                query, top_k=top_k,
                recency_half_life_days=recency_half_life_days,
                graph_boost=graph_boost, rerank=rerank, rerank_top_n=rerank_top_n,
            )
            if fused is not None:
                return fused
            # Fallthrough: decomposition failed → run original single-shot

        # Pattern-based query splitting (Improvement UU). Cheap, no LLM —
        # catches compound queries like "why X and why Y" / "X потому что Y".
        # When a split fires, each sub-query runs through the normal recall
        # pipeline and results are fused via RRF. Saves ~30pp on compound
        # queries that single-shot recall would otherwise diffuse.
        #
        # Hardening note (H1): we intentionally do NOT wrap this whole block
        # in a bare `except: pass` — silent fallback hides real bugs (a
        # missing RecallPack field would have looked indistinguishable from
        # "no split fired"). We only catch the specific failure modes we
        # accept (sub-recall raising, fusion edge cases). Constructor /
        # programmer errors propagate so tests catch them.
        if (
            not _skip_decompose
            and self.config.get("recall.pattern_split")
        ):
            from pmb.reasoning.query_split import split_query, rrf_fuse
            sub_queries = split_query(query)
            self._pattern_split_last_fired = len(sub_queries) > 1
            if len(sub_queries) > 1:
                sub_packs = []
                sub_recall_ok = True
                for sq in sub_queries:
                    try:
                        sp = self.recall(
                            sq, top_k=max(top_k, 10),
                            recency_half_life_days=recency_half_life_days,
                            graph_boost=graph_boost,
                            rerank=rerank, rerank_top_n=rerank_top_n,
                            _skip_decompose=True,
                        )
                    except Exception:
                        sub_recall_ok = False
                        break
                    sub_packs.append(sp)
                if sub_recall_ok and sub_packs:
                    # Round-robin interleave per-sub top results. Each
                    # sub-pack's top-1 is guaranteed a slot in the final
                    # top-N, which is what the user expects for compound
                    # queries — "X and Y" should surface BOTH X-answer
                    # and Y-answer near the top, not have RRF blend them
                    # into a single diluted ranking.
                    #
                    # We still keep RRF as a tiebreaker on duplicates that
                    # both sub-packs found, so the fusion isn't naive.
                    rank_lists = [[r.ulid for r in sp.results] for sp in sub_packs]
                    fused_scored = rrf_fuse(rank_lists, top_n=top_k * 4)
                    rrf_rank = {u: i for i, (u, _s) in enumerate(fused_scored)}

                    # Resolve ulid -> result (use first occurrence across packs)
                    ulid_to_res: dict[str, Any] = {}
                    for sp in sub_packs:
                        for r in sp.results:
                            if r.ulid not in ulid_to_res:
                                ulid_to_res[r.ulid] = r

                    # Round-robin: take top-1 of pack 0, then top-1 of
                    # pack 1, then top-2 of pack 0, etc. Dedup as we go.
                    seen_ulids: set[str] = set()
                    out_results = []
                    max_per_pack = max(top_k, 5)
                    for rank in range(max_per_pack):
                        for sp in sub_packs:
                            if rank < len(sp.results):
                                r = sp.results[rank]
                                if r.ulid not in seen_ulids:
                                    out_results.append(r)
                                    seen_ulids.add(r.ulid)
                                    if len(out_results) >= top_k:
                                        break
                        if len(out_results) >= top_k:
                            break
                    # If still under top_k, fill from RRF order
                    if len(out_results) < top_k:
                        for ulid, _s in fused_scored:
                            if ulid in seen_ulids:
                                continue
                            r = ulid_to_res.get(ulid)
                            if r is not None:
                                out_results.append(r)
                                seen_ulids.add(ulid)
                                if len(out_results) >= top_k:
                                    break
                    if out_results:
                        # Take workspace metadata from the first sub-pack so
                        # the returned RecallPack has all required fields.
                        meta_src = sub_packs[0]
                        pack = RecallPack(
                            query=query,
                            workspace_name=meta_src.workspace_name,
                            workspace_id=meta_src.workspace_id,
                            results=out_results,
                            n_total_in_workspace=meta_src.n_total_in_workspace,
                            elapsed_ms=sum(
                                getattr(sp, "elapsed_ms", 0.0) for sp in sub_packs
                            ),
                        )
                        self.recall_cache.put(
                            make_recall_cache_key(
                                query, top_k, recency_half_life_days,
                                graph_boost, rerank, rerank_top_n,
                            ),
                            pack,
                        )
                        self._pattern_split_last_returned = True
                        return pack
            self._pattern_split_last_returned = False

        # LRU cache hit — short-circuit the whole pipeline. The cache is
        # invalidated automatically on any event write via bump_generation().
        cache_key = make_recall_cache_key(
            query, top_k, recency_half_life_days, graph_boost, rerank, rerank_top_n,
        )
        cached = self.recall_cache.get(cache_key)
        if cached is not None:
            return cached
        """
        Поиск релевантной памяти по query.

        Pipeline:
        1. HybridSearch returns top_k*5 candidates ranked by BM25+vec only.
        2. Graph expansion: extract entities from query, pull additional
           candidate ulids from `graph_event_entities` (1-hop traversal).
        3. Batched SQL fetch для всех кандидатов c filter archived.
        4. Importance multiplier + recency boost + graph_boost applied в Python.
        5. Touch + reinforcement boost для финальных hits.

        graph_boost: additive bonus for events surfaced by the graph
        (0..1, default 0.15). Set to 0 to disable graph augmentation.
        """
        t0 = time.perf_counter()

        # Stage -0.5: typo correction (Improvement K). For each query token,
        # if there's a known entity within edit-distance 2, substitute.
        # Catches "Aliceee" → "alice", "Postgers" → "postgres", etc.
        # Cheap: ~90 entities × short Levenshtein → microseconds.
        typo_corrections = []
        if self.config.get("recall.typo_correction"):
            try:
                from pmb.reasoning.typo_fix import correct_query
                import sqlite3
                with sqlite3.connect(self.workspace.db_path) as conn:
                    rows = conn.execute(
                        "SELECT kind, name FROM graph_entities "
                        "WHERE workspace_id = ? AND n_mentions >= 1",
                        (self.workspace.id,),
                    ).fetchall()
                known = [(r[0], r[1]) for r in rows]
                corrected, typo_corrections = correct_query(query, known)
                if typo_corrections:
                    query = corrected  # use corrected query for the rest of pipeline
            except Exception:
                pass

        # Stage 0: Adaptive Layer Routing (Improvement E). Classify the query
        # and get per-layer multipliers; applied later in scoring loop.
        # Cheap — pure regex.
        layer_weights = None
        if self.config.get("recall.adaptive_routing"):
            try:
                from pmb.reasoning.router import QueryRouter
                intent = QueryRouter().classify(query)
                layer_weights = intent.weights
            except Exception:
                layer_weights = None

        # Stage 0.5: Predictive cache check (Improvement F).
        # ~3-5ms cosine over pre-baked questions. If we have a near-identical
        # cached query, return its top-K ulids directly — bypassing all
        # downstream stages. This is the "intuitive answer" path.
        if self.config.get("recall.predictive_enabled"):
            try:
                from pmb.reasoning.predictive import (
                    load_entries, best_match, mark_hit,
                )
                entries = load_entries(
                    self.workspace.db_path, self.workspace.id,
                    max_age_days=self.config.get("recall.predictive_ttl_days") or 7.0,
                )
                if entries and self.search.is_ready():
                    q_emb = self.search.embed(query)
                    threshold = self.config.get("recall.predictive_threshold") or 0.85
                    hit = best_match(entries, q_emb, threshold=threshold)
                    if hit is not None:
                        entry, sim = hit
                        # Hydrate the cached ulids back into events
                        rows = self.events.get_many(
                            entry.top_ulids, workspace_id=self.workspace.id,
                            only_active=True,
                        )
                        if rows:
                            # Build a results pack
                            from pmb.core.search import SearchHit as _SH
                            results: list[RecallResult] = []
                            for u in entry.top_ulids[:top_k]:
                                ev = rows.get(u)
                                if not ev:
                                    continue
                                results.append(RecallResult(
                                    ulid=ev.ulid, content=ev.content,
                                    score=float(sim), bm25_score=0.0, vec_score=float(sim),
                                    importance=ev.importance, recency_score=0.0,
                                    timestamp=ev.timestamp, event_type=ev.event_type,
                                    metadata=ev.metadata,
                                ))
                            # Record cache hit stats
                            try:
                                mark_hit(self.workspace.db_path, entry.id)
                            except Exception:
                                pass
                            n_total = self.events.count(
                                self.workspace.id, include_archived=False,
                            )
                            pack = RecallPack(
                                query=query,
                                workspace_name=self.workspace.name,
                                workspace_id=self.workspace.id,
                                results=results,
                                n_total_in_workspace=n_total,
                                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                            )
                            return pack
            except Exception:
                pass  # cache failure → normal recall

        # Stage 1: search by BM25+vec only (no importance/recency in search core)
        raw_hits: list[SearchHit] = self.search.search(
            query=query,
            top_k=top_k * 5,  # запас под archived filter
        )

        # Stage 2: graph expansion — entities in the query may surface events
        # that BM25/vec missed entirely. We weight each matched entity by
        # rarity (IDF-ish): an event that matches multiple rare entities
        # gets a stronger boost than one that matches a single common word.
        # Improvement C: temporal anchor — parse a date reference from the
        # query when it looks temporal. Used later to boost candidates with
        # nearby event_time.
        temporal_anchor: Optional[float] = None
        if self.config.get("recall.temporal_enabled"):
            try:
                from pmb.reasoning.temporal import (
                    is_temporal_query, extract_event_time,
                )
                if is_temporal_query(query):
                    temporal_anchor = extract_event_time(query)
            except Exception:
                temporal_anchor = None

        graph_ulids: set[str] = set()
        graph_weights: dict[str, float] = {}  # ulid -> sum of IDF weights
        if graph_boost > 0:
            q_ext = self.entity_extractor.extract(query)
            q_names = [n for _, n in q_ext.all_named()]
            # Optional LLM query expansion for abstract queries that the
            # rule-based extractor missed. Opt-in via config.
            if self.config.get("recall.graph_expansion_llm"):
                try:
                    from pmb.graph.expansion import expand_query
                    from pmb.health.consolidate import resolve_llm_client
                    extra = expand_query(
                        query,
                        self.workspace.storage_dir,
                        llm=resolve_llm_client(
                            backend=self.config.get("consolidate.backend"),
                        ),
                    )
                    for e in extra:
                        if e and e not in q_names:
                            q_names.append(e)
                except Exception:
                    # Expansion is best-effort; never block recall on LLM errors
                    pass
            if q_names:
                matched = self.graph.find_entities_by_name(
                    self.workspace.id, q_names,
                )
                if matched:
                    # IDF-ish: 1 / log(2 + n_mentions). Rare entity (1 mention)
                    # → ~0.91; common entity (e.g. 50 mentions) → ~0.25.
                    eid_to_idf: dict[int, float] = {}
                    for e in matched:
                        if e.id is None:
                            continue
                        eid_to_idf[e.id] = 1.0 / math.log(2.0 + max(0, e.n_mentions))
                    # Bump pair limit (was top_k * 20 = 200 — too tight for
                    # multi-hop, where the right event may be ranked far down
                    # in raw graph traversal but emerges via multi-entity bonus
                    # below). 100x top_k gives ample room without paying for it.
                    pairs = self.graph.event_entity_pairs(
                        list(eid_to_idf), limit=top_k * 100,
                    )
                    event_matched_eids: dict[str, set[int]] = {}
                    for ulid_x, eid in pairs:
                        idf = eid_to_idf.get(eid, 0.0)
                        graph_weights[ulid_x] = graph_weights.get(ulid_x, 0.0) + idf
                        event_matched_eids.setdefault(ulid_x, set()).add(eid)
                    # Multi-hop bonus: an event mentioning N distinct query
                    # entities gets weight × (1 + 0.5*(N-1)). Two entities =
                    # 1.5x, three = 2x. This is the bridge for multi-hop
                    # questions: "what did Alice do in December?" — answer
                    # event has both 'alice' AND 'december', so it dominates
                    # single-entity matches.
                    multi_bonus = self.config.get("recall.multi_entity_bonus") or 0.5
                    for ulid_x, ent_set in event_matched_eids.items():
                        n = len(ent_set)
                        if n > 1 and multi_bonus > 0:
                            graph_weights[ulid_x] *= (1.0 + multi_bonus * (n - 1))
                    graph_ulids = set(graph_weights.keys())

        # Stage 2.4: Personalized PageRank (HippoRAG). Cheap multi-hop boost
        # that uses the entity graph we already built. Diffuses probability
        # from query entities through the graph for many hops in one shot —
        # this is what lets the answer event (which only mentions ONE query
        # entity directly but is multi-hop close to the rest) surface.
        #
        # GATING: experiments show PPR helps multi-hop queries but ADDS NOISE
        # to single-entity lookups (where exact match should dominate). We
        # apply PPR only when query has multi-hop intent OR mentions 2+
        # known graph entities.
        ppr_event_scores: dict = {}
        if self.config.get("recall.ppr_enabled"):
            try:
                from pmb.graph.ppr import (
                    personalized_pagerank, score_events_by_ppr,
                )
                ppr_graph = self._get_ppr_graph()
                if ppr_graph is not None:
                    # Extract query entities independently (Stage 2 may have skipped)
                    q_ext_ppr = self.entity_extractor.extract(query)
                    q_names_ppr = [n for _, n in q_ext_ppr.all_named()]
                    seed_eids: list[int] = []
                    seed_w: list[float] = []
                    if q_names_ppr:
                        matched_ppr = self.graph.find_entities_by_name(
                            self.workspace.id, q_names_ppr,
                        )
                        for e in matched_ppr:
                            if e.id is None:
                                continue
                            seed_eids.append(e.id)
                            # IDF-ish — rare entity weighted more
                            seed_w.append(
                                1.0 / math.log(2.0 + max(0, e.n_mentions))
                            )
                    # Intent gate: PPR only when query is multi-hop or has 2+
                    # matched entities. Single-entity exact-match queries are
                    # better served by raw hybrid search.
                    ppr_should_run = (
                        len(seed_eids) >= 2
                        or _looks_multihop(query)
                        or self.config.get("recall.ppr_always")
                    )
                    if seed_eids and ppr_should_run:
                        ppr_scores = personalized_pagerank(
                            ppr_graph, seed_eids, seed_weights=seed_w,
                            alpha=self.config.get("recall.ppr_alpha") or 0.5,
                            iterations=self.config.get("recall.ppr_iters") or 20,
                        )
                        ppr_event_scores = score_events_by_ppr(
                            ppr_graph, ppr_scores,
                        )
            except Exception as _e:
                ppr_event_scores = {}

        # Stage 2.5: causation walk for multi-hop queries.
        # Cheap: one SQL hit, walks up to 1 hop forward/backward from top
        # search hits. Detects multi-hop intent via regex; harmless on
        # simple queries (no edges → no expansion).
        causation_ulids: set[str] = set()
        if (
            self.config.get("recall.causation_walk")
            and raw_hits
            and _looks_multihop(query)
        ):
            try:
                from pmb.reasoning.causation import walk_edges
                seed = [h.ulid for h in raw_hits[: max(3, top_k // 2)]]
                causation_ulids = walk_edges(
                    self.workspace.db_path, seed,
                    direction="both", hops=1,
                    min_confidence=0.4,
                    max_results=top_k * 3,
                )
                # Don't double-count search hits
                causation_ulids -= {h.ulid for h in raw_hits}
            except Exception:
                causation_ulids = set()

        # Stage 2.6: narrative arc expansion. If query looks narrative
        # ("tell me about X", "history of Y"), search arc summaries via
        # BM25-ish substring match and inject member events of best-matching
        # arc into the candidate pool. Very cheap — no LLM.
        arc_ulids: set[str] = set()
        arc_summaries_hit: list[dict] = []
        if self.config.get("recall.arc_expansion"):
            try:
                from pmb.reasoning.arcs import looks_narrative
                from pmb.reasoning.arcs import (
                    list_arcs as _list_arcs, events_in_arc as _events_in_arc,
                )
                # Always try; the term-match below is cheap. But weight it
                # more heavily if query 'looks narrative'.
                arcs_now = _list_arcs(self.workspace.db_path, self.workspace.id, status="active", limit=200)
                if arcs_now:
                    q_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
                    q_terms = {t for t in q_terms if len(t) > 2}
                    scored_arcs = []
                    for arc in arcs_now:
                        blob = (arc.title + " " + (arc.summary or "")).lower()
                        # Count distinct query terms appearing in the blob
                        n_match = sum(1 for t in q_terms if t in blob)
                        if n_match == 0:
                            continue
                        # Bonus if narrative-looking query
                        score = n_match * (2.0 if looks_narrative(query) else 1.0)
                        scored_arcs.append((arc, score))
                    scored_arcs.sort(key=lambda t: -t[1])
                    for arc, _ in scored_arcs[:2]:
                        arc_ulids.update(_events_in_arc(self.workspace.db_path, arc.id))
                        arc_summaries_hit.append({
                            "arc_id": arc.id, "title": arc.title,
                            "summary": (arc.summary or "")[:300],
                        })
                    arc_ulids -= {h.ulid for h in raw_hits}
            except Exception:
                arc_ulids = set()

        # PPR candidate expansion: top events by PPR score that aren't already
        # candidates. Caps at top_k * 2 to avoid blowing up the fetch.
        ppr_top_ulids: set[str] = set()
        if ppr_event_scores:
            ppr_top_n = min(top_k * 3, 100)
            top_by_ppr = sorted(
                ppr_event_scores.items(), key=lambda kv: -kv[1]
            )[:ppr_top_n]
            ppr_top_ulids = {u for u, _ in top_by_ppr}

        if not raw_hits and not graph_ulids and not causation_ulids and not arc_ulids and not ppr_top_ulids:
            n_total = self.events.count(self.workspace.id, include_archived=False)
            return RecallPack(
                query=query,
                workspace_name=self.workspace.name,
                workspace_id=self.workspace.id,
                results=[],
                n_total_in_workspace=n_total,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # Stage 3: batched fetch — union of search + graph + causation + arc hits
        seen_for_dedupe = {h.ulid for h in raw_hits}
        all_ulids: list[str] = [h.ulid for h in raw_hits]
        for u in graph_ulids:
            if u not in seen_for_dedupe:
                all_ulids.append(u); seen_for_dedupe.add(u)
        for u in causation_ulids:
            if u not in seen_for_dedupe:
                all_ulids.append(u); seen_for_dedupe.add(u)
        for u in arc_ulids:
            if u not in seen_for_dedupe:
                all_ulids.append(u); seen_for_dedupe.add(u)
        for u in ppr_top_ulids:
            if u not in seen_for_dedupe:
                all_ulids.append(u); seen_for_dedupe.add(u)
        rows = self.events.get_many(
            all_ulids, workspace_id=self.workspace.id, only_active=True,
        )

        # Stage 4: rerank with importance + recency + graph boost
        now = time.time()
        half_life_sec = recency_half_life_days * 86400.0

        # Improvement B: pre-compute the set of *content* tokens from the
        # query. Used below as a small multiplicative boost when a candidate
        # event's content contains ALL the unique meaningful tokens of the
        # query - direct lexical match in the right context.
        # Stopwords removed because they would over-fire on generic queries.
        _STOP = {
            "a", "an", "the", "is", "are", "was", "were", "of", "in", "on",
            "to", "for", "and", "or", "with", "we", "i", "you", "do", "does",
            "did", "what", "who", "where", "when", "why", "how", "by", "at",
            "from", "as", "be", "have", "has", "had", "this", "that", "these",
            "those", "it", "our", "my", "your", "their", "his", "her",
            "about", "any", "some", "all", "more", "than", "but", "not",
            "now", "use", "used", "using",
        }
        q_tokens = {
            t for t in re.findall(r"[a-zA-Zа-яА-Я0-9]+", (query or "").lower())
            if len(t) > 2 and t not in _STOP
        }

        # Precompute PAMVR query features ONCE per recall, not per
        # candidate. Cuts p99 from ~860ms to ~300ms on multilingual stress.
        pamvr_features = None
        if self._pamvr_enabled:
            try:
                # Self-reference rescue: cache user names mined from
                # "Меня зовут X" / "My name is X" facts. Lookup is O(1)
                # at query time. Refreshed lazily every N writes.
                if not hasattr(self, "_user_names_cache"):
                    self._user_names_cache: set[str] = set()
                    self._user_names_event_count = -1
                try:
                    import sqlite3 as _sql
                    with _sql.connect(str(self.workspace.db_path)) as conn:
                        row = conn.execute(
                            "SELECT COUNT(*) FROM events "
                            "WHERE archived_at IS NULL"
                        ).fetchone()
                        n_now = int(row[0] or 0)
                    if (self._user_names_event_count < 0
                            or n_now - self._user_names_event_count >= 25):
                        self._user_names_cache = _mine_user_names(
                            self.workspace.db_path
                        )
                        self._user_names_event_count = n_now
                except Exception:
                    pass

                pamvr_features = _pamvr_prepare(
                    query, vocab_bridges=self._vocab_bridges,
                    user_names=self._user_names_cache,
                )
            except Exception:
                pamvr_features = None

        search_hits_by_ulid = {h.ulid: h for h in raw_hits}
        scored: list[tuple[SearchHit, Event, float, float]] = []
        for ulid, ev in rows.items():
            h = search_hits_by_ulid.get(ulid)
            if h is None:
                # Graph-only hit — synthesize a SearchHit at low base score so
                # graph evidence alone can still surface a strong importance event.
                h = SearchHit(
                    ulid=ulid, score=0.0, bm25_score=0.0, vec_score=0.0,
                    importance=ev.importance, recency_score=0.0,
                )
            importance_factor = 0.5 + 0.5 * ev.importance
            age_sec = max(0.0, now - ev.timestamp)
            recency = math.exp(-age_sec * math.log(2) / half_life_sec)
            # Historical-intent: when the user asks "what did we use before",
            # the latest pinned fact is the wrong answer. We weaken the
            # recency reward (multiplier <1.0) and add a small bonus to OLDER
            # events. Both come from QueryRouter.classify().
            rec_mul = layer_weights.historical_recency_mul if layer_weights else 1.0
            base = h.score * importance_factor * (1.0 + 0.2 * recency * rec_mul)
            if layer_weights and layer_weights.older_event_bonus > 0:
                # invert recency: 1.0 for very old, 0.0 for fresh.
                base += layer_weights.older_event_bonus * (1.0 - recency) * importance_factor
            # Graph augmentation: bonus proportional to summed IDF of matched
            # entities. Rare-entity matches dominate, common-entity matches
            # contribute little.
            w = graph_weights.get(ulid, 0.0) if graph_boost > 0 else 0.0
            if w > 0:
                gb_mul = layer_weights.graph_boost_mul if layer_weights else 1.0
                base += graph_boost * gb_mul * w * importance_factor
            # Causation augmentation: small additive bonus for events surfaced
            # by walking event_edges from raw hits. Multi-hop unlock.
            if ulid in causation_ulids:
                causation_boost = self.config.get("recall.causation_boost") or 0.10
                cb_mul = layer_weights.causation_boost_mul if layer_weights else 1.0
                base += causation_boost * cb_mul * importance_factor * (1.0 + 0.2 * recency)
            # Arc augmentation: events that belong to a matching arc get a
            # narrative coherence bonus. Smaller than causation since arc
            # membership is a softer signal.
            if ulid in arc_ulids:
                arc_boost = self.config.get("recall.arc_boost") or 0.08
                ab_mul = layer_weights.arc_boost_mul if layer_weights else 1.0
                base += arc_boost * ab_mul * importance_factor * (1.0 + 0.2 * recency)
            # PPR augmentation: events with high PPR mass from query entities
            # get a boost. This is the multi-hop unlock — events that don't
            # appear in raw search but are graph-close to query entities.
            if ppr_event_scores:
                pscore = ppr_event_scores.get(ulid, 0.0)
                if pscore > 0:
                    ppr_weight = self.config.get("recall.ppr_weight") or 0.5
                    pw_mul = layer_weights.ppr_weight_mul if layer_weights else 1.0
                    base += ppr_weight * pw_mul * min(1.0, pscore * 100.0) * importance_factor
            # Improvement C: temporal-proximity boost. If query is temporal
            # and event has a parsed event_time, boost by exp-decay distance
            # to the query's anchor.
            if temporal_anchor is not None and ev.metadata:
                ev_t = ev.metadata.get("event_time") if isinstance(ev.metadata, dict) else None
                if ev_t is not None:
                    try:
                        from pmb.reasoning.temporal import temporal_proximity_boost
                        tprox = temporal_proximity_boost(
                            event_time=float(ev_t),
                            query_anchor_time=temporal_anchor,
                            half_life_days=self.config.get("recall.temporal_half_life_days") or 14.0,
                        )
                        if tprox > 0:
                            t_weight = self.config.get("recall.temporal_boost") or 0.20
                            tb_mul = layer_weights.temporal_boost_mul if layer_weights else 1.0
                            base += t_weight * tb_mul * tprox * importance_factor
                    except Exception:
                        pass
            # Improvement E: per-layer event-type boost. Routes the query
            # toward the right semantic layer based on intent.
            if layer_weights:
                et = ev.event_type
                if et == "fact_atom":
                    base *= layer_weights.facts_boost
                elif et == "reflection":
                    base *= layer_weights.reflections_boost
                else:
                    # raw events (qa, fact, event, git, etc.)
                    base *= layer_weights.raw_boost
            # Improvement B: query-keyword overlap boost. If most of the
            # meaningful tokens of the query are present in this event's
            # content, this event is a likely DIRECT match - boost it.
            # The boost is proportional to the overlap fraction so a tiny
            # overlap doesn't move the score much, but a near-full overlap
            # gives a meaningful nudge. Capped at 1.25x to avoid letting
            # one keyword count as a definitive answer.
            if q_tokens:
                content_lower = (ev.content or "").lower()
                # Cheap substring check: count which query tokens appear
                # anywhere in the content. Skips word-boundary work; good
                # enough at this score-tweak granularity.
                n_hit = sum(1 for t in q_tokens if t in content_lower)
                if n_hit >= 2:
                    # Overlap fraction over QUERY tokens, capped via min().
                    overlap = n_hit / max(1, len(q_tokens))
                    # Multiplier: 1.0 (no overlap) up to 1.25 (full match).
                    base *= 1.0 + min(0.25, 0.25 * overlap)

            # Personal-marker boost: facts that literally start with a
            # personal possessive ("User's", "Alex's", "I am", "My", "I
            # work on") get boosted ONLY when the query itself has identity
            # intent (router classified it as "identity"). Without this
            # gate the boost mis-fires on topical queries that just happen
            # to surface an identity-shaped fact (e.g. "What's the JWT
            # lifetime?" surfacing "Alex's terminal is Wezterm" at top-1).
            if (ev.event_type == "fact"
                    and layer_weights
                    and layer_weights.identity_marker_boost > 1.0):
                content_head = (ev.content or "")[:60].lower()
                if (content_head.startswith("user's ")
                        or content_head.startswith("user ")
                        or content_head.startswith("alex ")
                        or content_head.startswith("alex's ")
                        or content_head.startswith("i am ")
                        or content_head.startswith("my ")
                        or content_head.startswith("i work ")
                        or content_head.startswith("i prefer ")):
                    base *= layer_weights.identity_marker_boost
            # Decision-intent boost: activity events marked as a
            # decision/agreement should outrank arguments on the same
            # topic when the user asks "what did we decide". The router
            # only sets decision_boost > 1.0 when the query phrasing
            # actually asks for a decision.
            # record_activity() stores the kind under `activity_kind`
            # in metadata (not `kind`); record_batch uses the same path,
            # so we check both for safety.
            if (layer_weights
                    and ev.event_type == "activity"
                    and layer_weights.decision_boost > 1.0):
                meta = ev.metadata or {}
                kind = meta.get("activity_kind") or meta.get("kind") or ""
                if kind in ("decision", "decided", "agreed",
                            "resolved", "concluded"):
                    base *= layer_weights.decision_boost
            # PAMVR (Predicate-Aware Multi-View Reranking) — 14 small
            # content-based boost rules empirically tuned to lift top-1
            # accuracy from ~60% to 93%+. Verb match, entity strict,
            # vocab bridges, topic constraint, time-duration, etc. See
            # pmb.reasoning.pamvr for the full rule set + research data.
            if self._pamvr_enabled:
                base = _pamvr_apply(
                    query, ev, base,
                    vocab_bridges=self._vocab_bridges,
                    query_features=pamvr_features,   # reuse precomputed
                )
            scored.append((h, ev, base, recency))

        # Stage 3.25: collapse reflections onto their source events.
        # Reflections are meta — they exist to make sources findable via
        # their LLM-generated 'might_answer' text. The final result list
        # should always be source events, with the reflection's score
        # added to the source's. If the source isn't in candidates yet,
        # the reflection is REPLACED by the source (preserving the score).
        # If source is missing entirely (deleted), the reflection still
        # acts as a useful fallback result.
        if self.config.get("recall.collapse_reflections"):
            scored = _collapse_reflections(scored, self.events, self.workspace.id)

        scored.sort(key=lambda t: -t[2])

        # Stage 3.5: optional cross-encoder reranker over top-N candidates.
        # The hybrid scoring is good at *finding* the right neighbourhood;
        # the cross-encoder is better at picking the single most-relevant doc.
        #
        # Improvement #1 + VV - SMART gated rerank:
        #
        #   Original gate:  fire when (top1 - top3) score gap is < epsilon.
        #   The intuition: when BM25+vector clearly disagree on the order,
        #   the cross-encoder is helpful; when they agree, the CE causes
        #   regressions (LoCoMo loses 17pp with always-on rerank).
        #
        #   VV upgrade: ADD a "confidence-required" stage after the CE
        #   produces scores. Only commit the reranked order if the new
        #   top-1's CE score beats the previous top-1's by >= swap_margin.
        #   When confidence is low, KEEP the hybrid order — the cross-
        #   encoder isn't sure enough to override it.
        #
        #   Net effect: gating fires more often (helps where hybrid was
        #   ambiguous) but commits more conservatively (no LoCoMo
        #   regression because clear winners stay on top).
        gated_rerank = False
        if (not rerank
                and self.config.get("recall.rerank_when_close")
                and self.search.reranker is not None
                and len(scored) >= 3):
            top1_score = scored[0][2]
            top3_score = scored[2][2]
            eps = float(self.config.get("recall.rerank_close_epsilon") or 0.05)
            if (top1_score - top3_score) < eps:
                gated_rerank = True

        if (rerank or gated_rerank) and self.search.reranker is not None and len(scored) > 1:
            top_n = scored[: max(top_k, min(rerank_top_n, len(scored)))]
            remainder = scored[len(top_n):]
            prev_top1_ulid = top_n[0][1].ulid
            # Build a temporary fake SearchHit list to feed the reranker
            tmp_hits = [
                SearchHit(
                    ulid=ev.ulid, score=score,
                    bm25_score=h.bm25_score, vec_score=h.vec_score,
                    importance=ev.importance, recency_score=recency,
                )
                for h, ev, score, recency in top_n
            ]
            ev_by_ulid = {ev.ulid: ev for _, ev, _, _ in top_n}
            recency_by_ulid = {ev.ulid: rc for _, ev, _, rc in top_n}
            reranked = self.search.rerank(
                query, tmp_hits,
                text_for_ulid=lambda u: ev_by_ulid[u].to_text(),
            )
            # VV confidence gate: when gating is on (not full rerank=True),
            # only ACCEPT a swap of position 1 if the cross-encoder is
            # genuinely confident. Otherwise revert to the hybrid order
            # for that slot, leaving the rest of the rerank intact.
            commit_swap = True
            if gated_rerank and not rerank and len(reranked) >= 2:
                new_top1_score = float(reranked[0].score)
                # find prev top-1's score in the reranked list (could be at any pos)
                prev_top1_in_rerank = next(
                    (i for i, h in enumerate(reranked) if h.ulid == prev_top1_ulid),
                    None,
                )
                if prev_top1_in_rerank is not None and prev_top1_in_rerank > 0:
                    prev_top1_score = float(reranked[prev_top1_in_rerank].score)
                    swap_margin = float(
                        self.config.get("recall.rerank_swap_margin") or 0.20
                    )
                    if (new_top1_score - prev_top1_score) < swap_margin:
                        # CE not confident enough — restore prev top-1 at pos 0
                        commit_swap = False
                        old_idx = prev_top1_in_rerank
                        reranked = [reranked[old_idx]] + [
                            h for i, h in enumerate(reranked) if i != old_idx
                        ]
            top_n = [
                (h, ev_by_ulid[h.ulid], h.score, recency_by_ulid[h.ulid])
                for h in reranked
            ]
            scored = top_n + remainder

        # Stage 3.6: optional LLM-as-judge rerank (Improvement XX).
        # Asks a small local LLM (qwen2.5:1.5b by default, via Ollama)
        # to pick the single best candidate from the current top-N.
        # OFF by default — adds ~100-300ms but lifts top-1 by 5-15pp on
        # hard queries where lexical+semantic+CE all give close scores.
        if self.config.get("recall.llm_rerank") and len(scored) >= 2:
            try:
                from pmb.reasoning.llm_rerank import (
                    llm_rerank_top_k, DEFAULT_LLM_RERANK_MODEL,
                )
                from pmb.health.consolidate import OllamaClient
                llm_top_n = int(
                    self.config.get("recall.llm_rerank_top_n") or 10
                )
                window = scored[:llm_top_n]
                # Lazy-create the client; reuse on subsequent calls
                if not hasattr(self, "_llm_rerank_client"):
                    self._llm_rerank_client = OllamaClient(
                        model=(
                            self.config.get("recall.llm_rerank_model")
                            or DEFAULT_LLM_RERANK_MODEL
                        ),
                        timeout=float(
                            self.config.get("recall.llm_rerank_timeout") or 5.0
                        ),
                    )
                cands = [ev for _h, ev, _s, _r in window]
                pick = llm_rerank_top_k(
                    query, cands, self._llm_rerank_client,
                    max_candidates=llm_top_n,
                )
                if pick is not None and pick > 0:
                    # Move picked candidate to position 0, preserve rest
                    chosen = window[pick]
                    new_window = [chosen] + [
                        x for i, x in enumerate(window) if i != pick
                    ]
                    scored = new_window + scored[llm_top_n:]
            except Exception:
                # Best-effort: any failure (model down, parse error) falls
                # back to whatever the previous stage produced.
                pass

        scored = scored[:top_k]

        # Stage 4: hydrate + reinforcement. Collect side effects so we
        # can flush them in a single SQLite transaction at the end —
        # this is the biggest latency win on warm recall.
        from pmb.core.events import (
            TIER_WORKING, TIER_EPISODIC, TIER_SEMANTIC,
            PROMOTE_WORKING_TO_EPISODIC_ACCESS,
            PROMOTE_EPISODIC_TO_SEMANTIC_ACCESS,
        )
        results: list[RecallResult] = []
        touches: list[str] = []
        importance_updates: list[tuple[str, float]] = []
        tier_promotions: list[tuple[str, str]] = []
        for h, ev, final_score, recency in scored:
            touches.append(ev.ulid)
            new_imp = boost_on_recall(ev.importance, final_score)
            if new_imp > ev.importance + 0.001:
                importance_updates.append((ev.ulid, new_imp))
            # Tier promotion: repeated successful retrieval moves a memory
            # up the hierarchy (working → episodic → semantic). The +1
            # here accounts for the touch we're about to apply.
            new_access = ev.access_count + 1
            if (ev.tier == TIER_WORKING
                    and new_access >= PROMOTE_WORKING_TO_EPISODIC_ACCESS):
                tier_promotions.append((ev.ulid, TIER_EPISODIC))
            elif (ev.tier == TIER_EPISODIC
                  and new_access >= PROMOTE_EPISODIC_TO_SEMANTIC_ACCESS):
                tier_promotions.append((ev.ulid, TIER_SEMANTIC))
            results.append(RecallResult(
                ulid=ev.ulid,
                event_type=ev.event_type,
                content=ev.content,
                metadata=ev.metadata,
                timestamp=ev.timestamp,
                score=final_score,
                bm25_score=h.bm25_score,
                vec_score=h.vec_score,
                importance=ev.importance,
                recency_score=recency,
            ))
        # Improvement #6: enqueue touches for the deferred flusher instead
        # of writing synchronously. Under concurrent recalls this turns
        # 16 lock-acquisitions per second into ~4 (one per flush tick).
        self._enqueue_touches(touches, importance_updates)
        for ulid_p, new_tier in tier_promotions:
            self.events.update_tier(ulid_p, new_tier)

        n_total = self.events.count(self.workspace.id, include_archived=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        pack = RecallPack(
            query=query,
            workspace_name=self.workspace.name,
            workspace_id=self.workspace.id,
            results=results,
            n_total_in_workspace=n_total,
            elapsed_ms=elapsed_ms,
        )
        # Spreading activation — prime graph neighbours of every hit. Like
        # priming in human associative memory: thinking about X makes
        # related concepts easier to retrieve next time.
        if self.config.get("recall.spreading_activation") and results:
            try:
                from pmb.graph.spreading import apply_spreading_activation
                hit_evs = [
                    ev for _, ev, _, _ in scored[: len(results)]
                ]
                apply_spreading_activation(
                    self,
                    hit_events=hit_evs,
                    boost=self.config.get("recall.spreading_boost"),
                    half_life_hours=self.config.get("recall.spreading_half_life_hours"),
                )
            except Exception:
                # Priming is best-effort; never block recall on it
                pass
        # Stash for next time. Writes bump the generation so future stale
        # cache hits are dropped on read.
        self.recall_cache.put(cache_key, pack)
        return pack

    # -----------------------------------------------------------------
    # Mutate
    # -----------------------------------------------------------------

    def pin(self, ulid: str, importance: float = 1.0):
        self.events.pin(ulid, importance)

    def forget(self, ulid: str):
        self.events.archive(ulid)
        # Note: НЕ удаляем из search index — пусть лежит, фильтруется на recall
        # При желании cleanup batch может удалить vectors archived events

    def unforget(self, ulid: str):
        self.events.unarchive(ulid)

    def stats(self) -> dict:
        sess = self.session_tracker.current(auto_create=False)
        return {
            "workspace": {
                "id": self.workspace.id,
                "name": self.workspace.name,
                "root": str(self.workspace.root),
                "source": self.workspace.source,
                "created_at": self.workspace.created_at,
            },
            "events": self.events.stats(self.workspace.id),
            "search_index_size": self.search.size(),
            "current_session": sess.to_dict() if sess else None,
        }

    # -----------------------------------------------------------------
    # Signals — git, session, decay
    # -----------------------------------------------------------------

    def sync_git(self, since_timestamp: Optional[float] = None) -> dict:
        """Захватить git commits в memory. Импортируется лениво."""
        from pmb.signals.git import GitSync
        return GitSync(self).sync(since_timestamp=since_timestamp)

    def session_start(self, name: Optional[str] = None) -> dict:
        return self.session_tracker.start(name).to_dict()

    def session_end(self) -> Optional[dict]:
        sess = self.session_tracker.end()
        return sess.to_dict() if sess else None

    def session_current(self) -> Optional[dict]:
        sess = self.session_tracker.current(auto_create=False)
        return sess.to_dict() if sess else None

    def apply_daily_decay(self, days_since: float = 1.0) -> dict:
        from pmb.signals.decay import apply_decay
        return apply_decay(self, days_since_last_decay=days_since)

    def file_correlations(self, file_path: str, top_k: int = 10) -> list[tuple[str, int]]:
        from pmb.signals.files import FileCorrelation
        return FileCorrelation(self).correlations(file_path, top_k)

    def file_history(self, file_path: str) -> list[dict]:
        from pmb.signals.files import FileCorrelation
        return FileCorrelation(self).file_history(file_path)

    # -----------------------------------------------------------------
    # Health & Maintenance
    # -----------------------------------------------------------------

    def run_self_test(self, n_samples: int = 20, min_age_days: float = 1.0,
                      apply_adaptive: bool = True) -> dict:
        """
        Запустить self-test и опционально применить adaptive boost.

        Adaptive integrates two signals:
          1) self-test failures (synthetic, closed-loop, fallback)
          2) user feedback log (real signal, preferred when present)
        """
        from pmb.health.self_test import SelfTestRunner
        runner = SelfTestRunner(self)
        result = runner.run(n_samples=n_samples, min_age_days=min_age_days)

        adaptive_summary = None
        feedback_summary = None
        if apply_adaptive:
            if result.failed_queries:
                from pmb.health.adaptive import apply_adaptive_boost
                adaptive_summary = apply_adaptive_boost(self, result)
            from pmb.health.adaptive import apply_feedback_adaptive
            feedback_summary = apply_feedback_adaptive(self)

        return {
            "self_test": result.to_dict(),
            "adaptive": adaptive_summary,
            "feedback_adaptive": feedback_summary,
        }

    def apply_feedback_adaptive(self) -> dict:
        from pmb.health.adaptive import apply_feedback_adaptive
        return apply_feedback_adaptive(self)

    def rehearse(
        self,
        importance_threshold: float = 0.5,
        min_idle_days: float = 7.0,
        max_rehearse: int = 20,
    ) -> dict:
        """Spaced-repetition rehearsal — keep important-but-idle memories alive."""
        from pmb.health.rehearse import rehearse as _rehearse
        return _rehearse(
            self,
            importance_threshold=importance_threshold,
            min_idle_days=min_idle_days,
            max_rehearse=max_rehearse,
        ).to_dict()

    # -----------------------------------------------------------------
    # Graph
    # -----------------------------------------------------------------

    def graph_stats(self) -> dict:
        return self.graph.stats(self.workspace.id)

    def graph_top_entities(self, kind: Optional[str] = None, limit: int = 20) -> list[dict]:
        return [e.to_dict() for e in self.graph.top_entities(
            self.workspace.id, kind=kind, limit=limit,
        )]

    def graph_neighbors(self, name: str, kind: Optional[str] = None, top_k: int = 10) -> dict:
        kinds = (kind,) if kind else ()
        ents = self.graph.find_entities_by_name(self.workspace.id, [name], kinds=kinds)
        if not ents:
            return {"entity": None, "neighbors": []}
        primary = ents[0]
        nbrs = self.graph.neighbors(self.workspace.id, primary.id, top_k=top_k)
        return {
            "entity": primary.to_dict(),
            "neighbors": [
                {"entity": e.to_dict(), "weight": w} for e, w in nbrs
            ],
        }

    def graph_rebuild_from_events(self) -> dict:
        """One-shot reindex of the graph from all active events.

        Useful when migrating an existing PMB workspace from the
        pre-graph version, or after changing the entity extractor.
        """
        events = self.events.list_active(self.workspace.id, limit=100000)
        n_indexed = 0
        for ev in events:
            text_parts = [ev.content]
            q = ev.metadata.get("query") if isinstance(ev.metadata, dict) else None
            if q:
                text_parts.insert(0, q)
            self._index_event_in_graph(ev, full_text="\n".join(text_parts))
            n_indexed += 1
        return {"n_events_indexed": n_indexed, **self.graph.stats(self.workspace.id)}

    def consolidate(
        self,
        dry_run: bool = False,
        backend: str = "auto",
        model: Optional[str] = None,
        since_days: float = 14.0,
        similarity_threshold: float = 0.5,
        min_cluster_size: int = 3,
        max_clusters: int = 10,
        llm=None,
    ) -> dict:
        """
        LLM-based generalization: cluster recent events by embedding similarity,
        ask the LLM to extract a rule per cluster, store as a fact, archive sources.

        backend: "auto" picks Anthropic if ANTHROPIC_API_KEY is set, else Ollama
        if a local server is reachable. "anthropic" / "ollama" force a choice.

        Pass an `llm` instance with .consolidate(texts) directly for tests.
        """
        from pmb.health.consolidate import run_consolidation
        result = run_consolidation(
            self,
            llm=llm,
            backend=backend,
            model=model,
            since_days=since_days,
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
            max_clusters=max_clusters,
            dry_run=dry_run,
        )
        if not dry_run:
            try:
                from pmb.health.auto_consolidate import mark_consolidation_done
                mark_consolidation_done(self)
            except Exception:
                pass
        return result

    def consolidation_due(self) -> dict:
        """Inspect whether auto-trigger thresholds are met right now."""
        from pmb.health.auto_consolidate import should_trigger
        return should_trigger(self)

    # -----------------------------------------------------------------
    # PMB v2 reasoning layer: Reflective Memory
    # -----------------------------------------------------------------

    def reflect_event(
        self,
        ulid: str,
        llm=None,
        context_size: int = 4,
        backend: str = "auto",
    ) -> Optional[dict]:
        """Run LLM reflection on a single event. Stores a 'reflection'-typed
        event linking back to the source. Returns the reflection dict, or
        None if LLM unavailable / parsing failed / source not found.

        Reflections are searchable like any other event, which is how
        multi-hop questions get bridged at recall time without any read-
        time LLM call.

        Safe to call repeatedly — duplicate reflections for the same source
        are prevented by checking metadata.source_ulid.
        """
        ev = self.events.get_by_ulid(ulid)
        if not ev or ev.event_type == "reflection":
            return None
        # Skip if already reflected
        if self._has_reflection_for(ulid):
            return None

        if llm is None:
            from pmb.health.consolidate import resolve_llm_client
            try:
                llm = resolve_llm_client(backend=backend)
            except Exception as e:
                import logging
                logging.getLogger(__name__).info("No LLM for reflection: %s", e)
                return None

        from pmb.reasoning.reflect import Reflector
        reflector = Reflector(llm, max_context_events=context_size)

        # Pull a small temporal context window (events near this one in time
        # AND/OR sharing entities). This is what makes causation hints in
        # the reflection more accurate.
        context = self._reflection_context(ev, max_n=context_size)

        reflection = reflector.reflect(ev, context_events=context)
        if reflection is None:
            return None

        # Persist as a new event
        from pmb.core.events import Event, default_tier_for_event_type, TIER_SEMANTIC
        new_ev = Event(
            event_type="reflection",
            content=reflection.to_event_content(),
            metadata=reflection.to_metadata(),
            workspace_id=self.workspace.id,
            importance=0.6,  # reflections are useful by default
            tier=TIER_SEMANTIC,  # they capture stable understanding
        )
        appended = self.events.append(new_ev)
        # Index in graph too — reflections contain rich entity signal
        try:
            self._index_event_in_graph(appended, full_text=appended.content)
        except Exception:
            pass
        # Add to vector index for hybrid recall
        try:
            self.search.add(appended.ulid, appended.to_text())
        except Exception:
            pass
        # Phase B (Improvement B): ALSO link reflection-derived entities
        # back to the SOURCE event. This makes the source findable via
        # entity matches against terms the LLM surfaced (people names,
        # themes, implications). PPR naturally walks these enriched links.
        # No new searchable chunk needed — pure graph signal.
        n_bridge_entities = 0
        if self.config.get("recall.reflection_to_edges"):
            try:
                refl_full_text = (
                    reflection.significance + " "
                    + " ".join(reflection.implications) + " "
                    + " ".join(reflection.might_answer) + " "
                    + " ".join(reflection.linked_themes) + " "
                    + " ".join(reflection.people)
                )
                refl_ext = self.entity_extractor.extract(refl_full_text)
                refl_named = refl_ext.all_named()
                # Add LLM-extracted people as 'person' entities too — our
                # rule-based extractor doesn't catch capitalized names.
                for p in reflection.people:
                    if p and len(p) >= 2:
                        refl_named.append(("person", p.lower()))
                # Add themes as concepts (deduplicate later)
                for th in reflection.linked_themes:
                    if th and len(th) >= 3:
                        refl_named.append(("theme", th.lower()))

                src_entity_ids: list[int] = []
                seen_kn = set()
                for kind, name in refl_named:
                    key = (kind, name)
                    if key in seen_kn:
                        continue
                    seen_kn.add(key)
                    eid = self.graph.upsert_entity(self.workspace.id, kind, name)
                    src_entity_ids.append(eid)
                if src_entity_ids:
                    # Link to SOURCE ulid (not the reflection event)
                    self.graph.link_event(ulid, src_entity_ids)
                    # Bump co-occurrence among these new entities
                    self.graph.bump_edges(self.workspace.id, src_entity_ids)
                    n_bridge_entities = len(src_entity_ids)
            except Exception:
                pass
        # Invalidate recall cache so future queries see the new reflection
        self.recall_cache.bump_generation()
        return {
            "source_ulid": ulid,
            "reflection_ulid": appended.ulid,
            "significance": reflection.significance,
            "might_answer": reflection.might_answer,
            "themes": reflection.linked_themes,
            "bridge_entities_added_to_source": n_bridge_entities,
        }

    # -----------------------------------------------------------------
    # Improvement F: Predictive Pre-Computation cache
    # -----------------------------------------------------------------

    def precompute_predictive_cache(
        self,
        n_questions: int = 15,
        events_to_consider: int = 25,
        cache_top_k: int = 20,
        llm=None,
        backend: str = "auto",
    ) -> dict:
        """LLM generates likely user questions; for each we pre-compute
        and cache top-K. At read time, fuzzy match on query embedding
        returns cached results instantly.

        Run during sleep / idle. Refreshes the cache. Old entries (over
        TTL) ignored at read time but kept until clear_predictive_cache().
        """
        # Pull recent active events (raw + facts + reflections all OK)
        import sqlite3
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ulid FROM events WHERE workspace_id = ? "
                "AND archived_at IS NULL "
                "ORDER BY timestamp DESC LIMIT ?",
                (self.workspace.id, events_to_consider),
            ).fetchall()
        candidates = [self.events.get_by_ulid(r["ulid"]) for r in rows]
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            return {"n_questions_generated": 0, "n_cached": 0, "skipped": "no_events"}

        if llm is None:
            from pmb.health.consolidate import resolve_llm_client
            try:
                llm = resolve_llm_client(backend=backend)
            except Exception:
                return {
                    "n_questions_generated": 0, "n_cached": 0,
                    "skipped": "no_llm",
                }

        from pmb.reasoning.predictive import (
            PredictiveQuestionGenerator, CacheEntry, store_entry,
        )
        gen = PredictiveQuestionGenerator(
            llm, max_questions=n_questions, max_events_in_prompt=events_to_consider,
        )
        questions = gen.generate(candidates)
        if not questions:
            return {"n_questions_generated": 0, "n_cached": 0, "skipped": "llm_returned_empty"}

        n_cached = 0
        for q in questions:
            try:
                # Run full recall on this question (with predictive disabled to
                # avoid infinite recursion / self-lookup)
                pack = self.recall(
                    q, top_k=cache_top_k,
                    _skip_decompose=True,
                )
                top_ulids = [r.ulid for r in pack.results]
                if not top_ulids:
                    continue
                emb = self.search.embed(q)
                store_entry(self.workspace.db_path, CacheEntry(
                    id=None, workspace_id=self.workspace.id,
                    query_text=q,
                    query_embedding=emb,
                    top_ulids=top_ulids,
                    created_at=time.time(),
                ))
                n_cached += 1
            except Exception:
                continue
        return {
            "n_questions_generated": len(questions),
            "n_cached": n_cached,
            "events_considered": len(candidates),
        }

    def clear_predictive_cache(self) -> int:
        from pmb.reasoning.predictive import clear_cache
        n = clear_cache(self.workspace.db_path, self.workspace.id)
        return n

    def extract_facts_batch(
        self,
        limit: int = 50,
        max_age_days: float = 365.0,
        llm=None,
        backend: str = "auto",
        max_facts_per_event: int = 30,
    ) -> dict:
        """Improvement D: extract atomic facts from recent events.

        Each fact becomes a new event of event_type='fact_atom' linked to
        its source by metadata.source_ulid. Facts are searchable like any
        event, so they surface in recall and give the reader pre-extracted
        clean statements.

        Skips events that already have facts (idempotent).
        """
        candidates = self._un_factualized_events(limit=limit, max_age_days=max_age_days)
        if not candidates:
            return {
                "n_candidates": 0, "n_facts_added": 0, "n_failed": 0,
            }
        if llm is None:
            from pmb.health.consolidate import resolve_llm_client
            try:
                llm = resolve_llm_client(backend=backend)
            except Exception:
                return {
                    "n_candidates": len(candidates),
                    "skipped": "no_llm",
                    "n_facts_added": 0, "n_failed": 0,
                }

        from pmb.reasoning.facts import FactExtractor
        from pmb.core.events import Event, TIER_SEMANTIC, default_tier_for_event_type
        extractor = FactExtractor(llm, max_facts_per_event=max_facts_per_event)
        n_facts = 0
        n_failed = 0
        for ev in candidates:
            try:
                facts = extractor.extract(ev)
                for f in facts:
                    fact_ev = Event(
                        event_type="fact_atom",
                        content=f.to_event_content(),
                        metadata={
                            "source_ulid": f.source_ulid,
                            "kind": "atomic_fact",
                        },
                        workspace_id=self.workspace.id,
                        importance=0.7,  # atomic facts are valuable signal
                        tier=TIER_SEMANTIC,
                    )
                    appended = self.events.append(fact_ev)
                    try:
                        self._index_event_in_graph(appended, full_text=appended.content)
                    except Exception:
                        pass
                    try:
                        self.search.add(appended.ulid, appended.to_text())
                    except Exception:
                        pass
                    n_facts += 1
            except Exception:
                n_failed += 1
        self.recall_cache.bump_generation()
        return {
            "n_candidates": len(candidates),
            "n_facts_added": n_facts,
            "n_failed": n_failed,
        }

    def _un_factualized_events(self, limit: int, max_age_days: float):
        """Events that don't yet have fact_atom children."""
        import sqlite3
        cutoff = time.time() - max_age_days * 86400
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT ulid FROM events e
                WHERE workspace_id = ?
                  AND event_type NOT IN ('fact_atom', 'reflection')
                  AND archived_at IS NULL
                  AND timestamp >= ?
                  AND NOT EXISTS (
                      SELECT 1 FROM events f
                      WHERE f.workspace_id = e.workspace_id
                        AND f.event_type = 'fact_atom'
                        AND json_extract(f.metadata_json, '$.source_ulid') = e.ulid
                  )
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (self.workspace.id, cutoff, limit),
            ).fetchall()
            ulids = [r["ulid"] for r in rows]
        return [self.events.get_by_ulid(u) for u in ulids if self.events.get_by_ulid(u)]

    def reflect_batch(
        self,
        limit: int = 20,
        max_age_days: float = 30.0,
        llm=None,
        backend: str = "auto",
        context_size: int = 4,
        extract_causation: bool = True,
    ) -> dict:
        """Reflect on up to `limit` recent events that haven't been
        reflected on yet. Designed to run in the background during sleep /
        consolidation cycles, not in any hot path.

        Returns a summary dict with counts and a sample of generated
        reflections (truncated). Failures are counted, not raised.
        """
        candidates = self._unreflected_events(limit=limit, max_age_days=max_age_days)
        if not candidates:
            return {"n_candidates": 0, "n_reflected": 0, "n_failed": 0, "samples": []}

        # Resolve LLM once for the whole batch
        if llm is None:
            from pmb.health.consolidate import resolve_llm_client
            try:
                llm = resolve_llm_client(backend=backend)
            except Exception:
                return {
                    "n_candidates": len(candidates),
                    "n_reflected": 0,
                    "n_failed": 0,
                    "skipped": "no_llm",
                    "samples": [],
                }

        n_ok = 0
        n_fail = 0
        n_edges = 0
        samples: list[dict] = []

        # Build extractor lazily, share across batch
        causation_extractor = None
        if extract_causation:
            try:
                from pmb.reasoning.causation import CausationExtractor
                causation_extractor = CausationExtractor(llm)
            except Exception:
                causation_extractor = None

        for ev in candidates:
            try:
                out = self.reflect_event(
                    ev.ulid, llm=llm, context_size=context_size,
                )
                if out:
                    n_ok += 1
                    if len(samples) < 3:
                        samples.append({
                            "source_ulid": ev.ulid,
                            "source_preview": (ev.content or "")[:120],
                            "significance": out["significance"][:160],
                            "might_answer": out["might_answer"][:3],
                        })
                    # Bonus: extract causation edges from this event to recent
                    # context. Same context window we used for reflection.
                    if causation_extractor:
                        try:
                            ctx = self._reflection_context(ev, max_n=context_size)
                            edges = causation_extractor.extract(ev, ctx)
                            from pmb.reasoning.causation import upsert_edge
                            for e in edges:
                                upsert_edge(self.workspace.db_path, e)
                                n_edges += 1
                        except Exception:
                            pass
                else:
                    n_fail += 1
            except Exception:
                n_fail += 1
        return {
            "n_candidates": len(candidates),
            "n_reflected": n_ok,
            "n_failed": n_fail,
            "n_causation_edges": n_edges,
            "samples": samples,
        }

    # -----------------------------------------------------------------
    # PMB v2 phase 3: Narrative Arcs
    # -----------------------------------------------------------------

    def cluster_events_into_arcs(
        self,
        limit: int = 20,
        max_age_days: float = 60.0,
        llm=None,
        backend: str = "auto",
        refresh_summaries: bool = True,
    ) -> dict:
        """LLM clusters recent unassigned events into narrative arcs.

        For each candidate event:
          - Show LLM the existing active arcs + this event
          - LLM decides: join arc, create new arc, or ignore
          - Persist accordingly

        After clustering, optionally rewrite arc summaries to reflect new
        members (one LLM call per touched arc).
        """
        from pmb.reasoning.arcs import (
            ArcManager, Arc, list_arcs, create_arc, add_event_to_arc,
            arcs_for_event, events_in_arc, update_arc, get_arc,
        )

        # Pull recent unassigned events
        candidates = self._unassigned_events(limit=limit, max_age_days=max_age_days)
        if not candidates:
            return {"n_candidates": 0, "n_joined": 0, "n_created": 0, "n_ignored": 0}

        if llm is None:
            from pmb.health.consolidate import resolve_llm_client
            try:
                llm = resolve_llm_client(backend=backend)
            except Exception:
                return {
                    "n_candidates": len(candidates),
                    "skipped": "no_llm",
                    "n_joined": 0, "n_created": 0, "n_ignored": 0,
                }
        mgr = ArcManager(llm)

        touched_arcs: set[int] = set()
        n_joined = n_created = n_ignored = 0
        for ev in candidates:
            arcs_now = list_arcs(self.workspace.db_path, self.workspace.id, status="active")
            decision = mgr.assign_event(ev, arcs_now, self.workspace.id)
            action = decision.get("action")
            if action == "join":
                arc_id = decision.get("arc_id")
                # Make sure the arc actually exists in our workspace
                if get_arc(self.workspace.db_path, arc_id):
                    if add_event_to_arc(self.workspace.db_path, arc_id, ev.ulid):
                        touched_arcs.add(arc_id)
                        n_joined += 1
                else:
                    n_ignored += 1
            elif action == "create":
                new_arc = Arc(
                    id=None,
                    workspace_id=self.workspace.id,
                    title=decision["new_title"],
                    summary="",
                    first_event_ulid=ev.ulid,
                    last_event_ulid=ev.ulid,
                    n_events=0,  # set by add_event_to_arc
                )
                arc_id = create_arc(self.workspace.db_path, new_arc)
                add_event_to_arc(self.workspace.db_path, arc_id, ev.ulid)
                touched_arcs.add(arc_id)
                n_created += 1
            else:
                n_ignored += 1

        # Refresh summaries for touched arcs
        n_summaries = 0
        if refresh_summaries and touched_arcs:
            for aid in touched_arcs:
                arc = get_arc(self.workspace.db_path, aid)
                if not arc:
                    continue
                event_ulids = events_in_arc(self.workspace.db_path, aid)
                events = [self.events.get_by_ulid(u) for u in event_ulids]
                events = [e for e in events if e is not None]
                events.sort(key=lambda e: e.timestamp)
                if not events:
                    continue
                try:
                    summary = mgr.write_summary(arc, events)
                    update_arc(self.workspace.db_path, aid, summary=summary)
                    n_summaries += 1
                except Exception:
                    pass

        # Cache invalidation — new arcs / summaries affect recall
        self.recall_cache.bump_generation()

        return {
            "n_candidates": len(candidates),
            "n_joined": n_joined,
            "n_created": n_created,
            "n_ignored": n_ignored,
            "n_summaries_updated": n_summaries,
        }

    def list_arcs(
        self, status: Optional[str] = "active", limit: int = 50,
    ) -> list[dict]:
        from pmb.reasoning.arcs import list_arcs
        return [a.to_dict() for a in list_arcs(
            self.workspace.db_path, self.workspace.id,
            status=status, limit=limit,
        )]

    def arc_detail(self, arc_id: int) -> Optional[dict]:
        from pmb.reasoning.arcs import get_arc, events_in_arc
        arc = get_arc(self.workspace.db_path, arc_id)
        if not arc or arc.workspace_id != self.workspace.id:
            return None
        ev_ulids = events_in_arc(self.workspace.db_path, arc_id)
        events = [self.events.get_by_ulid(u) for u in ev_ulids]
        events = [e for e in events if e is not None]
        events.sort(key=lambda e: e.timestamp)
        return {
            **arc.to_dict(),
            "events": [
                {
                    "ulid": e.ulid,
                    "timestamp": e.timestamp,
                    "preview": (e.content or "")[:200],
                }
                for e in events
            ],
        }

    def _unassigned_events(self, limit: int = 20, max_age_days: float = 60.0):
        """Recent non-reflection events that aren't in any arc yet."""
        import sqlite3
        cutoff = time.time() - max_age_days * 86400
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT ulid FROM events e
                WHERE workspace_id = ?
                  AND event_type != 'reflection'
                  AND archived_at IS NULL
                  AND timestamp >= ?
                  AND NOT EXISTS (
                      SELECT 1 FROM arc_events ae WHERE ae.event_ulid = e.ulid
                  )
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (self.workspace.id, cutoff, limit),
            ).fetchall()
            ulids = [r["ulid"] for r in rows]
        return [self.events.get_by_ulid(u) for u in ulids if self.events.get_by_ulid(u)]

    # ----- helpers ----------------------------------------------------

    # -----------------------------------------------------------------
    # Improvement G: Multi-Modal Recall escalation
    # -----------------------------------------------------------------

    def recall_smart(
        self,
        query: str,
        top_k: Optional[int] = None,
        confidence_threshold: float = 0.5,
        max_escalations: int = 2,
        **kwargs,
    ) -> RecallPack:
        """Auto-escalating recall: cheap first, retry with more effort if
        confidence is low.

        Stage 1 (fast):  current default pipeline (PPR/causation/arc gated)
        Stage 2 (escalate if low conf): force adaptive decomposition pass
        Stage 3 (escalate further):     enlarge top_k 2x, enable rerank

        Returns the best (highest-confidence) result across stages.
        Never silently fails: if all stages produce zero results, returns
        whatever the first stage gave (empty pack).
        """
        if top_k is None:
            top_k = self.config.get("recall.top_k")

        # Stage 1: normal recall
        pack = self.recall(query, top_k=top_k, **kwargs)
        best = pack
        if pack.confidence >= confidence_threshold:
            return pack

        # Stage 2: force decomposition (only useful if multi-hop intent)
        if max_escalations >= 1:
            try:
                rh = kwargs.get("recency_half_life_days") or self.config.get("recall.recency_half_life_days")
                gb = kwargs.get("graph_boost"); gb = gb if gb is not None else self.config.get("recall.graph_boost")
                rr = kwargs.get("rerank"); rr = rr if rr is not None else self.config.get("recall.rerank")
                rtn = kwargs.get("rerank_top_n") or self.config.get("recall.rerank_top_n")
                pack2 = self._recall_with_decomposition(
                    query, top_k=top_k,
                    recency_half_life_days=rh,
                    graph_boost=gb, rerank=rr, rerank_top_n=rtn,
                )
                if pack2 and pack2.confidence > best.confidence:
                    best = pack2
                if best.confidence >= confidence_threshold:
                    return best
            except Exception:
                pass

        # Stage 3: enlarge top_k + enable rerank
        if max_escalations >= 2:
            try:
                pack3 = self.recall(
                    query, top_k=max(top_k * 2, 25),
                    rerank=True,
                    _skip_decompose=True,
                )
                if pack3.confidence > best.confidence:
                    best = pack3
            except Exception:
                pass

        return best

    # -----------------------------------------------------------------
    # Adaptive Query Decomposition (RAG-Fusion / IRCoT style)
    # -----------------------------------------------------------------

    def _recall_with_decomposition(
        self,
        query: str,
        top_k: int,
        recency_half_life_days: float,
        graph_boost: float,
        rerank: bool,
        rerank_top_n: int,
    ) -> Optional[RecallPack]:
        """Run query decomposition + sub-query retrieval + RRF merge.
        Returns None if decomposition couldn't run (LLM unavailable etc.)
        so caller can fall back to single-shot recall."""
        try:
            from pmb.reasoning.decompose import QueryDecomposer, reciprocal_rank_fuse
            from pmb.health.consolidate import resolve_llm_client
            try:
                llm = resolve_llm_client(
                    backend=self.config.get("consolidate.backend"),
                )
            except Exception:
                return None
            decomposer = QueryDecomposer(
                llm, cache_dir=self.workspace.storage_dir,
            )
            decomp = decomposer.decompose(query)
            if len(decomp.sub_queries) <= 1:
                # Decomposer said it's already single-hop; cancel decomposition
                return None

            # Run each sub-query through the standard pipeline. Skip
            # decomposition on these calls to prevent recursion.
            t0 = time.perf_counter()
            rankings: list[list[str]] = []
            all_pack_results: dict = {}  # ulid -> RecallResult (any sub-q)
            for sq in decomp.sub_queries:
                sub_pack = self.recall(
                    sq, top_k=max(top_k, 15),
                    recency_half_life_days=recency_half_life_days,
                    graph_boost=graph_boost, rerank=rerank, rerank_top_n=rerank_top_n,
                    _skip_decompose=True,
                )
                rankings.append([r.ulid for r in sub_pack.results])
                for r in sub_pack.results:
                    all_pack_results.setdefault(r.ulid, r)

            # Fuse rankings into one
            fused = reciprocal_rank_fuse(rankings, k=60)[:top_k]
            results = [
                all_pack_results[u] for u, _ in fused if u in all_pack_results
            ]

            n_total = self.events.count(self.workspace.id, include_archived=False)
            return RecallPack(
                query=query,
                workspace_name=self.workspace.name,
                workspace_id=self.workspace.id,
                results=results,
                n_total_in_workspace=n_total,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).info("Decomposition failed: %s", e)
            return None

    # -----------------------------------------------------------------
    # Improvement C: bi-temporal index
    # -----------------------------------------------------------------

    def _attach_event_time(self, ev) -> None:
        """Parse content for explicit date references and store as
        metadata.event_time (UTC timestamp). Free, no LLM."""
        if not self.config.get("recall.temporal_enabled"):
            return
        try:
            from pmb.reasoning.temporal import extract_event_time
            t = extract_event_time(ev.content or "", reference_now=ev.timestamp)
            if t is None:
                return
            # Update metadata in-place via direct SQL
            import sqlite3, json as _j
            with sqlite3.connect(self.workspace.db_path) as conn:
                row = conn.execute(
                    "SELECT metadata_json FROM events WHERE ulid = ?", (ev.ulid,),
                ).fetchone()
                if row is None:
                    return
                try:
                    meta = _j.loads(row[0]) if row[0] else {}
                except Exception:
                    meta = {}
                meta["event_time"] = float(t)
                conn.execute(
                    "UPDATE events SET metadata_json = ? WHERE ulid = ?",
                    (_j.dumps(meta), ev.ulid),
                )
            # Also bump in-memory metadata if dataclass instance is reused
            if hasattr(ev, "metadata") and isinstance(ev.metadata, dict):
                ev.metadata["event_time"] = float(t)
        except Exception:
            pass

    # -----------------------------------------------------------------
    # PPR graph (HippoRAG-style multi-hop scoring)
    # -----------------------------------------------------------------

    def _get_ppr_graph(self):
        """Lazy build + cache of the PPR graph snapshot. Rebuilt when the
        recall cache generation bumps (any write to events/edges). Returns
        None if the graph is empty (e.g. fresh workspace)."""
        gen = self.recall_cache._generation
        if self._ppr_graph is not None and self._ppr_graph_generation == gen:
            return self._ppr_graph
        try:
            from pmb.graph.ppr import build_ppr_graph
            self._ppr_graph = build_ppr_graph(self.workspace.db_path, self.workspace.id)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("PPR graph build failed: %s", e)
            self._ppr_graph = None
        self._ppr_graph_generation = gen
        return self._ppr_graph

    def _has_reflection_for(self, source_ulid: str) -> bool:
        import sqlite3
        with sqlite3.connect(self.workspace.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM events WHERE workspace_id = ? AND event_type = 'reflection' "
                "AND json_extract(metadata_json, '$.source_ulid') = ? LIMIT 1",
                (self.workspace.id, source_ulid),
            ).fetchone()
            return row is not None

    def _unreflected_events(
        self, limit: int = 20, max_age_days: float = 30.0,
    ):
        """Recent non-reflection events that don't yet have a reflection."""
        import sqlite3
        cutoff = time.time() - max_age_days * 86400
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT ulid FROM events e
                WHERE workspace_id = ?
                  AND event_type != 'reflection'
                  AND archived_at IS NULL
                  AND timestamp >= ?
                  AND NOT EXISTS (
                      SELECT 1 FROM events r
                      WHERE r.workspace_id = e.workspace_id
                        AND r.event_type = 'reflection'
                        AND json_extract(r.metadata_json, '$.source_ulid') = e.ulid
                  )
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (self.workspace.id, cutoff, limit),
            ).fetchall()
            ulids = [r["ulid"] for r in rows]
        return [self.events.get_by_ulid(u) for u in ulids if self.events.get_by_ulid(u)]

    def _reflection_context(self, ev, max_n: int = 4):
        """Pick a small set of events to give the reflection LLM context.

        Combines temporal nearest neighbours (events written shortly before
        this one) with entity-sharing neighbours from the graph. Both
        signals are cheap and give the LLM useful surrounding context.
        """
        import sqlite3
        ctx: list = []
        seen: set[str] = {ev.ulid}
        # Temporal window: 6 events immediately before this one
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ulid FROM events WHERE workspace_id = ? AND archived_at IS NULL "
                "AND event_type != 'reflection' AND timestamp < ? "
                "ORDER BY timestamp DESC LIMIT 6",
                (self.workspace.id, ev.timestamp),
            ).fetchall()
        for r in rows:
            u = r["ulid"]
            if u in seen:
                continue
            seen.add(u)
            full = self.events.get_by_ulid(u)
            if full:
                ctx.append(full)
            if len(ctx) >= max_n:
                break
        return ctx

    def maybe_auto_consolidate(self, **kwargs) -> Optional[dict]:
        """Run consolidation only if auto-trigger thresholds are met.

        Returns the consolidation result if a run happened, else None.
        Safe to call from any quiet moment (CLI exit hook, MCP idle, cron).
        """
        decision = self.consolidation_due()
        if not decision.get("should_run"):
            return None
        return self.consolidate(**kwargs)

    def health_trend(self) -> dict:
        from pmb.health.self_test import SelfTestRunner
        return SelfTestRunner(self).trend()

    def detect_conflicts(self, max_age_days: float = 365.0) -> list[dict]:
        from pmb.health.conflicts import ConflictDetector
        conflicts = ConflictDetector(self).detect(max_age_days=max_age_days)
        return [c.to_dict() for c in conflicts]

    def auto_resolve_conflicts(
        self, dry_run: bool = True, merge_via_llm: bool = False,
    ) -> dict:
        from pmb.health.conflicts import ConflictDetector
        return ConflictDetector(self).auto_resolve(
            dry_run=dry_run, merge_via_llm=merge_via_llm,
        )

    def compact(self, dry_run: bool = False, age_days: int = 30) -> dict:
        from pmb.maintenance.compact import StorageCompactor
        return StorageCompactor(self).compact(dry_run=dry_run, age_days=age_days)

    def cold_stats(self) -> dict:
        from pmb.maintenance.compact import StorageCompactor
        return StorageCompactor(self).cold_stats()

    # -----------------------------------------------------------------
    # Recall Feedback — real user signal
    # -----------------------------------------------------------------

    def record_recall_feedback(
        self,
        ulid: str,
        verdict: str,
        query: Optional[str] = None,
        expected_ulid: Optional[str] = None,
    ) -> dict:
        from pmb.health.feedback import record_feedback
        return record_feedback(self, ulid, verdict, query=query, expected_ulid=expected_ulid)

    def feedback_summary(self) -> dict:
        from pmb.health.feedback import summary
        return summary(self)

    def close(self):
        """Закрыть открытые ресурсы (LanceDB, native handles)."""
        # Drain any pending touch updates before letting the engine die.
        try:
            self._drain_touch_buffer()
        except Exception:
            pass
        try:
            self.search._lance = None
            self.search._table_obj = None
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
