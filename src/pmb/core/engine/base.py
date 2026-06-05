from __future__ import annotations

from pathlib import Path
from typing import Optional

from pmb.config import Config
from pmb.core.events import (
    EventStore,
)
from pmb.core.recall_cache import RecallCache
from pmb.core.search import HybridSearch
from pmb.core.workspace import Workspace, detect_workspace
from pmb.graph.entities import EntityExtractor
from pmb.graph.store import GraphStore
from pmb.signals.session import SessionTracker
from pmb.reasoning.pamvr import (
    VOCAB_BRIDGES as _PAMVR_DEFAULT_BRIDGES,
)
from pmb.reasoning.vocab_miner import (
    mine_workspace as _mine_workspace_bridges,
    merge_bridges as _merge_vocab_bridges,
)


from pmb.core.engine.write import WriteMixin
from pmb.core.engine.goals import GoalsMixin
from pmb.core.engine.dedup import DedupMixin
from pmb.core.engine.embed import EmbedMixin
from pmb.core.engine.graph import GraphMixin
from pmb.core.engine.recall import RecallMixin
from pmb.core.engine.reasoning import ReasoningMixin
from pmb.core.engine.health import HealthMixin


class Engine(
    WriteMixin,
    GoalsMixin,
    DedupMixin,
    EmbedMixin,
    GraphMixin,
    RecallMixin,
    ReasoningMixin,
    HealthMixin,
):
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
        _rerank_needed = self.config.get("recall.rerank") or self.config.get(
            "recall.rerank_when_close"
        )
        self.search = HybridSearch(
            vector_path=self.workspace.vector_path,
            model_name=emb_model,
            embedding_backend=emb_backend,
            embedding_base_url=emb_base_url,
            bm25_weight=self.config.get("recall.bm25_weight"),
            rerank_model_name=(self.config.get("recall.rerank_model") if _rerank_needed else None),
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
        self._touch_buffer: dict[str, float] = {}  # ulid -> last_accessed
        self._touch_imp_buffer: dict[str, float] = {}  # ulid -> latest importance
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
        self._auto_bridges_enabled = bool(self.config.get("recall.auto_vocab_bridges"))
        # Improvement WW: write-time atomic fact extraction.
        self._atomic_extract_enabled = bool(self.config.get("write.atomic_fact_extract"))
        self._vocab_bridges: dict[str, list[str]] = dict(_PAMVR_DEFAULT_BRIDGES)
        self._vocab_bridges_cache_path = self.workspace.storage_dir / "vocab_bridges.json"
        self._vocab_bridges_last_event_count = -1
        if self._auto_bridges_enabled:
            try:
                self._refresh_vocab_bridges()
            except Exception:
                # Auto-mining is a best-effort enhancement, never crash init.
                pass

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
            max_bridges_per_key=int(self.config.get("recall.auto_vocab_max_per_key") or 8),
            refresh_threshold=int(self.config.get("recall.auto_vocab_refresh_after") or 50),
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
            threshold = int(self.config.get("recall.auto_vocab_refresh_after") or 50)
            if n - self._vocab_bridges_last_event_count >= threshold:
                self._refresh_vocab_bridges()
        except Exception:
            pass

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

    def sync_git(self, since_timestamp: Optional[float] = None) -> dict:
        """Захватить git commits в memory. Импортируется лениво."""
        from pmb.signals.git import GitSync

        return GitSync(self).sync(since_timestamp=since_timestamp)

    def session_start(self, name: Optional[str] = None) -> dict:
        return self.session_tracker.start(name).to_dict()

    def session_end(self) -> Optional[dict]:
        sess = self.session_tracker.end()
        out = sess.to_dict() if sess else None
        # Zero-command auto-distill: if enabled, distill durable lessons from
        # the session that just ended. Off the recall path; never crashes end.
        if out and self.config.get("lessons.auto_distill_on_session_end"):
            try:
                d = self.distill_lessons(session_id=out.get("id"))
                out["distilled_lessons"] = d.get("n_recorded", 0)
            except Exception:
                pass
        return out

    def session_current(self) -> Optional[dict]:
        sess = self.session_tracker.current(auto_create=False)
        return sess.to_dict() if sess else None

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
