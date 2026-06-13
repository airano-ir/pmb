from __future__ import annotations


class DedupMixin:
    def _dedup_pre_write(
        self,
        content: str,
        event_type: str,
    ):
        """Run L1 (exact) then L2 (semantic) dedup checks.

        Returns (hit, borderline):
          hit:        DedupHit if a definite duplicate exists - caller skips write
          borderline: (candidate_ulid, similarity) if a borderline neighbor
                      was found; caller WRITES then enqueues this pair for
                      async LLM verify (L2.5)
        Both can be None (no signal, proceed normally).

        Improvement Y: when called INSIDE record_batch, skip L2 semantic check
        entirely. The items in one batch came from the same agent thinking
        pass - they're highly unlikely to dup against existing storage AND
        running L2 doubles the embedding cost (once for dedup search, once
        for the actual write). L1 (exact-text) still runs for safety, and
        the periodic `pmb dedupe` sweep catches anything L1 missed.
        """
        if not self.config.get("dedup.enable"):
            return None, None
        try:
            from pmb.reasoning.dedup import (
                find_exact_duplicate,
                find_semantic_duplicate,
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
        # agent turn). Skip - periodic `pmb dedupe` handles paraphrases.
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

    def _exact_dup_within_window(
        self,
        content: str,
        event_type: str,
        window_h: float,
    ):
        """L1-only exact-duplicate probe bounded by a short window (HOURS).

        Unlike `_dedup_pre_write` this never embeds - one indexed SQL scan over
        same-type ACTIVE events newer than `window_h`. Returns the canonical
        event's `DedupHit` or None. Used by `record_activity`, whose items are
        re-logged seconds apart and would otherwise pile up (0.2 / former E6).
        """
        if window_h <= 0 or not content:
            return None
        try:
            from pmb.reasoning.dedup import find_exact_duplicate
        except Exception:
            return None
        try:
            return find_exact_duplicate(
                db_path=self.workspace.db_path,
                workspace_id=self.workspace.id,
                event_type=event_type,
                content=content,
                lookback_days=window_h / 24.0,
            )
        except Exception:
            return None

    def _dedup_nearest_candidates(
        self,
        content: str,
        top_k: int = 20,
    ) -> list:
        """Query LanceDB for nearest events to `content` via COSINE metric.

        Returns [(ulid, cosine_similarity), ...] sorted descending.

        Used only by dedup - recall has its own (richer) pipeline.
        """
        import numpy as np

        q = np.asarray(self.search.embed(content), dtype=np.float32)
        try:
            # LanceDB cosine returns distance ∈ [0, 2] where 0 = identical.
            # Similarity = 1 - distance. We request a few extras and clip.
            results = self.search._table.search(q.tolist()).metric("cosine").limit(top_k).to_list()
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
        import sqlite3
        import time as _t

        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.execute(
                "UPDATE events SET access_count = access_count + 1, "
                "last_accessed = ? WHERE ulid = ?",
                (_t.time(), ulid),
            )

    def dedupe_sweep(
        self,
        threshold: float | None = None,
        event_types: list[str] | None = None,
    ) -> dict:
        """One-shot dedup pass over ALL active events in this workspace.

        Clusters by cosine ≥ threshold within each event_type; archives
        losers with metadata.merged_into → winner. Reversible via
        `dedupe_undo()`.

        Use after upgrading dedup logic or just to clean up an aged workspace.
        """
        from pmb.reasoning.dedup import sweep_workspace

        thr = float(threshold if threshold is not None else self.config.get("dedup.cosine_high"))

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
        import sqlite3

        import numpy as np

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
            out.append(
                (
                    r["ulid"],
                    r["event_type"],
                    float(r["importance"] or 0.0),
                    int(r["access_count"] or 0),
                    float(r["timestamp"] or 0.0),
                    v,
                )
            )
        return out

    def dedupe_undo(self) -> int:
        """Restore events archived by dedup (metadata.merged_into is set).
        Returns count of restored events.
        """
        from pmb.reasoning.dedup import undo_merges

        return undo_merges(self.workspace.db_path, self.workspace.id)

    def dedupe_run_pending(
        self,
        backend: str = "auto",
        limit: int = 50,
    ) -> dict:
        """L2.5: drain dedup_pending queue, ask LLM whether each pair is
        the same fact. Yes → archive newer; no → mark resolved.
        """
        from pmb.reasoning.dedup import run_pending

        return run_pending(
            db_path=self.workspace.db_path,
            workspace_id=self.workspace.id,
            backend=backend,
            limit=limit,
        )

    def dedupe_list_pending(self, limit: int = 100) -> list[dict]:
        """List borderline pairs awaiting LLM verdict (or user review)."""
        from pmb.reasoning.dedup import list_pending

        return list_pending(
            self.workspace.db_path,
            self.workspace.id,
            limit=limit,
        )
