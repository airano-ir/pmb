from __future__ import annotations

from typing import Optional


class HealthMixin:
    def apply_daily_decay(self, days_since: float = 1.0) -> dict:
        from pmb.signals.decay import apply_decay

        return apply_decay(self, days_since_last_decay=days_since)

    def archive_cold(
        self,
        days: Optional[int] = None,
        max_importance: Optional[float] = None,
        dry_run: bool = True,
    ) -> dict:
        """Time-based forgetting (#6): archive ACTIVE facts/activities that are
        cold AND low-value AND old, so junk that decay only ever down-weighted
        finally leaves recall. Archive-only (reversible, tagged archived_reason
        = "decay_cold").

        A candidate must be ALL of:
          * event_type in {fact, activity}  (never lessons/goals/preferences/…)
          * importance <= max_importance     (default config decay.archive_cold_max_importance)
          * access_count == 0                (never recalled / reinforced)
          * older than `days`                (default config decay.archive_cold_days)
          * NOT pinned (importance < 0.99)
          * NOT a current keyed fact         (keyed_fact_key in metadata)

        Returns {candidates: [{ulid, age_days, importance, content}], n, dry_run}.
        """
        import json as _json
        import sqlite3 as _sql
        import time as _time

        days = int(self.config.get("decay.archive_cold_days") if days is None else days)
        max_importance = float(
            self.config.get("decay.archive_cold_max_importance")
            if max_importance is None else max_importance
        )
        cutoff_ts = _time.time() - days * 86400.0

        candidates: list[dict] = []
        try:
            with _sql.connect(str(self.workspace.db_path)) as conn:
                conn.row_factory = _sql.Row
                rows = conn.execute(
                    "SELECT ulid, content, metadata_json, importance, "
                    "access_count, timestamp FROM events "
                    "WHERE workspace_id = ? AND archived_at IS NULL "
                    "AND event_type IN ('fact', 'activity') "
                    "AND importance <= ? AND access_count = 0 AND timestamp < ? "
                    "ORDER BY timestamp ASC",
                    (self.workspace.id, max_importance, cutoff_ts),
                ).fetchall()
        except Exception:
            return {"candidates": [], "n": 0, "dry_run": dry_run, "error": "scan_failed"}

        now = _time.time()
        for r in rows:
            if float(r["importance"] or 0.0) >= 0.99:
                continue  # pinned
            try:
                meta = _json.loads(r["metadata_json"] or "{}")
            except Exception:
                meta = {}
            if isinstance(meta, dict) and meta.get("keyed_fact_key"):
                continue  # current keyed attribute — keep
            if isinstance(meta, dict) and (
                meta.get("kind") == "lesson" or meta.get("source") == "lesson"
            ):
                continue  # lessons are stored as facts — never decay-archive them
            candidates.append({
                "ulid": r["ulid"],
                "age_days": round((now - r["timestamp"]) / 86400.0, 1),
                "importance": round(float(r["importance"] or 0.0), 3),
                "content": (r["content"] or "")[:100],
            })

        if not dry_run and candidates:
            for c in candidates:
                try:
                    self.events.archive(c["ulid"])
                    with _sql.connect(str(self.workspace.db_path)) as conn:
                        row = conn.execute(
                            "SELECT metadata_json FROM events WHERE ulid = ?",
                            (c["ulid"],),
                        ).fetchone()
                        m = _json.loads(row[0] or "{}") if row else {}
                        if not isinstance(m, dict):
                            m = {}
                        m["archived_reason"] = "decay_cold"
                        conn.execute(
                            "UPDATE events SET metadata_json = ? WHERE ulid = ?",
                            (_json.dumps(m), c["ulid"]),
                        )
                except Exception:
                    continue
            self.recall_cache.bump_generation()

        return {"candidates": candidates, "n": len(candidates), "dry_run": dry_run,
                "days": days, "max_importance": max_importance}

    def file_correlations(self, file_path: str, top_k: int = 10) -> list[tuple[str, int]]:
        from pmb.signals.files import FileCorrelation

        return FileCorrelation(self).correlations(file_path, top_k)

    def file_history(self, file_path: str) -> list[dict]:
        from pmb.signals.files import FileCorrelation

        return FileCorrelation(self).file_history(file_path)

    def run_self_test(
        self, n_samples: int = 20, min_age_days: float = 1.0, apply_adaptive: bool = True
    ) -> dict:
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

    def health_trend(self) -> dict:
        from pmb.health.self_test import SelfTestRunner

        return SelfTestRunner(self).trend()

    def detect_conflicts(self, max_age_days: float = 365.0) -> list[dict]:
        from pmb.health.conflicts import ConflictDetector

        conflicts = ConflictDetector(self).detect(max_age_days=max_age_days)
        return [c.to_dict() for c in conflicts]

    def auto_resolve_conflicts(
        self,
        dry_run: bool = True,
        merge_via_llm: bool = False,
    ) -> dict:
        from pmb.health.conflicts import ConflictDetector

        return ConflictDetector(self).auto_resolve(
            dry_run=dry_run,
            merge_via_llm=merge_via_llm,
        )

    def compact(self, dry_run: bool = False, age_days: int = 30) -> dict:
        from pmb.maintenance.compact import StorageCompactor

        return StorageCompactor(self).compact(dry_run=dry_run, age_days=age_days)

    def cold_stats(self) -> dict:
        from pmb.maintenance.compact import StorageCompactor

        return StorageCompactor(self).cold_stats()

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
