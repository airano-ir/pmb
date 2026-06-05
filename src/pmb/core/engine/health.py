from __future__ import annotations

from typing import Optional


class HealthMixin:
    def apply_daily_decay(self, days_since: float = 1.0) -> dict:
        from pmb.signals.decay import apply_decay

        return apply_decay(self, days_since_last_decay=days_since)

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
