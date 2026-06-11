"""Health checks: self-test, conflict detection, accuracy tracking."""

from pmb.health.adaptive import adaptive_history, apply_adaptive_boost
from pmb.health.conflicts import ConflictDetector, FactConflict
from pmb.health.self_test import SelfTestResult, SelfTestRunner

__all__ = [
    "SelfTestRunner",
    "SelfTestResult",
    "ConflictDetector",
    "FactConflict",
    "apply_adaptive_boost",
    "adaptive_history",
]
