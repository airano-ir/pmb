"""Signal capture modules - git, sessions, files, decay."""

from pmb.signals.decay import apply_decay, recompute_importance
from pmb.signals.files import FileCorrelation
from pmb.signals.git import GitSync, capture_recent_commits
from pmb.signals.session import Session, SessionTracker

__all__ = [
    "GitSync",
    "capture_recent_commits",
    "Session",
    "SessionTracker",
    "apply_decay",
    "recompute_importance",
    "FileCorrelation",
]
