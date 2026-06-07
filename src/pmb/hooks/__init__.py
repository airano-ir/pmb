"""Session-start / per-turn hook logic.

The CLI commands (`pmb hooks install`, `pmb prepare-context`) live in
`pmb.cli.hooks` / `pmb.cli.main`. This package holds the pure-logic
modules they call into:

- `auto_recall`: classify a user message into an intent, dispatch the
  matching PMB queries, return a structured context bundle.
"""

from pmb.hooks.auto_recall import (
    Intent,
    AutoContextResult,
    detect_intents,
    run_auto_context,
    format_context,
    is_trivial,
)
from pmb.hooks.session_restore import build_session_restore
from pmb.hooks.followcheck import (
    run_followcheck,
    FollowCheckResult,
    FollowVerdict,
)

__all__ = [
    "Intent",
    "AutoContextResult",
    "detect_intents",
    "run_auto_context",
    "format_context",
    "is_trivial",
    "build_session_restore",
    "run_followcheck",
    "FollowCheckResult",
    "FollowVerdict",
]
