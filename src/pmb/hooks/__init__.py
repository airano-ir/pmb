"""Session-start / per-turn hook logic.

The CLI commands (`pmb hooks install`, `pmb prepare-context`) live in
`pmb.cli.hooks` / `pmb.cli.main`. This package holds the pure-logic
modules they call into:

- `auto_recall`: classify a user message into an intent, dispatch the
  matching PMB queries, return a structured context bundle.
"""

from pmb.hooks.auto_recall import (
    AutoContextResult,
    Intent,
    compute_prepare_context_text,
    detect_intents,
    format_context,
    is_trivial,
    run_auto_context,
)
from pmb.hooks.autowrite import (
    AutoWriteResult,
    autowrite_gate,
    run_autowrite,
    synthesize_llm,
    synthesize_template,
)
from pmb.hooks.followcheck import (
    FollowCheckResult,
    FollowVerdict,
    run_followcheck,
)
from pmb.hooks.session_restore import build_session_restore

__all__ = [
    "Intent",
    "AutoContextResult",
    "detect_intents",
    "run_auto_context",
    "format_context",
    "compute_prepare_context_text",
    "is_trivial",
    "build_session_restore",
    "run_followcheck",
    "FollowCheckResult",
    "FollowVerdict",
    "run_autowrite",
    "autowrite_gate",
    "synthesize_template",
    "synthesize_llm",
    "AutoWriteResult",
]
