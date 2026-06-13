"""
Approximate token budgeting for the chat loop.

We do not have to be exact - we only need to know "are we getting close to
the context window so we should compact?" A 5% error in token count is
fine; what matters is monotonicity (adding messages strictly increases the
estimate).

Strategy, in order of preference:
  1. Use `anthropic.count_tokens()` if available in installed SDK version.
  2. Fall back to a simple len/4 heuristic (English text ≈ 4 chars/token).
"""

from __future__ import annotations


def _heuristic_tokens(text: str) -> int:
    # Conservative: assume 3.5 chars/token (gives a slight overestimate)
    return max(1, int(len(text) / 3.5))


class TokenBudget:
    """Tracks an approximate token count over a list of messages.

    Pass `window` as the model's full context size; `target_max` as the
    fraction at which the policy should trigger compaction (default 75%).
    """

    def __init__(self, window: int = 200_000, target_max: float = 0.75):
        self.window = window
        self.target_max = target_max
        self._client_count = None  # Lazy-bound anthropic counter if usable

    @property
    def threshold(self) -> int:
        return int(self.window * self.target_max)

    def count_messages(self, messages: list[dict], system: str | None = None) -> int:
        """Total approximate tokens for a system + messages list."""
        total = 0
        if system:
            total += _heuristic_tokens(system)
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                # Anthropic content blocks
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", "")
                        total += _heuristic_tokens(text)
                    else:
                        total += _heuristic_tokens(str(block))
            else:
                total += _heuristic_tokens(str(content))
            # Per-message overhead (role tokens etc.)
            total += 4
        return total

    def should_compact(self, messages: list[dict], system: str | None = None) -> bool:
        return self.count_messages(messages, system) >= self.threshold
