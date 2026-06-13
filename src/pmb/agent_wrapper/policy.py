"""
Compression policies.

The interface is intentionally tiny: given the current messages list, return
a new (shorter) messages list. The policy is the entire research question -
the value of this wrapper depends on whether selective compression actually
helps users vs. Claude Code's default auto-compact.

## What's here

- `CompressionPolicy` protocol.
- `DropOldestNarrative` - a placeholder that just drops the oldest non-pinned
  messages until the budget is met. This is NOT smart compression, it's a
  baseline to measure against.

## What's still to build (the real work)

A `SelectivePolicy` that:

1. Identifies *decisions* and *facts* in the message stream (heuristics:
   "we decided to ...", "the answer is ...", explicit `[FACT]` tags, tool
   results with low-cardinality outputs).
2. Keeps those verbatim.
3. Summarizes the *narrative* - back-and-forth exploration messages - into
   a single short paragraph, anchored to the original message IDs so the
   detail can be re-fetched from PMB if needed.
4. Drops messages whose information is already captured in PMB memory
   (because re-fetchable on demand).

Sketch of how that policy would look:

    class SelectivePolicy:
        def __init__(self, llm, pmb_engine, budget):
            ...

        def compact(self, messages):
            decisions, narrative, other = self._classify(messages)
            summary = self._llm_summarize(narrative)  # one paragraph
            # decisions kept verbatim
            return [*pinned, summary_message, *decisions, *recent_n_turns]

Doing this well is the open research problem.
"""

from __future__ import annotations

import re
from typing import Protocol


class CompressionPolicy(Protocol):
    def compact(self, messages: list[dict]) -> list[dict]:
        """Return a shorter messages list."""
        ...


# ----------------------------------------------------------------------
# Rule-based message classifier (no LLM)
# ----------------------------------------------------------------------


# Patterns we use to detect decisions/facts at compaction time. These are
# intentionally narrow - false positives are worse than misses since
# misclassified-as-narrative still gets summarized.
_DECISION_PHRASES = [
    # Bare "we" / contractions: we, we'll, we've, we're
    re.compile(
        r"\bwe(?:'(?:ll|ve|re)|\s+will)?\s+"
        r"(?:decided|chose|will go with|go with|are switching to|"
        r"switched to|picked|settled on)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:let'?s|let us) (?:go|switch|use) (?:with )?\b", re.IGNORECASE),
    re.compile(r"\b(?:final|finalised|finalized) (?:decision|choice|answer)\b", re.IGNORECASE),
    re.compile(r"\b(?:we'?ll|we will|we are)\s+going\s+with\b", re.IGNORECASE),
    re.compile(r"\bdecision:\s*", re.IGNORECASE),
]

_FACT_PHRASES = [
    re.compile(r"\b(?:the )?(?:answer|fix|root cause|reason) (?:is|was)\b", re.IGNORECASE),
    re.compile(r"\bthis project uses\b", re.IGNORECASE),
    re.compile(r"\b(?:remember|note) (?:that|this)\b", re.IGNORECASE),
    re.compile(r"\[FACT\]", re.IGNORECASE),
    re.compile(r"\[DECISION\]", re.IGNORECASE),
]


def classify_message(msg: dict) -> str:
    """
    Bucket a message into one of:
      'system' | 'tool_result' | 'decision' | 'fact' | 'error' | 'narrative'

    Pure heuristics. Misclassification falls into 'narrative' which means
    the message becomes eligible for summarization rather than verbatim
    preservation.
    """
    role = msg.get("role", "")
    content = msg.get("content", "")

    if role == "system":
        return "system"
    if role == "tool":
        return "tool_result"

    # Anthropic-style content blocks: presence of a tool_use/tool_result block
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                t = block.get("type", "")
                if t in ("tool_use", "tool_result"):
                    return "tool_result"
        text = " ".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    else:
        text = str(content)

    # Error markers
    if re.search(r"\b(?:Traceback|error:|exception:)\b", text, re.IGNORECASE):
        return "error"
    for pat in _DECISION_PHRASES:
        if pat.search(text):
            return "decision"
    for pat in _FACT_PHRASES:
        if pat.search(text):
            return "fact"
    return "narrative"


def _msg_text(msg: dict) -> str:
    c = msg.get("content", "")
    if isinstance(c, list):
        return " ".join(b.get("text", "") for b in c if isinstance(b, dict))
    return str(c)


# ----------------------------------------------------------------------
# Selective compression policy
# ----------------------------------------------------------------------


class SelectivePolicy:
    """
    Real selective compression - mechanic #1 from the pitch.

    Rule of thumb when budget pressure rises:

      - **Always keep verbatim:** system messages, the last N turns,
        any message classified as `decision` / `fact` / `tool_result` /
        `error`, and anything explicitly pinned.
      - **Summarize:** runs of `narrative` messages between pinned ones.
        Summary is one paragraph with anchors `[ulid:...]` to the original
        PMB events, so the full detail is re-fetchable.

    Backoff: if even after summarizing all narrative we're still over
    budget, drop oldest summarized narrative until we fit.

    The LLM call is only used to produce summaries - classification is
    purely rule-based to keep compaction cheap and deterministic.
    """

    def __init__(
        self,
        budget,
        engine=None,
        llm=None,
        keep_last_n: int = 6,
        max_summary_chars: int = 600,
    ):
        self.budget = budget
        self.engine = engine
        self.llm = llm
        self.keep_last_n = keep_last_n
        self.max_summary_chars = max_summary_chars

    # ------------ classification helpers --------

    def _verbatim(self, msg: dict, idx: int, n_total: int) -> bool:
        # Always keep last N
        if idx >= n_total - self.keep_last_n:
            return True
        kind = classify_message(msg)
        if kind in ("system", "tool_result", "decision", "fact", "error"):
            return True
        # Explicit pin marker
        c = msg.get("content")
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("metadata", {}).get("pin"):
                    return True
        return False

    # ------------ summary production --------

    def _summarize_narrative_run(self, run: list[dict]) -> dict:
        """Produce a single 'system' message summarizing a run of narrative.

        Uses self.llm if available (any object with .consolidate(list[str]) →
        dict, same protocol as `LLMClient` used for sleep-engine consolidation).
        Falls back to a deterministic head+tail extract if no LLM is wired -
        better than losing the info entirely.
        """
        snippets = [_msg_text(m) for m in run]
        joined = "\n\n---\n\n".join(snippets)

        if self.llm is not None:
            try:
                out = self.llm.consolidate(snippets)
                if out.get("summary"):
                    body = out["summary"].strip()
                    return self._summary_message(body[: self.max_summary_chars])
            except Exception:
                # Fall through to heuristic
                pass

        # Heuristic fallback: first sentence + last sentence
        body = joined.replace("\n", " ").strip()
        if len(body) > self.max_summary_chars:
            head = body[: self.max_summary_chars // 2]
            tail = body[-(self.max_summary_chars // 2):]
            body = f"{head} … {tail}"
        return self._summary_message(body)

    def _summary_message(self, body: str) -> dict:
        return {
            "role": "user",
            "content": f"[COMPRESSED NARRATIVE] {body}",
        }

    # ------------ main entry --------

    def compact(self, messages: list[dict]) -> list[dict]:
        if not self.budget.should_compact(messages):
            return messages
        if len(messages) <= self.keep_last_n:
            return messages

        n = len(messages)
        kept: list[dict] = []
        narrative_run: list[dict] = []

        def flush_run():
            if narrative_run:
                kept.append(self._summarize_narrative_run(narrative_run))
                narrative_run.clear()

        for idx, msg in enumerate(messages):
            if self._verbatim(msg, idx, n):
                flush_run()
                kept.append(msg)
            else:
                narrative_run.append(msg)
        flush_run()

        # Backoff: if STILL over budget, drop oldest summaries (but keep
        # decisions/facts/tool_results/recent_N intact).
        i = 0
        while self.budget.should_compact(kept) and i < len(kept):
            if kept[i].get("content", "").startswith("[COMPRESSED NARRATIVE]") if isinstance(kept[i].get("content"), str) else False:
                kept.pop(i)
                continue
            i += 1
        return kept


class DropOldestNarrative:
    """
    Baseline policy: drop oldest non-pinned messages until under budget.

    "Pinned" here means messages tagged with `metadata={"pin": True}` in their
    content block, or system messages. This is *not* what the pitch promised -
    real selective compression is `SelectivePolicy` which is not yet built.
    Use this as a no-LLM control for A/B comparison.
    """

    def __init__(self, budget, keep_last_n: int = 6):
        self.budget = budget
        self.keep_last_n = keep_last_n

    @staticmethod
    def _is_pinned(msg: dict) -> bool:
        c = msg.get("content")
        if isinstance(c, list):
            for block in c:
                if isinstance(block, dict) and block.get("metadata", {}).get("pin"):
                    return True
        return False

    def compact(self, messages: list[dict]) -> list[dict]:
        if not self.budget.should_compact(messages):
            return messages
        if len(messages) <= self.keep_last_n:
            return messages

        head = messages[: -self.keep_last_n]
        tail = messages[-self.keep_last_n :]
        # Keep pinned items from head; drop the rest oldest-first
        pinned = [m for m in head if self._is_pinned(m)]
        unpinned = [m for m in head if not self._is_pinned(m)]

        while unpinned and self.budget.should_compact(pinned + unpinned + tail):
            unpinned.pop(0)

        return pinned + unpinned + tail
