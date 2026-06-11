"""
Reflective Memory — the core of PMB v2.

What it does:
  For each new event (or a batch in sleep), an LLM asks itself:
    1. Why does this matter?
    2. What does it imply or set up?
    3. What questions might this event later help answer?
    4. Which recent events might it connect to?

The LLM's answers are saved as NEW events with event_type='reflection',
metadata.source_ulid pointing back to the original. Because they live in
the regular events table, the existing hybrid+graph+rerank recall pipeline
finds them automatically — no new index, no read-time LLM.

Why this beats classic RAG memory:
  Multi-hop questions like "what did Alice do after meeting Bob?" fail in
  classic RAG because the answer event mentions only Alice — not Bob, not
  'after'. Pure similarity search can't bridge.
  With reflection, the meeting event spawns a reflection containing
  'might-answer: what did Alice do after seeing Bob?'. That reflection
  ranks high on the query → query is routed to the meeting event AND to
  whatever it references in implications.

Where it lives in the lifecycle:
  - WRITE-TIME: event is stored normally, queued as 'unreflected'.
  - IDLE/SLEEP: a worker pulls batches of unreflected events, calls LLM,
    persists reflections + marks events reflected. Never blocks the user.
  - READ-TIME: standard recall. Reflections are just events to BM25/vec.

Cost model:
  ~1 LLM call per source event, batched. With Claude CLI this is ~5s per
  call; with Ollama on a decent machine ~1-2s. We do this in idle, so the
  user never waits.

Failure mode:
  If LLM is unavailable, reflections aren't generated. PMB still works —
  it just doesn't get the multi-hop boost. Graceful degradation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmb.core.events import Event


log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------

REFLECTION_PROMPT = """\
You are a memory system reflecting on a single event the user logged.

Your job: extract DEEP semantic context that makes this event findable for
future questions, even questions that don't share the event's exact words.

Output VALID JSON with these keys:
  "significance":   one sentence — why does this matter? what makes it notable?
  "implications":   array of 1-3 short strings — what this sets up, enables, or rules out
  "might_answer":   array of 2-4 short hypothetical user questions this event helps answer.
                    These should phrase the same fact in different ways and capture multi-hop angles.
  "linked_themes":  array of 1-3 short topic tags ("relationship_with_bob", "postgres_migration", etc.)
  "people":         array of person names mentioned (lowercase, distinct)
  "time_anchor":    short string — when did this happen? ("dec 2025", "after the trip", etc.) or "" if unclear

Rules:
- Output ONLY the JSON, no preamble, no markdown fences.
- Keep each string under 120 chars.
- Be specific. Don't paraphrase the event; extract what's IMPLIED.
- 'might_answer' is the most important field. Imagine someone reading
  this event 6 months later — what questions would point them here?

Event content:
\"\"\"
{content}
\"\"\"

Event metadata (for context, don't repeat verbatim):
{metadata_summary}

Optional context from recent related events:
{context_window}

JSON:"""


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

@dataclass
class Reflection:
    """Parsed LLM output for one event."""
    source_ulid: str
    significance: str
    implications: list[str]
    might_answer: list[str]
    linked_themes: list[str]
    people: list[str]
    time_anchor: str
    raw_json: str  # for debug / reprocessing

    def to_event_content(self) -> str:
        """Format as a searchable text blob for storage as an event."""
        parts = [f"Reflection: {self.significance}"]
        if self.might_answer:
            parts.append("Might answer: " + " | ".join(self.might_answer))
        if self.implications:
            parts.append("Implications: " + " | ".join(self.implications))
        if self.linked_themes:
            parts.append("Themes: " + ", ".join(self.linked_themes))
        if self.people:
            parts.append("People: " + ", ".join(self.people))
        if self.time_anchor:
            parts.append(f"Time: {self.time_anchor}")
        return "\n".join(parts)

    def to_metadata(self) -> dict:
        return {
            "source_ulid": self.source_ulid,
            "reflection_kind": "primary",
            "themes": self.linked_themes,
            "people": self.people,
            "time_anchor": self.time_anchor,
        }


# ----------------------------------------------------------------------
# LLM interface
# ----------------------------------------------------------------------

class Reflector:
    """LLM-driven reflection over single events.

    Usage:
        r = Reflector(llm_client)
        reflection = r.reflect(event, context_events=[...])
    """

    def __init__(self, llm_client, max_context_events: int = 4):
        """
        llm_client: any object with a generic 'complete(prompt: str) -> str'
        method, OR one of our consolidate clients (we'll use .consolidate
        with a single-event wrapper if needed). Tries common shapes.
        """
        self.llm = llm_client
        self.max_context_events = max_context_events

    def reflect(
        self, event: Event, context_events: list[Event] | None = None
    ) -> Reflection | None:
        if not event or not event.content or not event.content.strip():
            return None
        prompt = self._build_prompt(event, context_events or [])

        try:
            text = self._call_llm(prompt)
        except Exception as e:
            log.warning("LLM call failed in Reflector: %s", e)
            return None
        if not text:
            return None

        try:
            data = self._parse_json(text)
        except Exception as e:
            log.warning("LLM returned non-JSON in Reflector: %s | output: %s", e, text[:200])
            return None

        return Reflection(
            source_ulid=event.ulid,
            significance=str(data.get("significance", "")).strip()[:500],
            implications=[s.strip()[:200] for s in (data.get("implications") or []) if isinstance(s, str)],
            might_answer=[s.strip()[:200] for s in (data.get("might_answer") or []) if isinstance(s, str)],
            linked_themes=[s.strip().lower()[:80] for s in (data.get("linked_themes") or []) if isinstance(s, str)],
            people=[s.strip().lower()[:50] for s in (data.get("people") or []) if isinstance(s, str)],
            time_anchor=str(data.get("time_anchor", "")).strip()[:80],
            raw_json=text[:4000],
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_prompt(self, event: Event, context: list[Event]) -> str:
        meta = event.metadata or {}
        meta_summary = json.dumps(
            {k: v for k, v in meta.items() if k in (
                "speaker", "session", "dia_id", "files_changed",
            )},
            ensure_ascii=False,
        )[:300] if meta else "{}"
        if context:
            ctx_lines = [
                f"  [{i+1}] {(c.content or '')[:200]}" for i, c in enumerate(context[: self.max_context_events])
            ]
            ctx_str = "\n".join(ctx_lines)
        else:
            ctx_str = "  (no related context yet)"

        return REFLECTION_PROMPT.format(
            content=(event.content or "")[:2000],
            metadata_summary=meta_summary,
            context_window=ctx_str,
        )

    def _call_llm(self, prompt: str) -> str:
        """Adapt to whatever shape our LLM client exposes."""
        # Tries .complete(prompt), then .consolidate({texts:[...]}), then fallback.
        if hasattr(self.llm, "complete"):
            return self.llm.complete(prompt) or ""
        if hasattr(self.llm, "consolidate"):
            # Existing consolidate client returns dict — repurpose by stuffing
            # our prompt as the only text.
            out = self.llm.consolidate([prompt])
            if isinstance(out, dict):
                # Prefer 'summary' as the raw LLM output if present.
                return out.get("summary") or json.dumps(out)
            return str(out)
        if callable(self.llm):
            return self.llm(prompt) or ""
        raise RuntimeError(
            "Reflector: LLM client has neither .complete(), .consolidate(), nor __call__"
        )

    def _parse_json(self, text: str) -> dict:
        text = text.strip()
        # Strip common markdown fences.
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rsplit("```", 1)[0].strip()
        # Find first { ... } block
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("no JSON object found")
        return json.loads(text[start : end + 1])
