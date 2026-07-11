"""
Auto-distill lessons from a session (zero-command memory growth).

komi-learn's edge was automatic lesson extraction from session transcripts.
This closes that gap, PMB-native: take a session's recorded events (what the
agent did, decided, fixed, was corrected on), ask an LLM to distill the
DURABLE lessons + failures, and record them as kind=lesson / kind=failure
events. Wire it to session-end and it's zero-command.

Design / risk notes:
  - Off the recall hot path → cannot affect LoCoMo or recall latency.
  - Reuses the existing consolidation LLM backends (Claude CLI / Anthropic /
    OpenAI / Ollama) via `resolve_llm_client`. No new dependency.
  - Dedups against existing lessons so re-running a session doesn't pile up
    duplicates.
  - Testable without an LLM: inject a fake client with `.complete(prompt)`.
  - Graceful: no backend → {"skipped": "no_llm"}, never crashes a session end.
"""

from __future__ import annotations

import json
import re

_PROMPT = """\
You are extracting DURABLE LESSONS from a coding/work session, to help an AI
agent work better next time. From the session events below, output ONLY a JSON
array. Each item:

  {{"type": "lesson"|"failure", "content": "<one concise, reusable rule>"}}

Rules:
- "lesson" = a reusable convention/technique to apply going forward
  (e.g. "this repo uses pnpm, never npm").
- "failure" = something that was tried and did NOT work, to avoid repeating
  (e.g. "bumping numpy to 2.x broke lancedb; stay on 1.x").
- Only durable, transferable rules. SKIP one-off facts, task status, and
  anything tied to this single run.
- Max {max_lessons} items. If nothing durable, return [].
- Each content < 200 chars, imperative, self-contained.

SESSION EVENTS:
{events}

JSON array:"""


def _parse_items(text: str, max_lessons: int) -> list[dict]:
    """Pull the JSON array out of an LLM response, defensively."""
    if not text:
        return []
    # find the first [...] block
    m = re.search(r"\[.*\]", text, re.DOTALL)
    raw = m.group(0) if m else text
    try:
        data = json.loads(raw)
    except Exception:
        return []
    out: list[dict] = []
    for it in data if isinstance(data, list) else []:
        if not isinstance(it, dict):
            continue
        t = (it.get("type") or "lesson").lower().strip()
        if t not in ("lesson", "failure"):
            t = "lesson"
        content = (it.get("content") or "").strip()
        if 8 <= len(content) <= 400:
            out.append({"type": t, "content": content[:400]})
        if len(out) >= max_lessons:
            break
    return out


def _session_events_text(engine, session_id: str | None, limit: int) -> tuple[str, int]:
    """Collect a session's events (or recent events) as a transcript-like text."""
    events = engine.events.list_active(engine.workspace.id, limit=max(limit, 200))
    if session_id:
        events = [e for e in events if e.source_session_id == session_id]
    # newest first → chronological for the prompt
    events = list(reversed(events[:limit]))
    lines = []
    for e in events:
        kind = (e.metadata or {}).get("kind") or e.event_type
        lines.append(f"- [{kind}] {e.content[:300]}")
    return "\n".join(lines), len(events)


def distill_lessons(
    engine,
    *,
    session_id: str | None = None,
    backend: str = "auto",
    model: str | None = None,
    max_lessons: int = 8,
    limit: int = 60,
    dry_run: bool = False,
    llm=None,
) -> dict:
    """Distill durable lessons/failures from a session's events via an LLM.

    Returns {"skipped": "..."} | {"candidates": [...], "n_recorded": N}.
    Pass `llm` (anything with `.complete(prompt)`) to test without a backend.
    """
    events_text, n_events = _session_events_text(engine, session_id, limit)
    if n_events == 0:
        return {"skipped": "no_events", "n_recorded": 0}

    if llm is None:
        try:
            from pmb.health.consolidate import resolve_llm_client
            llm = resolve_llm_client(backend=backend, model=model)
        except Exception as e:  # noqa: BLE001
            return {"skipped": "no_llm", "detail": str(e), "n_recorded": 0}

    prompt = _PROMPT.format(events=events_text, max_lessons=max_lessons)
    try:
        text = llm.complete(prompt, max_tokens=600)
    except Exception as e:  # noqa: BLE001
        return {"skipped": "llm_error", "detail": str(e), "n_recorded": 0}

    candidates = _parse_items(text, max_lessons)
    if not candidates:
        return {"candidates": [], "n_recorded": 0}

    # dedup against existing lessons/failures (exact, case-insensitive)
    existing = {
        (e.content or "").strip().lower()
        for e in engine.events.list_active(engine.workspace.id, limit=2000)
        if (e.metadata or {}).get("kind") in ("lesson", "failure")
    }
    fresh = [c for c in candidates if c["content"].strip().lower() not in existing]

    if dry_run or not fresh:
        return {"candidates": candidates, "fresh": fresh, "n_recorded": 0,
                "dry_run": dry_run}

    items = [{"type": c["type"], "content": c["content"],
              "metadata": {"source": "lesson", "kind": c["type"],
                           "distilled": True}}
             for c in fresh]
    out = engine.record_batch(items)
    return {"candidates": candidates, "fresh": fresh,
            "n_recorded": out.get("n_ok", 0)}
