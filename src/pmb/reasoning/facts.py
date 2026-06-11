"""
Atomic Fact Extraction (Improvement D).

What this is:
  In contrast to Reflections (META-facts: "why does this matter?"), Atomic
  Facts are concrete subject-verb-object statements extracted from event
  text. mem0 builds its entire memory on this primitive. We add it as a
  layer on top of our raw-text + reflection store.

Why it matters:
  When a Reader LLM is asked "What did Caroline research?" and we give it
  raw dialogue ("Caroline: ... by the way I've been thinking about adoption,
  contacted my mentor for advice on agencies and lawyers..."), the Reader
  has to do work to extract the fact "Caroline researched adoption agencies".

  If instead we PRE-EXTRACT atomic facts at sleep time, the Reader sees:
    - Caroline researched adoption agencies
    - Caroline is transgender
    - Caroline plans to pursue counseling
  ... and the answer is trivial.

  This is the structural difference between PMB v1 (raw text retrieval) and
  mem0 (fact-tuple retrieval). We get both — facts as a layer, raw text
  preserved.

Storage:
  Each extracted fact = new event of event_type='fact_atom' with
  metadata.source_ulid = original event's ulid. Searchable via existing
  hybrid pipeline. The same `collapse_reflections` mechanism (renamed
  conceptually to "collapse derived events") sends scoring credit back
  to source if needed.

Cost:
  1 LLM call per source event (typically yields 5-30 facts). For LoCoMo
  conv-26 with 19 session-events × ~10s = ~3 min total via Claude CLI.
  This is in sleep / idle. Read-time cost: zero.

Difference from Reflection:
  Reflection prompt asks "what does this MEAN" (interpretation).
  Fact prompt asks "what ATOMIC TRUE STATEMENTS does this entail"
  (literal extraction). Both feed the same retrieval index but help
  different question types — facts help direct lookups, reflections
  help inferential / themed questions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmb.core.events import Event


log = logging.getLogger(__name__)


FACT_EXTRACTION_PROMPT = """\
Extract ATOMIC, STAND-ALONE facts from the dialogue/event below. Each fact
must be self-contained — readable without seeing the surrounding text.

Rules:
- One subject-verb-object statement per fact.
- Resolve pronouns (he/she/they → actual name).
- Resolve relative time ("yesterday" → absolute date if [Session date: ...]
  is given in the chunk).
- Skip greetings, small talk, hypotheticals, jokes.
- Include facts about: events that happened, decisions, preferences,
  relationships, places, dates, plans, possessions, professional details,
  health, family.

Output VALID JSON: a list of short strings, nothing else.

Examples of good atomic facts:
  ["Caroline researched adoption agencies",
   "Caroline is transgender",
   "Caroline moved from Sweden 4 years ago",
   "Melanie does pottery",
   "Melanie went to a concert with her daughter on 12 May 2023"]

Bad facts (do not emit):
  "Caroline said hi" (small talk)
  "Caroline and Melanie talked" (no atomic content)
  "Caroline might do X someday" (hypothetical)

Event session date (if present): {session_date}

Event content:
\"\"\"
{content}
\"\"\"

JSON list:"""


@dataclass
class AtomicFact:
    text: str
    source_ulid: str

    def to_event_content(self) -> str:
        return f"Fact: {self.text}"


class FactExtractor:
    """LLM-driven atomic-fact extractor."""

    def __init__(self, llm_client, max_facts_per_event: int = 30):
        self.llm = llm_client
        self.max_facts = max_facts_per_event

    def extract(self, event: Event) -> list[AtomicFact]:
        if not event or not (event.content or "").strip():
            return []
        meta = event.metadata or {}
        session_date = ""
        for k in ("session_dt", "session_date_time", "event_time"):
            v = meta.get(k)
            if v:
                session_date = str(v)
                break

        prompt = FACT_EXTRACTION_PROMPT.format(
            session_date=session_date or "(unknown)",
            content=(event.content or "")[:5000],
        )
        try:
            text = self._call_llm(prompt)
        except Exception as e:
            log.warning("FactExtractor LLM failed: %s", e)
            return []
        try:
            arr = self._parse_array(text)
        except Exception as e:
            log.warning("FactExtractor parse failed: %s | %s", e, text[:200])
            return []

        facts: list[AtomicFact] = []
        seen = set()
        for s in arr:
            if not isinstance(s, str):
                continue
            t = s.strip()
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            # Filter obvious garbage
            if len(t) < 8 or len(t) > 250:
                continue
            facts.append(AtomicFact(text=t, source_ulid=event.ulid))
            if len(facts) >= self.max_facts:
                break
        return facts

    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        if hasattr(self.llm, "complete"):
            return self.llm.complete(prompt) or ""
        if callable(self.llm):
            return self.llm(prompt) or ""
        raise RuntimeError("FactExtractor: llm must expose .complete() or be callable")

    def _parse_array(self, text: str) -> list:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rsplit("```", 1)[0].strip()
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("no JSON array in output")
        return json.loads(text[start : end + 1])
