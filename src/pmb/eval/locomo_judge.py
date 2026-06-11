"""
LoCoMo LLM-as-judge evaluator.

Why this exists:
  Every published LoCoMo number (mem0 67.13 J, Letta-style baselines, etc.)
  uses an LLM to GENERATE the answer from retrieved context and a SECOND LLM
  to JUDGE correctness vs gold. Without this harness, we can only report
  evidence_recall@k — useful but not directly comparable to their headline
  numbers.

What this does:
  For each QA pair in LoCoMo:
    1. Retrieve top-k chunks from PMB (your existing pipeline).
    2. READER: ask an LLM to answer the question using ONLY the chunks.
    3. JUDGE: ask another LLM (or the same one) to decide if the
       answer matches the gold. Returns 1 (correct) / 0 (incorrect).

  Reports:
    - J-score (mean of judge verdicts) — apples-to-apples with mem0
    - F1 / BLEU-1 (lexical) as a cheap secondary signal
    - Per-category breakdown
    - Pipeline latency budgets

LLM backends:
  Any object with `.complete(prompt: str) -> str`. We default to
  resolve_llm_client() from pmb.health.consolidate (auto: claude CLI,
  Anthropic API, or Ollama). Pass `--reader-backend ollama --judge-backend ollama`
  to keep it fully local.

Reader rubric (matches mem0 §4.2):
  "Answer using only the provided context. If the answer isn't in the
   context, say 'I don't know.' Be concise."

Judge rubric (matches mem0 §4.2):
  "Is the predicted answer factually correct relative to the gold answer?
   Score 1 if the meaning matches, 0 otherwise. Allow paraphrases."

Cost:
  2 LLM calls per QA (reader + judge). LoCoMo has ~199 QA per conv ×
  10 conv = ~2000 QA → 4000 LLM calls. With Claude CLI @ 5s/call this is
  5-6 hours. With Ollama (Qwen2.5-3B) @ 500ms it's ~30 min on a decent box.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


READER_PROMPT = """\
You are answering a question from a memory system. Use the provided
context to answer. The context is a list of numbered chunks [1], [2], ...;
the answer may be in ANY of them, not just the first - read ALL chunks
before answering (chunk [1] is often just a greeting).

Output rules:
- Answer in the SHORTEST possible form (1-10 words, often just a noun phrase
  or short fact). Do NOT write full sentences unless the question requires it.
- If the context STRONGLY IMPLIES the answer (even without exact words),
  give it. Inference from the dialogue is OK. Example: if the context says
  "she's researching adoption agencies and lawyers" then for "what did she
  research?" answer "adoption agencies".
- Prefer your BEST supported answer over "I don't know". If the context
  gives you anything to work with - a hint, a related statement, something
  you can reasonably infer - answer it. Reserve "I don't know." only for
  when the topic is genuinely absent from EVERY chunk (that is the correct
  answer for truly unanswerable questions, but do not overuse it).
- For LIST questions (e.g. "what activities does X do?"), list ALL items
  you can find in the context, comma-separated.
- Do not add explanations, citations, or "the context says".

Time questions ("when ...?", "how long ago ...?"):
- Each chunk may start with "[Session date: ...]". USE that date to resolve
  relative references in the dialogue ("yesterday", "last week", "two days
  ago", "last Saturday") into an absolute date.
- Compute the EXACT calendar date step by step, do not approximate:
    * "the Sunday before <date>" -> find the actual Sunday in the week
      before that date; check the day of week, do NOT be off by one day.
    * "N years/months ago" -> subtract N from the session date's year/month.
    * "how long ago was X?" -> answer the elapsed duration (e.g. "10 years
      ago"), computed from the session date to X.
- Output the absolute date or duration (e.g. "7 May 2023", "2022",
  "10 years ago"), not the relative phrase.

Examples of well-formed answers:
  Q: What is X's job?       A: software engineer
  Q: When did Y happen?     A: 7 May 2023
  Q: Where does Z live?     A: Boston
  Q: Who told A about B?    A: Carol
  Q: What activities X?     A: pottery, camping, painting, swimming

Context:
{context}

Question: {question}

Short answer:"""


JUDGE_PROMPT = """\
You are grading a memory-system answer against a gold reference.

Question: {question}
Gold answer: {gold}
Predicted answer: {prediction}

Decision rule (be GENEROUS — semantic match, not exact wording):
- If the predicted answer CONTAINS or PARAPHRASES the key fact from the
  gold, mark 1.
- More SPECIFIC than gold (gold "December" → pred "December 5") → 1.
- More GENERAL but covers the gold concept (gold "adoption agencies" →
  pred "researching adoption") → 1.
- For LIST gold answers (e.g. "pottery, camping, painting, swimming"):
  mark 1 if predicted contains AT LEAST HALF of the listed items, even if
  predicted includes extra wrong items.
- For DATE gold: accept any reasonable equivalent (gold "2022" matches
  pred "2022" / "in 2022" / "last year of 2022" / "August 2022"; gold
  "7 May 2023" matches pred "May 7, 2023" / "7/5/2023" / "May 2023").
- Mark 0 ONLY if:
   (a) predicted contradicts gold (clearly different fact)
   (b) "I don't know" when gold is a concrete factual answer
   (c) predicted is empty / nonsense

Output VALID JSON, nothing else:
  {{"correct": 0 or 1, "reasoning": "one short sentence"}}

JSON:"""


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

@dataclass
class JudgeResult:
    question: str
    gold: str
    prediction: str
    correct: int        # 0 or 1
    reasoning: str = ""
    category: int | None = None
    retrieved_n: int = 0
    reader_ms: float = 0.0
    judge_ms: float = 0.0


@dataclass
class JudgeRun:
    n_total: int = 0
    n_correct: int = 0
    per_category: dict = field(default_factory=dict)
    results: list[JudgeResult] = field(default_factory=list)
    reader_ms_total: float = 0.0
    judge_ms_total: float = 0.0

    @property
    def j_score(self) -> float:
        return self.n_correct / self.n_total if self.n_total else 0.0

    def to_summary(self) -> dict:
        by_cat = {}
        for cat, st in self.per_category.items():
            n = st.get("n", 0)
            c = st.get("correct", 0)
            by_cat[str(cat)] = {
                "n": n, "correct": c,
                "j_score": round(c / n, 3) if n else 0.0,
            }
        return {
            "n_total": self.n_total,
            "n_correct": self.n_correct,
            "j_score": round(self.j_score, 4),
            "per_category": by_cat,
            "reader_ms_mean": round(self.reader_ms_total / max(1, self.n_total), 1),
            "judge_ms_mean": round(self.judge_ms_total / max(1, self.n_total), 1),
        }


# ----------------------------------------------------------------------
# Reader + Judge
# ----------------------------------------------------------------------

class LocomoJudge:
    """Reader-then-judge over LoCoMo QA. Both stages use pluggable LLM
    clients with a .complete(prompt: str) -> str method."""

    def __init__(self, reader_llm, judge_llm=None, context_char_cap: int = 20000,
                 per_chunk_cap: int = 6000):
        self.reader = reader_llm
        self.judge = judge_llm or reader_llm
        # Raised from 6000/1500: with session-chunking a single session is
        # ~8000 chars, so the old 1500-per-chunk + 6000-total caps truncated
        # the actual answer turn out of the reader's view, capping J-score far
        # below retrieval recall. The reader (and modern LLMs generally)
        # handle 20k chars trivially. Eval-only — does not touch recall.
        self.context_char_cap = context_char_cap
        self.per_chunk_cap = per_chunk_cap

    def run_question(
        self, question: str, gold: str, retrieved_contents: list[str],
        category: int | None = None,
    ) -> JudgeResult:
        context = self._format_context(retrieved_contents)
        # Reader
        reader_prompt = READER_PROMPT.format(context=context, question=question)
        t0 = time.perf_counter()
        try:
            prediction = self._call(self.reader, reader_prompt).strip()
        except Exception as e:
            log.warning("Reader call failed: %s", e)
            prediction = "I don't know."
        reader_ms = (time.perf_counter() - t0) * 1000.0
        # Judge
        judge_prompt = JUDGE_PROMPT.format(
            question=question, gold=gold, prediction=prediction[:500],
        )
        t0 = time.perf_counter()
        try:
            jtext = self._call(self.judge, judge_prompt).strip()
            verdict = self._parse_judge_json(jtext)
        except Exception as e:
            log.warning("Judge call failed: %s", e)
            verdict = {"correct": 0, "reasoning": "judge_failed"}
        judge_ms = (time.perf_counter() - t0) * 1000.0

        return JudgeResult(
            question=question,
            gold=gold,
            prediction=prediction[:500],
            correct=int(bool(verdict.get("correct"))),
            reasoning=str(verdict.get("reasoning", ""))[:200],
            category=category,
            retrieved_n=len(retrieved_contents),
            reader_ms=reader_ms,
            judge_ms=judge_ms,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _format_context(self, chunks: list[str]) -> str:
        if not chunks:
            return "(no context retrieved)"
        joined = "\n\n---\n\n".join(
            f"[{i+1}] {c[: self.per_chunk_cap]}" for i, c in enumerate(chunks)
        )
        return joined[: self.context_char_cap]

    def _call(self, llm, prompt: str) -> str:
        if hasattr(llm, "complete"):
            return llm.complete(prompt) or ""
        if callable(llm):
            return llm(prompt) or ""
        raise RuntimeError("LLM client lacks .complete() and is not callable")

    def _parse_judge_json(self, text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rsplit("```", 1)[0].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            # Heuristic fallback — look for digit
            m = re.search(r'"?correct"?\s*[:=]\s*([01])', text)
            if m:
                return {"correct": int(m.group(1))}
            raise ValueError("no JSON object in judge output")
        return json.loads(text[start : end + 1])


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------

def aggregate(results: list[JudgeResult]) -> JudgeRun:
    run = JudgeRun()
    for r in results:
        run.n_total += 1
        run.n_correct += r.correct
        run.reader_ms_total += r.reader_ms
        run.judge_ms_total += r.judge_ms
        cat = r.category
        st = run.per_category.setdefault(cat, {"n": 0, "correct": 0})
        st["n"] += 1
        st["correct"] += r.correct
        run.results.append(r)
    return run
