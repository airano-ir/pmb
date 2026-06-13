"""Tests for the LLM-as-judge LoCoMo evaluator (Improvement A)."""
from __future__ import annotations

import json

from pmb.eval.locomo_judge import (
    JUDGE_PROMPT,
    READER_PROMPT,
    JudgeResult,
    LocomoJudge,
    aggregate,
)


class _ScriptedLLM:
    """Returns scripted responses in order. Used to stub reader and judge."""
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, prompt: str, **kw) -> str:
        out = self.responses[self.calls]
        self.calls += 1
        return out


def test_reader_judge_correct_answer():
    """End-to-end: reader produces an answer, judge says correct."""
    reader = _ScriptedLLM(["Postgres on port 5433"])
    judge = _ScriptedLLM([json.dumps({"correct": 1, "reasoning": "matches"})])
    j = LocomoJudge(reader_llm=reader, judge_llm=judge)
    r = j.run_question(
        question="What port does Postgres use?",
        gold="Postgres on port 5433",
        retrieved_contents=["Project: Postgres 17 on port 5433"],
        category=1,
    )
    assert r.correct == 1
    assert "matches" in r.reasoning
    assert r.category == 1


def test_reader_judge_incorrect_answer():
    reader = _ScriptedLLM(["I don't know."])
    judge = _ScriptedLLM([json.dumps({"correct": 0, "reasoning": "missing"})])
    j = LocomoJudge(reader_llm=reader, judge_llm=judge)
    r = j.run_question(
        question="When did Alice travel?",
        gold="December 5",
        retrieved_contents=["irrelevant note"],
        category=3,
    )
    assert r.correct == 0


def test_reader_judge_handles_markdown_fenced_json():
    """Many real LLMs wrap JSON in ```json fences. Must still parse."""
    reader = _ScriptedLLM(["answer"])
    judge = _ScriptedLLM(['```json\n{"correct": 1, "reasoning": "ok"}\n```'])
    j = LocomoJudge(reader_llm=reader, judge_llm=judge)
    r = j.run_question(question="q", gold="g", retrieved_contents=["c"])
    assert r.correct == 1


def test_judge_falls_back_to_regex_on_malformed_json():
    """If LLM emits malformed but parseable text, salvage the verdict."""
    reader = _ScriptedLLM(["x"])
    judge = _ScriptedLLM(['the answer is correct: 1'])
    j = LocomoJudge(reader_llm=reader, judge_llm=judge)
    r = j.run_question(question="q", gold="g", retrieved_contents=["c"])
    assert r.correct == 1


def test_aggregate_computes_per_category_j_score():
    results = [
        JudgeResult(question="q1", gold="g", prediction="p", correct=1, category=1),
        JudgeResult(question="q2", gold="g", prediction="p", correct=0, category=1),
        JudgeResult(question="q3", gold="g", prediction="p", correct=1, category=3),
        JudgeResult(question="q4", gold="g", prediction="p", correct=1, category=3),
    ]
    run = aggregate(results)
    assert run.n_total == 4
    assert run.n_correct == 3
    assert run.j_score == 0.75
    summary = run.to_summary()
    assert summary["per_category"]["1"]["j_score"] == 0.5
    assert summary["per_category"]["3"]["j_score"] == 1.0


def test_reader_failure_doesnt_crash():
    class _Bad:
        def complete(self, prompt, **kw):
            raise RuntimeError("reader down")

    judge = _ScriptedLLM([json.dumps({"correct": 0, "reasoning": "no answer"})])
    j = LocomoJudge(reader_llm=_Bad(), judge_llm=judge)
    r = j.run_question(question="q", gold="g", retrieved_contents=["c"])
    # Reader failure → predicted "I don't know", judge says wrong → correct=0
    assert r.correct == 0
    assert "don't know" in r.prediction.lower()


def test_prompts_contain_expected_anchors():
    """Make sure prompts actually carry the rubric we documented."""
    assert "I don't know" in READER_PROMPT
    assert "shortest" in READER_PROMPT.lower() or "short" in READER_PROMPT.lower()
    assert "contradicts" in JUDGE_PROMPT.lower() or "contains" in JUDGE_PROMPT.lower()
    assert "0 or 1" in JUDGE_PROMPT
