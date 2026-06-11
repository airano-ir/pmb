"""V3 + V4 — intent-classifier eval set and PAMVR multiplier freeze.

V3 gates `detect_intents` (the hook's zero-cooperation intent routing): a
labelled EN/RU/UK set asserts each message still routes to the right intent,
including WORK_REQUEST (R4) and trivial-ack negatives. A regression in the
lang-pack intent regexes (Phase L) or the R4 heuristics fails here.

V4 freezes the PAMVR boost multipliers. `apply_pamvr` is a pure float->float
reranker; with a `trace` list it reports WHICH rule applied WHICH multiplier.
We pin the trace for canonical cases so no multiplier can change silently —
any tweak to a `score *= X` line must update the golden here (the conscious
delta the plan asks for). Complements V1's end-to-end recall floors.
"""
from __future__ import annotations

import pytest

from pmb.hooks.auto_recall import detect_intents
from pmb.reasoning.pamvr import apply_pamvr

# ── V3: message -> an intent that MUST be present in the classification ──────
INTENT_CASES = [
    # RU
    ("что я делал вчера", "PAST_QUERY"),
    ("какие у меня цели", "GOALS_QUERY"),
    ("что мы только что обсуждали", "RECENT_QUERY"),
    ("какие правила проекта", "LESSONS_QUERY"),
    ("привет", "SKIP"),
    # UK
    ("які у мене цілі", "GOALS_QUERY"),
    ("дякую", "SKIP"),
    # EN
    ("what did I do yesterday", "PAST_QUERY"),
    ("what is left to do", "GOALS_QUERY"),
    ("do we have a rule about commits", "LESSONS_QUERY"),
    ("refactor the auth module", "WORK_REQUEST"),
    ("fix the login bug", "WORK_REQUEST"),
    ("thanks", "SKIP"),
]


@pytest.mark.eval
@pytest.mark.parametrize("msg,expected", INTENT_CASES)
def test_intent_routing(msg, expected):
    got = detect_intents(msg, known_projects=set())
    assert expected in got, f"{msg!r} -> {got}, expected {expected!r} present"


# ── V4: frozen PAMVR traces (rule -> multiplier) for canonical cases ─────────
class _Ev:
    def __init__(self, content, metadata=None, event_type="fact"):
        self.content = content
        self.metadata = metadata or {}
        self.event_type = event_type


PAMVR_GOLDEN = [
    # query, event, base, expected_trace
    (
        "where do we deploy the service",
        _Ev("we deploy the service to fargate"),
        1.0,
        [
            {"rule": "verb-match", "mult": 1.25},
            {"rule": "verb+topic combo (both agree)", "mult": 1.5},
            {"rule": "keyword-AND (query token overlap)", "mult": 1.5},
            {"rule": "vocab-bridge (domain synonym)", "mult": 1.35},
        ],
    ),
    (
        "quantum chromodynamics lecture notes",
        _Ev("we deploy the service to fargate"),
        1.0,
        [
            {"rule": "topic-intersection (zero overlap penalty)", "mult": 0.7},
            {"rule": "keyword-AND (query token overlap)", "mult": 0.92},
        ],
    ),
    (
        "what was the fix for the login bug",
        _Ev("Fix: login bug resolved by clearing the cache", {"kind": "fix"}),
        1.0,
        [
            {"rule": "keyword-AND (query token overlap)", "mult": 1.5},
            {"rule": "prefix-kind (fix:/decision marker)", "mult": 1.3},
        ],
    ),
]


@pytest.mark.eval
@pytest.mark.parametrize("query,event,base,expected", PAMVR_GOLDEN)
def test_pamvr_multiplier_freeze(query, event, base, expected):
    trace: list = []
    apply_pamvr(query, event, base, trace=trace)
    assert trace == expected, (
        "PAMVR multipliers changed — if this is intentional, update the golden "
        f"trace here.\n got: {trace}\n exp: {expected}"
    )
