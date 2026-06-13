"""Tests for PAMVR rule tracing (`explain_pamvr`) that powers `pmb why`.

The critical invariant: turning the trace ON must NOT change the score the
hot path computes with trace OFF. If these drift, recall correctness and the
explanation would disagree - so we assert they're numerically identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pmb.reasoning.pamvr import apply_pamvr, explain_pamvr


@dataclass
class FakeEvent:
    content: str
    metadata: dict = field(default_factory=dict)


SAMPLES = [
    ("where do I live now", "I live in Lisbon currently"),
    ("where do I live now", "I used to live in Berlin previously"),
    ("what does Alice think about Postgres", "Alice prefers Postgres for JSONB"),
    ("what does Alice think about Postgres", "Bob likes MySQL"),
    ("what's the typing policy", "Going forward we enforce mypy strict"),
    ("how long is the JWT lifetime", "JWT tokens are valid for 30 minutes"),
    ("who is on the backend team", "Alice, Bob and Carol own the backend"),
    ("did we use Fargate", "We deployed the API on Fargate"),
    ("random unrelated query text", "completely different content here"),
]


def test_trace_does_not_change_score():
    """trace ON must produce the exact same final score as trace OFF."""
    for q, c in SAMPLES:
        ev = FakeEvent(content=c)
        no_trace = apply_pamvr(q, ev, 1.0)
        explained = explain_pamvr(q, ev, 1.0)
        assert abs(no_trace - explained["final_score"]) < 1e-9, (
            f"score drift on {q!r}/{c!r}: "
            f"{no_trace} vs {explained['final_score']}"
        )


def test_net_multiplier_equals_product_of_rules():
    """The net multiplier must equal the product of the individual rule mults."""
    for q, c in SAMPLES:
        ev = FakeEvent(content=c)
        out = explain_pamvr(q, ev, 1.0)
        product = 1.0
        for step in out["rules_fired"]:
            product *= step["mult"]
        assert abs(product - out["net_multiplier"]) < 1e-2, (
            f"{q!r}: product {product} != net {out['net_multiplier']}"
        )


def test_present_tense_boosts_present_fact():
    ev_now = FakeEvent(content="I live in Lisbon currently")
    out = explain_pamvr("where do I live now", ev_now, 1.0)
    # a "now/current" rule should fire and push the score up
    rules = {r["rule"] for r in out["rules_fired"]}
    assert any("now/current" in r for r in rules)
    assert out["net_multiplier"] > 1.0


def test_past_tense_demoted_for_now_query():
    ev_now = FakeEvent(content="I live in Lisbon currently")
    ev_past = FakeEvent(content="I used to live in Berlin previously")
    now = explain_pamvr("where do I live now", ev_now, 1.0)["final_score"]
    past = explain_pamvr("where do I live now", ev_past, 1.0)["final_score"]
    assert now > past  # present-tense fact ranks above past-tense for "now"


def test_unrelated_content_is_penalised():
    ev = FakeEvent(content="completely different content here")
    out = explain_pamvr("random unrelated query text", ev, 1.0)
    assert out["net_multiplier"] <= 1.0


def test_empty_query_no_rules():
    ev = FakeEvent(content="anything")
    out = explain_pamvr("", ev, 1.0)
    assert out["rules_fired"] == []
    assert out["final_score"] == 1.0


def test_rules_have_expected_shape():
    ev = FakeEvent(content="Alice prefers Postgres for JSONB")
    out = explain_pamvr("what does Alice think about Postgres", ev, 1.0)
    assert out["rules_fired"], "expected at least one rule to fire"
    for step in out["rules_fired"]:
        assert set(step.keys()) == {"rule", "mult"}
        assert isinstance(step["rule"], str)
        assert isinstance(step["mult"], float)
