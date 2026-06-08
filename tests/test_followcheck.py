"""Tests for the Stop-hook follow-through inference (no model cooperation).

Covers token extraction, the strong-token gate (the thing that stops topical
coincidence from registering as a follow), and the dispatcher with a fake
engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pmb.hooks.followcheck import (
    _distinctive_tokens,
    _is_strong,
    run_followcheck,
)


# ─── token extraction ───────────────────────────────────────────────────


def test_distinctive_tokens_drops_stopwords_and_short():
    toks = _distinctive_tokens("The agent should use pnpm, never npm here")
    # 'the','should','use','here','agent','npm'(<4? no, 3)→ dropped; 'pnpm' kept
    assert "pnpm" in toks
    assert "the" not in toks
    assert "use" not in toks
    assert "agent" not in toks  # in stoplist


def test_distinctive_tokens_keeps_identifiers():
    toks = _distinctive_tokens("call record_batch and bump qwen2.5 on port 8765")
    assert "record_batch" in toks
    assert "qwen2.5" in toks
    # pure-digit 8765 is dropped by isdigit() guard
    assert "8765" not in toks


def test_is_strong():
    assert _is_strong("dashboard")        # ≥7 chars
    assert _is_strong("record_batch")     # underscore
    assert _is_strong("qwen2.5")          # dot + digit
    assert _is_strong("cold-start")       # hyphen
    assert _is_strong("port5")            # digit
    assert not _is_strong("flag")         # 4 chars, no identifier markers
    assert not _is_strong("queue")        # 5 chars, plain word


# ─── dispatcher ─────────────────────────────────────────────────────────


@dataclass
class FakeEngine:
    surfaces: list[dict] = field(default_factory=list)
    activity: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    just_happened: list[dict] = field(default_factory=list)
    marked: list[tuple] = field(default_factory=list)
    na_marks: list[tuple] = field(default_factory=list)

    def recent_unconfirmed_surfaces(self, minutes=30.0, limit=50):
        return self.surfaces

    def recent_activity(self, minutes=30.0, limit=100):
        return self.activity

    def recent_agent_actions(self, minutes=30.0, limit=200, significant_only=True):
        return [
            a for a in self.actions
            if not significant_only or a.get("significant", True)
        ]

    def what_just_happened(self, n=40):
        return self.just_happened

    def mark_lesson_followed(self, surface_id, followed=True, note=None):
        self.marked.append((surface_id, followed, note))
        return {"ok": True}

    def mark_lesson_not_applicable(self, surface_id, note=None):
        self.na_marks.append((surface_id, note))
        return {"ok": True, "not_applicable": True}


def test_followcheck_marks_on_strong_overlap():
    eng = FakeEngine(
        surfaces=[{
            "surface_id": 1, "lesson_ulid": "L1",
            "content": "Always run pmb warmup before recall to avoid cold-start latency",
        }],
        activity=[
            {"content": "Ran pmb warmup to fix cold-start; recall now fast"},
        ],
    )
    res = run_followcheck(eng, min_overlap=2, min_strong=1)
    assert res.marked_followed == 1
    assert eng.marked and eng.marked[0][0] == 1
    assert eng.marked[0][1] is True
    # note explains the basis honestly
    assert "auto-detected" in (eng.marked[0][2] or "")


def test_followcheck_skips_on_weak_overlap_only():
    # Overlap exists but only on common/short words → strong gate blocks it.
    eng = FakeEngine(
        surfaces=[{
            "surface_id": 2, "lesson_ulid": "L2",
            "content": "This rule is about the first case before any other",
        }],
        activity=[
            {"content": "I did the first thing before the other case here"},
        ],
    )
    res = run_followcheck(eng, min_overlap=2, min_strong=2)
    assert res.marked_followed == 0
    assert eng.marked == []


def test_followcheck_no_activity_skips():
    eng = FakeEngine(
        surfaces=[{"surface_id": 3, "lesson_ulid": "L3",
                   "content": "Use lancedb arm64 wheel on macos"}],
        activity=[],
        just_happened=[],
    )
    res = run_followcheck(eng)
    assert res.marked_followed == 0
    assert res.skipped_reason is not None


def test_followcheck_no_surfaces_skips():
    eng = FakeEngine(surfaces=[])
    res = run_followcheck(eng)
    assert res.checked == 0
    assert res.skipped_reason == "no unconfirmed surfaces in window"


def test_followcheck_dry_run_does_not_write():
    eng = FakeEngine(
        surfaces=[{
            "surface_id": 4, "lesson_ulid": "L4",
            "content": "Configure paraphrase-multilingual embedding model for recall",
        }],
        activity=[
            {"content": "Switched to paraphrase-multilingual model; recall improved"},
        ],
    )
    res = run_followcheck(eng, min_overlap=2, min_strong=1, apply=False)
    assert res.marked_followed == 1   # would-mark count
    assert eng.marked == []            # but nothing actually written


def test_followcheck_requires_min_overlap():
    # Only ONE strong token shared → below default min_overlap=3.
    eng = FakeEngine(
        surfaces=[{
            "surface_id": 5, "lesson_ulid": "L5",
            "content": "dashboard rendering uses requestanimationframe for svg",
        }],
        activity=[{"content": "touched the dashboard layout only"}],
    )
    res = run_followcheck(eng, min_overlap=3, min_strong=2)
    assert res.marked_followed == 0


def test_followcheck_broad_corroboration_one_strong_token():
    """Live-e2e regression: ONE killer identifier (is_warm) backed by many
    regular matches (recall, hook, cold, load, auth, zephyr) is a real follow
    — but only one token is 'strong', so the rigid 2-strong gate wrongly
    rejected it. The broad-corroboration clause must mark it at defaults."""
    eng = FakeEngine(
        surfaces=[{
            "surface_id": 7, "lesson_ulid": "L7",
            "content": ("Zephyr auth hook rule: always probe is_warm() before "
                        "calling recall() in hook code; a cold "
                        "sentence-transformers load adds 15s to that turn"),
        }],
        activity=[{
            "content": ("Added an is_warm() guard before the recall() call in "
                        "the Zephyr auth hook so cold turns skip the load"),
        }],
    )
    # defaults are min_overlap=3, min_strong=2 — must mark via the broad path
    res = run_followcheck(eng)
    assert res.marked_followed == 1
    assert eng.marked and eng.marked[0][0] == 7
    assert "is_warm" in (eng.marked[0][2] or "")


def test_followcheck_one_strong_thin_corroboration_still_skips():
    """Guardrail on the broad clause: overlap meets the base floor (3) with
    ONE strong token but only thin corroboration (< min_overlap+2 = 5) → must
    NOT mark. Keeps the broad path from degrading into coincidence."""
    eng = FakeEngine(
        surfaces=[{
            "surface_id": 8, "lesson_ulid": "L8",
            "content": "The is_warm guard helps cold paths",
        }],
        # shares exactly {is_warm, cold, guard} = 3 (one strong) → broad needs 5
        activity=[{"content": "added is_warm to the cold guard path"}],
    )
    res = run_followcheck(eng, min_overlap=3, min_strong=2)
    assert res.marked_followed == 0


def test_followcheck_marks_from_observed_actions_without_activity_record():
    """Ambient edits/commands close the loop even when the agent forgot to
    record an activity summary."""
    eng = FakeEngine(
        surfaces=[{
            "surface_id": 9, "lesson_ulid": "L9", "surfaced_at": 100.0,
            "content": "Use pnpm build when changing src/auth.py",
        }],
        actions=[
            {
                "tool": "Edit", "target": "src/auth.py", "status": "ok",
                "timestamp": 110.0, "significant": True, "surface_ids": "9",
            },
            {
                "tool": "Bash", "target": "pnpm build", "status": "ok",
                "timestamp": 120.0, "significant": True, "surface_ids": "9",
            },
        ],
    )
    res = run_followcheck(eng, min_overlap=2, min_strong=1)
    assert res.marked_followed == 1
    assert eng.marked and eng.marked[0][0] == 9
    assert "observed action" in (eng.marked[0][2] or "")


def test_followcheck_ignores_actions_before_surface():
    """Work done before the lesson appeared cannot prove it influenced work."""
    eng = FakeEngine(
        surfaces=[{
            "surface_id": 10, "lesson_ulid": "L10", "surfaced_at": 200.0,
            "content": "Use pnpm build when changing src/auth.py",
        }],
        actions=[{
            "tool": "Bash", "target": "pnpm build src/auth.py", "status": "ok",
            "timestamp": 100.0, "significant": True, "surface_ids": "",
        }],
    )
    res = run_followcheck(eng, min_overlap=2, min_strong=1)
    assert res.marked_followed == 0
    assert eng.marked == []


def test_followcheck_respects_explicit_surface_link():
    """An action linked to another active lesson must not confirm this one."""
    eng = FakeEngine(
        surfaces=[{
            "surface_id": 11, "lesson_ulid": "L11", "surfaced_at": 100.0,
            "content": "Use pnpm build when changing src/auth.py",
        }],
        actions=[{
            "tool": "Bash", "target": "pnpm build src/auth.py", "status": "ok",
            "timestamp": 110.0, "significant": True, "surface_ids": "12",
        }],
    )
    res = run_followcheck(eng, min_overlap=2, min_strong=1)
    assert res.marked_followed == 0
    assert eng.marked == []


# ─── not-applicable classification (adherence denominator) ────────────────


def test_followcheck_marks_not_applicable_on_zero_overlap():
    """A surfaced lesson with ZERO token overlap with the turn's work is marked
    not_applicable (followed=-1), so it's excluded from the adherence
    denominator instead of dragging it down as a phantom 'not followed'."""
    eng = FakeEngine(
        surfaces=[{
            "surface_id": 20, "lesson_ulid": "L20",
            "content": "Always run pmb warmup before recall to avoid cold-start",
        }],
        activity=[
            {"content": "Refactored the invoice PDF exporter and its currency rounding"},
        ],
    )
    res = run_followcheck(eng, min_overlap=2, min_strong=1)
    assert res.marked_followed == 0
    assert res.not_applicable == 1
    assert eng.na_marks and eng.na_marks[0][0] == 20
    assert eng.marked == []  # never counted as followed OR ignored


def test_followcheck_partial_overlap_stays_unconfirmed():
    """Some overlap but below the follow bar → NEITHER followed nor
    not_applicable. It was plausibly relevant, so it honestly stays '?'."""
    eng = FakeEngine(
        surfaces=[{
            "surface_id": 21, "lesson_ulid": "L21",
            "content": "Use lancedb arm64 wheel and paraphrase-multilingual model on macos",
        }],
        # shares only 'lancedb' (1) → < min_overlap=3, but NOT zero overlap
        activity=[{"content": "poked at the lancedb table layout briefly"}],
    )
    res = run_followcheck(eng, min_overlap=3, min_strong=2)
    assert res.marked_followed == 0
    assert res.not_applicable == 0          # NOT marked not-applicable
    assert eng.na_marks == [] and eng.marked == []


def test_followcheck_no_evidence_does_not_mark_not_applicable():
    """No recorded activity / actions → we can't judge relevance, so nothing
    is marked not_applicable (skips before the matching loop)."""
    eng = FakeEngine(
        surfaces=[{"surface_id": 22, "lesson_ulid": "L22",
                   "content": "Use lancedb arm64 wheel on macos"}],
        activity=[], actions=[], just_happened=[],
    )
    res = run_followcheck(eng)
    assert res.not_applicable == 0
    assert eng.na_marks == []


def test_followcheck_can_disable_not_applicable():
    """mark_not_applicable=False preserves the old behaviour (leave '?')."""
    eng = FakeEngine(
        surfaces=[{
            "surface_id": 23, "lesson_ulid": "L23",
            "content": "Always run pmb warmup before recall to avoid cold-start",
        }],
        activity=[{"content": "Refactored the invoice exporter and currency rounding"}],
    )
    res = run_followcheck(eng, mark_not_applicable=False)
    assert res.not_applicable == 0
    assert eng.na_marks == []
