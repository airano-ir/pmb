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
    just_happened: list[dict] = field(default_factory=list)
    marked: list[tuple] = field(default_factory=list)

    def recent_unconfirmed_surfaces(self, minutes=30.0, limit=50):
        return self.surfaces

    def recent_activity(self, minutes=30.0, limit=100):
        return self.activity

    def what_just_happened(self, n=40):
        return self.just_happened

    def mark_lesson_followed(self, surface_id, followed=True, note=None):
        self.marked.append((surface_id, followed, note))
        return {"ok": True}


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
