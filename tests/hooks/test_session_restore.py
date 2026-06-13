"""Tests for post-compact session-restore (build_session_restore)."""

from __future__ import annotations

from dataclasses import dataclass, field

from pmb.hooks.session_restore import build_session_restore


@dataclass
class FakeWS:
    name: str = "demo"


@dataclass
class FakeEngine:
    brief: dict = field(default_factory=dict)
    goals: list = field(default_factory=list)
    project: dict | None = None
    detected: dict | None = None
    workspace: FakeWS = field(default_factory=FakeWS)

    def session_brief(self, minutes=None):
        return self.brief

    def list_goals(self, status=None, limit=5):
        return self.goals

    def detect_project_in_text(self, text, min_mentions=1):
        return self.detected

    def project_overview(self, name):
        return self.project or {"empty": True}


def test_empty_brief_no_goals_no_project_returns_empty():
    eng = FakeEngine(brief={"empty": True}, goals=[], detected=None,
                     workspace=FakeWS(name="x"))
    # project detection falls back to workspace name 'x' (len 1 → skipped),
    # so nothing to show.
    out = build_session_restore(eng, include_project=False)
    assert out == ""


def test_brief_with_done_and_decisions_renders():
    eng = FakeEngine(brief={
        "scope": "session", "n_events": 12,
        "done": [{"content": "Fixed the auth bug"}],
        "decisions": [{"content": "Chose Postgres over Mongo"}],
        "lessons": [{"content": "Use pnpm never npm"}],
        "failures": [],
        "goals": [],
        "other": [],
    })
    out = build_session_restore(eng, include_project=False)
    assert "session restore" in out.lower()
    assert "Fixed the auth bug" in out
    assert "Chose Postgres over Mongo" in out
    assert "Use pnpm never npm" in out
    assert "pick the thread back up" in out.lower()


def test_open_goals_surface():
    eng = FakeEngine(
        brief={"done": [{"content": "did a thing"}]},
        goals=[{"title": "Ship v1"}],
    )
    out = build_session_restore(eng, include_project=False)
    assert "Ship v1" in out


def test_project_overview_attached_when_detected():
    eng = FakeEngine(
        brief={"done": [{"content": "worked on LoadGuard engine"}]},
        detected={"name": "LoadGuard"},
        project={
            "entity": {"name": "LoadGuard", "n_mentions": 30},
            "key_facts": [{"content": "TypeScript profit engine"}],
            "lessons": [{"content": "non-ASCII path breaks node --run"}],
        },
    )
    out = build_session_restore(eng, include_project=True)
    assert "LoadGuard" in out
    assert "TypeScript profit engine" in out
    assert "non-ASCII path breaks" in out


def test_max_chars_truncation():
    eng = FakeEngine(brief={
        "done": [{"content": "X" * 500} for _ in range(20)],
    })
    out = build_session_restore(eng, include_project=False, max_chars=300)
    assert len(out) <= 300
    assert "restore truncated" in out


def test_brief_failure_safe():
    # session_brief raising must not blow up the caller.
    class Boom(FakeEngine):
        def session_brief(self, minutes=None):
            raise RuntimeError("db locked")
    eng = Boom()
    assert build_session_restore(eng) == ""
