"""Surface-ids must reach the agent in the restore text, otherwise the
adherence loop cannot work after `/compact`.

Three concerns covered here:

1. `session_brief` items carry their `ulid` so `_log_lesson_surfaces` can
   mint a `surface_id` for them (previously the ulid was stripped by
   `_it()` and surface logging silently no-op'd).
2. `build_session_restore` logs surfaces BEFORE rendering and embeds the
   id in the rendered text as `[id:N]`.
3. A "Pending lesson confirmations" block appears, listing the ids the
   agent should pass to `mark_lesson_followed`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

from pmb.hooks.session_restore import build_session_restore


# --- session_brief --------------------------------------------------------

def test_session_brief_lessons_include_ulid(tmp_pmb_home, tmp_workspace_dir):
    """A lesson recorded in-session must come back from session_brief with
    its ulid, so session_restore can surface-log it. Otherwise the filter
    `if L.get("ulid")` drops every session lesson and adherence runs blind.
    """
    from pmb.core.engine import Engine
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
                 config_overrides={"recall.cache_size": 0})
    u = eng.record_fact(
        "always pin the python toolchain version",
        metadata={"kind": "lesson", "source": "lesson"},
    )
    brief = eng.session_brief()
    found = [L for L in (brief.get("lessons") or []) if L.get("ulid") == u]
    assert found, "session_brief stripped the lesson's ulid"


# --- build_session_restore ------------------------------------------------

@dataclass
class _FakeWS:
    name: str = "demo"


@dataclass
class _FakeEngine:
    brief: dict = field(default_factory=dict)
    goals: list = field(default_factory=list)
    project: dict | None = None
    detected: dict | None = None
    workspace: _FakeWS = field(default_factory=_FakeWS)
    # surface_id generator and a log so tests can inspect what got logged
    _ids: object = field(default_factory=lambda: count(1000))
    logged: list = field(default_factory=list)

    def session_brief(self, minutes=None):
        return self.brief

    def list_goals(self, status=None, limit=5):
        return self.goals

    def detect_project_in_text(self, text, min_mentions=1):
        return self.detected

    def project_overview(self, name):
        return self.project or {"empty": True}

    def _log_lesson_surfaces(self, lessons, query, source, session_id=None):
        # Match the real signature: mutate each lesson with a surface_id.
        for L in lessons:
            sid = next(self._ids)
            L["surface_id"] = sid
            self.logged.append(
                {"surface_id": sid, "ulid": L.get("ulid"), "source": source}
            )
        return lessons


def test_surface_id_embedded_in_session_lesson_line():
    eng = _FakeEngine(brief={
        "lessons": [{"ulid": "u-1", "content": "use pnpm not npm"}],
    })
    out = build_session_restore(eng, include_project=False)
    assert "[id:1000]" in out
    assert "use pnpm not npm" in out
    # And the surface was actually logged via the fake.
    assert any(r["ulid"] == "u-1" for r in eng.logged)


def test_surface_id_embedded_in_project_lesson_line():
    eng = _FakeEngine(
        brief={"done": [{"content": "worked on LoadGuard"}]},
        detected={"name": "LoadGuard"},
        project={
            "entity": {"name": "LoadGuard", "n_mentions": 30},
            "key_facts": [],
            "lessons": [{"ulid": "u-2", "content": "skip ASCII paths"}],
        },
    )
    out = build_session_restore(eng, include_project=True)
    assert "[id:1000]" in out
    assert "skip ASCII paths" in out


def test_pending_confirmations_block_lists_ids():
    eng = _FakeEngine(brief={
        "lessons": [
            {"ulid": "u-1", "content": "use pnpm not npm"},
            {"ulid": "u-2", "content": "no force push"},
        ],
    })
    out = build_session_restore(eng, include_project=False)
    assert "Pending lesson confirmations" in out
    assert "mark_lesson_followed" in out
    # Both ids referenced in the confirm-block.
    assert "1000" in out and "1001" in out


def test_no_confirm_block_when_no_lessons():
    eng = _FakeEngine(brief={"done": [{"content": "just shipped a thing"}]})
    out = build_session_restore(eng, include_project=False)
    assert "Pending lesson confirmations" not in out
    assert "mark_lesson_followed" not in out


def test_surfaces_logged_before_text_render():
    """Order matters: if surface logging ran AFTER text rendering, the
    surface_id would never appear in the text. This test pins that order
    in by asserting the rendered text references an id the fake produced."""
    eng = _FakeEngine(brief={
        "lessons": [{"ulid": "u-1", "content": "x"}],
    })
    out = build_session_restore(eng, include_project=False)
    sid = eng.logged[0]["surface_id"]
    assert f"[id:{sid}]" in out
