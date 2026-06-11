"""R10: lesson follow-through rate is over DECIDED surfaces, not all surfaces.

A surfaced-but-unmarked lesson is UNKNOWN, not "not followed". The rate must be
followed / (followed + ignored); the unknown bucket is reported separately so a
frequently-surfaced-but-unmarked rule can't drag the rate toward 0.
"""
from __future__ import annotations

from pmb.core.engine import Engine


def _engine(ws, home):
    return Engine(cwd=ws, pmb_home=home, config_overrides={"recall.cache_size": 0})


def test_followthrough_excludes_unknown(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    u = eng.record_fact("use ruff not flake8", metadata={"kind": "lesson"})
    L = {"ulid": u, "content": "use ruff not flake8"}
    # one surface per session (R1 dedups within a session-hour)
    sids = []
    for s in ("s1", "s2", "s3"):
        r = eng._log_lesson_surfaces([dict(L)], query="lint", source="recall",
                                     session_id=s)
        sids.append(r[0]["surface_id"])
    eng.mark_lesson_followed(sids[0], followed=True)   # followed
    eng.mark_lesson_followed(sids[1], followed=False)  # ignored
    # sids[2] left unknown

    s = eng.adherence_stats(days=1.0)
    assert s["lesson_followed"] == 1
    assert s["lesson_ignored"] == 1
    assert s["lesson_unknown"] == 1
    assert s["lesson_decided"] == 2
    # rate is 1/2 (decided), NOT 1/3 (all surfaces) — the unknown is excluded
    assert abs(s["lesson_followthrough"] - 0.5) < 1e-6


def test_all_unknown_does_not_tank_rate(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    u = eng.record_fact("commit in imperative mood", metadata={"kind": "lesson"})
    for s in ("a", "b", "c", "d"):
        eng._log_lesson_surfaces([{"ulid": u, "content": "x"}], query="q",
                                 source="recall", session_id=s)
    st = eng.adherence_stats(days=1.0)
    assert st["lesson_unknown"] == 4
    assert st["lesson_decided"] == 0
    # nothing decided → rate stays 0.0 but that's "no data", not "0% followed";
    # the unknown count tells the real story.
    assert st["lesson_followthrough"] == 0.0
