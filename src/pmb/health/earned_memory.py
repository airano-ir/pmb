"""Earned Memory - close the surface -> outcome loop.

PMB already logs WHICH lesson surfaces were active when each agent action ran
(`agent_actions.surface_ids`, set by ambient_log._active_surface_ids) and can
classify a turn's OUTCOME without an LLM (`autowrite.classify_actions`:
tests pass/fail, red->green resolution, build, deploy, edit churn). The one
missing edge is the join: surfaced-lesson -> outcome-of-the-turn-it-was-in.

This module computes that join, purely from existing tables (model-free), and
reports per-lesson EFFECTIVENESS:

  * success_rate  - of outcome-bearing turns where the lesson was active
  * lift          - success_rate minus the baseline (turns with NO lesson active)
  * avg_churn     - significant actions per turn (lower = the lesson made the
                    agent converge faster); churn_delta vs baseline

So PMB can finally answer "which memories pull weight, which are dead weight,
which are HARMFUL (precede failures)" - the thing that today requires manual
A/B. Only turns with a CLEAN outcome signal (a test/build ran, a deploy, or a
red->green) are counted, so `n_outcome_turns` doubles as a signal-density check.
"""
from __future__ import annotations

import sqlite3
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmb.core.engine import Engine


def _turn_outcome(classified: dict) -> tuple[bool, bool]:
    """(has_clean_outcome, success) for a classified turn."""
    c = classified
    has = bool(c["tests_passed"] or c["tests_failed"] or c["build_ok"]
               or c["deploy"] or c["error_resolution"])
    success = bool(
        (c["tests_passed"] or c["error_resolution"] or c["deploy"] or c["build_ok"])
        and not (c["tests_failed"] and not c["error_resolution"])
    )
    return has, success


def lesson_impact(engine: Engine, window_days: int = 30, gap_seconds: float = 300.0) -> dict:
    """Per-lesson effectiveness from the surface->outcome join. Model-free.

    Groups agent_actions into turns (same session, gaps < gap_seconds), keeps
    only turns with a clean outcome signal, attributes each turn's outcome to the
    lessons that were active during it, and compares against the baseline of
    turns where no lesson was active.
    """
    from pmb.hooks.autowrite import classify_actions

    cutoff = time.time() - window_days * 86400
    ws = engine.workspace.id

    rows: list = []
    surf_rows: list = []
    try:
        with sqlite3.connect(engine.workspace.db_path) as c:
            try:
                rows = c.execute(
                    "SELECT timestamp, tool, target, status, significant, "
                    "session_id, surface_ids FROM agent_actions "
                    "WHERE workspace_id=? AND timestamp>=? "
                    "ORDER BY COALESCE(session_id,''), timestamp",
                    (ws, cutoff),
                ).fetchall()
            except Exception:
                rows = []  # table may not exist yet on a fresh workspace
            try:
                surf_rows = c.execute(
                    "SELECT id, lesson_ulid FROM lesson_surfaces WHERE workspace_id=?",
                    (ws,),
                ).fetchall()
            except Exception:
                surf_rows = []
    except Exception:
        pass

    surf2lesson = {str(r[0]): r[1] for r in surf_rows if r[1]}

    # Group into turns: contiguous same-session actions with small gaps.
    turns: list[dict] = []
    cur: dict | None = None
    last_sess = object()
    last_ts: float | None = None
    for ts, tool, target, status, sig, sess, sids in rows:
        new_turn = (
            cur is None or sess != last_sess
            or (last_ts is not None and (ts - last_ts) > gap_seconds)
        )
        if new_turn:
            if cur:
                turns.append(cur)
            cur = {"actions": [], "surfaces": set()}
        cur["actions"].append({
            "significant": bool(sig), "tool": tool or "",
            "target": target or "", "status": status or "",
        })
        for s in (sids or "").split(","):
            s = s.strip()
            if s:
                cur["surfaces"].add(s)
        last_sess = sess
        last_ts = ts
    if cur:
        turns.append(cur)

    base: list[tuple[bool, int]] = []
    per_lesson: dict[str, list[tuple[bool, int]]] = {}
    n_outcome_turns = 0
    for t in turns:
        cl = classify_actions(t["actions"])
        has, success = _turn_outcome(cl)
        if not has:
            continue
        n_outcome_turns += 1
        churn = int(cl["n_significant"])
        lessons = {surf2lesson.get(s) for s in t["surfaces"]} - {None}
        if not lessons:
            base.append((success, churn))
        for lu in lessons:
            per_lesson.setdefault(lu, []).append((success, churn))

    base_rate = (sum(1 for s, _ in base if s) / len(base)) if base else None
    base_churn = (sum(ch for _, ch in base) / len(base)) if base else None

    # Lesson content for the report.
    content: dict[str, str] = {}
    if per_lesson:
        try:
            with sqlite3.connect(engine.workspace.db_path) as c:
                q = ",".join("?" for _ in per_lesson)
                for ulid, txt in c.execute(
                    f"SELECT ulid, content FROM events WHERE workspace_id=? "
                    f"AND ulid IN ({q})",
                    (ws, *per_lesson.keys()),
                ).fetchall():
                    content[ulid] = txt or ""
        except Exception:
            pass

    lessons_out = []
    for lu, vals in per_lesson.items():
        n = len(vals)
        sr = sum(1 for s, _ in vals if s) / n
        ch = sum(c for _, c in vals) / n
        lessons_out.append({
            "lesson_ulid": lu,
            "content": (content.get(lu, "") or "")[:160],
            "n": n,
            "success_rate": round(sr, 3),
            "lift": round(sr - base_rate, 3) if base_rate is not None else None,
            "avg_churn": round(ch, 2),
            "churn_delta": round(ch - base_churn, 2) if base_churn is not None else None,
        })
    # Worst lift first (harmful/dead weight surfaces to the top for review).
    lessons_out.sort(key=lambda x: (x["lift"] if x["lift"] is not None else 0, -x["n"]))

    return {
        "window_days": window_days,
        "n_outcome_turns": n_outcome_turns,
        "n_baseline_turns": len(base),
        "baseline_success_rate": round(base_rate, 3) if base_rate is not None else None,
        "baseline_avg_churn": round(base_churn, 2) if base_churn is not None else None,
        "lessons": lessons_out,
    }
