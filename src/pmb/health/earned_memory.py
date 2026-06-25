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

import math
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


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial success-rate k/n. Stdlib-only
    (no scipy). Returns (low, high) clamped to [0, 1]; (0.0, 1.0) for n == 0.

    Wilson, not the normal approximation: it stays inside [0, 1] and is sane at
    the tiny n the outcome signal actually produces, where the textbook
    p ± z·sqrt(p(1-p)/n) interval breaks (it can leave [0,1] and collapses to
    width 0 at p=0 or p=1).
    """
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def lesson_impact(
    engine: Engine,
    window_days: int = 30,
    gap_seconds: float = 300.0,
    min_n: int = 5,
) -> dict:
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
                    "SELECT id, lesson_ulid, followed FROM lesson_surfaces "
                    "WHERE workspace_id=?",
                    (ws,),
                ).fetchall()
            except Exception:
                # Older DBs may lack the `followed` column; fall back so the
                # baseline join still works (just without the causal arm).
                try:
                    surf_rows = c.execute(
                        "SELECT id, lesson_ulid FROM lesson_surfaces "
                        "WHERE workspace_id=?",
                        (ws,),
                    ).fetchall()
                except Exception:
                    surf_rows = []
    except Exception:
        pass

    surf2lesson = {str(r[0]): r[1] for r in surf_rows if len(r) >= 2 and r[1]}
    # Known follow-through per surface (1/0). NULL = not yet marked -> excluded
    # from the causal arm entirely (it is neither "followed" nor "ignored").
    surf2followed = {
        str(r[0]): bool(r[2])
        for r in surf_rows if len(r) >= 3 and r[2] is not None
    }

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
    # Within-lesson causal arm: lesson -> {"followed": [success...],
    # "ignored": [success...]}. Only surfaces with a KNOWN follow flag count.
    arms: dict[str, dict[str, list[bool]]] = {}
    n_outcome_turns = 0
    for t in turns:
        cl = classify_actions(t["actions"])
        has, success = _turn_outcome(cl)
        if not has:
            continue
        n_outcome_turns += 1
        churn = int(cl["n_significant"])
        # Per lesson active in this turn, resolve its follow-arm: "followed" if
        # ANY of its surfaces here was followed, else "ignored" if at least one
        # surface was explicitly not-followed, else unknown (skip the arm).
        lesson_arm: dict[str, str] = {}
        for s in t["surfaces"]:
            lu = surf2lesson.get(s)
            if lu is None:
                continue
            lesson_arm.setdefault(lu, "")  # lesson is active even if arm unknown
            if s in surf2followed:
                if surf2followed[s]:
                    lesson_arm[lu] = "followed"
                elif lesson_arm[lu] != "followed":
                    lesson_arm[lu] = "ignored"
        lessons = set(lesson_arm)
        if not lessons:
            base.append((success, churn))
        for lu in lessons:
            per_lesson.setdefault(lu, []).append((success, churn))
            arm = lesson_arm[lu]
            if arm:  # known follow-through only
                arms.setdefault(lu, {"followed": [], "ignored": []})[arm].append(success)

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
        k = sum(1 for s, _ in vals if s)            # successes
        sr = k / n
        ch = sum(c for _, c in vals) / n
        ci_low, ci_high = _wilson(k, n)
        # Verdict is deliberately CONSERVATIVE. A lesson is only called useful /
        # harmful when its success-rate CI clears the baseline AND it has enough
        # outcome turns to mean anything. Everything else is "unverified" (seen,
        # not yet provable) or "insufficient" (too few turns to judge). This is
        # what stops an n=2 fluke from being reported as a real effect - and why
        # the per-lesson `lift` above must not drive ranking/decay on its own.
        if base_rate is None or n < min_n:
            verdict = "insufficient"
        elif ci_low > base_rate:
            verdict = "useful"
        elif ci_high < base_rate:
            verdict = "harmful"
        else:
            verdict = "unverified"
        # Within-lesson causal read: same lesson, FOLLOWED turns vs IGNORED
        # turns. Holds the surfacing trigger fixed, so it is far less confounded
        # than `lift` (which compares against unrelated no-lesson turns). Fires
        # only when both arms clear min_n and their success CIs separate.
        fa = arms.get(lu, {}).get("followed", [])
        ia = arms.get(lu, {}).get("ignored", [])
        nf, ni = len(fa), len(ia)
        followed_lift = None
        causal_verdict = "insufficient"
        if nf >= min_n and ni >= min_n:
            f_lo, f_hi = _wilson(sum(fa), nf)
            i_lo, i_hi = _wilson(sum(ia), ni)
            followed_lift = round(sum(fa) / nf - sum(ia) / ni, 3)
            if f_lo > i_hi:
                causal_verdict = "helps"
            elif f_hi < i_lo:
                causal_verdict = "hurts"
            else:
                causal_verdict = "inconclusive"
        lessons_out.append({
            "lesson_ulid": lu,
            "content": (content.get(lu, "") or "")[:160],
            "n": n,
            "success_rate": round(sr, 3),
            "ci_low": round(ci_low, 3),
            "ci_high": round(ci_high, 3),
            "verdict": verdict,
            "lift": round(sr - base_rate, 3) if base_rate is not None else None,
            "avg_churn": round(ch, 2),
            "churn_delta": round(ch - base_churn, 2) if base_churn is not None else None,
            "n_followed": nf,
            "n_ignored": ni,
            "followed_success_rate": round(sum(fa) / nf, 3) if nf else None,
            "ignored_success_rate": round(sum(ia) / ni, 3) if ni else None,
            "followed_lift": followed_lift,
            "causal_verdict": causal_verdict,
        })
    # Worst lift first (harmful/dead weight surfaces to the top for review).
    lessons_out.sort(key=lambda x: (x["lift"] if x["lift"] is not None else 0, -x["n"]))

    # Top-level signal sufficiency. Outcome turns are RARE (each needs a test /
    # build / deploy / red->green), so the loop is only trustworthy once enough
    # of them have accrued AND a real no-lesson baseline exists. Until then the
    # per-lesson numbers are shown but must NOT drive ranking/decay - exactly the
    # "measurement only" stance the feature shipped with.
    n_confident = sum(1 for L in lessons_out if L["verdict"] in ("useful", "harmful"))
    n_causal = sum(1 for L in lessons_out if L["causal_verdict"] in ("helps", "hurts"))
    if n_outcome_turns < 20 or len(base) < 5:
        sufficiency = "insufficient"
    elif n_outcome_turns < 50:
        sufficiency = "thin"
    else:
        sufficiency = "ok"
    trustworthy = sufficiency == "ok" and n_confident > 0

    return {
        "window_days": window_days,
        "n_outcome_turns": n_outcome_turns,
        "n_baseline_turns": len(base),
        "baseline_success_rate": round(base_rate, 3) if base_rate is not None else None,
        "baseline_avg_churn": round(base_churn, 2) if base_churn is not None else None,
        "min_n": min_n,
        "n_confident": n_confident,
        "n_causal": n_causal,
        "signal_sufficiency": sufficiency,   # insufficient | thin | ok
        "trustworthy": trustworthy,
        "caveat": (
            "`lift`/`verdict` compare a lesson's turns to the no-lesson baseline "
            "and are ASSOCIATIONAL: lessons surface on harder, more specific "
            "turns, so a real positive effect can read as negative lift "
            "(confounding by indication). Prefer `causal_verdict` - a "
            "within-lesson followed-vs-ignored comparison that holds the "
            "surfacing trigger fixed. Its residual confound: the agent's CHOICE "
            "to follow may itself track task type."
        ),
        "lessons": lessons_out,
    }
