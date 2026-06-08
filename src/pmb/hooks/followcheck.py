"""Follow-through inference — close the lesson loop WITHOUT model cooperation.

The adherence dashboard shows follow-rate near 0% because models almost
never call `mark_lesson_followed`. That's the same class of problem
auto-recall solved for reads: don't depend on the model to self-report.

This module infers follow-through deterministically. When a turn ends
(Claude Code fires a Stop hook), we look at the lessons that surfaced in
the last few minutes, the activity the agent recorded, and the raw actions
the ambient observer saw (edits, commands, results). Actions are linked to
the lesson surfaces that were active when they happened, and timestamp
filtering prevents work done before a lesson surfaced from "proving" it was
followed. If a surfaced lesson's distinctive tokens show up in what the
agent did, that's honest (if weak) evidence the lesson influenced the work
— we mark it followed with an explicit auto-detected note. No fabrication:
absence of evidence → no mark → the surface stays unconfirmed.

Wired via `pmb hooks install` → Stop hook → `pmb lesson-followcheck`.
Pure SQL + token matching, no embeddings, runs in a few ms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


# Tokenizer, stopword set, and the "strong token" test live in
# pmb.core.text_match so the SURFACE side (find_lessons — should this lesson
# show at all?) and this CONFIRM side (did the work follow it?) judge by ONE
# yardstick — otherwise a lesson surfaces on a loose match it can never be
# confirmed against, and just drags the adherence denominator down. Re-exported
# under the historical underscore names this module and its tests reference.
from pmb.core.text_match import (  # noqa: E402
    STOPWORDS as _STOP,
    distinctive_tokens as _distinctive_tokens,
    is_strong as _is_strong,
)


def _surface_ids(action: dict) -> set[int]:
    """Surface ids attached by the ambient action logger."""
    raw = action.get("surface_ids") or ""
    if isinstance(raw, (list, tuple, set)):
        vals = raw
    else:
        vals = str(raw).split(",")
    out: set[int] = set()
    for value in vals:
        try:
            out.add(int(value))
        except (TypeError, ValueError):
            continue
    return out


def _action_text(action: dict) -> str:
    """Turn a raw observed action into token-matchable evidence."""
    parts = [
        str(action.get("tool") or ""),
        str(action.get("target") or ""),
        str(action.get("command") or ""),
        str(action.get("status") or ""),
    ]
    return " ".join(p for p in parts if p)


def _is_after_surface(item: dict, surfaced_at: Any) -> bool:
    """True when evidence happened after the lesson surfaced.

    Missing timestamps are accepted for backwards compatibility with older
    activity rows and lightweight fake engines; newly observed actions always
    carry timestamps.
    """
    if not isinstance(surfaced_at, (int, float)):
        return True
    timestamp = item.get("timestamp")
    if not isinstance(timestamp, (int, float)):
        return True
    return timestamp >= surfaced_at - 1.0


def _action_matches_surface(action: dict, surface_id: int, surfaced_at: Any) -> bool:
    """Prefer explicit surface linkage; fall back to timestamps for old rows."""
    linked = _surface_ids(action)
    if linked:
        return surface_id in linked
    return _is_after_surface(action, surfaced_at)


@dataclass
class FollowVerdict:
    surface_id: int
    lesson_ulid: str
    followed: bool
    overlap: list[str] = field(default_factory=list)
    note: str = ""
    not_applicable: bool = False


@dataclass
class FollowCheckResult:
    checked: int = 0
    marked_followed: int = 0
    not_applicable: int = 0
    verdicts: list[FollowVerdict] = field(default_factory=list)
    skipped_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "checked": self.checked,
            "marked_followed": self.marked_followed,
            "not_applicable": self.not_applicable,
            "skipped_reason": self.skipped_reason,
            "verdicts": [
                {
                    "surface_id": v.surface_id,
                    "lesson_ulid": v.lesson_ulid,
                    "followed": v.followed,
                    "overlap": v.overlap,
                    "note": v.note,
                    "not_applicable": v.not_applicable,
                }
                for v in self.verdicts
            ],
        }


def run_followcheck(
    engine,
    *,
    window_minutes: float = 30.0,
    activity_minutes: float = 30.0,
    min_overlap: int = 3,
    min_strong: int = 2,
    max_surfaces: int = 50,
    mark_not_applicable: bool = True,
    apply: bool = True,
) -> FollowCheckResult:
    """Infer follow-through for recently-surfaced, still-unconfirmed lessons.

    Algorithm:
      1. Pull lesson surfaces in the last `window_minutes` with no verdict.
      2. Pull recorded activity and ambient-observed actions.
      3. For each surface, keep only evidence created after it surfaced or
         explicitly linked to its `surface_id`.
      4. If the lesson's distinctive tokens overlap that evidence by
         ≥ `min_overlap` total AND ≥ `min_strong` of those are 'strong'
         tokens (long words / identifiers, not 4-6 char common words)
         → mark followed (auto), with a note naming the overlap.
      5. Else, if there WAS evidence but ZERO overlap (the turn wasn't about
         this lesson at all), mark it not_applicable (followed = -1) so it is
         excluded from the adherence denominator instead of counting as a
         phantom 'not followed'. Partial-but-weak overlap stays '?'.

    Conservative by design. The strong-token gate is what stops topical
    coincidence ("dashboard", "before", "first") from registering as a
    follow. Lessons with no qualifying overlap are left untouched (stay
    '?'), never marked ignored — absence of evidence is not evidence of
    ignoring.
    """
    res = FollowCheckResult()

    # 1. Unconfirmed surfaces.
    try:
        surfaces = engine.recent_unconfirmed_surfaces(
            minutes=window_minutes, limit=max_surfaces,
        )
    except Exception:
        res.skipped_reason = "recent_unconfirmed_surfaces unavailable"
        return res
    if not surfaces:
        res.skipped_reason = "no unconfirmed surfaces in window"
        return res

    # 2. What the agent actually DID — activity summaries + observed actions.
    #    Crucial: match against behaviour, not recorded facts/lessons.
    #    If we folded in what_just_happened() (which returns every event type)
    #    the surfaced lesson itself would be in the bag and "prove itself" —
    #    its own tokens would always overlap. recent_activity is scoped to
    #    event_type='activity' (edits / completed work / tool calls /
    #    decisions logged via record_activity), which is exactly the agent's
    #    behaviour this turn.
    activity_events: list[dict] = []
    try:
        acts = engine.recent_activity(minutes=activity_minutes, limit=100)
        for a in acts or []:
            activity_events.append(a)
    except Exception:
        pass

    action_events: list[dict] = []
    try:
        action_events = engine.recent_agent_actions(
            minutes=activity_minutes, limit=200, significant_only=True,
        ) or []
    except Exception:
        pass

    if not activity_events and not action_events:
        res.skipped_reason = "no recorded activity or observed actions to match against"
        res.checked = len(surfaces)
        return res

    # 3. Match each surface.
    for s in surfaces:
        res.checked += 1
        surface_id = s["surface_id"]
        surfaced_at = s.get("surfaced_at")
        matched_activity = [
            a for a in activity_events if _is_after_surface(a, surfaced_at)
        ]
        matched_actions = [
            a for a in action_events
            if _action_matches_surface(a, surface_id, surfaced_at)
        ]
        evidence_parts = [
            a.get("content", "") or "" for a in matched_activity
        ] + [
            _action_text(a) for a in matched_actions
        ]
        evidence_bag = _distinctive_tokens(" ".join(evidence_parts))
        if not evidence_bag:
            continue
        lesson_tokens = _distinctive_tokens(s.get("content", ""))
        if not lesson_tokens:
            continue
        overlap = sorted(lesson_tokens & evidence_bag)
        strong = [t for t in overlap if _is_strong(t)]
        # Credible follow if EITHER:
        #   • ≥ min_strong strong tokens overlap (the easy path), OR
        #   • ≥ 1 strong token AND broad corroboration (≥ min_overlap+2
        #     distinctive overlaps).
        # The second clause catches the common real case of ONE killer
        # identifier (e.g. `is_warm`) backed by many regular matches
        # (recall, hook, cold, load …) that the rigid 2-strong gate wrongly
        # rejected — while a single strong token alone (no corroboration)
        # still doesn't qualify, so topical coincidence stays out.
        enough_strong = len(strong) >= min_strong
        broad = len(strong) >= 1 and len(overlap) >= (min_overlap + 2)
        if len(overlap) >= min_overlap and (enough_strong or broad):
            evidence = strong + [t for t in overlap if t not in strong]
            sources = []
            if matched_actions:
                sources.append(f"{len(matched_actions)} observed action(s)")
            if matched_activity:
                sources.append(f"{len(matched_activity)} activity record(s)")
            note = (
                "auto-detected (post-surface evidence: "
                + ", ".join(sources)
                + "; overlap): "
                + ", ".join(evidence[:6])
            )
            v = FollowVerdict(
                surface_id=surface_id,
                lesson_ulid=s.get("lesson_ulid", ""),
                followed=True,
                overlap=overlap[:8],
                note=note,
            )
            res.verdicts.append(v)
            if apply:
                try:
                    engine.mark_lesson_followed(
                        surface_id=surface_id, followed=True, note=note,
                    )
                    res.marked_followed += 1
                except Exception:
                    pass
            else:
                res.marked_followed += 1  # would-mark count in dry-run
        elif not overlap and mark_not_applicable:
            # Zero topical overlap between the lesson and everything the agent
            # did this turn (evidence DID exist — we `continue`d above if it
            # didn't). The work simply wasn't about this lesson, so record it
            # as not_applicable instead of letting it silently count as 'not
            # followed' and drag down the adherence denominator. A surface with
            # SOME overlap but below the follow bar stays untouched ('?').
            na_note = (
                "auto-detected not-applicable: zero token overlap with this "
                "turn's work — lesson did not pertain to what was done"
            )
            res.not_applicable += 1
            res.verdicts.append(FollowVerdict(
                surface_id=surface_id,
                lesson_ulid=s.get("lesson_ulid", ""),
                followed=False,
                overlap=[],
                note=na_note,
                not_applicable=True,
            ))
            if apply:
                try:
                    engine.mark_lesson_not_applicable(
                        surface_id=surface_id, note=na_note,
                    )
                except Exception:
                    pass

    return res
