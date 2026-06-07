"""Follow-through inference — close the lesson loop WITHOUT model cooperation.

The adherence dashboard shows follow-rate near 0% because models almost
never call `mark_lesson_followed`. That's the same class of problem
auto-recall solved for reads: don't depend on the model to self-report.

This module infers follow-through deterministically. When a turn ends
(Claude Code fires a Stop hook), we look at the lessons that surfaced in
the last few minutes and the activity the agent actually recorded in the
same window. If a surfaced lesson's distinctive tokens show up in what the
agent did, that's honest (if weak) evidence the lesson influenced the work
— we mark it followed with an explicit auto-detected note. No fabrication:
absence of activity → no mark → the surface stays unconfirmed.

This only produces a signal when the agent records its activity (active
mode). If nothing was recorded, nothing is inferred — which is the honest
outcome, not a fake number.

Wired via `pmb hooks install` → Stop hook → `pmb lesson-followcheck`.
Pure SQL + token matching, no embeddings, runs in a few ms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


# Tokens too common to be evidence of anything — matching on these would
# mark every lesson followed. Multilingual stop-ish set + PMB-generic words +
# the high-frequency English fillers that caused false positives in testing
# (already / both / before / first / different / default ...).
_STOP = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "not", "but", "use",
    "using", "used", "via", "per", "you", "your", "our", "its", "are", "was",
    "were", "has", "have", "had", "will", "would", "should", "must", "never",
    "always", "into", "onto", "than", "then", "else", "when", "what", "which",
    "already", "both", "before", "after", "first", "last", "next", "different",
    "default", "defaults", "same", "other", "another", "every", "each", "some",
    "any", "all", "more", "most", "less", "only", "just", "also", "still",
    "here", "there", "where", "while", "until", "they", "them", "their",
    "make", "made", "need", "want", "like", "such", "very", "much", "many",
    "пмб", "это", "как", "что", "для", "при", "над", "под", "без", "все",
    "так", "там", "тут", "нет", "его", "она", "они", "под", "это", "этот",
    "может", "была", "были", "быть", "если", "чтобы", "когда", "потом",
    "pmb", "lesson", "rule", "agent", "code", "file", "test", "run", "from",
    "thing", "things", "stuff", "case", "cases", "time", "times", "work",
})

# A "distinctive" token: alnum (incl. unicode word chars), length ≥ 4, not a
# stopword, not pure digits. We also keep dotted/identifier-ish tokens
# (e.g. "record_batch", "pnpm", "lancedb") which are strong evidence.
_TOKEN = re.compile(r"[A-Za-z0-9_À-ɏЀ-ӿ][\w\-.]{2,}", re.UNICODE)


def _is_strong(tok: str) -> bool:
    """A 'strong' token is specific enough that a coincidental match is
    unlikely: a long word (≥7 chars) or an identifier (has _ . - or a digit,
    e.g. record_batch / qwen2.5 / 8765 / paraphrase-multilingual). Matching
    on strong tokens is real evidence; matching only on 4-6 char common
    words is not."""
    if len(tok) >= 7:
        return True
    if any(c in tok for c in ("_", ".", "-")):
        return True
    if any(c.isdigit() for c in tok):
        return True
    return False


def _distinctive_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for m in _TOKEN.finditer((text or "").lower()):
        tok = m.group(0).strip("-._")
        if len(tok) < 4:
            continue
        if tok.isdigit():
            continue
        if tok in _STOP:
            continue
        out.add(tok)
    return out


@dataclass
class FollowVerdict:
    surface_id: int
    lesson_ulid: str
    followed: bool
    overlap: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class FollowCheckResult:
    checked: int = 0
    marked_followed: int = 0
    verdicts: list[FollowVerdict] = field(default_factory=list)
    skipped_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "checked": self.checked,
            "marked_followed": self.marked_followed,
            "skipped_reason": self.skipped_reason,
            "verdicts": [
                {
                    "surface_id": v.surface_id,
                    "lesson_ulid": v.lesson_ulid,
                    "followed": v.followed,
                    "overlap": v.overlap,
                    "note": v.note,
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
    apply: bool = True,
) -> FollowCheckResult:
    """Infer follow-through for recently-surfaced, still-unconfirmed lessons.

    Algorithm:
      1. Pull lesson surfaces in the last `window_minutes` with no verdict.
      2. Pull the agent's recorded activity in the last `activity_minutes`.
      3. Build a token bag from all that activity.
      4. For each surfaced lesson, if its distinctive tokens overlap the
         activity bag by ≥ `min_overlap` total AND ≥ `min_strong` of those
         are 'strong' tokens (long words / identifiers, not 4-6 char common
         words) → mark followed (auto), with a note naming the overlap.

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

    # 2. What the agent actually DID — activity events only.
    #    Crucial: match against ACTIONS, not against recorded facts/lessons.
    #    If we folded in what_just_happened() (which returns every event type)
    #    the surfaced lesson itself would be in the bag and "prove itself" —
    #    its own tokens would always overlap. recent_activity is scoped to
    #    event_type='activity' (edits / completed work / tool calls /
    #    decisions logged via record_activity), which is exactly the agent's
    #    behaviour this turn.
    activity_text_parts: list[str] = []
    try:
        acts = engine.recent_activity(minutes=activity_minutes, limit=100)
        for a in acts or []:
            activity_text_parts.append(a.get("content", "") or "")
    except Exception:
        pass

    activity_bag = _distinctive_tokens(" ".join(activity_text_parts))
    if not activity_bag:
        res.skipped_reason = "no recorded activity to match against"
        res.checked = len(surfaces)
        return res

    # 3. Match each surface.
    for s in surfaces:
        res.checked += 1
        lesson_tokens = _distinctive_tokens(s.get("content", ""))
        if not lesson_tokens:
            continue
        overlap = sorted(lesson_tokens & activity_bag)
        strong = [t for t in overlap if _is_strong(t)]
        if len(overlap) >= min_overlap and len(strong) >= min_strong:
            note = (
                "auto-detected (activity overlap): "
                + ", ".join(strong[:5])
                + " appeared in recorded activity this turn"
            )
            v = FollowVerdict(
                surface_id=s["surface_id"],
                lesson_ulid=s.get("lesson_ulid", ""),
                followed=True,
                overlap=overlap[:8],
                note=note,
            )
            res.verdicts.append(v)
            if apply:
                try:
                    engine.mark_lesson_followed(
                        surface_id=s["surface_id"], followed=True, note=note,
                    )
                    res.marked_followed += 1
                except Exception:
                    pass
            else:
                res.marked_followed += 1  # would-mark count in dry-run

    return res
