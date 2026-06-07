"""Post-compact session restore.

When an agent's context window compacts (Claude Code fires a SessionStart
hook with source=compact) the agent loses the in-flight detail of what it
was doing. PMB is the durable session memory: this module rebuilds a
compact "here's where you were" block from what was recorded THIS session
(or the last N minutes) + the overview of whatever project the session was
about.

Wired via `pmb hooks install` → SessionStart hook → `pmb session-restore`.
Pure read, no embeddings, no LLM — runs in a few ms.
"""

from __future__ import annotations

from typing import Any, Optional


def _trim(s: Any, n: int = 200) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    return s.strip().replace("\n", " ")[:n]


def build_session_restore(
    engine,
    *,
    minutes: Optional[float] = None,
    include_project: bool = True,
    max_chars: int = 4000,
) -> str:
    """Return a plain-text 'where you left off' block, or '' if nothing.

    Sections:
      • What this session decided / did / learned (session_brief)
      • Open goals still in flight
      • Project overview, if the recent work clearly centred on one project
        (lessons = RULES, key facts, recent decisions)
    """
    try:
        brief = engine.session_brief(minutes=minutes)
    except Exception:
        return ""
    if not brief or brief.get("empty"):
        # Nothing recorded this session — but there might still be a project
        # the user is actively working on. Fall through to project detection
        # via recent activity only if include_project; otherwise bail.
        brief = {}

    buf: list[str] = []
    buf.append("== PMB session restore (context compacted — here's where you were) ==")
    scope = brief.get("scope")
    if scope:
        buf.append(f"(scope: {scope}, {brief.get('n_events', 0)} events)")

    decisions = brief.get("decisions") or []
    done = brief.get("done") or []
    lessons = brief.get("lessons") or []
    failures = brief.get("failures") or []
    goals = brief.get("goals") or []
    other = brief.get("other") or []

    if done:
        buf.append("\nDone this session:")
        for d in done[:8]:
            buf.append(f"  ✓ {_trim(d.get('content',''), 160)}")
    if decisions:
        buf.append("\nDecisions made (don't re-litigate):")
        for d in decisions[:6]:
            buf.append(f"  > {_trim(d.get('content',''), 160)}")
    if lessons:
        buf.append("\nLessons learned this session (RULES — keep following):")
        for L in lessons[:6]:
            buf.append(f"  ! {_trim(L.get('content',''), 180)}")
    if failures:
        buf.append("\nFailures (don't repeat):")
        for f in failures[:5]:
            buf.append(f"  ✗ {_trim(f.get('content',''), 160)}")
    if goals:
        buf.append("\nGoals touched:")
        for g in goals[:5]:
            buf.append(f"  * {_trim(g.get('content',''), 120)}")

    # Open goals (workspace-wide, not just this session) — what's still in flight.
    try:
        open_goals = engine.list_goals(status="in_progress", limit=5)
    except Exception:
        open_goals = []
    if open_goals:
        buf.append("\nStill open (in_progress goals):")
        for g in open_goals[:5]:
            buf.append(
                "  * " + _trim(g.get("content") or g.get("title", ""), 120)
            )

    # Project overview, if the recent work clearly centred on one project.
    if include_project:
        # Build a haystack from recent content and detect a project in it.
        haystack_parts: list[str] = []
        for grp in (done, decisions, lessons, other):
            for it in grp[:10]:
                haystack_parts.append(it.get("content", "") or "")
        haystack = " ".join(haystack_parts)
        project_name = None
        try:
            if haystack.strip():
                det = engine.detect_project_in_text(haystack, min_mentions=1)
                if det and det.get("name"):
                    project_name = det["name"]
        except Exception:
            project_name = None
        # Fallback: workspace name itself.
        if not project_name:
            try:
                wn = (engine.workspace.name or "").strip()
                if wn and len(wn) >= 2 and not wn.isdigit():
                    project_name = wn
            except Exception:
                project_name = None
        if project_name:
            try:
                ov = engine.project_overview(project_name)
                if ov and not ov.get("empty"):
                    ent = ov.get("entity") or {}
                    buf.append(
                        f"\nProject context: {ent.get('name','?')} "
                        f"({ent.get('n_mentions','?')} mentions)"
                    )
                    kf = ov.get("key_facts") or []
                    if kf:
                        buf.append("Key facts:")
                        for f in kf[:4]:
                            buf.append(f"  - {_trim(f.get('content',''), 150)}")
                    pls = ov.get("lessons") or []
                    if pls:
                        buf.append("Project lessons (RULES):")
                        for L in pls[:4]:
                            buf.append(f"  ! {_trim(L.get('content',''), 170)}")
            except Exception:
                pass

    # If we produced nothing but the header(s), treat as empty.
    meaningful = len(buf) > 2
    if not meaningful:
        return ""

    buf.append("")
    buf.append(
        "Pick the thread back up from the above — don't re-ask the user what "
        "you already did. Call session_brief / project_overview for more."
    )

    text = "\n".join(buf)
    if len(text) > max_chars:
        text = text[: max_chars - 30].rstrip() + "\n... [restore truncated]"
    return text
