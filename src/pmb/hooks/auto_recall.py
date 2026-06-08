"""Auto-recall — zero-cooperation memory injection.

PMB's biggest UX problem: the agent has to remember to call `recall` /
`prepare` / `project_overview`. It often doesn't. The Adherence dashboard
shows `prepare_rate` near 0% on most workspaces — instructions in
CLAUDE.md are good intentions, not enforcement.

This module fixes the dependency. It runs from the UserPromptSubmit hook
(`pmb hooks install`) and decides — without asking the model — which PMB
calls to pre-execute and inject as context. Pure regex intent
classification + parallel-safe dispatch over the engine. No LLM, no
network, no API keys. Sub-100ms p95 on a warm workspace.

Intents (multilingual, RU/UK/EN):

  PROJECT_PREP        a known project name + a work-verb (fix/add/refactor/
                      исправь/допиши). → full project_overview + arcs + goals.
  PROJECT_OVERVIEW    a known project name standalone. → project_overview.
  PAST_QUERY          "когда / what did I / why did we / какой у меня".
                      → recall(message).
  RECENT_QUERY        "что мы только что / what did I just / щойно".
                      → what_just_happened(5).
  GOALS_QUERY         "какие у меня цели / open goals".
                      → list_goals(in_progress).
  LESSONS_QUERY       "какие правила / lessons / conventions".
                      → find_lessons(message).
  GENERIC_FACTUAL     question mark + non-trivial content, no other match.
                      → recall(message, top_k=3), surface only if score > 0.3.
  SKIP                trivial (greeting / ack / very short / pure-code).

Lessons are *always* surfaced (cheap, high value) in addition to whatever
intent fires. The CLI formats the result into a plain-text block that the
agent's host appends to the prompt.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional


# ─── Intents ─────────────────────────────────────────────────────────────


class Intent:
    PROJECT_PREP = "PROJECT_PREP"
    PROJECT_OVERVIEW = "PROJECT_OVERVIEW"
    PAST_QUERY = "PAST_QUERY"
    RECENT_QUERY = "RECENT_QUERY"
    GOALS_QUERY = "GOALS_QUERY"
    LESSONS_QUERY = "LESSONS_QUERY"
    GENERIC_FACTUAL = "GENERIC_FACTUAL"
    SKIP = "SKIP"


# ─── Patterns ────────────────────────────────────────────────────────────
#
# Three-language coverage (en/ru/uk). Anchored to whole-word boundaries
# where reasonable. Tested in tests/test_auto_recall.py.

# "what did I / when did I / why did we / where did / who did" + Cyrillic
# equivalents. Includes mild typos ("когда-то я делал X").
_PAST_QUERY = re.compile(
    r"(?:\bкогда\s+(?:я|мы|это)|\bчто\s+(?:я|мы)\s+(?:делал|сделал|"
    r"говорил|записал|выбрал|решил|обсужд)|\bпочему\s+(?:мы|я)\s+"
    r"(?:выбра|реши|отказа)|\bкакой\s+у\s+меня|\bгде\s+(?:я|мы)\s+"
    r"(?:храни|записа|положи)|\bкто\s+так(?:ой|ая)|\bчто\s+я\s+"
    r"планировал|"
    r"\bколи\s+я|\bщо\s+я\s+(?:робив|зробив)|\bхто\s+так(?:ий|а)|"
    r"\bwhat\s+did\s+(?:i|we)\s|\bwhen\s+did\s+(?:i|we)|\bwhy\s+did\s+we|"
    r"\bwhere\s+did\s+(?:i|we)|\bwho\s+(?:is|was|said|did)|\bwhat'?s\s+"
    r"(?:my|the\s+)|\bhow\s+come|\bdo\s+we\s+have|\bhave\s+i\s+(?:ever|"
    r"already))",
    re.IGNORECASE,
)

# "что мы только что обсуждали / что мы сейчас делаем" — last few turns.
_RECENT_QUERY = re.compile(
    r"(?:\bчто\s+(?:мы|я)\s+(?:только\s+что|сейчас|щас|недавно)|"
    r"\bщо\s+ми\s+(?:щойно|тільки\s+що)|"
    r"\bwhat\s+(?:did|are)\s+(?:i|we)\s+(?:just|currently|right\s+now)|"
    r"\bwhat'?s\s+(?:going\s+on|happening)|"
    r"\bкакая\s+у\s+нас\s+(?:сейчас|щас)|"
    r"\bна\s+ч[её]м\s+я\s+остановил)",
    re.IGNORECASE,
)

_GOALS_QUERY = re.compile(
    r"(?:\bкакие\s+у\s+меня\s+(?:цели|задачи|планы)|"
    r"\bмои\s+(?:цели|задачи|планы)|"
    r"\bчто\s+я\s+планировал|"
    r"\bоткрытые\s+(?:цели|задачи)|"
    r"\bяк[іи]\s+у\s+мене\s+цілі|"
    r"\bmy\s+(?:open\s+)?goals?|\bopen\s+goals?|\bin\s+flight|"
    r"\bwhat\s+am\s+i\s+working\s+on|\bcurrent\s+goals?|"
    # 'what's left to do' — a natural goals question the old patterns missed
    r"\bдодела\w+|\bнедодела\w*|\bчто\s+остал\w+|"
    r"\bчто\s+(?:дел\w+\s+)?дальше\b|"
    r"\bщо\s+(?:залиш\w+|дал[іи]|дороб\w+)|"
    r"\bto-?do\b|\bwhat'?s\s+(?:left|next)\b|\bwhat\s+is\s+left\b|"
    r"\bremaining\s+(?:tasks?|work|items?)\b|"
    r"\bwhat\s+(?:do\s+i\s+still\s+need|should\s+i\s+do\s+next))",
    re.IGNORECASE,
)

_LESSONS_QUERY = re.compile(
    r"(?:\bкакие\s+(?:есть\s+)?правила|\bправила\s+проекта|"
    r"\bкакие\s+(?:есть\s+)?уроки|\bчто\s+я\s+учил|"
    r"\bconvention|\blesson|\brule\s+(?:about|for)|"
    r"\bdo\s+we\s+use\b|\bdo\s+we\s+have\s+a\s+rule|"
    r"\bяк[іи]\s+правила)",
    re.IGNORECASE,
)

# Work verbs that, combined with a project name, mean "I'm about to work
# on it". Multilingual; intentionally generous.
_WORK_VERB = re.compile(
    r"(?:\b(?:fix|fixing|add|adding|refactor|refactoring|implement|"
    r"implementing|build|building|write|writing|debug|debugging|"
    r"deploy|deploying|test|testing|continue|review|reviewing|"
    r"port|porting|migrate|migrating|update|updating|rewrite|"
    r"rewriting|wire|wiring|patch|patching|land|ship|push)\b|"
    r"\bworking\s+on\b|\bwork\s+on\b|"
    r"\b(?:исправ|почини|допиш|напиш|перепиш|добав|рефактор|"
    r"задеплой|оттестир|поработ|настро|внес|внеси|пробрось|"
    r"обнови|переведи|вынеси|вытащ|реализу|реши|"
    r"правил|правит|править|правлю)|"
    r"\b(?:виправ|напиши|допиши|додай|рефактор|оновити))",
    re.IGNORECASE,
)

# Trivial input: greetings, acks, single emoji, very short.
_TRIVIAL = re.compile(
    r"^[\s\W_]*(?:hi|hello|hey|yo|ok|okay|kk|got\s+it|sure|thanks|"
    r"thank\s+you|ty|tysm|cheers|nice|cool|np|good\s+morning|"
    r"good\s+night|gn|gm|"
    r"привет|здравствуй|спс|спасибо|ок|окей|круто|понятно|поняла?|"
    r"норм|клас|хорошо|давай|ну\s+давай|ага|"
    r"привіт|добрий\s+день|дякую|зрозуміло)"
    r"[\s\W_]*$",
    re.IGNORECASE,
)

# A '?' (incl. fullwidth) anywhere is a strong question signal.
_HAS_QUESTION = re.compile(r"[?？]")


# ─── API ─────────────────────────────────────────────────────────────────


def is_trivial(msg: str, min_chars: int = 5) -> bool:
    """True for greetings, acks, single emoji, or very-short input."""
    s = (msg or "").strip()
    if len(s) < min_chars:
        return True
    if _TRIVIAL.match(s):
        return True
    return False


def detect_intents(
    msg: str,
    known_projects: Optional[set[str]] = None,
    min_chars: int = 5,
) -> list[str]:
    """Classify a user message into a list of intents.

    `known_projects` is a set of project-entity names this workspace
    actually knows about. Without that hint we can't tell a project name
    apart from a random capitalized noun.

    Returns a list (order = priority). [Intent.SKIP] alone means
    nothing fires and the hook should output nothing.
    """
    s = (msg or "").strip()
    if is_trivial(s, min_chars=min_chars):
        return [Intent.SKIP]

    out: list[str] = []

    # Project detection — purely substring (case-insensitive). Avoid
    # spurious matches by requiring word-boundary. `known_projects` come
    # from engine.detect_project_in_text or the entity graph.
    project_hit = False
    if known_projects:
        s_l = s.lower()
        for name in known_projects:
            if not name:
                continue
            name_l = name.lower()
            # Use a regex word-boundary anchor so "PMBish" doesn't match "PMB"
            # but allow Cyrillic project names too.
            if re.search(rf"(?:^|\W){re.escape(name_l)}(?:$|\W)", s_l):
                project_hit = True
                break

    if project_hit:
        if _WORK_VERB.search(s):
            out.append(Intent.PROJECT_PREP)
        else:
            out.append(Intent.PROJECT_OVERVIEW)

    # Past / recent / goals / lessons — explicit question patterns.
    if _RECENT_QUERY.search(s):
        out.append(Intent.RECENT_QUERY)
    if _PAST_QUERY.search(s):
        out.append(Intent.PAST_QUERY)
    if _GOALS_QUERY.search(s):
        out.append(Intent.GOALS_QUERY)
    if _LESSONS_QUERY.search(s):
        out.append(Intent.LESSONS_QUERY)

    # Last resort: a question mark and no other intent fired. Likely a
    # factual ask. We'll do a low-cost recall and only surface if it
    # actually hits.
    if not out and _HAS_QUESTION.search(s):
        out.append(Intent.GENERIC_FACTUAL)

    return out or [Intent.SKIP]


# ─── Dispatch ────────────────────────────────────────────────────────────


@dataclass
class AutoContextResult:
    """What the dispatcher returns. CLI / hook formats this for the model."""

    message: str
    intents: list[str]
    project: Optional[dict] = None        # project_overview output
    arcs: list[dict] = field(default_factory=list)
    recall_hits: list[dict] = field(default_factory=list)
    recall_query: Optional[str] = None
    recall_confidence: float = 0.0
    recent: list[dict] = field(default_factory=list)
    open_goals: list[dict] = field(default_factory=list)
    lessons: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    latency_ms: int = 0
    skipped: bool = False
    skip_reason: Optional[str] = None

    def is_empty(self) -> bool:
        """True if nothing useful matched — hook should print nothing."""
        return not any([
            self.project,
            self.arcs,
            self.recall_hits,
            self.recent,
            self.open_goals,
            self.lessons,
            self.decisions,
        ])


def _known_projects(engine) -> set[str]:
    """Pull all known project-entity names this workspace has.

    Combines two sources:
      1. The workspace's own name (always treat it as a known project —
         on fresh workspaces the graph hasn't extracted entities yet,
         and we still want "fix bug in <workspace>" to trigger PREP).
      2. graph_top_entities — the canonical entity-set used by
         engine.detect_project_in_text. Only includes the kinds that
         could plausibly be projects (skip 1-char / pure-digit noise).
    """
    out: set[str] = set()

    # 1. Workspace name as a default project.
    try:
        ws_name = (engine.workspace.name or "").strip()
        if ws_name and len(ws_name) >= 2 and not ws_name.isdigit():
            out.add(ws_name)
    except Exception:
        pass

    # 2. Graph entities.
    try:
        entities = engine.graph_top_entities(kind=None, limit=200)
    except Exception:
        return out
    for e in entities or []:
        name = e.get("name") or e.get("normalized_name")
        if name and isinstance(name, str) and len(name) >= 2:
            if not name.isdigit():
                out.add(name)
    return out


def _resolve_project_name(
    engine,
    msg: str,
    known_projects: set[str],
) -> Optional[str]:
    """Pick the project name to dispatch on.

    Strategy:
      1. Ask engine.detect_project_in_text with a relaxed threshold —
         even 1 mention is enough for the auto-recall hook (we'd
         rather over-fire than miss).
      2. If that returns nothing, fall back to a longest-substring
         match against `known_projects` (which already includes the
         workspace name + graph entities).
    """
    try:
        det = engine.detect_project_in_text(msg, min_mentions=1)
        if det and det.get("name"):
            return det["name"]
    except Exception:
        pass
    s_l = (msg or "").lower()
    for name in sorted(known_projects, key=len, reverse=True):
        if not name:
            continue
        if re.search(rf"(?:^|\W){re.escape(name.lower())}(?:$|\W)", s_l):
            return name
    return None


def run_auto_context(
    engine,
    message: str,
    *,
    min_chars: int = 5,
    recall_top_k: int = 5,
    recall_min_score: float = 0.30,
    lessons_limit: int = 5,
    decisions_limit: int = 3,
    surface_decisions: bool = True,
    recent_minutes: float = 1440.0,
    recent_limit: int = 8,
    goals_limit: int = 5,
    log_surfaces: bool = True,
) -> AutoContextResult:
    """Classify the message and dispatch the matching PMB queries.

    Pure orchestration — no I/O outside the engine itself. All branches
    are wrapped in best-effort try/except so a single failure doesn't
    blank the whole context.
    """
    t0 = time.perf_counter()
    msg = (message or "").strip()
    res = AutoContextResult(message=msg, intents=[])

    if not msg:
        res.intents = [Intent.SKIP]
        res.skipped = True
        res.skip_reason = "empty message"
        res.latency_ms = int((time.perf_counter() - t0) * 1000)
        return res

    # Non-message noise: task-notification / system-reminder / raw tool-output
    # blocks get routed through the hook the same as user text, but they are
    # NOT requests. Surfacing memory on them is pure noise — on the real
    # workspace this was ~half of all lesson surfaces. Skip before classifying.
    _head = msg.lstrip()[:120].lower()
    if any(mk in _head for mk in (
        "<task-notification>", "<system-reminder>", "[system notification",
        "<tool-use-id>", "<command-name>", "<local-command",
    )):
        res.intents = [Intent.SKIP]
        res.skipped = True
        res.skip_reason = "non-message (system/tool block)"
        res.latency_ms = int((time.perf_counter() - t0) * 1000)
        return res

    # Trivial skip BEFORE touching the engine — saves a DB roundtrip on
    # greetings/acks/very-short input. `is_trivial` is regex-only.
    if is_trivial(msg, min_chars=min_chars):
        res.intents = [Intent.SKIP]
        res.skipped = True
        res.skip_reason = "trivial"
        res.latency_ms = int((time.perf_counter() - t0) * 1000)
        return res

    # Step 1: classify (non-trivial: load known projects from the graph).
    try:
        known = _known_projects(engine)
    except Exception:
        known = set()
    intents = detect_intents(msg, known_projects=known, min_chars=min_chars)
    res.intents = intents

    if intents == [Intent.SKIP]:
        # Safety net — detect_intents shouldn't return SKIP for non-trivial
        # input, but if it does, treat the same as trivial.
        res.skipped = True
        res.skip_reason = "no-intent-matched"
        res.latency_ms = int((time.perf_counter() - t0) * 1000)
        return res

    # Step 2: dispatch per intent.
    # PROJECT_PREP / OVERVIEW share the project_overview call; PREP also
    # pulls active arcs + open goals as extras.
    if Intent.PROJECT_PREP in intents or Intent.PROJECT_OVERVIEW in intents:
        try:
            project_name = _resolve_project_name(engine, msg, known)
            if project_name:
                ov = engine.project_overview(project_name)
                if ov and not ov.get("empty"):
                    res.project = ov
                    if Intent.PROJECT_PREP in intents:
                        try:
                            arcs = engine.active_arcs_for_project(
                                project_name, limit=2,
                            )
                            if arcs:
                                res.arcs = arcs
                        except Exception:
                            pass
        except Exception:
            pass

    # PAST_QUERY / GENERIC_FACTUAL: ask the recall layer.
    # Cold-start guard: the hook spawns a fresh Python process each turn,
    # so loading sentence-transformers from disk for every recall would
    # add 10-20s per user message. Skip the recall when the engine isn't
    # warm — the model loads in the BACKGROUND for the next turn, and
    # the other intents (project/lessons/recent/goals) still fire from
    # pure-SQL paths. To keep recall always-on, run `pmb warmup` once
    # after install to seed the on-disk cache.
    do_recall = (
        Intent.PAST_QUERY in intents
        or Intent.GENERIC_FACTUAL in intents
    )
    if do_recall:
        warm = True
        try:
            # is_warm() lives on the engine.search submodule (via embed mixin)
            # but historically callers ran `engine.is_warm()` too. Try both.
            if hasattr(engine, "is_warm"):
                warm = bool(engine.is_warm())
            elif hasattr(engine, "search") and hasattr(engine.search, "is_warm"):
                warm = bool(engine.search.is_warm())
        except Exception:
            warm = True  # be permissive if probe fails
        if not warm:
            # Note the cold-skip in intents so the agent / debug output
            # makes it obvious why recall didn't fire.
            res.intents = res.intents + ["RECALL_COLD_SKIP"]
        else:
            try:
                pack = engine.recall(query=msg, top_k=recall_top_k)
                pd = pack.to_dict() if pack else {}
                hits = pd.get("results") or []
                conf = float(pd.get("confidence") or 0.0)
                # GENERIC_FACTUAL is best-effort — only surface if the top
                # hit is reasonably confident. PAST_QUERY is explicit so
                # we always surface what we got.
                if hits and (
                    Intent.PAST_QUERY in intents
                    or hits[0].get("score", 0.0) >= recall_min_score
                ):
                    res.recall_hits = hits
                    res.recall_query = msg
                    res.recall_confidence = conf
            except Exception:
                pass

    if Intent.RECENT_QUERY in intents:
        try:
            recent = engine.what_just_happened(n=recent_limit)
            if recent:
                res.recent = recent
            else:
                act = engine.recent_activity(
                    minutes=recent_minutes, limit=recent_limit,
                )
                if act:
                    res.recent = act
        except Exception:
            pass

    if Intent.GOALS_QUERY in intents:
        try:
            goals = engine.list_goals(status="in_progress", limit=goals_limit)
            if goals:
                res.open_goals = goals
        except Exception:
            pass

    # Lessons: always run (cheap, high value). If LESSONS_QUERY was the
    # explicit intent we pull a bigger window; otherwise just 3.
    try:
        limit = lessons_limit if Intent.LESSONS_QUERY in intents else 3
        lessons = engine.find_lessons(query=msg, limit=limit)
        if lessons:
            res.lessons = lessons
            if log_surfaces:
                try:
                    engine._log_lesson_surfaces(
                        lessons, query=msg, source="hook.auto_recall",
                    )
                except Exception:
                    pass
    except Exception:
        pass

    # Decisions (the "why we did X" rationale): always run when enabled.
    # Surfacing past decisions next to lessons means "before you do this,
    # here's what we already decided about it" — the agent doesn't have to
    # think to ask. Only attached when find_decisions exists (newer engine).
    if surface_decisions and hasattr(engine, "find_decisions"):
        try:
            decs = engine.find_decisions(query=msg, limit=decisions_limit)
            if decs:
                # Don't repeat decisions the project_overview already surfaced.
                seen = set()
                pc = res.project or {}
                for d in (pc.get("decisions") or []):
                    u = d.get("ulid")
                    if u:
                        seen.add(u)
                res.decisions = [d for d in decs if d.get("ulid") not in seen]
        except Exception:
            pass

    res.latency_ms = int((time.perf_counter() - t0) * 1000)
    return res


# ─── Formatting ──────────────────────────────────────────────────────────


def _trim(s: Any, n: int = 200) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.strip().replace("\n", " ")
    return s[:n]


def format_context(
    res: AutoContextResult,
    *,
    max_chars: int = 4000,
    include_trace: bool = True,
) -> str:
    """Render an AutoContextResult into a plain-text block for the agent.

    Empty input → empty output. Hosts truncate long context, so we cap to
    `max_chars` and trim with an explicit marker.
    """
    if res.skipped or res.is_empty():
        return ""

    buf: list[str] = []
    header = "== PMB auto-context =="
    if include_trace:
        header += (
            f"  [intents={','.join(res.intents)} "
            f"latency={res.latency_ms}ms]"
        )
    buf.append(header)
    buf.append(f"(matched on message: {_trim(res.message, 100)!r})")

    if res.project and not res.project.get("empty"):
        pc = res.project
        ent = pc.get("entity") or {}
        buf.append(
            f"\nProject: {ent.get('name','?')} "
            f"({ent.get('n_mentions','?')} mentions)"
        )
        kf = pc.get("key_facts") or []
        if kf:
            buf.append("Key facts:")
            for f in kf[:5]:
                buf.append(f"  - {_trim(f.get('content',''), 160)}")
        proj_lessons = pc.get("lessons") or []
        if proj_lessons:
            buf.append("Project lessons (RULES — FOLLOW):")
            for L in proj_lessons[:5]:
                sid = L.get("surface_id")
                tag = f" [surface_id={sid}]" if sid else ""
                buf.append(f"  ! {_trim(L.get('content',''), 200)}{tag}")
        dec = pc.get("decisions") or []
        if dec:
            buf.append("Past decisions:")
            for d in dec[:3]:
                buf.append(f"  > {_trim(d.get('content',''), 160)}")
        og = pc.get("open_goals") or []
        if og:
            buf.append("Open goals on this project:")
            for g in og[:3]:
                buf.append(
                    "  * "
                    + _trim(g.get("content") or g.get("title", ""), 120)
                )

    if res.recall_hits:
        buf.append(
            f"\nRecall hits for {_trim(res.recall_query, 60)!r} "
            f"(confidence={res.recall_confidence:.2f}):"
        )
        for h in res.recall_hits[:5]:
            date = h.get("date") or ""
            stamp = f" [{date}]" if date else ""
            buf.append(
                f"  - {_trim(h.get('content',''), 180)}"
                f" (score={float(h.get('score') or 0.0):.2f}){stamp}"
            )

    if res.recent:
        buf.append("\nRecent activity:")
        for a in res.recent[:5]:
            buf.append(f"  - {_trim(a.get('content',''), 120)}")

    if res.open_goals and not (res.project and res.project.get("open_goals")):
        buf.append("\nOpen goals:")
        for g in res.open_goals[:5]:
            buf.append(
                "  * "
                + _trim(g.get("content") or g.get("title", ""), 120)
            )

    # Surface lessons separately ONLY if no project context already had them.
    proj_lessons = (res.project or {}).get("lessons") or []
    if res.lessons and not proj_lessons:
        buf.append("\nLessons matching this message:")
        for L in res.lessons[:5]:
            sid = L.get("surface_id")
            tag = f" [surface_id={sid}]" if sid else ""
            buf.append(f"  ! {_trim(L.get('content',''), 200)}{tag}")

    # Past decisions (rationale) — so the agent doesn't re-decide a settled
    # call. Surfaced only if project_overview didn't already include them.
    proj_decisions = (res.project or {}).get("decisions") or []
    if res.decisions and not proj_decisions:
        buf.append("\nPast decisions about this (don't re-litigate without reason):")
        for d in res.decisions[:5]:
            buf.append(f"  > {_trim(d.get('content',''), 200)}")

    if res.arcs:
        buf.append("\nActive narrative arcs:")
        for a in res.arcs[:2]:
            buf.append(
                f"  ~ {_trim(a.get('title',''), 120)}"
                f" ({a.get('n_events','?')} events)"
            )

    # Footer: explicit instruction to use lessons + close the feedback loop.
    if res.lessons or proj_lessons:
        buf.append("")
        buf.append(
            "If a lesson with [surface_id=N] applies, FOLLOW it. After acting,"
        )
        buf.append(
            "call mark_lesson_followed(surface_id=N, followed=True, note=\"...\")."
        )

    # Coordinate with the deliberate MCP channel: this block was injected by the
    # auto-recall HOOK, not requested. Tell the agent so it doesn't spend a turn
    # re-calling prepare()/recall() for what's already here.
    buf.append("")
    buf.append("(auto-recall ran for this message — only call recall() / "
               "find_lessons() for specifics not shown above.)")

    text = "\n".join(buf)
    if len(text) > max_chars:
        text = text[: max_chars - 30].rstrip() + "\n... [context truncated]"
    return text
