"""Build .pmb/resume.md from the current engine state.

The resume note is a compact, structured "where we are now" snapshot:
  - Open goals
  - Recent decisions
  - Key lessons (RULES the agent must follow)
  - Recent activity (last 24h significant work)
  - Latest exploration conclusions (from exploration-memo)
  - Project structure summary (if `pmb index project` was run)

Rendered as plain markdown so it is human-readable, diffable, and committable
to git - the same workflow Recall pioneered with `.recall/context.md`, but
sourced from PMB's typed memory instead of an extractive summary over a
transcript, so it carries real structure.

Pure-deterministic: no LLM call, no embedder. Same engine state -> same file.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmb.core.engine import Engine

_DEFAULT_PATH = ".pmb/resume.md"


def _safe(line: str, n: int = 160) -> str:
    s = (line or "").strip().replace("\r", " ").replace("\n", " · ")
    return s[:n]


def _recent_decisions(engine: Engine, limit: int = 6, days: int = 14) -> list[dict]:
    cutoff = time.time() - days * 86400
    rows: list[dict] = []
    try:
        with sqlite3.connect(engine.workspace.db_path) as c:
            cur = c.execute(
                "SELECT content, timestamp, metadata_json FROM events "
                "WHERE workspace_id=? AND archived_at IS NULL "
                "AND event_type='activity' AND timestamp>=? "
                "ORDER BY rowid DESC LIMIT 200",
                (engine.workspace.id, cutoff),
            )
            for content, ts, mj in cur.fetchall():
                kind = ""
                try:
                    md = json.loads(mj) if mj else {}
                    kind = (md.get("activity_kind") or "").lower()
                except Exception:
                    pass
                if kind in ("decision", "agreed", "resolved"):
                    rows.append({"content": content or "", "ts": ts})
                if len(rows) >= limit:
                    break
    except Exception:
        pass
    return rows


def _recent_files(engine: Engine, limit: int = 8, hours: int = 24) -> list[str]:
    cutoff = time.time() - hours * 3600
    seen: list[str] = []
    try:
        with sqlite3.connect(engine.workspace.db_path) as c:
            try:
                rows = c.execute(
                    "SELECT target FROM agent_actions WHERE workspace_id=? "
                    "AND timestamp>=? AND significant=1 ORDER BY timestamp DESC",
                    (engine.workspace.id, cutoff),
                ).fetchall()
            except Exception:
                return []
            for (t,) in rows:
                t = (t or "").strip()
                if not t or t in seen:
                    continue
                # keep only path-like targets
                if "/" in t or "\\" in t or "." in t.split()[-1]:
                    seen.append(t)
                if len(seen) >= limit:
                    break
    except Exception:
        pass
    return seen


def _latest_exploration(engine: Engine, limit: int = 3, project: str | None = None) -> list[dict]:
    out: list[dict] = []
    try:
        with sqlite3.connect(engine.workspace.db_path) as c:
            params: list = [engine.workspace.id]
            sql = (
                "SELECT metadata_json FROM events "
                "WHERE workspace_id=? AND archived_at IS NULL "
                "AND json_extract(metadata_json,'$.source')='exploration-memo'"
            )
            if project:
                sql += " AND json_extract(metadata_json,'$.project_path')=?"
                params.append(project)
            sql += " ORDER BY rowid DESC LIMIT ?"
            params.append(limit)
            for (mj,) in c.execute(sql, params).fetchall():
                try:
                    md = json.loads(mj) if mj else {}
                except Exception:
                    continue
                if md.get("intent") and md.get("conclusion"):
                    out.append({"intent": md["intent"], "conclusion": md["conclusion"]})
    except Exception:
        pass
    return out


def build_resume(engine: Engine, project: str | None = None) -> str:
    """Render the resume markdown. `project` (optional) scopes
    project-specific sections; resume is workspace-wide otherwise."""
    proj_path = str(Path(project).resolve()) if project else str(Path.cwd())
    proj_name = Path(proj_path).name

    parts: list[str] = []
    parts.append(f"# PMB resume — {proj_name}")
    parts.append("")
    parts.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"from workspace `{engine.workspace.id}`._")
    parts.append("")
    parts.append("> Auto-written by `pmb resume save` (or the optional Stop hook). "
                 "Edit by hand to refine; PMB will not overwrite untracked "
                 "content above the marker.")
    parts.append("")

    # Open goals
    try:
        goals = engine.list_goals(status="in_progress", limit=10)
    except Exception:
        goals = []
    parts.append("## Open goals")
    if goals:
        for g in goals:
            t = g.get("title") or g.get("content") or ""
            parts.append(f"- {_safe(t)}")
    else:
        parts.append("_None recorded._")
    parts.append("")

    # Recent decisions
    parts.append("## Recent decisions")
    decisions = _recent_decisions(engine)
    if decisions:
        for d in decisions:
            parts.append(f"- {_safe(d['content'])}")
    else:
        parts.append("_None in the last 14 days._")
    parts.append("")

    # Key lessons (RULES)
    parts.append("## Lessons / rules to follow")
    try:
        lessons = engine.find_lessons(query="", limit=8) or []
    except Exception:
        lessons = []
    if lessons:
        for L in lessons:
            parts.append(f"- {_safe((L.get('content') or ''), 200)}")
    else:
        parts.append("_None._")
    parts.append("")

    # Recent activity (last 24h)
    parts.append("## Recent activity (last 24h)")
    try:
        recent = engine.recent_activity(hours=24, limit=10) or []
    except Exception:
        recent = []
    if recent:
        for r in recent:
            parts.append(f"- {_safe(r.get('content') or '')}")
    else:
        parts.append("_Quiet._")
    parts.append("")

    # Recent files touched
    files = _recent_files(engine)
    if files:
        parts.append("## Files touched (last 24h)")
        for f in files:
            parts.append(f"- `{_safe(f, 200)}`")
        parts.append("")

    # Latest exploration conclusions for this project (if any)
    exp = _latest_exploration(engine, project=proj_path)
    if exp:
        parts.append("## Latest exploration conclusions")
        for e in exp:
            parts.append(f"- **{_safe(e['intent'], 120)}** — {_safe(e['conclusion'], 240)}")
        parts.append("")

    # Project structure (only if indexed for this name)
    try:
        st = engine.project_structure(proj_name) if proj_name else {}
    except Exception:
        st = {}
    if st and not st.get("empty"):
        parts.append(f"## Project structure ({proj_name})")
        langs = ", ".join(f"{k}={v}" for k, v in (st.get("languages") or {}).items())
        parts.append(f"- files indexed: {st.get('n_files')}; languages: {langs or '-'}")
        km = st.get("key_modules") or []
        if km:
            parts.append("- key modules:")
            for m in km[:8]:
                pu = m.get("purpose") or "(no purpose recorded)"
                parts.append(f"  - `{_safe(m.get('file',''), 80)}` — {_safe(pu, 160)}")
        rc = st.get("recent_changes") or []
        if rc:
            parts.append("- recent change intents:")
            for c in rc[:3]:
                parts.append(f"  - {_safe(c.get('summary',''), 200)}")
        parts.append("")

    parts.append("---")
    parts.append("<!-- PMB-RESUME-MARKER: content below this line is preserved across regenerations -->")
    parts.append("")
    return "\n".join(parts)


_MARKER = "<!-- PMB-RESUME-MARKER:"


def save_resume(engine: Engine, path: str | None = None, project: str | None = None) -> dict:
    """Write the resume note. Anything BELOW the PMB-RESUME-MARKER line in an
    existing file is preserved verbatim (so the user's hand-edited additions
    survive regeneration).
    """
    p = Path(path or _DEFAULT_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = build_resume(engine, project=project)
    user_tail = ""
    if p.exists():
        try:
            prev = p.read_text(encoding="utf-8", errors="replace")
            i = prev.find(_MARKER)
            if i != -1:
                nl = prev.find("\n", i)
                user_tail = prev[nl + 1:] if nl != -1 else ""
        except OSError:
            pass
    final = body + (user_tail.lstrip("\n") if user_tail.strip() else "")
    try:
        p.write_text(final, encoding="utf-8")
    except OSError as e:
        return {"saved": False, "error": str(e), "path": str(p)}
    return {"saved": True, "path": str(p), "bytes": len(final)}
