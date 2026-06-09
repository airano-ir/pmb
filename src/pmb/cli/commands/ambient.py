"""Ambient-memory + hook-entry root commands (prepare-context, auto-context,
session-restore, lesson-followcheck, track-action, autowrite, forget-auto,
ambient-watch, codex-notify).

Extracted from cli/main.py (no behavior change). cli/main.py imports this
module so these @app.command registrations run on the shared app."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer

from pmb.cli._common import _humanize_time, app, console  # noqa: F401

# prepare-context — the line that hooks call to inject memory at session
# start. Reads the user message from stdin or a positional arg, prints a
# compact context block.
# ═══════════════════════════════════════════════════════════════════════

def _read_stdin_utf8() -> str:
    """Read stdin as UTF-8, regardless of the platform locale.

    Critical for the hook path: Claude Code pipes the user message as
    UTF-8, but on Windows `sys.stdin.read()` defaults to the locale
    codepage (cp1251 on a RU system), which mangles Cyrillic/non-ASCII
    into mojibake — and then the intent regexes never match. Read the raw
    bytes and decode UTF-8 explicitly (replace on error so we never crash
    the hook).
    """
    import sys as _sys
    try:
        data = _sys.stdin.buffer.read()
        return data.decode("utf-8", errors="replace")
    except Exception:
        # Fallback to the text stream if .buffer isn't available (e.g.
        # pytest capture replaces stdin with a StringIO).
        try:
            return _sys.stdin.read()
        except Exception:
            return ""


def _extract_hook_prompt(raw: str) -> str:
    """Pull the user message out of whatever the host piped on stdin.

    Claude Code (and other hosts) send a JSON hook payload, with the user's
    text under `prompt`. If we treat that whole JSON blob as the message, the
    intent classifier matches on `{"hook_event_name":...}` instead of the
    user text — and worse, json.dumps escapes non-ASCII, so Cyrillic/Unicode
    prompts arrive as `\\uXXXX` and NO intent ever fires. Manual callers and
    older wiring pipe raw text. Accept both: parse JSON and take the prompt
    field; otherwise use the raw string unchanged.
    """
    if not raw:
        return ""
    s = raw.lstrip()
    if s[:1] in ("{", "["):
        try:
            import json as _json
            obj = _json.loads(s)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            for k in ("prompt", "message", "user_message", "user_prompt",
                      "text", "content"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            # A JSON object with no recognizable text field → nothing to do;
            # fall through to the raw string (better than an empty message).
    return raw


@app.command("prepare-context")
def prepare_context_cmd(
    message: Optional[str] = typer.Argument(
        None,
        help="The user message to prepare context for. If omitted, read from stdin.",
    ),
    stdin_flag: bool = typer.Option(
        False, "--stdin",
        help="Read the message from stdin instead of the positional arg.",
    ),
    max_chars: int = typer.Option(
        4000, "--max-chars",
        help="Hard cap on output size. Hosts truncate long hook output.",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q",
        help="Suppress empty / no-context output (silent if nothing to inject).",
    ),
    legacy: bool = typer.Option(
        False, "--legacy",
        help="Force the pre-auto-recall always-on bundle (project + lessons + "
             "recent + goals on every turn). Useful for A/B vs the new "
             "intent-classified path.",
    ),
):
    """Print a compact PMB context block for a user message.

    Designed to be wired into session-start hooks (`pmb hooks install`).
    Output is plain text — the hook host appends it to the prompt as
    an additional system note before the model thinks.

    The default behaviour goes through `pmb.hooks.auto_recall`:

      - skip silently on trivial input (greetings, acks, <5 chars)
      - PROJECT_PREP / PROJECT_OVERVIEW on known project names
      - PAST_QUERY → recall()  (this is the big one — fixes the
        "agent forgot to call recall" hole)
      - RECENT_QUERY → what_just_happened()
      - GOALS_QUERY → list_goals(in_progress)
      - LESSONS_QUERY → wider find_lessons() window
      - GENERIC_FACTUAL → low-cost recall, surface only if confident
      - lessons always run as a side-dish

    Pass `--legacy` to fall back to the older always-on bundle.
    Pass `auto_recall.enabled=false` in config for the same effect at
    the workspace level.
    """
    import sys as _sys
    if stdin_flag or message is None:
        # Hosts pipe a JSON hook payload; extract the prompt (and don't choke
        # on \uXXXX-escaped Cyrillic). Manual callers may pipe raw text.
        msg = _extract_hook_prompt(_read_stdin_utf8().strip()).strip()
    else:
        msg = (message or "").strip()
    if not msg:
        if quiet:
            return
        _sys.stdout.write("[pmb hook] no user message; nothing to prepare\n")
        return

    try:
        from pmb.core.engine import Engine
        eng = Engine()
    except Exception as e:
        if quiet:
            return
        _sys.stdout.write(f"[pmb hook] engine init failed: {e}\n")
        return

    # Respect the config switch and the CLI override.
    use_auto = not legacy and bool(eng.config.get("auto_recall.enabled"))

    if use_auto:
        from pmb.hooks import format_context, run_auto_context
        try:
            res = run_auto_context(
                eng, msg,
                min_chars=int(eng.config.get("auto_recall.min_message_chars") or 5),
                recall_top_k=int(eng.config.get("auto_recall.recall_top_k") or 5),
                recall_min_score=float(
                    eng.config.get("auto_recall.recall_min_score") or 0.30
                ),
                surface_decisions=bool(
                    eng.config.get("auto_recall.surface_decisions")
                ),
            )
        except Exception as e:
            if quiet:
                return
            _sys.stdout.write(f"[pmb hook] auto-recall failed: {e}\n")
            return

        if res.skipped or res.is_empty():
            if quiet:
                return
            _sys.stdout.write(
                f"[pmb hook] no context to inject "
                f"(intents={','.join(res.intents)}, "
                f"reason={res.skip_reason or 'no-match'}).\n"
            )
            return

        # Allow either CLI flag or config to bound output size; smaller wins.
        cap = min(
            max_chars,
            int(eng.config.get("auto_recall.budget_chars") or 4000),
        )
        include_trace = bool(eng.config.get("auto_recall.include_trace"))
        text = format_context(res, max_chars=cap, include_trace=include_trace)
        if not text:
            if quiet:
                return
            _sys.stdout.write("[pmb hook] auto-recall produced empty output.\n")
            return
        _sys.stdout.write(text + "\n")
        return

    # ── Legacy path: always-on bundle (kept for --legacy A/B testing) ──
    try:
        out: dict = {}
        det = eng.detect_project_in_text(msg)
        if det:
            ov = eng.project_overview(det["name"])
            out["project_context"] = ov
            try:
                arcs = eng.active_arcs_for_project(det["name"], limit=2)
                if arcs:
                    out["active_arcs"] = arcs
            except Exception:
                pass
        try:
            ls = eng.find_lessons(query=msg, limit=5)
            if ls:
                eng._log_lesson_surfaces(ls, query=msg, source="hook.prepare")
                out["lessons"] = ls
        except Exception:
            pass
        try:
            act = eng.recent_activity(minutes=1440.0, limit=8)
            if act:
                out["recent_activity"] = act
        except Exception:
            pass
        try:
            goals = eng.list_goals(status="in_progress", limit=5)
            if goals:
                out["open_goals"] = goals
        except Exception:
            pass
    except Exception as e:
        if quiet:
            return
        _sys.stdout.write(f"[pmb hook] prepare failed: {e}\n")
        return

    if not out:
        if quiet:
            return
        _sys.stdout.write(
            "[pmb hook] no project / lesson / activity matched. "
            "Answer normally.\n"
        )
        return

    buf: list[str] = []
    buf.append("== PMB context for this turn ==")
    buf.append(f"(matched on message: {msg[:80]!r})")

    pc = out.get("project_context")
    if pc and not pc.get("empty"):
        ent = pc.get("entity") or {}
        buf.append(f"\nProject: {ent.get('name')} ({ent.get('n_mentions')} mentions)")
        kf = pc.get("key_facts", [])[:5]
        if kf:
            buf.append("Key facts:")
            for f in kf:
                buf.append(f"  - {f.get('content','')[:160]}")
        ls_in_pc = pc.get("lessons", [])[:5]
        if ls_in_pc:
            buf.append("Lessons (RULES to follow):")
            for L in ls_in_pc:
                sid = L.get("surface_id")
                tag = f" [surface_id={sid}]" if sid else ""
                buf.append(f"  ! {L.get('content','')[:200]}{tag}")
        dec = pc.get("decisions", [])[:3]
        if dec:
            buf.append("Past decisions:")
            for d in dec:
                buf.append(f"  > {d.get('content','')[:160]}")
        og = pc.get("open_goals", [])[:3]
        if og:
            buf.append("Open goals:")
            for g in og:
                buf.append(f"  * {g.get('content', g.get('title',''))[:120]}")

    ls_flat = out.get("lessons", [])
    if ls_flat and (not pc or not pc.get("lessons")):
        buf.append("\nLessons matching this message:")
        for L in ls_flat[:5]:
            sid = L.get("surface_id")
            tag = f" [surface_id={sid}]" if sid else ""
            buf.append(f"  ! {L.get('content','')[:200]}{tag}")

    ra = out.get("recent_activity", [])
    if ra:
        buf.append("\nRecent activity (last 24h):")
        for a in ra[:5]:
            buf.append(f"  - {a.get('content','')[:120]}")

    arcs = out.get("active_arcs", [])
    if arcs:
        buf.append("\nActive narrative arcs:")
        for a in arcs[:2]:
            buf.append(f"  ~ {a.get('title','')[:120]} ({a.get('n_events')} events)")

    buf.append("")
    buf.append("If a Lesson with [surface_id=N] applies, FOLLOW it and after")
    buf.append("acting call mark_lesson_followed(surface_id=N, followed=True,")
    buf.append("note=\"<one line: what you did>\").")

    text = "\n".join(buf)
    if len(text) > max_chars:
        text = text[: max_chars - 40] + "\n... [context truncated]"
    _sys.stdout.write(text + "\n")


# ─── pmb auto-context — debug-friendly inspection of the hook output ──

@app.command("auto-context")
def auto_context_cmd(
    message: str = typer.Argument(
        ..., help="The message to classify and dispatch on.",
    ),
    show_json: bool = typer.Option(
        False, "--json",
        help="Print the structured AutoContextResult as JSON instead of the "
             "formatted context block.",
    ),
    max_chars: int = typer.Option(
        4000, "--max-chars",
        help="Cap on the formatted context block (ignored when --json).",
    ),
):
    """Inspect what the auto-recall hook would inject for a given message.

    Useful for debugging "why didn't PMB fire here?" / "why did it fire
    on this trivial input?". The classifier is regex-only so behaviour is
    deterministic and reproducible.

    Examples:

      pmb auto-context "fix the recall bug in PMB"
      pmb auto-context "when did I last edit docker-compose"
      pmb auto-context "what are my open goals" --json
    """
    from pmb.core.engine import Engine
    from pmb.hooks import format_context, run_auto_context

    eng = Engine()
    res = run_auto_context(
        eng, message,
        min_chars=int(eng.config.get("auto_recall.min_message_chars") or 5),
        recall_top_k=int(eng.config.get("auto_recall.recall_top_k") or 5),
        recall_min_score=float(
            eng.config.get("auto_recall.recall_min_score") or 0.30
        ),
        surface_decisions=bool(eng.config.get("auto_recall.surface_decisions")),
    )
    if show_json:
        import json
        from dataclasses import asdict
        console.print_json(json.dumps(asdict(res), default=str, ensure_ascii=False))
        return
    text = format_context(res, max_chars=max_chars, include_trace=True)
    if not text:
        console.print(
            f"[dim]no context (intents={','.join(res.intents)}, "
            f"reason={res.skip_reason or 'no-match'}, "
            f"latency={res.latency_ms}ms)[/]"
        )
        return
    # markup=False: the formatted block contains `[surface_id=N]` and
    # `[intents=...]` brackets that Rich would otherwise eat as style tags.
    console.print(text, markup=False, highlight=False)


# ─── pmb session-restore — rebuild context after a compaction ─────────

@app.command("session-restore")
def session_restore_cmd(
    minutes: Optional[int] = typer.Option(
        None, "--minutes", "-m",
        help="Window to summarise (default: config session.brief_minutes). "
             "Used when no active session is bound (e.g. fresh hook process).",
    ),
    max_chars: int = typer.Option(
        4000, "--max-chars",
        help="Hard cap on output size.",
    ),
    no_project: bool = typer.Option(
        False, "--no-project",
        help="Skip the project_overview section (session_brief only).",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q",
        help="Print nothing when there's no session to restore.",
    ),
):
    """Rebuild 'where you left off' after the agent's context compacts.

    Designed for a SessionStart(compact|resume) hook (`pmb hooks install`):
    when Claude Code compacts the conversation, this prints what THIS
    session decided / did / learned + the project overview, so the agent
    picks the thread back up instead of re-asking the user.

    Run it by hand to preview:  pmb session-restore -m 180
    """
    import sys as _sys
    try:
        from pmb.core.engine import Engine
        from pmb.hooks import build_session_restore
        eng = Engine()
    except Exception as e:
        if not quiet:
            _sys.stdout.write(f"[pmb hook] engine init failed: {e}\n")
        return
    try:
        text = build_session_restore(
            eng,
            minutes=float(minutes) if minutes else None,
            include_project=not no_project,
            max_chars=max_chars,
        )
    except Exception as e:
        if not quiet:
            _sys.stdout.write(f"[pmb hook] session-restore failed: {e}\n")
        return
    if not text:
        if not quiet:
            _sys.stdout.write(
                "[pmb hook] nothing to restore (no recent session activity).\n"
            )
        return
    _sys.stdout.write(text + "\n")


# ─── pmb lesson-followcheck — infer follow-through, no model cooperation ──

@app.command("lesson-followcheck")
def lesson_followcheck_cmd(
    window: int = typer.Option(
        30, "--window", "-w",
        help="Minutes back to scan for unconfirmed lesson surfaces.",
    ),
    min_overlap: int = typer.Option(
        2, "--min-overlap",
        help="Distinctive-token overlap required to count a lesson as "
             "followed. Higher = stricter (fewer false positives).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Show what would be marked without writing.",
    ),
    show_json: bool = typer.Option(
        False, "--json", help="Emit the FollowCheckResult as JSON.",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q",
        help="Print nothing (for the Stop hook — it runs silently).",
    ),
):
    """Infer lesson follow-through from recorded activity — no model needed.

    For each lesson that surfaced recently but was never confirmed, check
    whether its distinctive tokens show up in what the agent actually did
    this turn (recorded activity). If so, mark it followed with an honest
    'auto-detected' note. Lessons with no evidence stay unconfirmed — we
    never fabricate a follow.

    Designed for a Stop hook (`pmb hooks install`) so the adherence
    dashboard's follow-rate reflects reality instead of sitting at 0%
    because models don't self-report.

    Preview:  pmb lesson-followcheck --dry-run --json
    """
    import sys as _sys
    try:
        from pmb.core.engine import Engine
        from pmb.hooks import run_followcheck
        eng = Engine()
    except Exception as e:
        if not quiet:
            _sys.stdout.write(f"[pmb hook] engine init failed: {e}\n")
        return
    try:
        res = run_followcheck(
            eng,
            window_minutes=float(window),
            activity_minutes=float(window),
            min_overlap=min_overlap,
            apply=not dry_run,
        )
    except Exception as e:
        if not quiet:
            _sys.stdout.write(f"[pmb hook] followcheck failed: {e}\n")
        return
    if show_json:
        import json
        console.print_json(json.dumps(res.to_dict(), ensure_ascii=False))
        return
    if quiet:
        return
    verb = "would mark" if dry_run else "marked"
    if res.marked_followed:
        console.print(
            f"[green]{verb} {res.marked_followed} lesson(s) followed[/] "
            f"(checked {res.checked})"
        )
        for v in res.verdicts:
            console.print(
                f"  ✓ surface {v.surface_id} · overlap: "
                f"{', '.join(v.overlap[:5])}", markup=False, highlight=False,
            )
    else:
        console.print(
            f"[dim]no follow-through inferred "
            f"(checked {res.checked}, reason={res.skipped_reason or 'no overlap'})[/]"
        )


# ─── pmb track-action — ambient: log one agent action (PostToolUse) ──────

@app.command("track-action")
def track_action_cmd(
    quiet: bool = typer.Option(
        True, "--quiet/--verbose", "-q",
        help="Silent by default — this runs on every tool call.",
    ),
):
    """Log one observed agent action. Reads the PostToolUse hook JSON from
    stdin and writes a single row to the agent_actions log. Instant, no
    model. Significance (edit vs read) is judged at write time.

    Designed for a PostToolUse hook; harmless to call by hand with a JSON
    payload on stdin.
    """
    import json as _json
    import sys as _sys
    raw = _read_stdin_utf8()
    if not raw.strip():
        return
    try:
        payload = _json.loads(raw)
    except Exception:
        return
    tool = payload.get("tool_name") or payload.get("tool") or ""
    ti = payload.get("tool_input") or {}
    tr = payload.get("tool_response") or payload.get("tool_result") or {}
    # target: file path / command / url depending on the tool
    target = (ti.get("file_path") or ti.get("path") or ti.get("command")
              or ti.get("notebook_path") or ti.get("url") or "")
    command = ti.get("command") or ""
    # status: best-effort from the tool response
    status = ""
    if isinstance(tr, dict):
        if tr.get("success") is True or tr.get("is_error") is False:
            status = "ok"
        elif tr.get("success") is False or tr.get("is_error") is True:
            status = "error"
        elif "exit_code" in tr:
            status = "ok" if tr.get("exit_code") == 0 else str(tr.get("exit_code"))
    session_id = payload.get("session_id")
    if not tool:
        return
    # Hot path: this runs on EVERY tool call, so do NOT load the engine
    # (numpy / lancedb). Resolve the workspace and INSERT directly via the
    # dependency-light ambient_log module — ~1s vs ~2s, no heavy imports.
    try:
        from pmb.core.ambient_log import insert_agent_action, is_significant_action
        from pmb.core.workspace import detect_workspace
        ws = detect_workspace()
        rid = insert_agent_action(
            ws.db_path, ws.id, tool=tool, target=str(target),
            status=status, command=str(command), session_id=session_id,
        )
        if not quiet:
            sig = is_significant_action(tool, str(target), str(command))
            console.print(f"[dim]tracked {tool} {str(target)[:50]} "
                          f"(significant={sig}, id={rid})[/]")
    except Exception as e:
        if not quiet:
            _sys.stdout.write(f"[pmb hook] track-action failed: {e}\n")


# ─── pmb autowrite — ambient: journal the turn if the agent didn't ───────

def _spawn_detached_autowrite(window: int) -> bool:
    """Spawn `pmb autowrite --worker` as a DETACHED process that outlives the
    current (Stop-hook) process, so an LLM synthesis round-trip never blocks
    the turn. Inherits this process's env (PMB_HOME / PMB_WORKSPACE).
    Returns True if the spawn succeeded.
    """
    import subprocess as _sp
    import sys as _sys
    from pathlib import Path as _Path
    py = _Path(_sys.executable)
    pmb_exe = None
    for c in (py.parent / "pmb.exe", py.parent / "pmb"):
        if c.exists():
            pmb_exe = str(c)
            break
    if not pmb_exe:
        return False  # caller falls back to inline template
    argv = [pmb_exe, "autowrite", "--worker", "--quiet", "--window", str(window)]
    try:
        kwargs = dict(
            stdin=_sp.DEVNULL, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
            close_fds=True,
        )
        if os.name == "nt":
            # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP — fully cut from the
            # parent so it survives the Stop hook returning.
            kwargs["creationflags"] = (
                getattr(_sp, "DETACHED_PROCESS", 0x00000008)
                | getattr(_sp, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
            )
        else:
            kwargs["start_new_session"] = True
        _sp.Popen(argv, **kwargs)
        return True
    except Exception:
        return False


@app.command("autowrite")
def autowrite_cmd(
    window: int = typer.Option(
        30, "--window", "-w",
        help="Minutes back = roughly this turn.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be written, write nothing.",
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Run even if autowrite.enabled is false in config (for testing).",
    ),
    worker: bool = typer.Option(
        False, "--worker",
        help="Internal: the detached background process that does the LLM "
             "synthesis. Not meant to be called by hand.",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Silent (for the Stop hook).",
    ),
):
    """Journal what the agent did this turn — but ONLY if the agent didn't
    record anything itself. Synthesizes one activity entry from the observed
    action log (template by default; LLM if autowrite.synthesizer is set).

    Template synthesis runs inline (instant). An LLM synthesizer is deferred
    to a DETACHED background process so the Stop hook never waits on a model
    round-trip. Off unless `autowrite.enabled` is true. Designed for a Stop
    hook.
    """
    import sys as _sys
    try:
        from pmb.core.engine import Engine
        from pmb.hooks import autowrite_gate, run_autowrite
        eng = Engine()
    except Exception as e:
        if not quiet:
            _sys.stdout.write(f"[pmb hook] autowrite init failed: {e}\n")
        return

    if not force and not bool(eng.config.get("autowrite.enabled")):
        if not quiet:
            console.print("[dim]autowrite disabled "
                          "(config autowrite.enabled=false)[/]")
        return

    win = float(window)
    min_acts = int(eng.config.get("autowrite.min_actions") or 2)
    min_imp = float(eng.config.get("autowrite.min_importance") or 0.0)
    synth = str(eng.config.get("autowrite.synthesizer") or "template")

    # Phase 2: if the synthesizer is an LLM, don't run it inline — it would
    # block the Stop hook for up to autowrite.llm_timeout_s. Instead: do a
    # cheap gate-check, and if the turn qualifies, spawn a DETACHED worker
    # that survives this process and does the LLM synthesis + record on its
    # own. The Stop hook returns in ~ms. (dry-run / explicit --worker skip
    # the detach and run inline.)
    if synth.startswith("llm:") and not dry_run and not worker:
        skip = autowrite_gate(eng, window_minutes=win, min_actions=min_acts,
                              min_importance=min_imp)
        if skip:
            if not quiet:
                console.print(f"[dim]nothing journaled — {skip}[/]")
            return
        ok = _spawn_detached_autowrite(window=int(win))
        if not quiet:
            if ok:
                console.print("[green]journaling in background[/] "
                              f"(synthesizer={synth})")
            else:
                # Detach failed — fall back to inline template so we don't
                # silently lose the journal entry.
                res = run_autowrite(
                    eng, window_minutes=win, min_actions=min_acts,
                    min_importance=min_imp, synthesizer="template", apply=True,
                )
                console.print(
                    f"[yellow]background spawn failed, wrote template[/]: "
                    f"{res.summary or res.skipped_reason}",
                    markup=False, highlight=False,
                )
        elif not ok:
            # quiet + detach failed → still record a template entry inline.
            run_autowrite(eng, window_minutes=win, min_actions=min_acts,
                          min_importance=min_imp, synthesizer="template",
                          apply=True)
        return

    # Inline path: template synthesizer, OR the detached --worker itself, OR
    # --dry-run. The worker uses the configured (LLM) synthesizer.
    res = run_autowrite(
        eng,
        window_minutes=win,
        min_actions=min_acts,
        min_importance=min_imp,
        synthesizer=synth,
        llm_model=str(eng.config.get("autowrite.llm_model") or ""),
        llm_timeout=float(eng.config.get("autowrite.llm_timeout_s") or 20.0),
        apply=not dry_run,
    )
    if quiet:
        return
    if res.wrote:
        verb = "would journal" if dry_run else "journaled"
        console.print(
            f"[green]{verb}[/] ({res.synthesizer}, {res.n_actions} actions, "
            f"importance={res.importance:.2f}): ",
            end="",
        )
        console.print(res.summary or "", markup=False, highlight=False)
    else:
        console.print(
            f"[dim]nothing journaled — {res.skipped_reason or 'no actions'}[/]"
        )


# ─── pmb forget-auto — drop memory the ambient writer made (user control) ─

@app.command("forget-auto")
def forget_auto_cmd(
    minutes: Optional[int] = typer.Option(
        None, "--minutes", "-m",
        help="Only auto-written entries from the last N minutes (default: all).",
    ),
):
    """Archive everything ambient auto-write recorded (source=autowrite).

    Your memory, your call: this removes only what PMB journaled on its own
    — never what you or the agent recorded explicitly. Archived, not
    hard-deleted.
    """
    from pmb.core.engine import Engine
    eng = Engine()
    n = eng.forget_auto_written(minutes=float(minutes) if minutes else None)
    console.print(f"[green]Archived[/] {n} auto-written entr"
                  f"{'y' if n == 1 else 'ies'}.")


# ─── pmb ambient-watch — ambient for MCP-only agents (observe the project) ─

@app.command("ambient-watch")
def ambient_watch_cmd(
    path: Optional[str] = typer.Argument(
        None, help="Project root to watch (default: current directory).",
    ),
    idle: float = typer.Option(
        25.0, "--idle",
        help="Seconds of no file changes that counts as 'turn ended' → "
             "triggers auto-write.",
    ),
    interval: float = typer.Option(
        5.0, "--interval", help="Seconds between git snapshots.",
    ),
    once: bool = typer.Option(
        False, "--once",
        help="Single scan + maybe-write, then exit (for cron / a trigger). "
             "No daemon loop.",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Less chatter.",
    ),
):
    """Ambient memory for MCP-only agents (Cursor, VS Code, Zed, Gemini, …).

    Those hosts give no hooks, so we watch the PROJECT instead of the agent:
    poll git for changed files, record them as actions, and auto-write the
    turn once the project goes idle. Coordination still holds — if the agent
    journaled via PMB's MCP tools, auto-write stays silent.

    Run it next to your editor:  `pmb ambient-watch .`
    (Claude Code / Codex don't need this — they have precise hooks.)
    """
    import time as _t
    from pathlib import Path as _P
    repo = _P(path).resolve() if path else _P.cwd()

    try:
        from pmb.core.ambient_log import insert_agent_action
        from pmb.core.workspace import detect_workspace
        from pmb.hooks.project_observer import is_git_repo, snapshot_changes
    except Exception as e:
        console.print(f"[red]ambient-watch init failed: {e}[/]")
        raise typer.Exit(1)

    if not is_git_repo(repo):
        console.print(
            f"[yellow]{repo} is not a git repo.[/] The project observer uses "
            "git to detect changes. Run `git init` or point at a repo."
        )
        raise typer.Exit(2)

    ws = detect_workspace(cwd=repo)
    state: dict[str, float] = {}
    last_change = 0.0
    last_write = 0.0
    pending_since_write = False

    def _maybe_autowrite():
        nonlocal pending_since_write, last_write
        try:
            from pmb.core.engine import Engine
            from pmb.hooks import run_autowrite
            eng = Engine(cwd=repo)
        except Exception:
            return
        if not bool(eng.config.get("autowrite.enabled")):
            return
        res = run_autowrite(
            eng,
            window_minutes=float(eng.config.get("autowrite.window_minutes") or 30.0),
            min_actions=int(eng.config.get("autowrite.min_actions") or 2),
            min_importance=float(eng.config.get("autowrite.min_importance") or 0.0),
            synthesizer=str(eng.config.get("autowrite.synthesizer") or "template"),
            apply=True,
        )
        pending_since_write = False
        last_write = _t.time()
        if not quiet and res.wrote:
            console.print("[green]journaled:[/] ", end="")
            console.print(res.summary or "", markup=False, highlight=False)

    def _scan():
        nonlocal state, last_change, pending_since_write
        actions, state = snapshot_changes(repo, state)
        for a in actions:
            insert_agent_action(ws.db_path, ws.id, tool=a["tool"],
                                target=a["target"], status=a["status"])
        if actions:
            last_change = _t.time()
            pending_since_write = True
            if not quiet:
                names = ", ".join(a["target"].split("/")[-1].split("\\")[-1]
                                  for a in actions[:4])
                console.print(f"[dim]observed {len(actions)} change(s): {names}[/]")

    if once:
        _scan()
        # In one-shot mode, write immediately if there's anything pending.
        if pending_since_write:
            _maybe_autowrite()
        return

    if not quiet:
        console.print(
            f"[bold]Watching[/] {repo} · idle={idle:g}s · interval={interval:g}s"
            f"\n[dim]ambient memory for MCP-only agents · Ctrl-C to stop[/]"
        )
    # Seed the snapshot so we don't journal the whole existing dirty tree.
    _, state = snapshot_changes(repo, {})
    try:
        while True:
            _t.sleep(interval)
            _scan()
            now = _t.time()
            if pending_since_write and (now - last_change) >= idle:
                _maybe_autowrite()
    except KeyboardInterrupt:
        if not quiet:
            console.print("\n[dim]stopped watching.[/]")


# ─── pmb codex-notify — ambient observer + auto-write for OpenAI Codex ────

@app.command("codex-notify")
def codex_notify_cmd(
    payload: Optional[str] = typer.Argument(
        None, help="The notify JSON Codex passes (agent-turn-complete). "
                   "If omitted, read from stdin. Unused fields are ignored.",
    ),
    quiet: bool = typer.Option(
        True, "--quiet/--verbose", "-q",
        help="Silent by default (this is a notify hook).",
    ),
):
    """Ambient memory for Codex: observe the turn's actions from the rollout
    log and journal them if the agent didn't.

    Codex has no per-tool hook, but it writes every action to
    ~/.codex/sessions/.../rollout-*.jsonl. On `agent-turn-complete` Codex
    calls this; we parse the NEW function_calls since last turn (offset-
    tracked), record them, and run the same auto-write as on Claude Code.
    """
    import sys as _sys
    # The notify event payload is informational; we don't strictly need it
    # (we find the active rollout by mtime). Read it so Codex's contract is
    # honoured and so a future version can use turn-id / cwd.
    if payload is None and not _sys.stdin.isatty():
        try:
            payload = _read_stdin_utf8()
        except Exception:
            payload = None
    _ = payload  # reserved for future (turn-id, cwd hints)

    try:
        from pmb.core.ambient_log import insert_agent_action, is_significant_action
        from pmb.core.workspace import detect_workspace
        from pmb.hooks.codex_rollout import find_latest_rollout, parse_rollout_actions
    except Exception as e:
        if not quiet:
            _sys.stdout.write(f"[pmb codex] init failed: {e}\n")
        return

    ws = detect_workspace()
    rollout = find_latest_rollout()
    if rollout is None:
        if not quiet:
            console.print("[dim]no Codex rollout found[/]")
        return

    # Offset state: per-rollout last-processed line, so we only record the
    # NEW actions each turn (no dupes across turns).
    import json as _j
    state_path = Path(ws.db_path).parent / "codex_rollout_offset.json"
    try:
        state = _j.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    except Exception:
        state = {}
    since = int(state.get(str(rollout), 0))

    scan = parse_rollout_actions(rollout, since_offset=since)
    # Record the new significant actions (lightweight, no engine).
    n_sig = 0
    for a in scan.actions:
        if is_significant_action(a["tool"], a.get("target", ""), a.get("command", "")):
            n_sig += 1
        insert_agent_action(
            ws.db_path, ws.id, tool=a["tool"], target=a.get("target", ""),
            status=a.get("status", ""), command=a.get("command", ""),
        )
    # Persist the new offset.
    try:
        state[str(rollout)] = scan.new_offset
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(_j.dumps(state), encoding="utf-8")
    except Exception:
        pass

    # The agent journaled itself this turn (record_* in the rollout) → silent.
    if scan.agent_recorded:
        if not quiet:
            console.print(f"[dim]agent recorded itself — silent "
                          f"({len(scan.actions)} actions seen)[/]")
        return

    # Auto-write (respects autowrite.enabled + coordination + threshold).
    try:
        from pmb.core.engine import Engine
        from pmb.hooks import run_autowrite
        eng = Engine()
    except Exception:
        return
    if not bool(eng.config.get("autowrite.enabled")):
        if not quiet:
            console.print("[dim]autowrite disabled[/]")
        return
    res = run_autowrite(
        eng,
        window_minutes=float(eng.config.get("autowrite.window_minutes") or 30.0),
        min_actions=int(eng.config.get("autowrite.min_actions") or 2),
        min_importance=float(eng.config.get("autowrite.min_importance") or 0.0),
        # Codex notify is already off the agent's critical path, but keep the
        # synthesizer inline-safe: template is instant; an LLM synthesizer
        # runs here directly (notify is fire-and-forget from Codex's view).
        synthesizer=str(eng.config.get("autowrite.synthesizer") or "template"),
        llm_model=str(eng.config.get("autowrite.llm_model") or ""),
        llm_timeout=float(eng.config.get("autowrite.llm_timeout_s") or 20.0),
        apply=True,
    )
    if not quiet:
        if res.wrote:
            console.print(f"[green]journaled[/] ({res.synthesizer}, "
                          f"{n_sig} significant): ", end="")
            console.print(res.summary or "", markup=False, highlight=False)
        else:
            console.print(f"[dim]nothing journaled — {res.skipped_reason}[/]")


# ═══════════════════════════════════════════════════════════════════════
# pmb hooks install / list / uninstall — force-feed prepare() into the
# agent's session-start hook so the READ-FIRST workflow is not optional.
# ═══════════════════════════════════════════════════════════════════════

