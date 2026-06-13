"""Ambient auto-write - the memory journals the agent's work if it didn't.

Runs from the Stop hook. Policy:

    1. Did the agent call a record_* tool this turn?  → STAY SILENT.
       (the agent's own summary beats anything we'd synthesize)
    2. Else, are there enough SIGNIFICANT observed actions?  → synthesize
       an activity entry and record it with metadata.source = 'autowrite'.

Synthesis is template-based by default (instant, deterministic, no model).
A local/CLI LLM can be opted in (`autowrite.synthesizer = llm:ollama` etc.)
for a nicer human summary - with a timeout and an automatic fall back to
the template, so it never blocks or fails the turn.

Honesty + control: every ambient entry is tagged `source=autowrite`, shown
as auto, and removable via `engine.forget_auto_written()`. ON by default
(`autowrite.enabled`) - capturing work the agent forgot to record is PMB's
signature; flip it off with `pmb config set autowrite.enabled false`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_EDIT_TOOLS = {
    "Edit", "Write", "MultiEdit", "NotebookEdit", "Update", "Create",
    "str_replace_editor", "create_file", "edit_file", "apply_patch",
}

# Outcome detectors (git-free on purpose - see score_turn_importance). These
# decide whether a turn produced a *result* worth remembering, not just churn.
_TEST_RE = re.compile(
    r"\b(pytest|tox|nox|unittest|jest|vitest|mocha|"
    r"go\s+test|cargo\s+test|npm\s+(?:run\s+)?test|pnpm\s+(?:run\s+)?test|"
    r"yarn\s+test|rspec|phpunit|dotnet\s+test)\b",
    re.IGNORECASE,
)
_DEPLOY_RE = re.compile(
    r"\b(deploy|migrate|alembic\s+upgrade|terraform\s+apply|kubectl\s+apply|"
    r"docker\s+push|helm\s+(?:install|upgrade)|flyctl\s+deploy|"
    r"vercel\s+(?:deploy|--prod))\b",
    re.IGNORECASE,
)
_BUILD_RE = re.compile(
    r"\b(make|cmake|cargo\s+build|go\s+build|docker\s+build|"
    r"npm\s+run\s+build|pnpm\s+(?:run\s+)?build|yarn\s+build|"
    r"tsc|webpack|vite\s+build)\b",
    re.IGNORECASE,
)


def _basename(path: str) -> str:
    if not path:
        return ""
    return re.split(r"[\\/]", path.strip())[-1] or path.strip()


def _status_ok(status: str) -> bool:
    s = (status or "").strip().lower()
    return s in ("ok", "0", "pass", "passed", "success", "done")


def _status_failed(status: str) -> bool:
    s = (status or "").strip().lower()
    if s in ("error", "fail", "failed", "failure"):
        return True
    # a non-zero exit code, possibly negative (signal)
    return bool(s) and (s.lstrip("-").isdigit() and s != "0")


# ── turn classification + importance (git-free) ──────────────────────────


def classify_actions(actions: list[dict]) -> dict:
    """Bucket a turn's SIGNIFICANT actions into outcome signals.

    Deliberately git-agnostic: commits/pushes are not inspected here - a
    turn earns importance by producing a *verifiable result* (tests pass, a
    failure gets resolved, a deploy runs), not by how it's version-controlled.
    Actions are expected newest-first (as `recent_agent_actions` returns).
    """
    sig = [a for a in actions if a.get("significant")]
    chron = list(reversed(sig))  # recent_agent_actions is newest-first
    edits: list[str] = []
    test_runs: list[bool] = []   # chronological ok/fail of each test run
    build_runs: list[bool] = []
    deploy = False
    for a in chron:
        tool = a.get("tool", "")
        target = a.get("target", "") or ""
        status = a.get("status", "") or ""
        if tool in _EDIT_TOOLS:
            bn = _basename(target)
            if bn:
                edits.append(bn)
            continue
        if _TEST_RE.search(target):
            if _status_ok(status):
                test_runs.append(True)
            elif _status_failed(status):
                test_runs.append(False)
        if _BUILD_RE.search(target):
            if _status_ok(status):
                build_runs.append(True)
            elif _status_failed(status):
                build_runs.append(False)
        if _DEPLOY_RE.search(target):
            deploy = True

    # tests/build reflect the LAST run, not "ever" - a turn that ends green is
    # passing even if it was red mid-way.
    tests_passed = bool(test_runs) and test_runs[-1] is True
    tests_failed = bool(test_runs) and test_runs[-1] is False
    build_ok = bool(build_runs) and build_runs[-1] is True

    # error → resolution: a real red→green transition (a run failed, then a
    # later run of the same kind succeeded). NOT just "an edit succeeded after
    # a failure" - that doesn't re-verify anything.
    def _resolved(runs: list[bool]) -> bool:
        seen_fail = False
        for ok in runs:
            if not ok:
                seen_fail = True
            elif seen_fail:
                return True
        return False
    error_resolution = _resolved(test_runs) or _resolved(build_runs)

    uniq_edits = list(dict.fromkeys(f for f in edits if f))
    return {
        "edits": uniq_edits,
        "n_files": len(uniq_edits),
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "deploy": deploy,
        "build_ok": build_ok,
        "error_resolution": error_resolution,
        "n_significant": len(sig),
    }


def score_turn_importance(actions: list[dict]) -> tuple[float, dict]:
    """Estimate how *important* this turn is, in [0, 1], git-free.

    The point of ambient memory is to capture important work the agent forgot
    to record - not to log every file touch. So outcome signals dominate and
    raw edits barely move the needle:

        base                         0.30
        + tests passed               0.25   (verified working)
        + tests left failing         0.10   (work happened, but red)
        + deploy / migrate           0.20   (lasting consequence)
        + a failure was resolved     0.18   (something was learned)
        + build succeeded            0.10
        + breadth: >=5 files         0.12   / >=3 files  0.06

    A turn of two small edits and nothing else scores ~0.30 and falls below
    the default 0.45 bar - exactly the 'just info' we want to drop. Returns
    (score, signals) so the caller can both gate and record the real number.
    """
    sig = [a for a in actions if a.get("significant")]
    if not sig:
        return 0.0, {"n_significant": 0, "score": 0.0}
    c = classify_actions(actions)
    score = 0.30
    if c["tests_passed"]:
        score += 0.25
    elif c["tests_failed"]:
        score += 0.10
    if c["deploy"]:
        score += 0.20
    if c["error_resolution"]:
        score += 0.18
    if c["build_ok"]:
        score += 0.10
    if c["n_files"] >= 5:
        score += 0.12
    elif c["n_files"] >= 3:
        score += 0.06
    score = min(score, 0.9)
    c["score"] = round(score, 3)
    return score, c


# ── template synthesis (default, no model) ───────────────────────────────


def synthesize_template(actions: list[dict]) -> str | None:
    """Build a one-line summary, leading with the OUTCOME (a failure fixed,
    tests passed, a deploy) and only then the files touched - so the entry
    reads as 'what was accomplished', not 'what was mechanically edited'.
    Returns None if there's nothing worth recording.
    """
    sig = [a for a in actions if a.get("significant")]
    if not sig:
        return None
    c = classify_actions(actions)

    # Commits / uncategorised actions detected separately so existing wording
    # is preserved (git is mentioned if it happened, but never drives
    # importance - see score_turn_importance).
    commits, other = [], []
    for a in sig:
        if a.get("tool", "") in _EDIT_TOOLS:
            continue
        low = (a.get("target", "") or "").lower()
        if "git commit" in low or "git push" in low:
            commits.append(a)
        elif not (_TEST_RE.search(low) or _DEPLOY_RE.search(low)
                  or _BUILD_RE.search(low)):
            other.append(a)

    parts: list[str] = []
    # ── outcome first ──
    if c["error_resolution"]:
        parts.append("fixed a failing run")
    elif c["tests_passed"]:
        parts.append("ran tests (passed)")
    elif c["tests_failed"]:
        parts.append("ran tests (still failing)")
    if c["build_ok"] and not c["error_resolution"]:
        parts.append("built")
    if c["deploy"]:
        parts.append("deployed/migrated")
    # ── then the work itself ──
    if c["edits"]:
        shown = ", ".join(c["edits"][:5])
        more = f" (+{len(c['edits']) - 5} more)" if len(c["edits"]) > 5 else ""
        parts.append(f"edited {c['n_files']} file(s): {shown}{more}")
    if commits:
        parts.append("committed")
    if other and not parts:
        tools = list(dict.fromkeys(a.get("tool", "?") for a in other))
        parts.append("ran " + ", ".join(tools[:4]))

    if not parts:
        return None
    summary = "; ".join(parts)
    return summary[0].upper() + summary[1:]


# ── optional LLM synthesis (opt-in, with template fallback) ──────────────


def synthesize_llm(
    actions: list[dict],
    backend: str,
    model: str = "",
    timeout: float = 20.0,
) -> str | None:
    """Ask a CLI LLM to write a one-line human summary of the actions.
    `backend` is 'llm:ollama' / 'llm:claude' / 'llm:codex'. Returns None on
    any failure so the caller can fall back to the template."""
    sig = [a for a in actions if a.get("significant")]
    if not sig:
        return None
    lines = []
    for a in sig[:40]:
        lines.append(f"- {a.get('tool','?')}: {(a.get('target') or '')[:120]} "
                     f"[{a.get('status') or ''}]")
    prompt = (
        "You are journaling what a coding agent just did, in ONE concise past-"
        "tense line (max 25 words). No preamble, no markdown, just the line.\n\n"
        "Actions:\n" + "\n".join(lines) + "\n\nSummary:"
    )
    try:
        from pmb.graph.extractors_llm import (
            _run_claude_cli,
            _run_codex_cli,
            _run_ollama_cli,
        )
        provider = backend.split(":", 1)[1] if ":" in backend else backend
        if provider == "ollama":
            out = _run_ollama_cli(prompt, model or "qwen2.5:3b", timeout)
        elif provider == "claude":
            out = _run_claude_cli(prompt, timeout, model or "haiku")
        elif provider == "codex":
            out = _run_codex_cli(prompt, timeout)
        else:
            return None
        line = (out or "").strip().splitlines()
        line = line[0].strip() if line else ""
        line = line.strip('"').strip()
        return line[:300] or None
    except Exception:
        return None


# ── orchestration ────────────────────────────────────────────────────────


@dataclass
class AutoWriteResult:
    wrote: bool = False
    skipped_reason: str | None = None
    summary: str | None = None
    ulid: str | None = None
    n_actions: int = 0
    synthesizer: str = "template"
    importance: float = 0.0

    def to_dict(self) -> dict:
        return {
            "wrote": self.wrote, "skipped_reason": self.skipped_reason,
            "summary": self.summary, "ulid": self.ulid,
            "n_actions": self.n_actions, "synthesizer": self.synthesizer,
            "importance": self.importance,
        }


def autowrite_gate(
    engine,
    *,
    window_minutes: float = 30.0,
    min_actions: int = 2,
    min_importance: float = 0.0,
) -> str | None:
    """Cheap pre-check (no synthesis): should we journal this turn at all?

    Three gates, cheapest first:
      1. did the agent journal itself?      → stay silent (coordination)
      2. enough significant actions?         → count floor
      3. is the turn important enough?       → quality bar (git-free score)

    Returns a skip-reason string if we should stay silent, or None if the
    turn qualifies. Lets the Stop hook decide whether it's even worth
    spawning a background LLM worker - without doing any model work itself.
    `min_importance <= 0` disables the quality bar (count gate only).
    """
    try:
        if engine.agent_wrote_recently(minutes=window_minutes):
            return "agent recorded its own work this turn"
    except Exception:
        pass
    actions = engine.recent_agent_actions(
        minutes=window_minutes, significant_only=True,
    )
    n = len(actions)
    if n < min_actions:
        return f"only {n} significant action(s) (< {min_actions})"
    if min_importance > 0:
        score, _ = score_turn_importance(actions)
        if score < min_importance:
            return (f"low importance ({score:.2f} < {min_importance:.2f}) - "
                    f"mechanical churn, no clear outcome")
    return None


def run_autowrite(
    engine,
    *,
    window_minutes: float = 30.0,
    min_actions: int = 2,
    min_importance: float = 0.0,
    synthesizer: str = "template",
    llm_model: str = "",
    llm_timeout: float = 20.0,
    apply: bool = True,
) -> AutoWriteResult:
    """Decide whether to journal this turn, and do it.

    Coordination gate first: if the agent already called record_* in the
    window, we stay silent. Then a count floor + a git-free importance bar
    (`min_importance`) keep out mechanical churn. Whatever clears the bar is
    synthesized outcome-first and recorded with its real importance score, so
    recall ranks a shipped fix above a two-file tweak.
    """
    res = AutoWriteResult(synthesizer=synthesizer)

    # 1 + 2 + 3. Coordination + count floor + importance bar.
    skip = autowrite_gate(
        engine, window_minutes=window_minutes, min_actions=min_actions,
        min_importance=min_importance,
    )
    if skip:
        res.skipped_reason = skip
        return res
    actions = engine.recent_agent_actions(
        minutes=window_minutes, significant_only=True,
    )
    res.n_actions = len(actions)
    importance, signals = score_turn_importance(actions)

    # 3. Synthesize. LLM if opted in, template otherwise / on LLM failure.
    summary = None
    if synthesizer and synthesizer.startswith("llm:"):
        summary = synthesize_llm(actions, synthesizer, llm_model, llm_timeout)
        if summary:
            res.synthesizer = synthesizer
    if not summary:
        summary = synthesize_template(actions)
        res.synthesizer = "template"
    if not summary:
        res.skipped_reason = "nothing meaningful to synthesize"
        return res
    res.summary = summary

    # 3b. Dedup: if we already auto-wrote this exact summary in the window
    # (two Stop fires for one logical turn, or an unchanged action set), don't
    # journal it again. Compares against recent source=autowrite activity.
    try:
        recent = engine.recent_activity(minutes=window_minutes, limit=20)
        norm = " ".join(summary.lower().split())
        for a in recent or []:
            if " ".join((a.get("content") or "").lower().split()) == norm:
                res.skipped_reason = "already auto-wrote this summary in window"
                return res
    except Exception:
        pass

    # 4. Record (unless dry-run) with the REAL importance - not a flat 0.4 -
    # so recall ranks a verified fix above a mechanical edit. Floor at 0.35 so
    # an ambient entry is still findable even at the low end.
    res.importance = max(0.35, round(importance, 2))
    if apply:
        try:
            ulid = engine.record_activity(
                summary=summary,
                actor="agent",
                kind="completed",
                details={
                    "source": "autowrite",
                    "n_actions": len(actions),
                    "synthesizer": res.synthesizer,
                    "importance_score": round(importance, 3),
                    "signals": {k: signals.get(k) for k in (
                        "tests_passed", "tests_failed", "deploy",
                        "build_ok", "error_resolution", "n_files",
                    ) if signals.get(k)},
                },
                importance=res.importance,
            )
            res.ulid = ulid
            res.wrote = True
        except Exception as e:
            res.skipped_reason = f"record failed: {e}"
    else:
        res.wrote = True  # would-write in dry-run
    return res
