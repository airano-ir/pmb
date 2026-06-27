"""Capture-on-correction + repeat guard.

Three layers:
  1. detect_correction / strong_lesson_matches — pure detection (no DB).
  2. Engine.capture_correction — records a draft lesson + dedups.
  3. run_auto_context / format_context — the correction banner + loud guard
     reach the rendered block even when the message has no other intent.

Real RR/ApplyPilot complaints are used as fixtures so the detector is tuned
to the traffic it actually has to catch.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from pmb.hooks.correction_capture import (
    detect_correction,
    strong_lesson_matches,
)

# ── 1. Detector ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("msg", [
    "схуяли оно с geminy не заполняет поля",          # profanity
    "блять что ты наделал оно нихуя не заполнило",     # profanity
    "снова теже грабли geminy не заполнил поле",       # "те же грабли"
    "БЛЯТЬ ТЫ ИНСТРУКЦИЮ СЛЫШИШЬ снова не отправило",  # caps + profanity
    "УЖЕ 5 РАЗ В ЭТУ КОНТОРУ ПОДАЛ ДАВАЙ ДРУГИЕ",      # caps shout
    "я не вижу окна ничего не вижу ты хоть думаешь что делаешь",  # rhetorical
    "how many times do I have to tell you this",        # en explicit
])
def test_strong_corrections_detected(msg):
    sig = detect_correction(msg)
    assert sig is not None and sig.is_correction
    assert sig.severity == "strong"


def test_weak_correction_needs_corroboration():
    # repeat marker + negated action → weak correction
    sig = detect_correction("снова не заполнило это поле")
    assert sig is not None and sig.severity == "weak"


def test_lone_repeat_is_not_a_correction():
    # "снова" with no negation and nothing else is just a normal request
    assert detect_correction("снова открой дашборд пожалуйста") is None


@pytest.mark.parametrize("msg", [
    "что там по вакансиям сейчас",
    "сколько там вакансий уже нашли",
    "дай ссылку на repo",
    "окей запусти поиск релевантных вакансий",
])
def test_benign_messages_not_flagged(msg):
    assert detect_correction(msg) is None


def test_caps_shout_requires_minimum_length():
    # short uppercase acks must not register as shouting
    assert detect_correction("OK") is None
    assert detect_correction("DONE") is None


# ── 2. strong_lesson_matches ───────────────────────────────────────────────

def test_strong_lesson_match_on_overlap():
    lessons = [
        {"ulid": "u-1", "content": "Always validate every greenhouse form "
         "field before submit on applypilot, never submit unvalidated"},
        {"ulid": "u-2", "content": "Commit messages should be imperative mood"},
    ]
    msg = "опять submit на greenhouse без validate полей applypilot"
    out = strong_lesson_matches(msg, lessons, min_overlap=2, min_strong=2)
    assert out and out[0]["ulid"] == "u-1"


def test_no_match_when_unrelated():
    lessons = [{"ulid": "u-2", "content": "Commit messages imperative mood"}]
    out = strong_lesson_matches("the weather is nice today", lessons)
    assert out == []


# ── 3. Engine.capture_correction ───────────────────────────────────────────

def _engine(ws, home):
    from pmb.core.engine import Engine
    return Engine(cwd=ws, pmb_home=home, config_overrides={"recall.cache_size": 0})


def _lessons_by_source(eng, source):
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        rows = c.execute(
            "SELECT ulid, content FROM events "
            "WHERE json_extract(metadata_json,'$.source')=?", (source,),
        ).fetchall()
    return rows


def test_capture_records_draft_and_surface(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    out = eng.capture_correction(
        "снова не заполнило motivation textarea перед submit",
        severity="weak", markers=["weak:снова", "neg:не заполн"],
    )
    assert out["lesson_ulid"]
    assert out["surface_id"]            # surfaced → agent gets an id to confirm
    assert out["reused"] is False
    rows = _lessons_by_source(eng, "correction-capture")
    assert len(rows) == 1
    assert rows[0][1].startswith("[DRAFT lesson")


def test_capture_dedups_within_window(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    a = eng.capture_correction(
        "снова geminy не заполнил motivation textarea на greenhouse applypilot",
        severity="weak")
    b = eng.capture_correction(
        "опять geminy не заполняет motivation textarea greenhouse applypilot форму",
        severity="weak")
    assert a["reused"] is False
    assert b["reused"] is True
    assert b["lesson_ulid"] == a["lesson_ulid"]
    # one underlying draft, two surfaces (each angry message gets an id)
    assert len(_lessons_by_source(eng, "correction-capture")) == 1


def _draft_metadata(eng, ulid):
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        raw = c.execute(
            "SELECT metadata_json FROM events WHERE ulid=?", (ulid,)).fetchone()[0]
    return json.loads(raw) if raw else {}


def test_capture_tags_explicit_project(tmp_pmb_home, tmp_workspace_dir):
    """An explicit project hint is embedded in the draft - both as structured
    metadata (so find_lessons(project=) keys off it) and inline in the text (so
    the agent reading the correction banner sees what it relates to)."""
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    out = eng.capture_correction(
        "снова не заполнило поле перед submit",
        severity="weak", markers=["weak:снова", "neg:не заполн"],
        project="LoadGuard",
    )
    assert out["lesson_ulid"]
    assert out.get("project") == "LoadGuard"
    ulid, content = _lessons_by_source(eng, "correction-capture")[0]
    assert "[project: LoadGuard]" in content
    assert _draft_metadata(eng, ulid).get("project_name") == "LoadGuard"


def test_capture_detects_project_from_message(tmp_pmb_home, tmp_workspace_dir):
    """With no explicit hint, the engine scopes the draft to the project NAMED in
    the complaint - detected from the workspace's known project entities."""
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    for _ in range(3):                       # seed two known projects
        eng.graph.upsert_entity(eng.workspace.id, "project", "LoadGuard")
        eng.graph.upsert_entity(eng.workspace.id, "project", "ApplyPilot")
    out = eng.capture_correction(
        "опять LoadGuard pricing снова не считает правильно",
        severity="weak", markers=["weak:опять", "neg:не счит"],
    )
    assert (out.get("project") or "") == "LoadGuard"
    ulid, content = _lessons_by_source(eng, "correction-capture")[0]
    assert "[project: LoadGuard]" in content
    assert _draft_metadata(eng, ulid).get("project_name") == "LoadGuard"


def test_auto_context_tags_correction_with_resolved_project(
    tmp_pmb_home, tmp_workspace_dir,
):
    """End-to-end through the hook resolver: a correction about PMB is tagged
    project=PMB even though 'pmb' is a dev-noise stopword the engine's own
    detector drops - the hook uses the same lenient resolver as the overview
    path. This is the real user case (shared 'personal' store, many projects)."""
    from pmb.hooks.auto_recall import run_auto_context
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.workspace.name = "personal"          # shared store → text-detection only
    for _ in range(4):
        eng.graph.upsert_entity(eng.workspace.id, "project", "PMB")
    res = run_auto_context(eng, "схуяли PMB опять снова не находит правило")
    assert res.correction is not None
    rows = _lessons_by_source(eng, "correction-capture")
    assert rows, "a draft should have been captured"
    assert _draft_metadata(eng, rows[0][0]).get("project_name") == "PMB"
    assert "[project: PMB]" in rows[0][1]


def test_capture_fallback_uses_workspace_basename_not_full_path(
    tmp_pmb_home, tmp_workspace_dir,
):
    """Regression (found on a real cold-hook run): when detection fails and the
    workspace name is a filesystem PATH (no git repo -> cwd path is the name),
    the fallback must tag the basename, not the whole path - otherwise the draft
    reads `[project: C:\\Users\\...\\myrepo]`, the exact 'кривой' output we are
    fixing."""
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.workspace.name = r"C:\Users\me\code\myrepo"
    out = eng.capture_correction("снова не заполнило поле перед submit",
                                 severity="weak")
    assert out.get("project") == "myrepo"
    _, content = _lessons_by_source(eng, "correction-capture")[0]
    assert "[project: myrepo]" in content
    tag = content.split("[project:")[1].split("]")[0]
    assert "\\" not in tag and "/" not in tag        # no path separators leak


def test_capture_untagged_when_no_project_known(tmp_pmb_home, tmp_workspace_dir):
    """No hint + nothing detectable → no project_name (don't invent one). The
    draft is still recorded; it just stays workspace-wide."""
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.workspace.name = "personal"          # catch-all store: no project fallback
    out = eng.capture_correction(
        "снова не заполнило поле перед submit", severity="weak")
    assert out.get("project") in (None, "")
    ulid, content = _lessons_by_source(eng, "correction-capture")[0]
    assert "[project:" not in content
    assert "project_name" not in _draft_metadata(eng, ulid)


# ── 4. Integration: run_auto_context + format_context ──────────────────────

def test_correction_message_injects_banner(tmp_pmb_home, tmp_workspace_dir):
    from pmb.hooks.auto_recall import format_context, run_auto_context
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    res = run_auto_context(eng, "схуяли оно снова не заполнило поле motivation")
    assert res.correction is not None
    assert not res.skipped
    text = format_context(res, include_trace=False)
    assert "CORRECTION DETECTED" in text
    # a draft was recorded → its surface_id is offered for confirmation
    if res.correction.get("surface_id"):
        assert "mark_lesson_followed" in text


def test_repeat_guard_promotes_existing_lesson(tmp_pmb_home, tmp_workspace_dir):
    from pmb.hooks.auto_recall import format_context, run_auto_context
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.record_fact(
        "Always click 'locate me' FIRST for the location field on applypilot "
        "greenhouse forms, never type the city by hand",
        metadata={"kind": "lesson", "source": "lesson"},
    )
    res = run_auto_context(
        eng, "почему опять location field applypilot greenhouse не через locate me")
    text = format_context(res, include_trace=False)
    assert "YOU'VE HIT THIS BEFORE" in text
    assert "locate me" in text.lower()
