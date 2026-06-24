"""Tests for the resume note builder (Recall-style committable snapshot,
sourced from PMB's typed memory)."""
from __future__ import annotations

from pmb.resume.builder import _MARKER, build_resume, save_resume


def test_build_resume_has_required_sections(isolated_engine):
    md = build_resume(isolated_engine)
    assert "# PMB resume" in md
    assert "## Open goals" in md
    assert "## Recent decisions" in md
    assert "## Lessons / rules to follow" in md
    assert "## Recent activity (last 24h)" in md
    assert _MARKER in md  # the preservation marker


def test_save_resume_writes_file_and_idempotent(isolated_engine, tmp_path):
    p = tmp_path / ".pmb/resume.md"
    r1 = save_resume(isolated_engine, path=str(p))
    assert r1["saved"] is True
    assert p.exists()
    body1 = p.read_text(encoding="utf-8")
    assert "# PMB resume" in body1

    # Second save: same engine state -> still well-formed, still has marker.
    r2 = save_resume(isolated_engine, path=str(p))
    assert r2["saved"] is True
    assert _MARKER in p.read_text(encoding="utf-8")


def test_save_resume_preserves_user_tail(isolated_engine, tmp_path):
    """Anything below the PMB-RESUME-MARKER must survive a regeneration -
    that is the contract that lets the user hand-edit additions."""
    p = tmp_path / "resume.md"
    save_resume(isolated_engine, path=str(p))
    original = p.read_text(encoding="utf-8")
    # User appends their own notes after the marker.
    user_tail = "\n## My notes\n- remember: the auth refactor is paused\n"
    p.write_text(original + user_tail, encoding="utf-8")

    save_resume(isolated_engine, path=str(p))
    body = p.read_text(encoding="utf-8")
    assert "My notes" in body
    assert "auth refactor is paused" in body
