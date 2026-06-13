"""E2 — PDF heading detection via str.isupper() (no uppercase char-class list)."""
from __future__ import annotations

from pmb.ingest.pdf import _detect_section, _looks_allcaps_heading


def test_chapter_marker_heading():
    out = _detect_section("Chapter 3: The Beginning\n\nsome body text follows here")
    assert out and "Chapter 3" in out


def test_numbered_heading():
    assert _detect_section("2.1 Introduction to Widgets\n\nbody") == "2.1 Introduction to Widgets"


def test_allcaps_heading_any_script():
    assert _looks_allcaps_heading("INTRODUCTION")
    assert _looks_allcaps_heading("ГЛАВА ПЕРВАЯ")        # Cyrillic — via isupper()
    assert not _looks_allcaps_heading("Mixed Case Title")
    assert _detect_section("OVERVIEW SECTION\n\nbody text") == "OVERVIEW SECTION"
    assert _detect_section("ВВЕДЕНИЕ\n\nтекст") == "ВВЕДЕНИЕ"


def test_prose_is_not_a_heading():
    assert _detect_section("this is a normal sentence of prose, nothing heading-like.") is None
