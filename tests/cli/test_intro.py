"""First-run experience: the brain-forming boot animation + marketing welcome.

  * the brain bitmap is left-right symmetric (so it centers cleanly);
  * a frame renders as a block-shaded Text that grows with the fill phase;
  * the welcome carries the pitch + the three commands worth knowing;
  * the animation is gated (PMB_NO_ANIM / non-TTY -> instant, no escape codes);
  * the marker makes the welcome fire exactly once, then yield to the palette.
"""
from __future__ import annotations

import io

from rich.console import Console

from pmb.cli.intro import _BLOCKS, _BRAIN, _brain_frame, maybe_first_run, play_boot, welcome


def _cap(width: int = 64) -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=width)


def test_brain_bitmap_is_symmetric():
    # left-right symmetric -> the blob centres cleanly; levels are in range.
    for row in _BRAIN:
        assert row == row[::-1], f"asymmetric row: {row}"
        assert all(0 <= lv < len(_BLOCKS) for lv in row)


def test_brain_frame_grows_with_phase():
    # more fill phase -> more total "ink" (denser block glyphs).
    def ink(ph):
        return sum(_BLOCKS.index(ch) for ch in _brain_frame(ph).plain if ch in _BLOCKS)
    assert ink(0.2) < ink(0.6) < ink(1.0)


def test_welcome_has_pitch_and_three_commands():
    c = _cap()
    welcome(c)
    out = c.file.getvalue()
    assert "✦" in out and "pmb" in out          # wordmark
    assert "memory" in out.lower()               # the pitch
    for cmd in ("pmb setup", "pmb --help"):
        assert cmd in out, f"missing CTA {cmd!r}"


def test_play_boot_off_tty_is_instant_and_clean(monkeypatch):
    # force_terminal=False -> the static branch; no Live, no escape codes.
    c = _cap()
    play_boot(c, settle=True)
    assert "pmb" in c.file.getvalue()
    c2 = _cap()
    play_boot(c2, settle=False)                  # no settle -> emits nothing
    assert c2.file.getvalue().strip() == ""


def test_play_boot_honours_no_anim(monkeypatch):
    monkeypatch.setenv("PMB_NO_ANIM", "1")
    c = Console(file=io.StringIO(), force_terminal=True, width=64)
    play_boot(c, settle=True)                    # must NOT animate (no hang/frames)
    out = c.file.getvalue()
    assert "pmb" in out
    assert not any(b in out for b in "░▒▓█"), "brain glyphs leaked"  # static only


def test_first_run_fires_once(tmp_path, monkeypatch):
    monkeypatch.setenv("PMB_HOME", str(tmp_path))
    monkeypatch.delenv("PMB_WORKSPACE", raising=False)
    monkeypatch.setenv("PMB_NO_ANIM", "1")
    c = _cap()
    assert maybe_first_run(c) is True            # first run -> welcome
    assert "pmb setup" in c.file.getvalue()
    assert maybe_first_run(_cap()) is False       # marker written -> palette path
