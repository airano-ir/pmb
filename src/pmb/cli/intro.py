"""First-impression surface: a boot animation + a marketing welcome.

The first time you run `pmb` in a terminal (and at the top of `pmb setup`), the
✦ spark *ignites* and breathes a ripple out from its core - PMB's signature
"memory pulse" as a one-time boot, then a one-screen welcome that says, in a
sentence, why pmb exists and the three commands worth knowing.

Everything here is TTY-gated and honors `PMB_NO_ANIM=1` (CI, screenshots, pipes
fall back to a static wordmark instantly), so it never slows or pollutes scripts.
"""
from __future__ import annotations

import os
import time

# One symmetric "breath": a dot ignites, the pulse expands from the ✦ core, then
# contracts back. Centered each frame, so it reads as a ring rippling outward.
# A small brain that assembles from a faint core and gently breathes - PMB's
# "memory forming" motif (its take on Claude's breathing cloud). Rendered with
# block-shading for volume + a soft pink -> magenta -> violet radial gradient, so
# it reads as a glowing 3-D blob, not flat. Left-right symmetric; the top-centre
# notch suggests the two hemispheres.
#   each cell is a level 0..5 -> how solid it is (0 = empty, 5 = full block)
_BRAIN = [
    [0, 0, 2, 3, 2, 3, 2, 0, 0],
    [0, 3, 5, 5, 3, 5, 5, 3, 0],
    [0, 4, 5, 5, 3, 5, 5, 4, 0],
    [0, 3, 5, 5, 3, 5, 5, 3, 0],
    [0, 0, 2, 4, 4, 4, 2, 0, 0],
]
_BLOCKS = " ·░▒▓█"                          # index == level
# radial gradient stops (core-distance -> rgb): warm bright core to cool edge
_BRAIN_STOPS = [
    (0.0, (250, 205, 255)),                 # near-white pink core
    (0.5, (214, 92, 235)),                  # magenta
    (1.0, (123, 78, 230)),                  # soft violet edge
]

_TAGLINE = "local-first memory for your AI agents"


def _grad(d: float) -> str:
    """Hex colour for a normalised core-distance `d` in [0,1] along _BRAIN_STOPS."""
    d = max(0.0, min(1.0, d))
    for (d0, c0), (d1, c1) in zip(_BRAIN_STOPS, _BRAIN_STOPS[1:]):
        if d <= d1:
            t = 0.0 if d1 == d0 else (d - d0) / (d1 - d0)
            r = round(c0[0] + (c1[0] - c0[0]) * t)
            g = round(c0[1] + (c1[1] - c0[1]) * t)
            b = round(c0[2] + (c1[2] - c0[2]) * t)
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#7b4ee6"


def _brain_frame(phase: float, breathe: float = 1.0):
    """One frame of the brain at fill `phase` (0..1, grows from the core) and a
    `breathe` brightness multiplier. Returns a Rich Text block (2 cells wide per
    pixel, so the blob looks round, not squished)."""
    from rich.text import Text

    h, w = len(_BRAIN), len(_BRAIN[0])
    cx, cy = (w - 1) / 2, (h - 1) / 2
    maxd = (cx * cx + cy * cy) ** 0.5
    t = Text()
    for y, row in enumerate(_BRAIN):
        for x, lv in enumerate(row):
            if lv == 0:
                t.append("  ")
                continue
            d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 / maxd
            eff = phase * (1.35 - 0.35 * d) * breathe        # core fills first
            idx = max(0, min(5, round(lv * eff)))
            t.append("  " if idx == 0 else _BLOCKS[idx] * 2, style=_grad(d))
        t.append("\n")
    return t


def _brain_sequence():
    """Assemble (core -> full) then two gentle breaths."""
    import math
    frames = [_brain_frame(phase=i / 9.0) for i in range(10)]
    for i in range(16):
        b = 0.78 + 0.22 * (0.5 + 0.5 * math.cos(math.pi * 2 * i / 8))
        frames.append(_brain_frame(phase=1.0, breathe=b))
    return frames


def play_boot(console, *, tagline: str = _TAGLINE, settle: bool = True) -> None:
    """Play the brain-forming animation (~1.8s): it assembles from a glowing core
    and breathes. With `settle`, leave the brain + wordmark on screen (standalone
    use); without it, it's transient, leaving a clean screen for a panel that
    follows. No-op to a static wordmark off a TTY or when PMB_NO_ANIM is set."""
    from rich.align import Align
    from rich.text import Text

    from pmb.cli._common import wordmark

    if os.environ.get("PMB_NO_ANIM") or not getattr(console, "is_terminal", False):
        if settle:
            console.print()
            console.print(Align.center(Text.from_markup(wordmark(tagline))))
            console.print()
        return

    from rich.live import Live
    width = min(getattr(console, "width", 64) or 64, 64)
    console.print()
    with Live(console=console, auto_refresh=False, transient=True) as live:
        for frame in _brain_sequence():
            live.update(Align.center(frame, width=width), refresh=True)
            time.sleep(0.07)
    if settle:  # leave the brain + wordmark on screen
        console.print(Align.center(_brain_frame(1.0), width=width))
        console.print(Align.center(Text.from_markup(wordmark(tagline)), width=width))
        console.print()


def welcome(console) -> None:
    """The one-screen marketing welcome: the formed brain, the pitch in a
    sentence, and the three commands worth knowing."""
    from rich.align import Align
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    from pmb.cli._common import wordmark

    brain = Align.center(_brain_frame(1.0))

    pitch = Text.from_markup(
        "Your coding agents forget everything between sessions.\n"
        "[bold]pmb[/] is the memory layer that doesn't - [#d65ceb]private, local,\n"
        "zero tokens[/], shared across every agent you use.", justify="center")

    steps = Group(
        Text.from_markup("[#a78bfa]▸[/] [bold]set up in 20s[/]    [cyan]pmb setup[/]"),
        Text.from_markup("[#a78bfa]▸[/] [bold]explore commands[/] [cyan]pmb[/]"
                         "          [dim]the command palette[/]"),
        Text.from_markup("[#a78bfa]▸[/] [bold]everything[/]       [cyan]pmb --help[/]"),
    )

    body = Group(brain, "", pitch, "", steps)
    console.print(Panel(body, title=wordmark(_TAGLINE), border_style="#a78bfa",
                        padding=(1, 3)))


def _marker_path():
    """`~/.pmb/.welcomed` - written once, so the welcome shows only on first run.
    Cheap to locate (no Engine/model load)."""
    try:
        from pmb.core.workspace import detect_workspace
        ws = detect_workspace()
        return ws.pmb_home / ".welcomed"
    except Exception:
        return None


def mark_welcomed() -> None:
    """Stamp the first-run marker so the welcome won't show again. Called by the
    welcome itself and by `pmb setup` (setting up counts as onboarded)."""
    marker = _marker_path()
    if marker is None:
        return
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("welcomed\n", encoding="utf-8")
    except Exception:
        pass


def maybe_first_run(console) -> bool:
    """If this looks like the first ever run, play the boot + show the welcome,
    stamp the marker, and return True (caller should stop). Else return False so
    the caller falls through to the command palette."""
    marker = _marker_path()
    if marker is None or marker.exists():
        return False
    play_boot(console, settle=False)  # ripple resolves straight into the welcome
    welcome(console)
    mark_welcomed()
    return True
