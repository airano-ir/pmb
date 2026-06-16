"""The command palette: bare `pmb` (in a terminal) opens a fuzzy + synonym
launcher over every command. These lock the matching so a refactor can't quietly
break "type what you mean -> get the right command":

  * the catalog covers the real command surface and is well-shaped;
  * intent queries surface the right commands by MEANING, not just substring
    ("make it fast" -> warmup, "wire my agent" -> setup/connect);
  * exact names win, nonsense matches nothing, empty query lists the head;
  * the rendered palette carries the ✦ wordmark + the matched command.

All pure-Python and instant: no Engine, no model, no TTY.
"""
from __future__ import annotations

import io

import pytest
from rich.console import Console

from pmb.cli.main import app
from pmb.cli.palette import command_catalog, rank, render_palette, score


@pytest.fixture(scope="module")
def catalog():
    return command_catalog(app)


def test_catalog_is_populated_and_shaped(catalog):
    assert len(catalog) >= 50, f"only {len(catalog)} commands catalogued"
    for c in catalog:
        assert isinstance(c["name"], str) and c["name"], c
        assert "help" in c
    names = [c["name"] for c in catalog]
    assert names == sorted(names), "catalog should be stable (alpha) order"
    # a few staples must be present
    for staple in ("setup", "connect", "recall", "stats", "warmup"):
        assert staple in names, f"missing {staple}"


@pytest.mark.parametrize(
    ("query", "expected_any"),
    [
        ("forget old stuff", {"declutter", "decay", "forget", "forget-topic"}),
        ("make it fast", {"warmup"}),
        ("wire my agent", {"setup", "connect"}),
        ("clean junk", {"declutter", "prune-expired", "compact"}),
        ("what do i know", {"audit", "overview"}),
        ("set things up", {"setup", "connect"}),
        ("search my memory", {"recall", "overview"}),
    ],
)
def test_intent_queries_surface_the_right_commands(catalog, query, expected_any):
    """Intent -> command, via the curated synonym map (no model)."""
    top = {c["name"] for c in rank(query, catalog, limit=5)}
    assert top & expected_any, f"{query!r} -> {top}, expected any of {expected_any}"


def test_exact_name_ranks_first(catalog):
    for name in ("stats", "warmup", "recall"):
        assert rank(name, catalog, limit=1)[0]["name"] == name


def test_nonsense_query_matches_nothing(catalog):
    assert rank("zzzqqxnotacommand", catalog) == []


def test_empty_query_returns_alpha_head(catalog):
    head = rank("", catalog, limit=6)
    assert len(head) == 6
    assert [c["name"] for c in head] == [c["name"] for c in catalog[:6]]


def test_score_prefers_exact_then_prefix_then_substring():
    cmd = {"name": "warmup", "help": "load the model"}
    exact = score("warmup", cmd)
    prefix = score("warm", cmd)
    substr = score("rmu", cmd)
    miss = score("xyzzy", cmd)
    assert exact > prefix > substr > miss == 0


def test_render_palette_carries_wordmark_and_match(catalog):
    results = rank("stats", catalog, limit=5)
    buf = io.StringIO()
    Console(file=buf, force_terminal=False, width=100).print(
        render_palette("stats", results, selected=0))
    out = buf.getvalue()
    assert "pmb" in out          # ✦ pmb wordmark in the panel title
    assert "stats" in out        # the matched command is listed
    assert "type to find" in out  # the keymap hint


def test_render_palette_handles_no_results():
    buf = io.StringIO()
    Console(file=buf, force_terminal=False, width=100).print(
        render_palette("zzzz", [], selected=0))
    assert "no match" in buf.getvalue()


# ── interactive loop (headless: real keyboard mocked, no TTY needed) ──────────
# run_palette() reads one key at a time via palette._read_key. We feed a scripted
# sequence and a non-terminal console (Rich Live no-ops off a TTY) to drive the
# state machine end-to-end and assert what Enter returns.
import pmb.cli.palette as palette_mod  # noqa: E402


def _drive(catalog, keys, monkeypatch):
    seq = iter(keys)
    monkeypatch.setattr(palette_mod, "_read_key",
                        lambda: next(seq, "esc"))  # bail on over-read
    console = Console(file=io.StringIO(), force_terminal=False, width=100)
    return palette_mod.run_palette(console, catalog)


def test_loop_type_then_enter_runs_exact(catalog, monkeypatch):
    assert _drive(catalog, [*"stats", "enter"], monkeypatch) == "stats"


def test_loop_type_intent_then_enter(catalog, monkeypatch):
    assert _drive(catalog, [*"warm", "enter"], monkeypatch) == "warmup"


def test_loop_arrow_down_moves_selection(catalog, monkeypatch):
    # empty query -> alpha head; one 'down' selects the second entry.
    assert _drive(catalog, ["down", "enter"], monkeypatch) == rank("", catalog)[1]["name"]


def test_loop_backspace_edits_the_query(catalog, monkeypatch):
    keys = [*"statszz", "backspace", "backspace", "enter"]  # -> "stats"
    assert _drive(catalog, keys, monkeypatch) == "stats"


def test_loop_esc_returns_none(catalog, monkeypatch):
    assert _drive(catalog, ["esc"], monkeypatch) is None


def test_loop_enter_on_no_match_returns_none(catalog, monkeypatch):
    assert _drive(catalog, [*"zzzq", "enter"], monkeypatch) is None


# ── live_select: the in-place arrow menu reused by `pmb setup` (embedder /
# offline-brain choice). Same headless drive: mock _read_key, non-TTY console. ──
from pmb.cli.palette import live_select  # noqa: E402

_SEL_ROWS = [("Light", "~0.5 GB", "+ tiny\n- weak"),
             ("Balanced", "~1.1 GB", "+ better\n- 2x RAM"),
             ("Best", "~2 GB", "+ sharpest\n- heavy")]
_SEL_H = ["Option", "RAM", "Plus / minus"]


def _drive_select(keys, monkeypatch, default=0):
    seq = iter(keys)
    monkeypatch.setattr(palette_mod, "_read_key", lambda: next(seq, "esc"))
    console = Console(file=io.StringIO(), force_terminal=False, width=100)
    return live_select(console, title="memory model", subtitle="pick",
                       headers=_SEL_H, rows=_SEL_ROWS, default=default)


def test_select_arrows_then_enter(monkeypatch):
    assert _drive_select(["down", "down", "enter"], monkeypatch) == 2


def test_select_up_clamps_at_top(monkeypatch):
    assert _drive_select(["up", "up", "enter"], monkeypatch) == 0


def test_select_down_clamps_at_bottom(monkeypatch):
    assert _drive_select(["down", "down", "down", "down", "enter"], monkeypatch) == 2


def test_select_number_quick_picks(monkeypatch):
    assert _drive_select(["3"], monkeypatch) == 2


def test_select_esc_returns_none(monkeypatch):
    assert _drive_select(["esc"], monkeypatch) is None


def test_select_honors_default_start(monkeypatch):
    # default=1, immediate Enter -> 1 (no movement)
    assert _drive_select(["enter"], monkeypatch, default=1) == 1


def test_select_empty_rows_returns_none(monkeypatch):
    console = Console(file=io.StringIO(), force_terminal=False, width=100)
    assert live_select(console, title="t", subtitle="s",
                       headers=_SEL_H, rows=[], default=0) is None
