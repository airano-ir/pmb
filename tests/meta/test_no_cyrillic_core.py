"""L5 — zero-Cyrillic ratchet gate for the core.

Phase L removed all Russian/Ukrainian PROSE (comments, docstrings, agent-facing
instructions, examples, CLI templates) from `src/pmb`. What remains is a small,
explicit allowlist of files that still hold FUNCTIONAL RU/UK language DATA —
verb stems, stopword sets, and regex patterns that MATCH Russian/Ukrainian user
text. Removing those means relocating the data into the `pmb/lang` packs (the
L1 data→pack task, which is parity-critical because it touches recall) — NOT
translating them in place (that would break Russian recall for the user).

This test:
  * FAILS if Cyrillic appears in any `src/pmb/**.py` NOT on the allowlist —
    so prose Cyrillic can never creep back in.
  * Is a RATCHET: as L1 moves a file's data into the lang packs, delete that
    file from `_DATA_ALLOWLIST`. The goal is a MINIMAL allowlist — only files
    whose RU/UK data genuinely cannot live in the off-by-default packs.
"""
from __future__ import annotations

import re
from pathlib import Path

_CYRILLIC = re.compile(r"[Ѐ-ӿԀ-ԯⷠ-ⷿꙀ-ꚟ]")
_SRC = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "src" / "pmb"

# L1 COMPLETE (2026-06-10): every RU/UK matching datum — verb stems, stopword
# sets, intent/heading/relation regexes, fact-extraction templates — was moved
# into pmb/lang/packs/{ru,uk}.yaml (active-by-default), merged back in by each
# module from an English inline floor; relocations pinned by
# tests/test_regex_parity.py.
#
# The ONE deliberate exception is the correction/profanity floor: it MUST match
# Russian/Ukrainian frustration out-of-the-box on a DEFAULT install, but the
# lang-pack mechanism is OFF by default (G3) and profanity does not "self-heal"
# via ALD — so that lexicon stays inline (see the module's own docstring). It is
# FUNCTIONAL matching DATA, not prose, so it is allowlisted, not translated. Any
# OTHER Cyrillic in src/pmb (prose, or new inline data) is still a hard failure.
_DATA_ALLOWLIST: set[str] = {"hooks/correction_capture.py"}


def _cyrillic_files() -> dict[str, int]:
    out: dict[str, int] = {}
    for f in _SRC.rglob("*.py"):
        rel = f.relative_to(_SRC).as_posix()
        n = sum(1 for ln in f.read_text(encoding="utf-8").splitlines()
                if _CYRILLIC.search(ln))
        if n:
            out[rel] = n
    return out


def test_no_cyrillic_outside_data_allowlist():
    offenders = {k: v for k, v in _cyrillic_files().items()
                 if k not in _DATA_ALLOWLIST}
    assert not offenders, (
        "Cyrillic (likely untranslated prose) found outside the data "
        f"allowlist — translate it to English: {offenders}"
    )


def test_allowlist_has_no_dead_entries():
    # keep the ratchet honest: an allowlisted file that no longer has any
    # Cyrillic must be removed from the list (its L1 relocation is done).
    have = _cyrillic_files()
    dead = sorted(p for p in _DATA_ALLOWLIST if p not in have)
    assert not dead, (
        f"these files are clean now — remove them from _DATA_ALLOWLIST: {dead}"
    )
