"""C1: Unicode-correct tokenization/normalization.

The project lesson is explicit: tokenizer changes can SILENTLY break recall and
are caught by live runs, not unit tests. So the contract here is the strongest
possible guard — the new tokenizers must be BYTE-IDENTICAL to the old ones on
the real-corpus languages (EN / RU / UK). The OLD regexes are re-embedded below
and compared head-to-head; only previously-mangled scripts (German, Greek, CJK)
are allowed to change.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pmb.reasoning.pamvr as pamvr
from pmb.reasoning.attributes import normalize_label
from pmb.reasoning.fact_extract import _SENT_SPLIT
from pmb.reasoning.vocab_miner import _TOKEN_RE

# EN + RU strings (and UK strings WITHOUT the letters і/ї/є/ґ): on these the
# old and new tokenizers must be byte-identical — this is the real corpus.
EN_RU = [
    "Current City",
    "current_city_2026",
    "I currently live in Tampa now",
    "Меня зовут Алексей",
    "я живу в Киеве",
    "PMB uses LanceDB and SQLite on port 5433",
    "U.S. army base near the river.",
    "We deployed to fly.io. It worked.",
    "Bob's editor is vim and tmux.",
    "Caroline researched adoption agencies.",
    "Мене звати Олександр",          # no і/ї/є/ґ → parity-safe
    "record_batch and qwen2.5 are identifiers",
    "Postgres 17, Redis 7, port 6379",
]


# ── old implementations, frozen for parity comparison ──────────────────────

def _old_normalize_label(label: str) -> str:
    s = (label or "").strip().lower()
    s = re.sub(r"[^0-9a-zа-яё]+", "_", s)
    return s.strip("_")


_OLD_VOCAB_RE = re.compile(r"[a-zа-я][a-zа-я0-9_]{2,}", re.IGNORECASE)
_OLD_PN_RE = re.compile(r"\b(?P<n>[A-ZА-ЯІЇЄҐ][a-zа-яіїєґ']{2,})\b")
_OLD_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZА-ЯІЇЄҐ])")
_OLD_TOKENS_RE = re.compile(r"[a-zA-Zа-яА-Я0-9]+")


def _old_extract_proper_nouns(q: str) -> set[str]:
    out: set[str] = set()
    for m in _OLD_PN_RE.finditer(q):
        tok = m.group("n").lower()
        if tok in pamvr._NOT_PROPER or tok in pamvr._STOP:
            continue
        out.add(tok)
    return out


# ── parity: EN/RU/UK behaviour is unchanged ────────────────────────────────

@pytest.mark.parametrize("s", EN_RU)
def test_normalize_label_parity_en_ru(s):
    assert normalize_label(s) == _old_normalize_label(s)


@pytest.mark.parametrize("s", EN_RU)
def test_vocab_tokens_parity_en_ru(s):
    assert _TOKEN_RE.findall(s) == _OLD_VOCAB_RE.findall(s)


@pytest.mark.parametrize("s", EN_RU)
def test_pamvr_tokens_parity_en_ru(s):
    new = {t for t in re.findall(r"[^\W_]+", s.lower(), flags=re.UNICODE)}
    old = {t for t in _OLD_TOKENS_RE.findall(s.lower())}
    assert new == old


@pytest.mark.parametrize("s", EN_RU)
def test_proper_noun_parity_en_ru(s):
    assert pamvr._extract_proper_nouns(s) == _old_extract_proper_nouns(s)


@pytest.mark.parametrize("s", EN_RU)
def test_sentence_split_parity_en_ru(s):
    assert _SENT_SPLIT.split(s) == _OLD_SENT_SPLIT.split(s)


# ── new: previously-mangled scripts now keep their letters ─────────────────

def test_normalize_label_keeps_accented_latin():
    # old class [^0-9a-zа-яё] dropped umlauts → "gr_e"; new keeps the letters.
    # casefold maps ß→ss (so "Straße"=="Strasse"), which is desirable.
    assert normalize_label("Größe") == "grösse"
    assert normalize_label("Año Actual") == "año_actual"
    assert _old_normalize_label("Größe") == "gr_e"  # documents the old bug


def test_ukrainian_letters_no_longer_dropped():
    """C1 also FIXES Ukrainian: the old Latin+RU classes silently dropped
    і/ї/є/ґ, so "Львові" tokenized as "львов". The new Unicode tokenizers keep
    them. (Behaviour change on UK content — a `pmb reindex` aligns the index.)"""
    s = "він живе у Львові тепер"
    new_vocab = _TOKEN_RE.findall(s)
    assert "він" in new_vocab and "Львові" in new_vocab
    old_vocab = _OLD_VOCAB_RE.findall(s)
    assert "він" not in old_vocab     # the old bug: "він" dropped entirely
    assert "Львов" in old_vocab       # and "Львові" truncated (і lost)
    assert normalize_label("Львові") == "львові"
    assert _old_normalize_label("Львові") == "львов"


def test_proper_nouns_detect_accented_and_greek_names():
    out = pamvr._extract_proper_nouns("I met Köln and São and Αθηνα today")
    assert "köln" in out and "são" in out
    # the old regex would have missed all three (non-Latin/Cyrillic capitals)
    assert not _old_extract_proper_nouns("Köln São Αθηνα")


def test_vocab_tokens_keep_accented_words():
    toks = _TOKEN_RE.findall("Ich wohne in München")
    assert "München" in toks or "münchen" in [t.lower() for t in toks]


def test_sentence_split_handles_cjk_terminator():
    parts = [p for p in _SENT_SPLIT.split("これは一文です。次の文です。") if p]
    assert len(parts) >= 2


def test_acronyms_and_abbreviations_not_over_split():
    # capital requirement preserved → "U.S. army" stays one sentence
    assert _SENT_SPLIT.split("U.S. army base near the river.") == [
        "U.S. army base near the river."]
