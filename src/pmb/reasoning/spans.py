"""C1 - universal value-span detector (no language data).

Extraction needs candidate VALUE spans (a city, an employer, a name, a number)
out of a sentence, in ANY language, with no per-language regex. Two universal
sources, both already proven script-agnostic in Phase L:

  * proper-noun spans - a capitalised, 3+ char token whose first letter is upper
    and the rest lower (Unicode `str.isupper/islower`, so Kyiv / München / Athina
    all qualify; acronyms like NASA do not). Reuses pamvr's tokenizer + the
    not-proper function-word set. Case + inflection are PRESERVED ("Kieve", not
    "kiev") because C2 injects the raw span into an English hypothesis and the
    keyed store clusters inflected variants by cosine.
  * number/date spans - a universal digit run (`42`, `2026-06-12`, `10:30`).

Identifier-shaped tokens (`record_batch`, `qwen2.5`, `gpt4`) are excluded so a
code token can never become a city - the normal case for this user's code-mixed
messages (RU prose + EN identifiers).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from pmb.core.text_match import STOPWORDS
from pmb.reasoning.pamvr import _NOT_PROPER, _PN_TOKEN_RE


@dataclass(frozen=True)
class Span:
    text: str          # RAW surface form (case + inflection preserved)
    kind: str          # "proper" | "number"


_NUM_RE = re.compile(r"\d[\d .,:/\-]*")


def looks_like_identifier(text: str) -> bool:
    """A code identifier, not a natural-language value: underscores, a
    letters+digits mix (qwen2.5, gpt4, h100), or a dotted path (file.py).
    Pure numbers ("42", "2026") are NOT identifiers - they are number spans."""
    if "_" in text:
        return True
    has_alpha = any(c.isalpha() for c in text)
    has_digit = any(c.isdigit() for c in text)
    if has_alpha and has_digit:
        return True
    if "." in text and has_alpha:
        return True
    return False


def _proper_spans(sentence: str) -> list[Span]:
    out: list[Span] = []
    for m in _PN_TOKEN_RE.finditer(sentence or ""):
        raw = m.group(0).strip("'")
        if len(raw) < 3:
            continue
        # Capitalised word: first upper, rest lower (Unicode-aware) - same shape
        # as pamvr._extract_proper_nouns, but we keep the RAW case.
        if not (raw[0].isupper() and raw[1:].islower()):
            continue
        low = raw.lower()
        if low in _NOT_PROPER or low in STOPWORDS:
            continue
        out.append(Span(raw, "proper"))
    return out


def _number_spans(sentence: str) -> list[Span]:
    out: list[Span] = []
    for m in _NUM_RE.finditer(sentence or ""):
        t = m.group(0).strip(" .,:/-")
        if t:
            out.append(Span(t, "number"))
    return out


def value_spans(sentence: str, max_spans: int = 8) -> list[Span]:
    """All candidate value spans in `sentence`, de-duplicated, identifiers
    removed, capped. Proper-noun spans first (the common keyed-fact values),
    then numbers."""
    out: list[Span] = []
    seen: set[tuple[str, str]] = set()
    for sp in _proper_spans(sentence) + _number_spans(sentence):
        if looks_like_identifier(sp.text):
            continue
        key = (sp.text.lower(), sp.kind)
        if key in seen:
            continue
        seen.add(key)
        out.append(sp)
        if len(out) >= max_spans:
            break
    return out
