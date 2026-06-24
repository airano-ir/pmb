"""Correction capture + repeat guard - turn user pushback into a lesson NOW.

The failure this fixes (observed live, RR/ApplyPilot, June 2026): the user
corrected the agent on the SAME thing 7 times over 6 hours; the lesson was
finally recorded only after the 7th complaint, so the auto-recall injection
had nothing to inject for the hours it mattered. Repetition wasn't "the agent
ignored the lesson" - the lesson did not exist yet.

Two mechanisms, both fired from the UserPromptSubmit hook (auto_recall):

  1. CAPTURE-ON-CORRECTION. Detect that the user is pushing back / frustrated
     / repeating themselves, and record a DRAFT lesson on the FIRST occurrence
     (hybrid: PMB saves the signal immediately, the agent refines it into a
     durable rule). Stage-1 of the lesson funnel - "write it in time".

  2. REPEAT GUARD. When the incoming message strongly overlaps a lesson that
     ALREADY exists, surface it LOUD and at the TOP of the context block, not
     as one line in a wall the agent habituates to.

This module is PURE detection (regex + token overlap, no DB, no model, no
network) so it runs on the cold per-process hook path in well under a
millisecond. The DB side (recording the draft, dedup) lives in the engine
(`Engine.capture_correction`).

Why the markers are inlined here (and Cyrillic, unlike auto_recall.py): a
frustration/profanity lexicon is small, fixed, and high-signal, and it MUST
work out of the box in Russian on a default install. The lang-pack mechanism
auto_recall.py relies on is OFF by default (G3) and self-heals only slowly via
ALD - profanity does not "self-heal". So the floor lives in code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pmb.core.text_match import distinctive_tokens as _distinctive_tokens
from pmb.core.text_match import is_strong as _is_strong

# ── Markers ────────────────────────────────────────────────────────────────
#
# STRONG: any single hit means "the user is correcting / angry". These are
# unambiguous - profanity, explicit "you did this again / I already told you",
# rhetorical "do you even know what you're doing".
_STRONG = re.compile(
    "|".join([
        # profanity / exasperation (ru/uk)
        r"бл[яa]+ть?", r"бля\b", r"\bблин\b", r"нах(?:уй|уя|рен)", r"схуял", r"\bхуй",
        r"пизд", r"\bе[бп]ан", r"\bёбан", r"заеб", r"охуе", r"какого\s+хуя",
        r"задолбал", r"достал[оа]?\b", r"твою\s+мать", r"что\s+за\s+(?:хрень|херня|хуйня)",
        # profanity (en)
        r"\bfuck", r"\bwtf\b", r"\bgoddamn", r"\bthe\s+hell\b", r"\bdamn\s+it\b",
        r"\bbull?shit\b", r"\bcrap\b",
        # explicit "again / how many times / i already said"
        r"те\s*же\s+грабли", r"опять\s+то\s*же", r"снова\s+то\s*же",
        r"сколько\s+раз", r"в\s+któ?рый\s+раз", r"я\s+же\s+(?:сказал|говорил|просил)",
        r"уже\s+(?:говорил|сказал|просил)", r"инструкци\w*\s+\w*\s*(?:слыш|читал|понял)",
        r"\bsame\s+(?:mistake|thing|issue|bug)\s+again", r"\bagain\?+",
        r"\bi\s+(?:already\s+)?(?:told|said|asked)\s+you", r"\bhow\s+many\s+times",
        # rhetorical frustration
        r"ты\s+хоть\s+\w+", r"думаешь\s+что\s+делаешь", r"что\s+ты\s+(?:делаешь|творишь|наделал)",
        r"\bdo\s+you\s+even\b", r"\bwhat\s+are\s+you\s+doing\b",
    ]),
    re.IGNORECASE,
)

# WEAK: needs corroboration (two weak hits, OR one weak + a negated action).
_WEAK_REPEAT = re.compile(
    "|".join([
        r"\bснова\b", r"\bопять\b", r"\bповторно\b", r"\bвновь\b",
        r"\bagain\b", r"\bonce\s+more\b", r"\byet\s+again\b", r"\bstill\b",
        r"всё\s+равно", r"все\s+равно", r"всё\s+ещё", r"все\s+еще",
        r"\bстоп\b", r"\bstop\b", r"\bwait\b", r"\bno+\b,",
        r"почему\s+(?:опять|снова|до\s+сих\s+пор)", r"why\s+(?:again|still)",
    ]),
    re.IGNORECASE,
)

# Negated action - "it did NOT do X". Targeted (not a blanket "не") to avoid
# firing on "не надо / не знаю / don't worry".
_NEG_ACTION = re.compile(
    "|".join([
        r"не\s+заполн", r"не\s+работа", r"не\s+отправ", r"не\s+валидир",
        r"не\s+вижу", r"не\s+нажал", r"не\s+проверил", r"не\s+сделал",
        r"не\s+до\s+конца", r"не\s+так\b", r"не\s+то\b", r"не\s+все\b",
        r"не\s+всё\b", r"\bне\s+должно\b", r"перестал[оа]?\b",
        r"does(?:n'?t|\s+not)\s+(?:work|fill|submit|validate)",
        r"did(?:n'?t|\s+not)\s+(?:fill|submit|work|validate|check)",
        r"\bnot\s+(?:working|filling|filled|validated|submitted)\b",
        r"\bbroke\b", r"\bbroken\b",
    ]),
    re.IGNORECASE,
)


@dataclass
class CorrectionSignal:
    """Outcome of correction detection."""
    is_correction: bool
    severity: str = "weak"          # "strong" | "weak"
    markers: list[str] = field(default_factory=list)


def _caps_shout(message: str, min_alpha: int = 12, ratio: float = 0.7) -> bool:
    """True when the message is mostly UPPERCASE (the user is shouting).

    Counts cased letters (Latin + Cyrillic) and checks the uppercase share.
    Requires a minimum count so a short 'OK' / 'DONE' doesn't register.
    """
    letters = [c for c in message if c.isalpha()]
    if len(letters) < min_alpha:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return (upper / len(letters)) >= ratio


def detect_correction(message: str) -> CorrectionSignal | None:
    """Detect that the user is correcting / frustrated. Returns a
    CorrectionSignal when so, else None.

    Strong if ANY strong marker hits or the message is a CAPS shout.
    Weak (still a correction) if a repeat-marker co-occurs with either a
    negated action or a second weak marker - "снова не заполнило",
    "опять стоп". A lone "снова открой дашборд" stays None (no negation,
    one weak marker only).
    """
    msg = (message or "").strip()
    if len(msg) < 4:
        return None

    markers: list[str] = []
    strong = False

    m = _STRONG.search(msg)
    if m:
        strong = True
        markers.append(f"strong:{m.group(0).strip().lower()[:24]}")

    if _caps_shout(msg):
        strong = True
        markers.append("strong:CAPS_SHOUT")

    weak_hits = [mm.group(0).strip().lower() for mm in _WEAK_REPEAT.finditer(msg)]
    neg = _NEG_ACTION.search(msg)
    if weak_hits:
        markers.extend(f"weak:{w[:16]}" for w in weak_hits[:3])
    if neg:
        markers.append(f"neg:{neg.group(0).strip().lower()[:24]}")

    if strong:
        return CorrectionSignal(True, "strong", markers)

    # Weak path: a repeat marker must be corroborated.
    if weak_hits and (neg or len(weak_hits) >= 2):
        return CorrectionSignal(True, "weak", markers)

    return None


def strong_lesson_matches(
    message: str,
    lessons: list[dict],
    *,
    min_overlap: int = 3,
    min_strong: int = 2,
    limit: int = 3,
) -> list[dict]:
    """Subset of `lessons` whose distinctive tokens strongly overlap the
    message - i.e. "you have hit this exact thing before". Same yardstick as
    followcheck (text_match) so a lesson surfaced loudly here is one that could
    actually be confirmed against the work. Sorted by overlap strength.
    """
    msg_tokens = _distinctive_tokens(message or "")
    if not msg_tokens or not lessons:
        return []
    scored: list[tuple[int, int, dict]] = []
    for L in lessons:
        ltok = _distinctive_tokens(L.get("content", "") or "")
        if not ltok:
            continue
        overlap = msg_tokens & ltok
        if len(overlap) < min_overlap:
            continue
        nstrong = sum(1 for t in overlap if _is_strong(t))
        if nstrong < min_strong:
            continue
        scored.append((nstrong, len(overlap), L))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return [L for _, _, L in scored[:limit]]
