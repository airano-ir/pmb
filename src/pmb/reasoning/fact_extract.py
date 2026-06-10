"""
Write-time atomic fact extraction (mem0-style, no LLM).

When the user writes a long message like:

    "Today I met Alice at the coffee shop. She's the new tech lead at Stripe
    and lives in Berlin. We discussed onboarding for the Q3 hire."

…retrieval on "where does Alice live?" or "who is the new Stripe tech lead?"
performs much better when those atomic facts exist as standalone, indexable
events alongside the original — not buried inside a 3-sentence paragraph
whose embedding averages all of them out.

This module does the no-LLM version of mem0's fact decomposition:

  - Sentence-level split (handles . ! ? and Russian variants).
  - Per-sentence "atomic enough?" filter (length, single-clause).
  - Subject-verb pattern matchers for high-precision extraction:
        * "X lives in Y", "X is the Z at W", "X owns Y", "X leads Y"
        * "We use X for Y", "We chose X over Y because Z"
        * "X is Y years old", "X was founded in Y"
  - Result: a list of `AtomicFact(content, kind, confidence)`.

Each atomic fact is recorded as a sibling event with metadata.parent_ulid
pointing back to the source. Recall ranks them like any other event (BM25
+ vector + PAMVR), so a question about Alice's location pulls the short
"Alice lives in Berlin" fact, not the 3-sentence wall.

Triggered from `record_batch` when:
  - content has >= MIN_LEN_FOR_EXTRACT chars
  - AND splits into >= MIN_SENTENCES sentences
  - AND `consolidate.write_atomic_facts` is True (default OFF to keep
    record_batch fast for the MCP fire-and-forget path; opt-in per workspace).

Cost: ~1-3 ms per message on typical paragraph. Pure regex, no model load.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


MIN_LEN_FOR_EXTRACT = 60       # chars
MIN_SENTENCES = 2              # below 2 sentences, nothing to split
MAX_ATOMIC_FACTS = 10          # safety cap


@dataclass(frozen=True)
class AtomicFact:
    content: str                 # the atomic fact text
    kind: str                    # which pattern fired ('identity', 'location', etc.)
    confidence: float            # 0.0-1.0


# Sentence boundary regex. Latin/Cyrillic/Ukrainian + accented-Latin + Greek
# capitals after a terminator (keeps the capital requirement so abbreviations
# like "U.S. army" don't over-split), PLUS CJK terminators (。！？) which carry
# no following space. EN/RU/UK behaviour is unchanged; other scripts are added.
_SENT_SPLIT = re.compile(
    r"(?<=[.!?])\s+(?=[A-ZА-ЯІЇЄҐÀ-ÖØ-ÞĀ-ŽΆ-Ώ])"
    r"|(?<=[。！？])"
)


# High-precision patterns. Each entry: (regex, kind, template_for_atomic).
# The template uses named groups; we extract subject/object and re-form a
# clean, short, indexable fact.
_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # Identity / role: "X is the (new) tech lead at Y". Role allows 2-4
    # tokens (modifier + role noun). Stops at "and"/"."/"," so we don't
    # bleed into the next clause.
    (re.compile(
        r"(?:^|(?<=[.,])\s+)(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+is\s+(?:the\s+|a\s+|an\s+)?(?P<role>(?:[a-z]+\s+){0,3}(?:lead|head|owner|manager|engineer|developer|designer|director|founder|cto|ceo|cfo))(?:\s+at\s+(?P<org>[A-Z][A-Za-z]+))?\b"),
     "role", "{subj} is {role}{at_org}"),

    # Location: "X lives in Y", "X is based in Y" — subject must be at
    # start of sentence or preceded by "and"/"."/"," to avoid capturing
    # a noun deep inside a clause as the subject.
    (re.compile(
        r"(?:^|(?<=[.,])\s+|\band\s+)(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?|[Hh]e|[Ss]he|[Tt]hey|[Ii])\s+(?:lives|live|is\s+based|resides|moved)\s+in\s+(?P<place>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)\b"),
     "location", "{subj} lives in {place}"),

    # Ownership / management
    (re.compile(
        r"\b(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:owns|manages|leads|heads|runs)\s+(?P<obj>[A-Za-z][A-Za-z\s]+)\b",
        re.IGNORECASE), "ownership", "{subj} leads {obj}"),

    # "We use X for Y" — tech choices. Tool can be multi-word
    # ("Cloud Run", "GitHub Actions") via capitalised continuation.
    (re.compile(
        r"\b[Ww]e\s+(?:use|are\s+using)\s+(?P<tool>[A-Z][A-Za-z0-9\-_.]+(?:\s+[A-Z][A-Za-z0-9\-_.]+)?)(?:\s+for\s+(?P<purpose>[a-z][a-z\s]+?)(?=[.,;]|$))?"),
     "tool_choice",
     "We use {tool}{for_purpose}"),

    # "We chose X over Y" — decisions
    (re.compile(
        r"\b[Ww]e\s+(?:chose|picked|selected)\s+(?P<a>[A-Za-z][A-Za-z0-9\-_.]+)\s+over\s+(?P<b>[A-Za-z][A-Za-z0-9\-_.]+)",
        re.IGNORECASE), "decision",
     "We picked {a} over {b}"),

    # Age / numeric attributes: "X is 30 years old"
    (re.compile(
        r"\b(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+is\s+(?P<n>\d+)\s+years?\s+old",
        re.IGNORECASE), "attribute_age", "{subj} is {n} years old"),

    # Relationship status: "X is single / married / divorced / engaged".
    # No IGNORECASE so the subject must be a real proper noun / pronoun
    # (avoids matching "the file is single-..."). Closed status set.
    (re.compile(
        r"(?:^|(?<=[.,])\s+|\band\s+)(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?|[Hh]e|[Ss]he|[Tt]hey|[Ii])\s+(?:is|am|are)\s+(?P<status>single|married|divorced|engaged|widowed)\b"),
     "relationship_status", "{subj} is {status}"),

    # Origin: "X moved from Y" / "X is from Y" / "X came from Y". Captures
    # where someone is originally from (distinct from current "lives in").
    (re.compile(
        r"(?:^|(?<=[.,])\s+|\band\s+)(?P<subj>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?|[Hh]e|[Ss]he|[Tt]hey|[Ii])\s+(?:moved\s+(?:from|away\s+from)|is\s+from|came\s+from|originally\s+from)\s+(?P<place>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)\b"),
     "origin", "{subj} is from {place}"),

    # ==================================================================
    # RUSSIAN patterns (P0-1 hardening)
    # ==================================================================
    # Identity: "Меня зовут X" / "Его зовут X" / "Я Х"
    (re.compile(
        r"\b(?:меня|его|её|их|нас)\s+зовут\s+(?P<name>[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)",
        re.IGNORECASE), "ru_identity", "Имя: {name}"),
    # "Я работаю X" / "X работает Y"
    (re.compile(
        r"(?:^|(?<=[.,])\s+)(?:я|он|она|они)\s+работа(?:ю|ет|ют)\s+(?P<role>[а-яё][а-яё\s]{2,30}?)(?=[.,;]|\s+(?:в|на|и|но|а)\s+|$)",
        re.IGNORECASE), "ru_role", "Работа: {role}"),
    (re.compile(
        r"(?:^|(?<=[.,])\s+)(?P<subj>[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)\s+работает\s+(?P<role>[а-яё][а-яё\s]{2,30}?)(?=[.,;]|\s+(?:в|на|и|но)\s+|$)"),
     "ru_role_named", "{subj} работает {role}"),
    # Location: "Я живу в X" / "X живёт в Y" / "X переехал в Y".
    # Conjugation: жив-у/-ёшь/-ёт/-ём/-ёте/-ут — \w{1,3} covers all forms.
    (re.compile(
        r"(?:^|(?<=[.,])\s+)(?:я|он|она|они)\s+жив\w{1,3}\s+(?:в|во|на)\s+(?P<place>[А-ЯЁ][а-яёА-ЯЁ\-\s]{1,30}?)(?=[.,;]|\s+(?:и|но|а)\s+|$)",
        re.IGNORECASE), "ru_location", "Живёт в {place}"),
    (re.compile(
        r"(?:^|(?<=[.,])\s+)(?P<subj>[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)\s+жив\w{1,3}\s+(?:в|во|на)\s+(?P<place>[А-ЯЁ][а-яёА-ЯЁ\-\s]{1,30}?)(?=[.,;]|\s+(?:и|но|а)\s+|$)"),
     "ru_location_named", "{subj} живёт в {place}"),
    (re.compile(
        r"(?:^|(?<=[.,])\s+)(?P<subj>[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)\s+(?:переехал[аи]?|перебрался)\s+(?:в|во|на)\s+(?P<place>[А-ЯЁ][а-яёА-ЯЁ\-\s]{1,30}?)(?=[.,;]|$)"),
     "ru_moved", "{subj} переехал в {place}"),
    # Birthday: "Мой день рождения X" / "X июня" / "День рождения - X"
    (re.compile(
        r"(?:^|(?<=[.,])\s+|\b)(?:мой|его|её|их)\s+день\s+рождени[яе]\s*[-—:]?\s*(?P<date>\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)(?:\s+\d{4})?)",
        re.IGNORECASE), "ru_birthday", "День рождения: {date}"),
    # Preference: "Я люблю X" / "Я предпочитаю X"
    (re.compile(
        r"(?:^|(?<=[.,])\s+|\b)(?:я|мне)\s+(?:люблю|нравится|нравятся|предпочитаю)\s+(?P<thing>[а-яёА-ЯЁ][а-яёА-ЯЁ\s]{2,40}?)(?=[.,;]|$)",
        re.IGNORECASE), "ru_preference", "Предпочтение: {thing}"),
    # Family / relations: "Мой друг X" / "Моя жена X"
    (re.compile(
        r"(?:^|(?<=[.,])\s+|\b)(?:мой|моя|мои|её|его)\s+(?P<rel>друг|подруга|жена|муж|сестра|брат|мама|папа|сын|дочь|кот|кошка|собака)\s+(?:\-|—|зовут)?\s*(?P<name>[А-ЯЁ][а-яё]+)",
        re.IGNORECASE), "ru_relation", "{rel}: {name}"),
    # "У меня X" / "У него Y зовут Z" — possession of pet/person/thing
    (re.compile(
        r"(?:^|(?<=[.,])\s+|\b)[уУ]\s+(?:меня|него|неё|нас|них)\s+(?P<rel>кот|кошка|собака|друг|подруга|жена|муж|сын|дочь|сестра|брат)\s+(?:по\s+имени\s+|зовут\s+)?(?P<name>[А-ЯЁ][а-яё]+)",
        re.IGNORECASE), "ru_possession", "{rel}: {name}"),

    # ==================================================================
    # UKRAINIAN patterns (P0-1 hardening)
    # ==================================================================
    # Identity: "Мене звати X" / "Його звуть X"
    (re.compile(
        r"\b(?:мене|його|її|їх|нас)\s+зват[иь]?\s+(?P<name>[А-ЯІЇЄҐ][а-яіїєґ']+(?:\s+[А-ЯІЇЄҐ][а-яіїєґ']+)?)",
        re.IGNORECASE), "uk_identity", "Ім'я: {name}"),
    # "Я працюю X" / "X працює Y"
    (re.compile(
        r"(?:^|(?<=[.,])\s+)(?:я|він|вона|вони)\s+працю(?:ю|є|ють)\s+(?P<role>[а-яіїєґ'][а-яіїєґ'\s]{2,30}?)(?=[.,;]|\s+(?:у|в|на|і|та|але)\s+|$)",
        re.IGNORECASE), "uk_role", "Робота: {role}"),
    (re.compile(
        r"(?:^|(?<=[.,])\s+)(?P<subj>[А-ЯІЇЄҐ][а-яіїєґ']+(?:\s+[А-ЯІЇЄҐ][а-яіїєґ']+)?)\s+працює\s+(?P<role>[а-яіїєґ'][а-яіїєґ'\s]{2,30}?)(?=[.,;]|\s+(?:у|в|на|і|та)\s+|$)"),
     "uk_role_named", "{subj} працює {role}"),
    # Location: "Я живу в X" / "X живе в Y" / "X переїхав до Y".
    # Conjugation: жив-у/-еш/-е/-емо/-ете/-уть.
    (re.compile(
        r"(?:^|(?<=[.,])\s+)(?:я|він|вона|вони)\s+жив\w{1,3}\s+(?:у|в|на)\s+(?P<place>[А-ЯІЇЄҐ][а-яіїєґА-ЯІЇЄҐ'\-\s]{1,30}?)(?=[.,;]|\s+(?:і|та|але)\s+|$)",
        re.IGNORECASE), "uk_location", "Живе у {place}"),
    (re.compile(
        r"(?:^|(?<=[.,])\s+)(?P<subj>[А-ЯІЇЄҐ][а-яіїєґ']+(?:\s+[А-ЯІЇЄҐ][а-яіїєґ']+)?)\s+жив\w{1,3}\s+(?:у|в|на)\s+(?P<place>[А-ЯІЇЄҐ][а-яіїєґА-ЯІЇЄҐ'\-\s]{1,30}?)(?=[.,;]|\s+(?:і|та|але)\s+|$)"),
     "uk_location_named", "{subj} живе у {place}"),
    (re.compile(
        r"(?:^|(?<=[.,])\s+)(?P<subj>[А-ЯІЇЄҐ][а-яіїєґ']+(?:\s+[А-ЯІЇЄҐ][а-яіїєґ']+)?)\s+(?:переїхав|переїхала|переїхали|перебрався)\s+(?:до|в|у)\s+(?P<place>[А-ЯІЇЄҐ][а-яіїєґА-ЯІЇЄҐ'\-\s]{1,30}?)(?=[.,;]|$)"),
     "uk_moved", "{subj} переїхав до {place}"),
    # Birthday: "Мій день народження X"
    (re.compile(
        r"(?:^|(?<=[.,])\s+|\b)(?:мій|його|її)\s+день\s+народженн[яі]\s*[-—:]?\s*(?P<date>\d{1,2}\s+(?:січня|лютого|березня|квітня|травня|червня|липня|серпня|вересня|жовтня|листопада|грудня)(?:\s+\d{4})?)",
        re.IGNORECASE), "uk_birthday", "День народження: {date}"),
    # Preference: "Я люблю X" / "Мені подобається X"
    (re.compile(
        r"(?:^|(?<=[.,])\s+|\b)(?:я\s+люблю|мені\s+подобає(?:ться|ться)|мені\s+подобаються)\s+(?P<thing>[а-яіїєґ'А-ЯІЇЄҐ][а-яіїєґ'А-ЯІЇЄҐ\s]{2,40}?)(?=[.,;]|$)",
        re.IGNORECASE), "uk_preference", "Вподобання: {thing}"),
    # Family / relations
    (re.compile(
        r"(?:^|(?<=[.,])\s+|\b)(?:мій|моя|мої|її|його)\s+(?P<rel>друг|подруга|дружина|чоловік|сестра|брат|мама|тато|син|донька|кіт|кішка|собака)\s+(?:\-|—|звати|звуть)?\s*(?P<name>[А-ЯІЇЄҐ][а-яіїєґ']+)",
        re.IGNORECASE), "uk_relation", "{rel}: {name}"),
]


def split_sentences(text: str) -> list[str]:
    """Split into sentences, EN + RU aware."""
    if not text:
        return []
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _clean(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip(" ,.!?;:")


def _atomic_from_match(m: re.Match, kind: str, template: str) -> Optional[AtomicFact]:
    """Build an AtomicFact from a regex match using a template."""
    try:
        d = {k: _clean(v) for k, v in m.groupdict().items() if v}
        # Synthesise optional bits
        d["at_org"] = (" at " + d["org"]) if d.get("org") else ""
        d["for_purpose"] = (" for " + d["purpose"]) if d.get("purpose") else ""
        text = template.format(**{**{k: "" for k in (
            "subj", "role", "org", "place", "obj", "tool", "purpose",
            "a", "b", "n", "at_org", "for_purpose", "status",
            # RU/UK named groups
            "name", "date", "thing", "rel",
        )}, **d}).strip()
        text = re.sub(r"\s+", " ", text)
        # Quality filter: avoid mostly-empty extractions
        # Min 6 chars / 2 words — short atoms like "Имя: Алексей" or
        # "Живёт в Киеве" are valid. Cuts off truly noisy single-word fragments.
        if len(text) < 6 or len(text.split()) < 2:
            return None
        # Bound length (no full paragraphs masquerading as atomic facts)
        if len(text) > 140:
            return None
        # Capitalise first letter for cleanliness
        text = text[0].upper() + text[1:]
        return AtomicFact(content=text, kind=kind, confidence=0.85)
    except Exception:
        return None


def extract_atomic_facts(
    text: str,
    min_len: int = MIN_LEN_FOR_EXTRACT,
    min_sentences: int = MIN_SENTENCES,
    max_facts: int = MAX_ATOMIC_FACTS,
) -> list[AtomicFact]:
    """Run the pattern bank over a paragraph and return atomic facts.

    Returns an empty list when:
      - text shorter than `min_len` chars
      - text splits into fewer than `min_sentences` sentences
      - no patterns fire (most of the time on actual chat)
    """
    if not text or len(text) < min_len:
        return []
    sentences = split_sentences(text)
    if len(sentences) < min_sentences:
        return []

    facts: list[AtomicFact] = []
    seen_texts: set[str] = set()
    for sent in sentences:
        for pat, kind, template in _PATTERNS:
            for m in pat.finditer(sent):
                af = _atomic_from_match(m, kind, template)
                if af is None:
                    continue
                key = af.content.lower()
                if key in seen_texts:
                    continue
                seen_texts.add(key)
                facts.append(af)
                if len(facts) >= max_facts:
                    return facts
    return facts
