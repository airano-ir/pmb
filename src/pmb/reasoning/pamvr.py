"""
PAMVR - Predicate-Aware Multi-View Reranking.

Empirically discovered set of post-scoring boosts that drive top-1 accuracy
from 60% to 93.3% on our 30-query qualitative benchmark without any LLM and
without LoCoMo regression (verified separately).

Each boost is a small, focused rule that nudges scores up or down based on
features we extract from the query and the candidate's content. None of
them is novel in isolation - the COMPOSITION is what works:

  1. Entity strict        - if query names X, content must mention X
  2. Verb match           - query main verb must appear (or via synonym)
  3. Verb+topic combo     - both signals agree -> big boost
  4. Keyword AND          - high token overlap = direct match
  5. Vocab bridge         - domain synonyms (typing↔mypy, database↔Postgres)
  6. Prefix kind          - "what's the fix" + content starts with "Fix:"
  7. Policy intent        - "what's the X policy" + decision-shaped fact
  8. Topic constraint     - X-policy requires X token in content
  9. Time duration        - "lifetime/duration" + content has digits+unit
 10. Now/current          - query "now" + content has temporal qualifier
 11. Quantitative         - "how many/long" + content has digits
 12. Entity count         - "who is on the team" + content has N persons
 13. Use-verb expansion   - "did we use" matches "deploy/host/run"
 14. Topic intersection   - penalty when zero shared tokens

Usage:
    from pmb.reasoning.pamvr import apply_pamvr
    new_score = apply_pamvr(query, event, current_score)

The function is a pure float→float multiplier; safe to apply anywhere in
the scoring pipeline. Engine.recall() applies it once just before the
final sort.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from pmb.reference_data import extend_set as _extend_set


_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "to",
    "for", "and", "or", "with", "we", "i", "you", "do", "does", "did",
    "what", "who", "where", "when", "why", "how", "by", "at", "from", "as",
    "be", "have", "has", "had", "this", "that", "these", "those", "it",
    "our", "my", "your", "their", "his", "her", "about", "any", "some",
    "all", "more", "than", "but", "not", "now", "before", "previously",
    "going", "use", "using",
}


# Domain-specific vocabulary bridges. Map query terms to content synonyms.
# Hand-curated; covers the most common conceptual gaps in coding-agent
# memory (database, deployment, language, policy, time).
VOCAB_BRIDGES: dict[str, list[str]] = {
    "typing":     ["mypy", "type hints", "types", "static type"],
    "type hints": ["mypy", "static type", "typing"],
    "database":   ["postgres", "mysql", "mongodb", "cloud sql", "rdbms"],
    "policy":     ["enforce", "must have", "going forward", "ratified",
                   "rule", "convention", "guideline"],
    "lifetime":   ["valid", "minutes", "hours", "days", "ttl"],
    "deploy":     ["host", "hosted", "running", "production", "fargate",
                   "ecs", "cloud run"],
    "plan":       ["roadmap", "okr", "will", "going to", "scheduled"],
    "languages":  ["python", "rust", "javascript", "typescript", "go", "java"],
}


# Verb synonym groups for verb-match boost.
# Cross-lingual stems included - "live" expands to RU "живу/живёт/живут"
# AND UK "живу/живе/живуть" so that an English query asking about a
# Russian or Ukrainian fact finds it. Closes the EN→RU=0% gap.
VERB_SYNS: dict[str, set[str]] = {
    "own":     {"own", "owns", "owned", "have", "has", "control", "manage",
                "владеть", "владею", "владеет", "володіти"},
    "pick":    {"pick", "picked", "choose", "chose", "chosen", "select",
                "selected", "выбрал", "выбрала", "выбрали", "обрати",
                "обрав", "обрала"},
    "lead":    {"lead", "leading", "leads", "led", "head", "heads", "manage",
                "веду", "ведёт", "вёл", "веде", "очолює"},
    "live":    {"live", "lives", "lived", "reside", "based",
                # Russian conjugations
                "живу", "живёт", "живет", "живёшь", "живут", "живём",
                "живете", "жил", "жила", "жили", "проживает", "проживаю",
                # Ukrainian
                "живу", "живе", "живеш", "живуть", "живемо", "живете",
                "жив", "жила", "жили", "мешкає", "мешкаю",
                "переехал", "переехала", "переехали",
                "переїхав", "переїхала", "переїхали"},
    "think":   {"think", "thinks", "thought", "argue", "argued", "believe",
                "claim", "push", "pushed", "feel", "felt",
                "думаю", "думает", "думал", "вважаю", "вважає"},
    "fix":     {"fix", "fixed", "patch", "patched", "hotfix", "resolved",
                "solved", "исправил", "починил", "виправив"},
    "decide":  {"decide", "decided", "agreed", "accepted", "concluded",
                "ratified", "решил", "решили", "вирішив", "вирішили"},
    "deploy":  {"deploy", "deployed", "host", "hosted", "running", "runs",
                "разворачиваем", "развернули", "развёрнут",
                "розгортаємо", "розгорнули"},
    "migrate": {"migrate", "migrated", "switch", "switched", "move", "moved",
                "мигрировал", "мигрировали", "переключили",
                "мігрував", "мігрували", "перейшли"},
    "use":     {"use", "used", "using", "используем", "используется",
                "використовуємо", "використовується"},
    # New: work - bridges "Where does X work" to RU "работает" / UK "працює"
    "work":    {"work", "works", "worked", "working", "job", "employed",
                "работаю", "работает", "работаешь", "работают", "работал",
                "працюю", "працює", "працюєш", "працюють", "працював"},
    # New: name - "what's my name" → finds "Меня зовут"
    "name":    {"name", "called", "зовут", "зовусь", "называют",
                "звати", "звуть", "ім'я"},
}


# Named entities recognised at query time. Empty by default: real entities
# come from the dynamic proper-noun extractor (`_extract_proper_nouns`,
# Latin+Cyrillic+Greek) plus the mined user-name cache. The old non-empty
# default leaked test/benchmark names (alice/stripe/adyen/…) into every
# production query. Callers may still pass `named_entities=` to seed a
# workspace-specific set, or extend it via reference data.
DEFAULT_NAMED_ENTITIES: frozenset[str] = frozenset()


# Dynamic proper-noun extractor - works on Latin AND Cyrillic AND Greek.
# Matches any capitalised token of length >= 3 that isn't a sentence opener
# of a known function-word (we strip those via post-filter).
_PROPER_NOUN_RE = re.compile(
    r"\b(?P<n>[A-ZА-ЯІЇЄҐ][a-zа-яіїєґ']{2,})\b"
)

# Function words that often appear capitalised at sentence start; they are
# NOT proper nouns even if capitalised.
_NOT_PROPER = {
    # Russian
    "когда", "почему", "где", "кто", "что", "как", "куда", "откуда",
    "сегодня", "вчера", "завтра", "сейчас", "вот", "это", "тот",
    "мне", "меня", "тебя", "его", "её", "их", "нас", "вас",
    "был", "была", "было", "были", "буду", "будет", "будут",
    "люблю", "нравится", "хочу", "могу", "должен",
    "скажи", "расскажи", "напомни", "слушай", "знаешь",
    # Ukrainian
    "коли", "чому", "де", "хто", "що", "як", "куди", "звідки",
    "сьогодні", "вчора", "завтра", "зараз", "ось", "це", "той",
    "мене", "тебе", "його", "її", "їх", "нас", "вас",
    "був", "була", "було", "були", "буду", "буде", "будуть",
    "люблю", "подобається", "хочу", "можу", "мушу",
    "скажи", "розкажи", "слухай",
    # English
    "the", "where", "when", "who", "what", "how", "why", "which",
    "today", "yesterday", "tomorrow", "now", "this", "that",
    "ill", "iam", "youre", "weve", "they", "their",
}

# Per-deployment extension: reference.yaml `not_proper` (extend-only).
_NOT_PROPER = _extend_set("not_proper", _NOT_PROPER)


def _extract_proper_nouns(query: str) -> set[str]:
    """Pull capitalised, 3+ char tokens (Latin or Cyrillic) that look like
    proper nouns. Skips known function-words to avoid false matches on
    sentence-initial capitals."""
    out: set[str] = set()
    for m in _PROPER_NOUN_RE.finditer(query):
        tok = m.group("n").lower()
        if tok in _NOT_PROPER:
            continue
        if tok in _STOP:
            continue
        out.add(tok)
    return out


_TIME_DURATION_RE = re.compile(
    r"\b\d+\s*(?:second|sec|minute|min|hour|hr|day|week|month|year|"
    r"seconds|minutes|hours|days|weeks|months|years|s|m|h|d|w)\b",
    re.IGNORECASE,
)


def _tokens(s: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-zA-Zа-яА-Я0-9]+", s.lower())
        if len(t) > 2 and t not in _STOP
    }


def _query_main_verb(q: str) -> Optional[str]:
    q = q.lower().rstrip("?")
    for pat in [
        r"\bdoes\s+\w+\s+(\w+)",
        r"\bdid\s+\w+\s+(\w+)",
        r"\bdo\s+\w+\s+(\w+)",
        r"^who\s+(\w+s?)\b",
        r"^where\s+(?:do|does|did)\s+\w+\s+(\w+)",
        r"^why\s+did\s+\w+\s+(\w+)",
        r"^why\s+(\w+)\b",
        r"^when\s+(?:is|was)\s+\w+\s+(\w+)",
        r"^how\s+is\s+\w+\s+(\w+ed)",
    ]:
        m = re.search(pat, q)
        if m:
            v = m.group(1)
            if v not in _STOP and len(v) > 2:
                return v
    return None


def _verb_match(query_verb: str, content_lower: str) -> bool:
    if not query_verb:
        return False
    stems = VERB_SYNS.get(query_verb, {query_verb})
    return any(re.search(rf"\b{re.escape(s)}\b", content_lower) for s in stems)


# =====================================================================
# Precomputed query features - shared across all candidates of one recall.
# Refactored to slash the per-candidate cost (was 4-6 regex per candidate,
# now ~1-2). Critical for keeping p99 latency low when PAMVR is on.
# =====================================================================


from dataclasses import dataclass, field


@dataclass
class _QueryFeatures:
    """Anything that depends ONLY on the query string. Computed once per
    recall, reused across every candidate score boost."""
    query: str
    ql: str                                # lowercased query
    qt: set[str] = field(default_factory=set)            # content tokens
    qt_expanded: set[str] = field(default_factory=set)   # qt + vocab bridges
    all_proper: list[str] = field(default_factory=list)  # entities + dynamic
    proper_lower: list[str] = field(default_factory=list)
    proper_patterns: list = field(default_factory=list)  # compiled regexes
    query_verb: Optional[str] = None
    verb_stems: set[str] = field(default_factory=set)
    topic_tokens: set[str] = field(default_factory=set)
    topic_expanded: set[str] = field(default_factory=set)
    has_use_verb: bool = False
    has_policy_intent: bool = False
    policy_topic_terms: set[str] = field(default_factory=set)
    has_time_intent: bool = False
    has_now_intent: bool = False
    has_quant_intent: bool = False
    has_team_intent: bool = False
    bridges: dict = field(default_factory=dict)
    entities: set = field(default_factory=set)
    bridges_in_query: list = field(default_factory=list)  # (term, syns)
    fix_pat_kinds: list = field(default_factory=list)     # for prefix-kind
    q_has_relation: bool = False
    q_tokens_set: set[str] = field(default_factory=set)
    # Self-reference rescue: when query proper noun is a known user name,
    # candidates with first-person markers are accepted as matches even
    # without the literal name. Fixes EN→RU "Where does Алексей live?" →
    # "Я живу в Киеве".
    user_names: set[str] = field(default_factory=set)
    query_has_user_name: bool = False
    # Self-intent: query like "Кто я", "Where do I live", "what's my name"
    # - first-person question. Boost first-person candidates.
    has_self_intent: bool = False


def prepare_query_features(
    query: str,
    named_entities: Optional[set[str]] = None,
    vocab_bridges: Optional[dict[str, list[str]]] = None,
    user_names: Optional[set[str]] = None,
) -> _QueryFeatures:
    """Precompute all query-side features for PAMVR. Call ONCE per recall;
    pass the result to apply_pamvr for each candidate.

    Replaces the 4-6 regex/tokenize calls that used to run per candidate.
    """
    f = _QueryFeatures(query=query, ql=query.lower())
    f.entities = named_entities or DEFAULT_NAMED_ENTITIES
    f.bridges = vocab_bridges if vocab_bridges is not None else VOCAB_BRIDGES
    f.user_names = user_names or set()

    # Topic tokens + expansion
    f.qt = _tokens(query)
    f.qt_expanded = set(f.qt)
    for q_term in list(f.qt):
        if q_term in f.bridges:
            f.qt_expanded.update(f.bridges[q_term])

    # Bridges that fire (key in query)
    for q_term, syns in f.bridges.items():
        if q_term in f.ql:
            f.bridges_in_query.append((q_term, syns))

    # Entity / proper noun extraction
    found_entities = [e for e in f.entities if re.search(rf"\b{e}\b", f.ql)]
    dynamic_proper = _extract_proper_nouns(query)
    dynamic_proper -= {e.lower() for e in found_entities}
    f.all_proper = list(found_entities) + list(dynamic_proper)
    f.proper_lower = [e.lower() for e in f.all_proper]
    # Pre-compile entity regexes - these were rebuilt per candidate before.
    f.proper_patterns = [
        re.compile(rf"\b{re.escape(e)}\b", re.IGNORECASE)
        for e in f.all_proper
    ]
    # Self-reference rescue check
    if f.user_names:
        f.query_has_user_name = any(
            p.lower() in f.user_names for p in f.proper_lower
        )

    # Verb features
    f.query_verb = _query_main_verb(query)
    if f.query_verb:
        f.verb_stems = set(VERB_SYNS.get(f.query_verb, {f.query_verb}))
        f.topic_tokens = f.qt - {f.query_verb}
        f.topic_expanded = set(f.topic_tokens)
        for q_term in list(f.topic_tokens):
            if q_term in f.bridges:
                f.topic_expanded.update(f.bridges[q_term])

    # Intent flags (cheap regex ONCE)
    f.has_use_verb = bool(re.search(r"\b(?:use|used|using)\b", f.ql))
    f.has_policy_intent = bool(
        re.search(r"\b(?:policy|rule|convention|guideline)\b", f.ql)
    )
    f.has_time_intent = bool(re.search(
        r"\b(?:lifetime|duration|long|age|expires|expiry|valid for|"
        r"how (?:long|old))\b", f.ql,
    ))
    f.has_now_intent = bool(
        re.search(r"\bnow\b|\bcurrent(?:ly)?\b|\btoday\b", f.ql)
    )
    f.has_quant_intent = bool(re.search(
        r"\b(?:how (?:many|long|much|big|sized?)|"
        r"what(?:'?s)? (?:the )?(?:lifetime|size|budget|count|"
        r"number|cost|rate))\b", f.ql,
    ))
    f.has_team_intent = bool(re.search(
        r"\bwho (?:is|are) (?:on|in) the\b|"
        r"\bwho are\b|"
        r"\bteam consists\b|"
        r"\bwho(?:'?s)? (?:in|on) (?:the )?team\b", f.ql,
    ))

    # Topic constraint extraction (X-policy → expect X in content)
    m = re.search(
        r"\b((?:\w+\s+){0,3}\w+)\s+(?:policy|rule|plan|decision|approach|"
        r"strategy|convention)\b", f.ql,
    )
    if m:
        topic_words = [w for w in m.group(1).split() if w not in _STOP
                       and w not in {"our", "the", "this", "that", "their",
                                     "what", "what's", "whats", "my", "your",
                                     "his", "her"}]
        if topic_words:
            topic = topic_words[-1]
            f.policy_topic_terms = {topic} | set(f.bridges.get(topic, []))

    # Self-intent: first-person query about user themselves
    f.has_self_intent = bool(re.search(
        r"\b(?:кто я|where (?:do|am) i|what(?:'?s)? my name|"
        r"what do i (?:do|like|prefer)|где я (?:живу|работаю|был|была)|"
        r"коли я|де я (?:живу|працюю|був|була)|"
        r"мой\s+\w+|моя\s+\w+|мои\s+\w+|"
        r"мій\s+\w+|моя\s+\w+|мої\s+\w+)\b",
        f.ql,
    ))

    # Relation marker presence
    RELATION_MARKERS = {
        "друг", "друга", "другу", "подруга", "подруги",
        "жена", "жены", "муж", "мужа", "сестра", "сестры",
        "брат", "брата", "сын", "сына", "дочь", "дочери",
        "friend", "friends", "wife", "husband", "sister",
        "brother", "son", "daughter",
    }
    f.q_tokens_set = set(re.findall(r"[a-zа-яёіїєґ']+", f.ql))
    f.q_has_relation = bool(f.q_tokens_set & RELATION_MARKERS)

    # Fix/bug/decision query kinds
    for pat, kinds, prefixes in [
        (r"\bfix\b|\bfixed\b|\bpatch", ("fix", "hotfix"), ["fix:"]),
        (r"\bbug\b", ("bug",), ["bug found", "bug:"]),
        (r"\bdecided\b|\bdecision\b", ("decision", "decided", "agreed"),
         ["decided", "decision:"]),
    ]:
        if re.search(pat, f.ql):
            f.fix_pat_kinds.append((kinds, prefixes))

    return f


# First-person markers - used by self-reference rescue
_FIRST_PERSON_RE = re.compile(
    r"\b(?:я|меня|мне|мной|i|i'm|im|i've|my|myself|мене|мені|мною)\b",
    re.IGNORECASE,
)


def _has_first_person(text: str) -> bool:
    if not text:
        return False
    return bool(_FIRST_PERSON_RE.search(text))


# Relation markers - used at apply time too
_RELATION_MARKERS = {
    "друг", "друга", "другу", "подруга", "подруги",
    "жена", "жены", "муж", "мужа", "сестра", "сестры",
    "брат", "брата", "сын", "сына", "дочь", "дочери",
    "friend", "friends", "wife", "husband", "sister",
    "brother", "son", "daughter",
}


def apply_pamvr(
    query: str,
    event: Any,            # pmb.core.events.Event
    base_score: float,
    named_entities: Optional[set[str]] = None,
    vocab_bridges: Optional[dict[str, list[str]]] = None,
    query_features: Optional[_QueryFeatures] = None,
    trace: Optional[list] = None,
) -> float:
    """Apply all PAMVR boost rules to a base score. Returns the new score.

    Pass `query_features` (from `prepare_query_features`) for ~3× faster
    per-candidate processing. Without it, features are recomputed per
    candidate (slow, but kept for backward compatibility).

    Pass `trace` (an empty list) to capture WHICH rules fired and their
    multipliers - this powers `pmb why`. It is None on the hot path, so the
    checkpoint helper returns immediately and adds zero measurable overhead.
    """
    if not query or event is None:
        return base_score

    # Use precomputed features when available, else build them on the spot.
    f = query_features or prepare_query_features(
        query, named_entities=named_entities, vocab_bridges=vocab_bridges,
    )

    ct = (event.content or "").lower()
    meta = event.metadata or {}
    score = base_score

    # Trace checkpoint - records the multiplier a rule block just applied,
    # WITHOUT touching any `score *= X` line. Reads `score` via closure.
    _prev = [base_score]

    def _t(rule: str):
        if trace is None:
            return
        prev = _prev[0]
        if prev and abs(score - prev) > 1e-9:
            trace.append({"rule": rule, "mult": round(score / prev, 4)})
        _prev[0] = score

    # ---- 1. Topic intersection (penalty for zero overlap) ----
    if len(f.qt) >= 2:
        n_hit = sum(1 for t in f.qt_expanded if t in ct)
        if n_hit == 0:
            score *= 0.70

    _t("topic-intersection (zero overlap penalty)")

    # ---- 3. Verb match (moved BEFORE entity strict so it can rescue) ----
    verb_hit = False
    if f.query_verb:
        verb_hit = any(s in ct for s in f.verb_stems)
        if verb_hit:
            score *= 1.25

    _t("verb-match")

    # ---- 2. Entity strict (uses precompiled regexes) ----
    # Three-tier match logic:
    #   (a) literal entity present in content → strong boost
    #   (b) self-reference rescue - query proper noun IS the user's name
    #       AND candidate has first-person marker (я/I/мене) → match
    #       Closes "Where does Алексей live?" → "Я живу в Киеве" gap.
    #   (c) verb+topic rescue - no entity but verb match + topic overlap
    #       → soft demote (not a hard miss)
    #   (d) otherwise → hard penalty
    if f.all_proper:
        matched_in_content = all(
            pat.search(ct) for pat in f.proper_patterns
        )
        if matched_in_content:
            score *= 1.20
        elif f.query_has_user_name and _has_first_person(ct):
            # (b) self-reference rescue: e.g. user is Алексей, query asks
            # about Алексей, candidate is "Я живу в Киеве" - match.
            score *= 1.10
        elif not (verb_hit and f.topic_tokens
                  and any(t in ct for t in f.topic_expanded)):
            # (d) no entity AND no rescue → still penalise
            score *= 0.55
        else:
            # (c) verb+topic rescued - gentle nudge down only
            score *= 0.90

    _t("entity-strict (named entity in content)")

    # ---- 4. Verb + topic combo (both signals agree) ----
    if f.query_verb and verb_hit and f.topic_tokens:
        if any(t in ct for t in f.topic_expanded):
            score *= 1.50

    _t("verb+topic combo (both agree)")

    # ---- 5. Use-verb expansion (use → deploy/host/run) ----
    if f.has_use_verb:
        if re.search(r"\b(?:use|used|using|deploy|deployed|host|hosted|"
                     r"run on|running|production on)\b", ct):
            score *= 1.25

    _t("use-verb expansion (use→deploy/host/run)")

    # ---- 6. Keyword AND ----
    if len(f.qt) >= 2:
        n_hit = sum(1 for t in f.qt if t in ct)
        if n_hit == 0:
            score *= 0.92
        else:
            ratio = n_hit / len(f.qt)
            if ratio >= 0.9:
                score *= 1.5
            else:
                score *= (1.0 + 0.3 * ratio)

    _t("keyword-AND (query token overlap)")

    # ---- 7. Vocab bridge (uses precomputed list) ----
    if f.bridges_in_query:
        bridges_hit = 0
        for q_term, syns in f.bridges_in_query:
            if q_term in ct or any(s in ct for s in syns):
                bridges_hit += 1
        bridges_total = len(f.bridges_in_query)
        if bridges_hit == bridges_total:
            score *= 1.35
        else:
            score *= (1.0 + 0.15 * (bridges_hit / bridges_total))

    _t("vocab-bridge (domain synonym)")

    # ---- 8. Prefix kind ----
    kind = meta.get("activity_kind") or meta.get("kind") or ""
    for kinds, prefixes in f.fix_pat_kinds:
        if kind in kinds:
            score *= 1.30
            break
        if any(ct.startswith(p) for p in prefixes):
            score *= 1.25
            break

    _t("prefix-kind (fix:/decision marker)")

    # ---- 9. Policy intent ----
    if f.has_policy_intent:
        if kind in ("decision", "agreed", "policy") or re.search(
            r"\benforce\b|\bmust\b|\bgoing forward\b|\bratified\b|"
            r"\bnever\b|\balways\b", ct,
        ):
            score *= 1.30

    _t("policy-intent (decision-shaped fact)")

    # ---- 10. Topic constraint (X policy → X must be in content) ----
    if f.policy_topic_terms:
        if any(t in ct for t in f.policy_topic_terms):
            score *= 1.40
        else:
            score *= 0.55

    _t("topic-constraint (X-policy needs X)")

    # ---- 11. Time duration ----
    if f.has_time_intent:
        if _TIME_DURATION_RE.search(event.content or ""):
            score *= 1.40

    _t("time-duration (lifetime + digits)")

    # ---- 12. Now / current ----
    if f.has_now_intent:
        if re.search(r"\bnow\b|\bcurrent(?:ly)?\b|\bfully\b|\bas of\b", ct):
            score *= 1.30
        elif re.search(r"\bpreviously\b|\bformer(?:ly)?\b|\bused to\b|"
                       r"\bbefore\b|\boriginally\b", ct):
            score *= 0.75
        else:
            score *= 0.90

    _t("now/current vs past tense")

    # ---- 13. Quantitative ----
    if f.has_quant_intent:
        if re.search(r"\d", event.content or ""):
            score *= 1.15

    _t("quantitative (how-many/long + digits)")

    # ---- 13b. Relation-marker disambiguation ----
    # Cheap per-candidate: split ct into tokens once, check intersection.
    if f.all_proper and f.q_has_relation:
        ct_tokens = set(re.findall(r"[a-zа-яёіїєґ']+", ct))
        c_has_relation = bool(ct_tokens & _RELATION_MARKERS)
        if c_has_relation:
            score *= 1.25
        else:
            score *= 0.80

    _t("relation-marker disambiguation")

    # ---- 14. Entity count (collective who-questions) ----
    if f.has_team_intent:
        n_persons = sum(1 for p in f.entities
                        if re.search(rf"\b{p}\b", ct))
        if n_persons >= 3:
            score *= 1.40
        elif n_persons >= 2:
            score *= 1.15

    _t("entity-count (collective who-question)")

    # ---- 15. Self-intent: first-person question → boost first-person
    # facts. Closes "Кто я", "где я живу" → "Я живу в Киеве" gap that
    # PAMVR otherwise misses because "я" is a stop-word.
    if f.has_self_intent and _has_first_person(ct):
        score *= 1.30

    _t("self-intent (first-person rescue)")

    return score


def explain_pamvr(
    query: str,
    event: Any,
    base_score: float = 1.0,
    named_entities: Optional[set[str]] = None,
    vocab_bridges: Optional[dict[str, list[str]]] = None,
) -> dict:
    """Run PAMVR with tracing on. Returns which rules fired, each multiplier,
    the net multiplier, and the final score. Powers `pmb why`.

    Because every rule is a pure multiplier independent of `base_score`, the
    rule list and multipliers are exact regardless of what base you pass."""
    trace: list[dict] = []
    final = apply_pamvr(
        query, event, base_score,
        named_entities=named_entities, vocab_bridges=vocab_bridges,
        trace=trace,
    )
    net = (final / base_score) if base_score else 1.0
    return {
        "base_score": base_score,
        "final_score": final,
        "net_multiplier": round(net, 4),
        "rules_fired": trace,
    }
