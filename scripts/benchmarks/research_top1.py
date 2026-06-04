"""
Research harness: 200-event workspace + 30 queries, measure top-1 accuracy.

Each experiment patches the recall pipeline via monkey-patches BEFORE
fresh engine creation, so we can iterate fast without touching engine.py.

Pipeline (latest engine.py):
  raw search -> graph -> causation/arcs -> PPR -> SCORING LOOP -> rerank
We monkey-patch the SCORING LOOP via a wrapper that adds extra boosts
to each (h, ev, base, recency) tuple before they get sorted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _bench_data import data_path

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parent.parent.parent / "src"))
sys.path.insert(0, str(_here.parent))

from quality_demo import CHATS, QUERIES


# Manually labeled expected substring(s) that MUST appear (lowercase) in
# the correct top-1 content.
EXPECTED = {
    "Who is the user?":                              ["alex, senior", "user's name"],
    "What languages does Alex use?":                 ["python 3.12", "python and rust",
                                                     "alex uses python"],
    "Where does Alex live?":                         ["lives in kyiv", "alex lives"],
    "Who is on the team?":                           ["team consists", "alex (lead), bob"],
    "What's the sprint length?":                     ["sprint length is two weeks"],
    "Who owns the database?":                        ["bob owns postgres"],
    "Who owns the API design?":                      ["carol owns api design"],
    "Why did we pick Postgres?":                     ["picked postgres", "postgres 15 over mongo"],
    "Why FastAPI?":                                  ["fastapi over flask"],
    "What did we decide about observability":        ["focus first sprint on observability"],
    "Have we hit a JSONB null bug?":                 ["bug found: orders-service crashes",
                                                     "jsonb null bug fixed",
                                                     "another null jsonb"],
    "What was the fix for the JSONB issue?":         ["coalesce", "migration 0042"],
    "What's our JSONB policy now?":                  ["enforce not null on all jsonb",
                                                     "new jsonb columns must"],
    "Where do we store refresh tokens?":             ["httponly cookies"],
    "Why httpOnly cookies?":                         ["httponly prevents xss",
                                                     "store refresh tokens in httponly"],
    "What's the JWT lifetime?":                      ["15 minutes; refresh tokens valid 7 days",
                                                     "jwt access tokens valid"],
    "Why are we using raw asyncpg?":                 ["raw asyncpg path uses prepared",
                                                     "switched to raw asyncpg"],
    "What's our latency budget per endpoint?":       ["p95 latency budget for any single"],
    "How is the connection pool sized?":             ["connection pool size = 2",
                                                     "2 * num_cores + 5"],
    "What did Alice think about typing?":            ["alice argued strongly"],
    "What was Bob's position on type hints?":        ["bob disagreed and pushed for gradual"],
    "What did the team decide about typing?":        ["team accepted carol",
                                                     "accepted carol's"],
    "Where do we deploy now?":                       ["production is now fully on gcp",
                                                     "cloud run, region"],
    "What did we use before?":                       ["using aws ecs fargate",
                                                     "aws ecs"],
    "Why did we migrate to GCP?":                    ["cost analysis", "migrate from aws ecs",
                                                     "decided to migrate"],
    "What was the cost reduction from GCP?":         ["38% lower", "35% cost reduction",
                                                     "~$300/month vs aws"],
    "What's the Q3 roadmap?":                        ["q3 okr", "payment processor v2",
                                                     "q3 roadmap drafted"],
    "Who's leading payments?":                       ["carol will lead the payments"],
    "What's the fraud detection plan?":              ["fraud detection will use",
                                                     "rule-based and ml scoring"],
    "When is the production cut-over scheduled?":    ["cut-over scheduled for next tuesday",
                                                     "production cut-over scheduled"],
}


# ----------------------------------------------------------------------
# Engine setup
# ----------------------------------------------------------------------


def fresh_engine(overrides: dict | None = None):
    from pmb.core.engine import Engine
    tmp_home = Path(tempfile.mkdtemp())
    tmp_ws = Path(tempfile.mkdtemp())
    os.environ["PMB_HOME"] = str(tmp_home)
    eng = Engine(cwd=tmp_ws, pmb_home=tmp_home, rerank_model=None,
                 config_overrides=overrides or {})
    _ = eng.search.model
    return eng


def ingest(eng):
    for chat in CHATS:
        eng.record_batch(chat["items"])
        time.sleep(0.3)
    time.sleep(3.0)


def matches_expected(content: str, expected_keys: list[str]) -> bool:
    c = content.lower()
    return any(k.lower() in c for k in expected_keys)


def evaluate(eng, label: str) -> dict:
    top1 = top3 = total = 0
    fails: list[dict] = []
    for q_text, q_cat in QUERIES:
        exp_keys = None
        for key, val in EXPECTED.items():
            if key.lower()[:30] in q_text.lower() or q_text.lower()[:30] in key.lower():
                exp_keys = val
                break
        if exp_keys is None:
            continue
        total += 1
        pack = eng.recall(q_text, top_k=3)
        if not pack.results:
            fails.append({"q": q_text, "top1": "(empty)", "exp": exp_keys})
            continue
        ok1 = matches_expected(pack.results[0].content, exp_keys)
        ok3 = any(matches_expected(r.content, exp_keys) for r in pack.results)
        if ok1:
            top1 += 1
        if ok3:
            top3 += 1
        if not ok1:
            fails.append({
                "q": q_text, "cat": q_cat,
                "top1": pack.results[0].content[:90],
                "top2": pack.results[1].content[:90] if len(pack.results) > 1 else "",
                "top3": pack.results[2].content[:90] if len(pack.results) > 2 else "",
                "exp": exp_keys[0][:60],
                "in3": ok3,
            })
    return {
        "label": label, "total": total, "top1": top1, "top3": top3,
        "top1_pct": round(100 * top1 / total, 1) if total else 0,
        "top3_pct": round(100 * top3 / total, 1) if total else 0,
        "fails": fails,
    }


# ----------------------------------------------------------------------
# MONKEY-PATCH FRAMEWORK
#
# Each experiment installs a `post_score_hook(query, ev, h, base) -> base`
# that runs after the regular scoring loop but before sort. We patch by
# wrapping the existing scoring loop.
# ----------------------------------------------------------------------


_CUSTOM_HOOK = None


def install_hook(hook):
    """Hook signature: (query, ev, base) -> new_base."""
    global _CUSTOM_HOOK
    _CUSTOM_HOOK = hook
    # Monkey-patch into Engine.recall via a wrapper. We intercept the
    # scored list after sort but before top_k slice.
    from pmb.core import engine as _eng_mod
    if not hasattr(_eng_mod.Engine, "_orig_recall"):
        _eng_mod.Engine._orig_recall = _eng_mod.Engine.recall

    def patched_recall(self, query, top_k=5, **kw):
        # Run original with top_k * 5 to get a larger pool; we'll re-rank
        # via hook then truncate.
        big_k = max(top_k, 15)
        # The original recall's `top_k` controls the final slice; we can't
        # easily intercept inside without huge refactor. Easier approach:
        # call original with top_k=big_k, then re-apply our hook on its
        # results.
        pack = type(self)._orig_recall(self, query, top_k=big_k, **kw)
        if _CUSTOM_HOOK is None:
            pack.results = pack.results[:top_k]
            return pack
        # Re-score
        rescored = []
        for r in pack.results:
            ev = self.events.get_by_ulid(r.ulid)
            if ev is None:
                continue
            new_score = _CUSTOM_HOOK(query, ev, r.score)
            r.score = new_score
            rescored.append(r)
        rescored.sort(key=lambda r: -r.score)
        pack.results = rescored[:top_k]
        return pack

    _eng_mod.Engine.recall = patched_recall


def uninstall_hook():
    global _CUSTOM_HOOK
    _CUSTOM_HOOK = None
    from pmb.core import engine as _eng_mod
    if hasattr(_eng_mod.Engine, "_orig_recall"):
        _eng_mod.Engine.recall = _eng_mod.Engine._orig_recall
        del _eng_mod.Engine._orig_recall


# ----------------------------------------------------------------------
# Hypothesis hooks
# ----------------------------------------------------------------------


_NAMED_ENTITIES = {"alex", "bob", "carol", "dana", "alice", "stripe", "adyen"}

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "to",
    "for", "and", "or", "with", "we", "i", "you", "do", "does", "did",
    "what", "who", "where", "when", "why", "how", "by", "at", "from", "as",
    "be", "have", "has", "had", "this", "that", "these", "those", "it",
    "our", "my", "your", "their", "his", "her", "about", "any", "some",
    "all", "more", "than", "but", "not", "now", "before", "previously",
    "going", "use", "using",
}


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Zа-яА-Я0-9]+", s.lower()) if len(t) > 2 and t not in _STOP}


def _query_main_verb(q: str) -> str | None:
    """Heuristic: extract verb following 'does/did/do' / 'is/was' / 'how'."""
    q = q.lower().rstrip("?")
    # Patterns: "what did X verb", "who verbs the X", "how is X verbed"
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


# Verb synonym groups for boost.
_VERB_SYNS = {
    "own": {"own", "owns", "owned", "have", "has", "control", "manage"},
    "pick": {"pick", "picked", "choose", "chose", "chosen", "select", "selected"},
    "lead": {"lead", "leading", "leads", "led", "head", "heads", "manage"},
    "live": {"live", "lives", "lived", "reside", "based"},
    "think": {"think", "thinks", "thought", "argue", "argued", "believe",
              "claim", "push", "pushed", "feel", "felt"},
    "fix": {"fix", "fixed", "patch", "patched", "hotfix", "resolved", "solved"},
    "decide": {"decide", "decided", "agreed", "accepted", "concluded", "ratified"},
    "deploy": {"deploy", "deployed", "host", "hosted", "running", "runs"},
    "migrate": {"migrate", "migrated", "switch", "switched", "move", "moved"},
    "use": {"use", "used", "using"},
}


def _verb_match(query_verb: str, content_lower: str) -> bool:
    if not query_verb:
        return False
    stems = _VERB_SYNS.get(query_verb, {query_verb})
    return any(re.search(rf"\b{re.escape(s)}\b", content_lower) for s in stems)


def hypo_keyword_AND(query: str, ev, base: float) -> float:
    """Strong multi-token AND: if content contains ALL meaningful query
    tokens, big boost. Partial overlap: small boost."""
    qt = _tokens(query)
    if len(qt) < 2:
        return base
    ct = (ev.content or "").lower()
    n_hit = sum(1 for t in qt if t in ct)
    if n_hit == 0:
        return base * 0.92
    ratio = n_hit / len(qt)
    if ratio >= 0.9:
        return base * 1.5
    return base * (1.0 + 0.3 * ratio)


def hypo_entity_strict(query: str, ev, base: float) -> float:
    """If query has a Named Entity, content MUST contain it (else penalty)."""
    qt_lower = query.lower()
    found_entities = [e for e in _NAMED_ENTITIES if re.search(rf"\b{e}\b", qt_lower)]
    if not found_entities:
        return base
    ct = (ev.content or "").lower()
    if not all(re.search(rf"\b{e}\b", ct) for e in found_entities):
        return base * 0.55  # hard penalty for missing required entity
    return base * 1.20


def hypo_verb_match(query: str, ev, base: float) -> float:
    """Verb alignment between query and content."""
    v = _query_main_verb(query)
    if not v:
        return base
    ct = (ev.content or "").lower()
    if _verb_match(v, ct):
        return base * 1.25
    return base


def hypo_prefix_kind(query: str, ev, base: float) -> float:
    """'What was the fix' → boost activity kind=fix or content starting with
    'Fix:' / 'Bug:' / 'Decision:'."""
    ql = query.lower()
    kind_map = [
        (r"\bfix\b|\bfixed\b|\bpatch", ("fix", "hotfix"), ["fix:"]),
        (r"\bbug\b", ("bug",), ["bug found", "bug:"]),
        (r"\bdecided\b|\bdecision\b", ("decision", "decided", "agreed"),
         ["decided", "decision:"]),
    ]
    for pat, kinds, prefixes in kind_map:
        if re.search(pat, ql):
            meta = ev.metadata or {}
            kind = meta.get("activity_kind") or meta.get("kind") or ""
            if kind in kinds:
                return base * 1.30
            ct = (ev.content or "").lower()
            for p in prefixes:
                if ct.startswith(p):
                    return base * 1.25
    return base


def hypo_now_current(query: str, ev, base: float) -> float:
    """Query contains 'now/currently' → boost content with similar markers."""
    ql = query.lower()
    if not re.search(r"\bnow\b|\bcurrent(?:ly)?\b|\btoday\b", ql):
        return base
    ct = (ev.content or "").lower()
    if re.search(r"\bnow\b|\bcurrent(?:ly)?\b|\bfully\b|\b(?:as|of) today\b", ct):
        return base * 1.20
    if re.search(r"\bpreviously\b|\bformer(?:ly)?\b|\bused to\b|\bbefore\b", ct):
        return base * 0.85
    return base


def hypo_quantitative(query: str, ev, base: float) -> float:
    """Query asks for a number/measurement → boost content with digits."""
    ql = query.lower()
    if not re.search(r"\b(?:how (?:many|long|much|big|sized?)|what(?:'?s)? "
                     r"(?:the )?(?:lifetime|size|budget|count|number|cost|rate))\b", ql):
        return base
    ct = ev.content or ""
    if re.search(r"\d", ct):
        return base * 1.15
    return base


def hypo_entity_count(query: str, ev, base: float) -> float:
    """'Who is on the team' / 'Who are X' → prefer events with multiple person names."""
    ql = query.lower()
    if not re.search(r"\bwho (?:is|are) (?:on|in) the\b|"
                     r"\bwho are\b|"
                     r"\bteam consists\b|"
                     r"\bwho(?:'?s)? (?:in|on) (?:the )?team\b", ql):
        return base
    ct = (ev.content or "").lower()
    n_persons = sum(1 for p in _NAMED_ENTITIES if re.search(rf"\b{p}\b", ct))
    if n_persons >= 3:
        return base * 1.40
    if n_persons >= 2:
        return base * 1.15
    return base


# Time-duration patterns for "lifetime / age / duration" questions.
_TIME_DURATION = re.compile(
    r"\b\d+\s*(?:second|sec|minute|min|hour|hr|day|week|month|year|"
    r"seconds|minutes|hours|days|weeks|months|years|s|m|h|d|w)\b",
    re.IGNORECASE,
)


def hypo_time_duration(query: str, ev, base: float) -> float:
    """Query asks for a duration/lifetime → boost content with explicit time amounts."""
    ql = query.lower()
    if not re.search(r"\b(?:lifetime|duration|long|age|expires|expiry|valid for|"
                     r"how (?:long|old))\b", ql):
        return base
    if _TIME_DURATION.search(ev.content or ""):
        return base * 1.40
    return base


def hypo_topic_constraint(query: str, ev, base: float) -> float:
    """When query has a SPECIFIC topical noun + 'policy/rule/decision/plan',
    require the topical noun (or its synonym) to be in content.

    Bug-fix: previously captured "our JSONB" as topic, now skips leading
    stopwords (our/the/their/etc.) to find the real topical noun.
    """
    ql = query.lower()
    # Capture all words before policy/rule/plan/etc., then strip stopwords
    m = re.search(r"\b((?:\w+\s+){0,3}\w+)\s+(?:policy|rule|plan|decision|approach|"
                  r"strategy|convention)\b", ql)
    if not m:
        return base
    topic_words = [w for w in m.group(1).split() if w not in _STOP
                   and w not in {"our", "the", "this", "that", "their", "what",
                                 "what's", "whats", "my", "your", "his", "her"}]
    if not topic_words:
        return base
    topic = topic_words[-1]  # use the LAST meaningful word as the topic
    topic_terms = {topic} | set(_VOCAB_BRIDGES.get(topic, []))
    ct = (ev.content or "").lower()
    if any(t in ct for t in topic_terms):
        return base * 1.40
    return base * 0.55  # stronger penalty if topic missing


def hypo_verb_topic_combo(query: str, ev, base: float) -> float:
    """Compound boost: when BOTH verb-match AND topic-token are in content."""
    v = _query_main_verb(query)
    if not v:
        return base
    qt = _tokens(query) - {v}
    if not qt:
        return base
    ct = (ev.content or "").lower()
    verb_hit = _verb_match(v, ct)
    qt_expanded = set(qt)
    for q_term in list(qt):
        if q_term in _VOCAB_BRIDGES:
            qt_expanded.update(_VOCAB_BRIDGES[q_term])
    topic_hit = any(t in ct for t in qt_expanded)
    if verb_hit and topic_hit:
        return base * 1.50  # both signals agree — strong boost
    return base


def hypo_now_strong(query: str, ev, base: float) -> float:
    """Stronger now/current handling than hypo_now_current."""
    ql = query.lower()
    has_now = bool(re.search(r"\bnow\b|\bcurrent(?:ly)?\b|\btoday\b", ql))
    if not has_now:
        return base
    ct = (ev.content or "").lower()
    if re.search(r"\bnow\b|\bcurrent(?:ly)?\b|\bfully\b|\bas of\b", ct):
        return base * 1.30
    # Penalize content with past markers
    if re.search(r"\bpreviously\b|\bformer(?:ly)?\b|\bused to\b|\bbefore\b|"
                 r"\boriginally\b", ct):
        return base * 0.75
    return base * 0.90  # mild penalty for facts without temporal qualifier


def hypo_use_verb_expand(query: str, ev, base: float) -> float:
    """'What did we use before' style: 'use' verb should match 'using',
    'used', 'are using', 'were using'."""
    ql = query.lower()
    if not re.search(r"\b(?:use|used|using)\b", ql):
        return base
    ct = (ev.content or "").lower()
    if re.search(r"\b(?:use|used|using|deploy|deployed|host|hosted|"
                 r"run on|running|production on)\b", ct):
        return base * 1.25
    return base


def hypo_ALL_v3(query, ev, base):
    """v2 + time duration + topic constraint + stronger now + use-verb expansion."""
    base = hypo_topic_intersection(query, ev, base)
    base = hypo_entity_strict(query, ev, base)
    base = hypo_verb_match(query, ev, base)
    base = hypo_use_verb_expand(query, ev, base)
    base = hypo_keyword_AND(query, ev, base)
    base = hypo_vocab_bridge(query, ev, base)
    base = hypo_prefix_kind(query, ev, base)
    base = hypo_policy_intent(query, ev, base)
    base = hypo_topic_constraint(query, ev, base)
    base = hypo_time_duration(query, ev, base)
    base = hypo_now_strong(query, ev, base)
    base = hypo_quantitative(query, ev, base)
    base = hypo_entity_count(query, ev, base)
    return base


def hypo_ALL_v4(query, ev, base):
    """v3 + verb+topic compound boost (×1.50 when both signals agree)."""
    base = hypo_topic_intersection(query, ev, base)
    base = hypo_entity_strict(query, ev, base)
    base = hypo_verb_match(query, ev, base)
    base = hypo_verb_topic_combo(query, ev, base)
    base = hypo_use_verb_expand(query, ev, base)
    base = hypo_keyword_AND(query, ev, base)
    base = hypo_vocab_bridge(query, ev, base)
    base = hypo_prefix_kind(query, ev, base)
    base = hypo_policy_intent(query, ev, base)
    base = hypo_topic_constraint(query, ev, base)
    base = hypo_time_duration(query, ev, base)
    base = hypo_now_strong(query, ev, base)
    base = hypo_quantitative(query, ev, base)
    base = hypo_entity_count(query, ev, base)
    return base


# NEW: vocabulary-bridge synonyms (domain-specific knowledge map)
# Maps query terms to content terms that mean the same thing.
_VOCAB_BRIDGES = {
    "typing": ["mypy", "type hints", "types", "static type"],
    "type hints": ["mypy", "static type", "typing"],
    "database": ["postgres", "mysql", "mongodb", "cloud sql", "rdbms"],
    "policy": ["enforce", "must have", "going forward", "ratified", "rule",
               "convention", "guideline"],
    "lifetime": ["valid", "minutes", "hours", "days", "ttl"],
    "deploy": ["host", "hosted", "running", "production", "fargate", "ecs",
               "cloud run"],
    "plan": ["roadmap", "okr", "will", "going to", "scheduled"],
    "languages": ["python", "rust", "javascript", "typescript", "go", "java"],
}


def hypo_vocab_bridge(query: str, ev, base: float) -> float:
    """For query tokens with known content-synonyms, boost events that have
    EITHER the literal token OR the synonyms."""
    ct = (ev.content or "").lower()
    ql = query.lower()
    bridges_hit = 0
    bridges_total = 0
    for q_term, syns in _VOCAB_BRIDGES.items():
        if q_term in ql:
            bridges_total += 1
            if q_term in ct or any(s in ct for s in syns):
                bridges_hit += 1
    if bridges_total == 0:
        return base
    if bridges_hit == bridges_total:
        return base * 1.35
    return base * (1.0 + 0.15 * (bridges_hit / bridges_total))


def hypo_policy_intent(query: str, ev, base: float) -> float:
    """Query 'what's our X policy / what's the X rule' → boost decision/rule events."""
    if not re.search(r"\b(?:policy|rule|convention|guideline)\b", query.lower()):
        return base
    ct = (ev.content or "").lower()
    meta = ev.metadata or {}
    kind = meta.get("activity_kind") or meta.get("kind") or ""
    # Boost: events that ARE policy-shaped
    if kind in ("decision", "agreed", "policy") or re.search(
        r"\benforce\b|\bmust\b|\bgoing forward\b|\bratified\b|\bnever\b|\balways\b",
        ct,
    ):
        return base * 1.30
    return base


def hypo_topic_intersection(query: str, ev, base: float) -> float:
    """Strong constraint: query CONTENT nouns (excluding question words)
    must intersect with content nouns. Heavily penalize content with zero
    intersection."""
    qt = _tokens(query)
    if len(qt) < 2:
        return base
    # Apply vocab bridges
    qt_expanded = set(qt)
    for q_term, syns in _VOCAB_BRIDGES.items():
        if q_term in query.lower():
            qt_expanded.update(syns)
    ct = (ev.content or "").lower()
    n_hit = sum(1 for t in qt_expanded if t in ct)
    if n_hit == 0:
        return base * 0.70
    return base


def hypo_ALL_v2(query, ev, base):
    """All winning hooks + vocab bridge + policy intent + topic intersection."""
    base = hypo_topic_intersection(query, ev, base)
    base = hypo_entity_strict(query, ev, base)
    base = hypo_verb_match(query, ev, base)
    base = hypo_keyword_AND(query, ev, base)
    base = hypo_vocab_bridge(query, ev, base)
    base = hypo_prefix_kind(query, ev, base)
    base = hypo_policy_intent(query, ev, base)
    base = hypo_now_current(query, ev, base)
    base = hypo_quantitative(query, ev, base)
    base = hypo_entity_count(query, ev, base)
    return base


# Composite hooks
def hypo_ABCD(query, ev, base):
    """Entity strict + verb match + keyword AND + prefix kind."""
    base = hypo_entity_strict(query, ev, base)
    base = hypo_verb_match(query, ev, base)
    base = hypo_keyword_AND(query, ev, base)
    base = hypo_prefix_kind(query, ev, base)
    return base


def hypo_ALL(query, ev, base):
    """Entity + verb + AND + prefix + now/current + quantitative + entity-count."""
    base = hypo_entity_strict(query, ev, base)
    base = hypo_verb_match(query, ev, base)
    base = hypo_keyword_AND(query, ev, base)
    base = hypo_prefix_kind(query, ev, base)
    base = hypo_now_current(query, ev, base)
    base = hypo_quantitative(query, ev, base)
    base = hypo_entity_count(query, ev, base)
    return base


# ----------------------------------------------------------------------
# Experiment runners
# ----------------------------------------------------------------------


def run_experiment(name: str, hook, overrides=None):
    uninstall_hook()
    if hook:
        install_hook(hook)
    eng = fresh_engine(overrides)
    ingest(eng)
    r = evaluate(eng, name)
    print(f"  [{name:<22}] top-1 {r['top1']}/{r['total']} = {r['top1_pct']:>5}%   "
          f"top-3 {r['top3']}/{r['total']} = {r['top3_pct']:>5}%")
    try:
        eng.close()
    except Exception:
        pass
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default=data_path("pmb_research_top1.json"))
    args = ap.parse_args()

    print(f"Research harness — {len(EXPECTED)} queries\n")
    all_results = []

    experiments = [
        ("baseline",          None),
        ("verb_topic_combo",  hypo_verb_topic_combo),
        ("topic_constraint",  hypo_topic_constraint),
        ("composite_ALL_v3",  hypo_ALL_v3),
        ("composite_ALL_v4",  hypo_ALL_v4),
    ]
    for name, hook in experiments:
        all_results.append(run_experiment(name, hook))

    # Print best fails for the best experiment
    print()
    best = max(all_results, key=lambda r: r["top1_pct"])
    print(f"=== Best: {best['label']} at {best['top1_pct']}% top-1 ===")
    print("Remaining fails:")
    for f in best["fails"][:20]:
        mark = "[in3]" if f.get("in3") else "[----]"
        print(f"  {mark} {f['q'][:55]:<55}  ->  {f.get('top1', '')[:80]}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
