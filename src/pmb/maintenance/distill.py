"""Phase D — Anchor→Lexicon Distillation (ALD): the self-compiling lexical cache.

The Semantic Anchor Engine (Phases A/B) needs the embedder, so it only runs on
the WARM daemon path. ALD closes that gap. It watches which anchors fire on real
traffic (D1 — the `anchor_fires` log), then in the maintenance tick mines the
n-grams that reliably PREDICT each anchor and compiles them into
``$PMB_HOME/lang/auto.yaml`` in the existing pack schema (D2). After a week of a
Polish user's traffic, "co mi zostało do zrobienia" then classifies LEXICALLY —
in microseconds, on a COLD hook with no model loaded and no Polish anywhere in
the codebase. The cache is local-only, built from the user's own messages.

Precision is measured ACROSS anchors within the fire log: an n-gram is a signal
for anchor A only if, among all fires it appeared in, ≥ ``min_precision`` were A
(with ≥ ``min_support`` messages). Generic n-grams spread across many anchors and
self-prune; stopword-only n-grams are dropped. No per-language data is written by
hand — the only inputs are English anchors + the user's traffic.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict

# Anchor set name → the lexical pack CATEGORY its compiled fragment lands in —
# the SAME category the inline EN regex in auto_recall.py reads via
# `_lang.merged_list(category)`. `self_intent` has no lexical category of its
# own; like B1 it folds into the past/recall path.
ANCHOR_TO_PACK_CATEGORY: dict[str, str] = {
    "intent.goals_query": "intent_goals_query",
    "intent.past_query": "intent_past_query",
    "intent.self_intent": "intent_past_query",
    "intent.recent_query": "intent_recent_query",
    "intent.lessons_query": "intent_lessons_query",
    "intent.work_request": "work_verb_markers",
    "intent.trivial_ack": "trivial_acks",
}

_WORD = re.compile(r"\w+", re.UNICODE)


def message_ngrams(text: str, max_n: int = 3, cap: int = 96) -> list[str]:
    """Lowercased 1..max_n word n-grams. Unicode ``\\w`` covers Latin/Cyrillic/
    accented scripts; CJK (no spaces) collapses to one token and is intentionally
    out of scope for the lexical cache (distant scripts stay on the warm anchor
    path — see the Phase D note)."""
    toks = _WORD.findall((text or "").lower())
    out: list[str] = []
    for n in range(1, max_n + 1):
        for i in range(len(toks) - n + 1):
            out.append(" ".join(toks[i:i + n]))
            if len(out) >= cap:
                return out
    return out


def _table_ready(conn) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS anchor_fires ("
        " id INTEGER PRIMARY KEY,"
        " ts REAL NOT NULL,"
        " workspace_id TEXT,"
        " anchor TEXT NOT NULL,"
        " text_hash TEXT,"
        " ngrams_json TEXT NOT NULL)")


def log_anchor_fire(engine, anchor: str, text: str) -> None:
    """D1: append one row per anchor fire. Best-effort — never raises into the
    hook. The caller gates on ``lang.anchor_log``."""
    try:
        import hashlib
        import sqlite3 as _sql
        ngrams = message_ngrams(text)
        if not ngrams:
            return
        h = hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:16]
        ws = getattr(engine.workspace, "id", None)
        with _sql.connect(str(engine.workspace.db_path)) as conn:
            _table_ready(conn)
            conn.execute(
                "INSERT INTO anchor_fires (ts, workspace_id, anchor, text_hash,"
                " ngrams_json) VALUES (?,?,?,?,?)",
                (time.time(), ws, anchor, h,
                 json.dumps(ngrams, ensure_ascii=False)))
    except Exception:
        pass


# D3 shadow-T1: the cold lexical tier (T0, partly auto.yaml-driven) is sampled
# against the warm anchor tier (T1) on live traffic; per-intent agreement is the
# live precision that prunes a distilled category gone wrong. Intent → the
# auto.yaml category its fragments live in (mirror of ANCHOR_TO_PACK_CATEGORY).
_INTENT_TO_CATEGORY: dict[str, str] = {
    "GOALS_QUERY": "intent_goals_query",
    "PAST_QUERY": "intent_past_query",
    "RECENT_QUERY": "intent_recent_query",
    "LESSONS_QUERY": "intent_lessons_query",
    "WORK_REQUEST": "work_verb_markers",
}
_CATEGORY_TO_INTENT = {v: k for k, v in _INTENT_TO_CATEGORY.items()}


def _shadow_table_ready(conn) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS shadow_t1 ("
        " id INTEGER PRIMARY KEY, ts REAL NOT NULL, workspace_id TEXT,"
        " intent TEXT NOT NULL, agreed INTEGER NOT NULL)")


def record_shadow_t1(engine, intent: str, agreed: bool) -> None:
    """D3: log one sampled T0-vs-T1 outcome (did the warm anchor agree with the
    cold lexical intent?). Best-effort, never raises into the hook."""
    try:
        import sqlite3 as _sql
        ws = getattr(engine.workspace, "id", None)
        with _sql.connect(str(engine.workspace.db_path)) as conn:
            _shadow_table_ready(conn)
            conn.execute(
                "INSERT INTO shadow_t1 (ts, workspace_id, intent, agreed) "
                "VALUES (?,?,?,?)", (time.time(), ws, intent, 1 if agreed else 0))
    except Exception:
        pass


def load_shadow_precision(engine) -> dict[str, tuple[int, int]]:
    """intent -> (n_agreed, n_total) over the shadow log for this workspace."""
    import sqlite3 as _sql
    out: dict[str, tuple[int, int]] = {}
    try:
        ws = getattr(engine.workspace, "id", None)
        with _sql.connect(str(engine.workspace.db_path)) as conn:
            _shadow_table_ready(conn)
            rows = conn.execute(
                "SELECT intent, SUM(agreed), COUNT(*) FROM shadow_t1 "
                "WHERE workspace_id IS ? GROUP BY intent", (ws,)).fetchall()
    except Exception:
        return {}
    for intent, agreed, total in rows:
        out[str(intent)] = (int(agreed or 0), int(total or 0))
    return out


def prune_anchor_log(engine, retention_days: float = 30.0) -> int:
    """D3: delete fire-log rows older than `retention_days` so distillation only
    sees RECENT traffic. This is the recency half of the self-healing cache —
    a phrasing the user stopped using ages out of auto.yaml on the next tick,
    which D2's precision rebuild alone (it re-derives from whatever is logged)
    cannot do. Best-effort; returns the number of rows deleted."""
    import sqlite3 as _sql
    cutoff = time.time() - max(0.0, float(retention_days)) * 86400.0
    try:
        with _sql.connect(str(engine.workspace.db_path)) as conn:
            _table_ready(conn)
            cur = conn.execute("DELETE FROM anchor_fires WHERE ts < ?", (cutoff,))
            return int(cur.rowcount or 0)
    except Exception:
        return 0


def load_anchor_log(engine) -> tuple[dict[str, Counter], Counter]:
    """Aggregate the fire log → (anchor → Counter(ngram), total Counter(ngram)
    across ALL fires). Each n-gram is counted ONCE per message, so a repeated
    word can't inflate support."""
    import sqlite3 as _sql
    per: dict[str, Counter] = defaultdict(Counter)
    total: Counter = Counter()
    try:
        ws = getattr(engine.workspace, "id", None)
        with _sql.connect(str(engine.workspace.db_path)) as conn:
            _table_ready(conn)
            rows = conn.execute(
                "SELECT anchor, ngrams_json FROM anchor_fires "
                "WHERE workspace_id IS ?", (ws,)).fetchall()
    except Exception:
        return {}, Counter()
    for anchor, ngrams_json in rows:
        try:
            ngrams = set(json.loads(ngrams_json))
        except Exception:
            continue
        for g in ngrams:
            per[anchor][g] += 1
            total[g] += 1
    return per, total


def _contentful(ngram: str, stopwords) -> bool:
    """An n-gram worth compiling has at least one token that is long enough and
    not a stopword (so "queda"/"zostało"/"obiettivi aperti" survive but "do" and
    "what is" do not)."""
    return any(len(t) >= 4 and t not in stopwords for t in ngram.split(" "))


def distill_lexicon(engine, min_support: int = 6,
                    min_precision: float = 0.95) -> dict:
    """D2 + D3: prune the stale fire log (recency self-healing), then mine the
    remaining recent log and compile predictive n-grams into auto.yaml. Returns
    a summary dict. Writes nothing when no n-gram clears the bars."""
    try:
        retention = float(engine.config.get("lang.anchor_log_retention_days") or 30)
    except Exception:
        retention = 30.0
    pruned = prune_anchor_log(engine, retention)
    per, total = load_anchor_log(engine)
    if not total:
        return {"categories": 0, "entries": 0, "ngrams": 0,
                "pruned": pruned, "skipped": "empty log"}

    from pmb.core.text_match import STOPWORDS
    pack: dict[str, list[str]] = defaultdict(list)
    provenance: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for anchor, counter in per.items():
        cat = ANCHOR_TO_PACK_CATEGORY.get(anchor)
        if not cat:
            continue
        for ngram, n_with in counter.items():
            n_total = int(total.get(ngram, 0))
            if n_total < min_support:
                continue
            prec = n_with / n_total
            if prec < min_precision:
                continue
            if not _contentful(ngram, STOPWORDS):
                continue
            key = (cat, ngram)
            if key in seen:
                continue
            seen.add(key)
            pack[cat].append(re.escape(ngram))
            provenance.append({"cat": cat, "ngram": ngram, "anchor": anchor,
                               "support": n_total, "precision": round(prec, 4)})

    # D3 shadow-T1 prune: drop a whole category whose COLD lexical classification
    # disagreed with the WARM anchor below min_precision on sampled live traffic
    # (>= min_shadow samples) — self-healing against a distilled fragment that
    # turned out to mislead the cold path.
    shadow = load_shadow_precision(engine)
    shadow_dropped: list[str] = []
    for cat in list(pack.keys()):
        intent = _CATEGORY_TO_INTENT.get(cat)
        if not intent:
            continue
        agreed, tot = shadow.get(intent, (0, 0))
        if tot >= 10 and (agreed / tot) < 0.9:
            pack.pop(cat, None)
            provenance = [p for p in provenance if p.get("cat") != cat]
            shadow_dropped.append(cat)

    n_entries = sum(len(v) for v in pack.values())
    if n_entries:
        _write_auto_pack(dict(pack), provenance, min_support, min_precision)
    return {"categories": len(pack), "entries": n_entries,
            "ngrams": len(total), "pruned": pruned,
            "shadow_dropped": shadow_dropped}


def _write_auto_pack(pack: dict, provenance: list[dict],
                     min_support: int, min_precision: float) -> None:
    """Write ``$PMB_HOME/lang/auto.yaml`` in the pack schema + a ``_ald_meta``
    block carrying provenance (D3 pruning reads it). Clears the pack cache so
    ``active_packs()`` sees it immediately (in-process; the cold hook re-imports
    a fresh process anyway)."""
    import yaml

    from pmb import lang as _lang
    d = _lang.user_dir()
    d.mkdir(parents=True, exist_ok=True)
    doc: dict = dict(pack)
    doc["_ald_meta"] = {
        "generated_ts": time.time(),
        "generator": "pmb.maintenance.distill",
        "min_support": min_support,
        "min_precision": min_precision,
        "entries": provenance,
    }
    header = (
        "# AUTO-GENERATED by PMB Anchor->Lexicon Distillation (Phase D).\n"
        "# Regenerated each maintenance tick from THIS machine's anchor-fire\n"
        "# log; local-only; safe to delete. Do not edit by hand.\n")
    path = d / "auto.yaml"
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=True)
    try:
        _lang.clear_cache()
    except Exception:
        pass
