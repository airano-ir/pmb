"""`pmb declutter` — sweep obvious junk out of memory (archive-only).

Heuristic-first: test artifacts, near-empty / pure-stopword content, exact
duplicates, and negation tombstones already obsoleted by a positive keyed
value. An optional bounded-LLM judge (--llm) reviews ONLY the low-value
borderline remainder, capped and behind the SAME circuit breaker as recall's
decomposition — never on a hot path (this is a manual maintenance command).

Everything is archive-only (reversible) and tagged
metadata.archived_reason="declutter" + declutter_reason=<why>.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict

# Keys/content that scream "test data". Content markers are deliberately
# HIGH-PRECISION (declutter ARCHIVES): we dropped collision-prone keyboard
# mashes — "asdf" is a real tool (asdf-vm), "qwerty"/"foobar" appear in real
# notes — and keep only unambiguous test/placeholder strings.
_TEST_KEY_RE = re.compile(r"test_attr|tmp_|dummy|placeholder|__test__", re.IGNORECASE)
_TEST_CONTENT_RE = re.compile(
    r"\b(?:final_value|lorem ipsum|test123|placeholder text)\b|x{5,}",
    re.IGNORECASE,
)
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "of", "and", "to", "in", "it",
    "this", "that", "ok", "okay", "yes", "no", "да", "нет", "это",
}

LLM_CANDIDATE_CAP = 50
LLM_TIMEOUT_S = 15.0
BORDERLINE_MAX_IMPORTANCE = 0.3


def _meta(raw) -> dict:
    try:
        m = json.loads(raw or "{}")
        return m if isinstance(m, dict) else {}
    except Exception:
        return {}


def _is_pure_stopwords(content: str) -> bool:
    toks = re.findall(r"[0-9a-zA-Zа-яёА-ЯЁ']+", (content or "").lower())
    return bool(toks) and all(t in _STOPWORDS for t in toks)


def is_suspect_junk(content: str) -> bool:
    """Cheap write-time quality heuristic shared with the declutter sweeper:
    True if CONTENT looks like junk (too short, placeholder/test text, or pure
    stopwords). Used by the optional write-time quality gate (config
    write.quality_gate). Conservative — it only FLAGS, never rejects."""
    s = (content or "").strip()
    if len(s) < 8:
        return True
    if _TEST_CONTENT_RE.search(s):
        return True
    return _is_pure_stopwords(s)


def _protected(meta: dict, importance: float) -> bool:
    """Pinned events and lessons are never decluttered."""
    if float(importance or 0.0) >= 0.99:
        return True
    return meta.get("kind") == "lesson" or meta.get("source") == "lesson"


def find_heuristic_candidates(engine) -> list[dict]:
    """Return [{ulid, reason, content}] for clear junk — no LLM."""
    out: list[dict] = []
    chosen: set[str] = set()
    dup_groups: dict[str, list[tuple]] = defaultdict(list)

    with sqlite3.connect(str(engine.workspace.db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT ulid, content, metadata_json, importance, timestamp, event_type "
            "FROM events WHERE workspace_id = ? AND archived_at IS NULL "
            "ORDER BY timestamp ASC",
            (engine.workspace.id,),
        ).fetchall()

    for r in rows:
        meta = _meta(r["metadata_json"])
        if _protected(meta, r["importance"]):
            continue
        content = r["content"] or ""
        key = meta.get("keyed_fact_key", "") or ""
        # 0. already flagged by the write-time quality gate (#8)
        if meta.get("quality_flag") == "suspect_junk":
            out.append({"ulid": r["ulid"], "reason": "quality_flag",
                        "content": content[:100]})
            chosen.add(r["ulid"])
            continue
        # 1. test artifacts (keyed test keys OR test-y content)
        if (key and _TEST_KEY_RE.search(key)) or _TEST_CONTENT_RE.search(content):
            out.append({"ulid": r["ulid"], "reason": "test_artifact",
                        "content": content[:100]})
            chosen.add(r["ulid"])
            continue
        # 2. near-empty / pure-stopword plain facts & activities
        if r["event_type"] in ("fact", "activity"):
            stripped = content.strip()
            if len(stripped) < 8 or _is_pure_stopwords(stripped):
                out.append({"ulid": r["ulid"], "reason": "near_empty",
                            "content": content[:100]})
                chosen.add(r["ulid"])
                continue
        # 3. collect plain (non-keyed) facts for exact-duplicate detection
        if r["event_type"] == "fact" and not key:
            norm = re.sub(r"\s+", " ", content.strip().lower())
            if norm:
                dup_groups[norm].append((r["timestamp"], r["ulid"]))

    # 3b. duplicates — keep the NEWEST, archive the older copies
    for norm, lst in dup_groups.items():
        if len(lst) < 2:
            continue
        lst.sort()  # ascending timestamp
        for _ts, ulid in lst[:-1]:
            if ulid not in chosen:
                out.append({"ulid": ulid, "reason": "exact_duplicate",
                            "content": norm[:100]})
                chosen.add(ulid)
    return out


def _borderline_pool(engine, exclude: set[str], cap: int) -> list[dict]:
    """Low-value, non-flagged plain facts the LLM judge may review."""
    pool: list[dict] = []
    with sqlite3.connect(str(engine.workspace.db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT ulid, content, metadata_json, importance FROM events "
            "WHERE workspace_id = ? AND archived_at IS NULL "
            "AND event_type = 'fact' AND importance <= ? "
            "ORDER BY importance ASC, timestamp ASC LIMIT ?",
            (engine.workspace.id, BORDERLINE_MAX_IMPORTANCE, cap * 3),
        ).fetchall()
    for r in rows:
        if r["ulid"] in exclude:
            continue
        meta = _meta(r["metadata_json"])
        if _protected(meta, r["importance"]) or meta.get("keyed_fact_key"):
            continue
        pool.append({"ulid": r["ulid"], "content": (r["content"] or "")[:200]})
        if len(pool) >= cap:
            break
    return pool


def _parse_judge(resp: str) -> dict:
    try:
        m = re.search(r"\{.*\}", resp or "", re.DOTALL)
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


def _llm_judge(engine, pool: list[dict]) -> list[dict]:
    """Ask a bounded LLM to confirm junk among the borderline pool. Returns the
    subset judged junk. Breaker-protected + timeout-clamped; never raises."""
    from pmb.core import circuit_breaker as _breaker
    if not pool or _breaker.is_open("llm"):
        return []
    thr = engine.config.get("recall.breaker_threshold") or 2
    cd = engine.config.get("recall.breaker_cooldown_s") or 60.0
    judged: list[dict] = []
    try:
        from pmb.health.consolidate import resolve_llm_client
        llm = resolve_llm_client(backend=engine.config.get("consolidate.backend"))
    except Exception:
        _breaker.record_failure("llm", thr, cd, "resolve_llm_client failed")
        return []
    if hasattr(llm, "timeout"):
        try:
            llm.timeout = max(1.0, min(float(llm.timeout), LLM_TIMEOUT_S))
        except Exception:
            pass
    for c in pool[:LLM_CANDIDATE_CAP]:
        prompt = (
            "You are auditing an AI assistant's long-term memory. Is the "
            "stored memory below JUNK (test data, placeholder, meaningless, or "
            "accidental) rather than a useful fact? Answer ONLY a JSON object "
            '{"junk": true|false, "reason": "<short>"}.\n\n'
            f"MEMORY: {c['content']}"
        )
        try:
            resp = llm.complete(prompt, max_tokens=120)
        except Exception as e:
            _breaker.record_failure("llm", thr, cd, str(e))
            break
        _breaker.record_success("llm")
        verdict = _parse_judge(resp)
        if verdict.get("junk") is True:
            judged.append({"ulid": c["ulid"], "content": c["content"][:100],
                           "reason": "llm:" + str(verdict.get("reason") or "junk")[:60]})
    return judged


def declutter(engine, apply: bool = False, use_llm: bool = False) -> dict:
    """Find (and optionally archive) junk memories.

    Returns {candidates: [{ulid, reason, content}], n, dry_run, llm_used,
    by_reason: {reason: count}}.
    """
    candidates = find_heuristic_candidates(engine)
    seen = {c["ulid"] for c in candidates}

    # Fold in negation tombstones already obsoleted by a positive keyed value
    # (reuses the Task-5 detector so the two stay consistent).
    try:
        neg = engine.archive_negations_for_current_keys(dry_run=True)
        for p in neg.get("plan", []):
            if p["ulid"] not in seen:
                candidates.append({"ulid": p["ulid"], "reason": "negation_obsolete",
                                   "content": p["content"]})
                seen.add(p["ulid"])
    except Exception:
        pass

    llm_used = False
    if use_llm:
        pool = _borderline_pool(engine, exclude=seen, cap=LLM_CANDIDATE_CAP)
        llm_used = bool(pool)
        for j in _llm_judge(engine, pool):
            if j["ulid"] not in seen:
                candidates.append(j)
                seen.add(j["ulid"])

    if apply and candidates:
        for c in candidates:
            try:
                engine.events.archive(c["ulid"])
                with sqlite3.connect(str(engine.workspace.db_path)) as conn:
                    row = conn.execute(
                        "SELECT metadata_json FROM events WHERE ulid=?",
                        (c["ulid"],)).fetchone()
                    m = _meta(row[0]) if row else {}
                    m["archived_reason"] = "declutter"
                    m["declutter_reason"] = c["reason"]
                    conn.execute("UPDATE events SET metadata_json=? WHERE ulid=?",
                                 (json.dumps(m), c["ulid"]))
            except Exception:
                continue
        try:
            engine.recall_cache.bump_generation()
        except Exception:
            pass

    by_reason: dict[str, int] = defaultdict(int)
    for c in candidates:
        by_reason[c["reason"].split(":")[0]] += 1
    return {"candidates": candidates, "n": len(candidates), "dry_run": not apply,
            "llm_used": llm_used, "by_reason": dict(by_reason)}
