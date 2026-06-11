"""
Regression tests for the hardening pass (TT/UU/VV/WW/XX silent failures).

Each test asserts not just "the right answer surfaces" but "the feature
actually fires the code path it claims". This catches the failure mode
where a silent except-pass falls through to a single-shot fallback that
incidentally produces a similar-looking result.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest

# -------------------- shared fixtures --------------------


@pytest.fixture
def tmp_engine():
    """Fresh Engine on a temp workspace."""
    from pmb.core.engine import Engine
    tmp_home = Path(tempfile.mkdtemp())
    tmp_ws = Path(tempfile.mkdtemp())
    os.environ["PMB_HOME"] = str(tmp_home)
    eng = Engine(
        cwd=tmp_ws, pmb_home=tmp_home, rerank_model=None,
        config_overrides={
            "recall.pattern_split": True,
            "recall.auto_vocab_bridges": False,
            "recall.cache_size": 0,
        },
    )
    _ = eng.search.model
    yield eng
    try:
        eng.close()
    except Exception:
        pass


# -------------------- H1: pattern_split must actually fire --------------------


def test_pattern_split_actually_fans_out(tmp_engine):
    """If pattern_split silently fell back to single-shot we'd never know
    from output alone. Assert the feature flag tracks an actual fan-out.
    """
    eng = tmp_engine
    eng.record_batch([
        {"type": "fact",
         "content": "We picked Postgres because of JSONB support."},
        {"type": "fact",
         "content": "We dropped MySQL because of replication issues."},
    ])
    time.sleep(1.5)

    # Single-clause: must NOT fire fan-out
    eng.recall("what database do we use", top_k=5)
    assert getattr(eng, "_pattern_split_last_fired", None) is False, (
        "single-clause query incorrectly triggered split"
    )

    # Compound: MUST fire fan-out AND return a RecallPack with all required fields
    eng.recall(
        "why did we pick postgres and why did we drop mysql",
        top_k=5,
    )
    assert eng._pattern_split_last_fired is True
    assert eng._pattern_split_last_returned is True, (
        "fan-out fired but did not return a fused pack — RecallPack ctor "
        "probably broken again (regression of original H1 bug)"
    )


def test_pattern_split_returns_pack_with_all_required_fields(tmp_engine):
    """Direct check: the RecallPack from the fan-out path must have all
    six required dataclass fields populated.
    """
    eng = tmp_engine
    eng.record_batch([
        {"type": "fact", "content": "Postgres is our primary database."},
        {"type": "fact", "content": "MySQL was deprecated last quarter."},
    ])
    time.sleep(1.5)

    pack = eng.recall(
        "why did we pick postgres and why did we drop mysql",
        top_k=5,
    )
    # All RecallPack fields must be set — no AttributeError, no None for required
    assert pack.query
    assert pack.workspace_name
    assert pack.workspace_id
    assert pack.results, "fan-out path returned empty results"
    assert isinstance(pack.n_total_in_workspace, int)
    assert isinstance(pack.elapsed_ms, float)


# -------------------- H2: tomli_w is importable --------------------


def test_tomli_w_dependency_present():
    """`pmb connect codex` writes Codex's TOML config and needs tomli_w.
    Was previously imported but missing from pyproject deps.
    """
    import tomli_w  # must not ImportError
    assert hasattr(tomli_w, "dumps")


# -------------------- H3: durable embed queue --------------------


def test_durable_embed_queue_persists_across_restart(tmp_path):
    """Queue rows must survive process death — that's the whole point.
    Simulate by creating, enqueueing, then opening fresh."""
    from pmb.core.embed_queue import PersistentEmbedQueue
    db = tmp_path / "events.sqlite"
    q1 = PersistentEmbedQueue(db)
    q1.enqueue("ulid_a", "text a")
    q1.enqueue("ulid_b", "text b")
    assert q1.pending_count() == 2
    del q1

    # Fresh process simulation
    q2 = PersistentEmbedQueue(db)
    assert q2.pending_count() == 2, "pending embeds lost across restart"


def test_durable_queue_dead_letters_after_max_attempts(tmp_path):
    """Failing adder must eventually mark the row as dead-letter, not
    silently drop or infinitely retry."""
    from pmb.core.embed_queue import PersistentEmbedQueue
    db = tmp_path / "events.sqlite"
    q = PersistentEmbedQueue(db, max_attempts=3, backoff_base=0.0)
    q.enqueue("bad_ulid", "bad text")

    def failing(_u, _t):
        raise RuntimeError("simulated index failure")

    # Run drain enough times to exceed max_attempts
    for _ in range(5):
        q.drain_once(failing)

    assert q.pending_count() == 0
    assert q.dead_letter_count() == 1
    rows = q.list_dead_letters()
    assert rows[0]["ulid"] == "bad_ulid"
    assert "RuntimeError" in (rows[0]["last_error"] or "")


def test_durable_queue_retry_dead_letter(tmp_path):
    """Operator must be able to revive dead letters after fixing the cause."""
    from pmb.core.embed_queue import PersistentEmbedQueue
    db = tmp_path / "events.sqlite"
    q = PersistentEmbedQueue(db, max_attempts=2, backoff_base=0.0)
    q.enqueue("ulid_x", "x")
    q.drain_once(lambda u, t: (_ for _ in ()).throw(RuntimeError("fail")))
    q.drain_once(lambda u, t: (_ for _ in ()).throw(RuntimeError("fail")))
    assert q.dead_letter_count() == 1

    moved = q.retry_dead_letter()
    assert moved == 1
    assert q.dead_letter_count() == 0
    assert q.pending_count() == 1


# -------------------- H5: multilingual concept extraction --------------------


def test_multilingual_concept_regex_matches_cyrillic():
    """Old [a-z] regex dropped every Cyrillic word silently. New \\w
    regex must match Cyrillic, accented Latin, etc."""
    from pmb.graph.entities import _CONCEPT_RE
    assert _CONCEPT_RE.findall("кошку зовут мурка") == [
        "кошку", "зовут", "мурка",
    ]
    assert "café" in _CONCEPT_RE.findall("я люблю café")
    assert "münchen" in _CONCEPT_RE.findall("берлин и münchen")


def test_multilingual_concept_regex_still_skips_short_and_numeric():
    """The 3+ char head + non-digit-non-underscore start are intentional —
    don't regress and start indexing 'is', '4', '_x'."""
    from pmb.graph.entities import _CONCEPT_RE
    assert "is" not in _CONCEPT_RE.findall("this is short")
    assert "4" not in _CONCEPT_RE.findall("кошке 4 года")
    assert _CONCEPT_RE.findall("_x_y_z") == []


# -------------------- H6: chunked IN clause --------------------


def test_get_many_handles_more_than_999_ulids(tmp_engine):
    """SQLite default SQLITE_MAX_VARIABLE_NUMBER is 999. Calling
    get_many with 1500 ulids must not crash."""
    eng = tmp_engine
    # We don't need real events — get_many with non-existent ulids
    # exercises the chunking path; returns empty dict, no error.
    fake_ulids = [f"fake_{i:04d}" for i in range(1500)]
    result = eng.events.get_many(fake_ulids)
    assert isinstance(result, dict)


def test_get_many_dedups_input(tmp_engine):
    """Duplicate ulids in input shouldn't multiply the IN clause size."""
    eng = tmp_engine
    eng.record_batch([{"type": "fact", "content": "Test fact for dedup."}])
    time.sleep(1.0)
    # Get the ulid back via recall
    pack = eng.recall("test fact", top_k=1)
    assert pack.results, "test fact not findable"
    ulid = pack.results[0].ulid

    # Pass it 50 times — should still return 1 row
    result = eng.events.get_many([ulid] * 50)
    assert len(result) == 1
    assert ulid in result


# -------------------- P0-1: RU/UK atomic fact extraction --------------------


def test_ru_atomic_facts_extracted():
    """Reviewer feedback (Alternix 2026-05-27): English-only extraction was
    a critical blocker for personal-assistant use. RU patterns must fire.
    """
    from pmb.reasoning.fact_extract import extract_atomic_facts
    text = (
        "Меня зовут Алексей. Я живу в Киеве. Мой день рождения 7 июня. "
        "Я люблю спокойные игры. Мой друг Алексей переехал в Варшаву."
    )
    facts = extract_atomic_facts(text, min_len=20, min_sentences=1)
    kinds = {f.kind for f in facts}
    assert any(k.startswith("ru_") for k in kinds), (
        f"no RU patterns fired: kinds={kinds}"
    )
    # Specifically: identity + location + birthday should all be there
    assert "ru_identity" in kinds, f"missing ru_identity: {kinds}"
    assert any("location" in k for k in kinds), f"missing location: {kinds}"
    assert "ru_birthday" in kinds, f"missing ru_birthday: {kinds}"


def test_uk_atomic_facts_extracted():
    """Same coverage but Ukrainian."""
    from pmb.reasoning.fact_extract import extract_atomic_facts
    text = (
        "Мене звати Олексій. Я живу у Києві. Мій день народження 7 червня. "
        "Друг користувача Олексій переїхав до США."
    )
    facts = extract_atomic_facts(text, min_len=20, min_sentences=1)
    kinds = {f.kind for f in facts}
    assert any(k.startswith("uk_") for k in kinds), (
        f"no UK patterns fired: kinds={kinds}"
    )
    assert "uk_identity" in kinds, f"missing uk_identity: {kinds}"
    assert any("location" in k for k in kinds), f"missing location: {kinds}"


def test_english_atomic_extraction_still_works():
    """Don't regress the existing English coverage when adding RU/UK."""
    from pmb.reasoning.fact_extract import extract_atomic_facts
    text = (
        "Today I met Alice. She is the tech lead at Stripe. "
        "She lives in Berlin. We use Cloud Run for deployment."
    )
    facts = extract_atomic_facts(text, min_len=20, min_sentences=1)
    kinds = {f.kind for f in facts}
    assert any(k in {"role", "location", "tool_choice"} for k in kinds), (
        f"English patterns broken: kinds={kinds}"
    )


# -------------------- P0-2: keyed-upsert supersession --------------------


def test_keyed_upsert_archives_prior_value(tmp_engine):
    """Reviewer scenario: 'I live in Kyiv' → 'I live in Warsaw'. Old fact
    must be archived, new fact must be the only active one for that key.
    """
    eng = tmp_engine
    r1 = eng.record_keyed_fact("user", "residence", "Kyiv")
    assert r1["new_ulid"]
    assert r1["superseded_ulids"] == [], "first call should have nothing to supersede"

    r2 = eng.record_keyed_fact("user", "residence", "Warsaw")
    assert r2["new_ulid"] != r1["new_ulid"]
    assert r1["new_ulid"] in r2["superseded_ulids"], (
        "prior keyed fact must be archived on second upsert"
    )

    # History endpoint sees both
    hist = eng.get_keyed_fact_history("user", "residence")
    assert len(hist) == 2
    current = [h for h in hist if h["is_current"]]
    archived = [h for h in hist if not h["is_current"]]
    assert len(current) == 1
    assert current[0]["value"] == "Warsaw"
    assert len(archived) == 1
    assert archived[0]["value"] == "Kyiv"


def test_keyed_upsert_recall_returns_only_current(tmp_engine):
    """After supersession, normal recall must return only the current value."""
    eng = tmp_engine
    eng.record_keyed_fact("user", "residence", "Kyiv")
    eng.record_keyed_fact("user", "residence", "Warsaw")
    time.sleep(0.5)
    pack = eng.recall("where does user live", top_k=5)
    contents = [r.content.lower() for r in pack.results]
    has_warsaw = any("warsaw" in c for c in contents)
    has_kyiv = any("kyiv" in c for c in contents)
    assert has_warsaw, f"current value 'Warsaw' missing from results: {contents}"
    assert not has_kyiv, f"archived 'Kyiv' surfaced in results: {contents}"


# -------------------- P1-1: warmup API --------------------


def test_warmup_marks_engine_as_warm(tmp_engine):
    """warmup() should set is_warm() True and return timing breakdown."""
    eng = tmp_engine
    assert not eng.is_warm()  # fresh engine
    result = eng.warmup(with_first_query=True)
    assert eng.is_warm()
    # Required keys present
    for k in ("total_ms", "model_load_ms", "bm25_load_ms",
              "lance_open_ms", "first_query_ms"):
        assert k in result, f"warmup result missing {k}"
        assert result[k] >= 0


def test_warmup_is_idempotent(tmp_engine):
    """Calling warmup twice should not crash or break state."""
    eng = tmp_engine
    eng.warmup()
    eng.warmup()
    assert eng.is_warm()


# -------------------- P1-2: multilingual model warning --------------------


def test_multilingual_check_flags_english_only_on_cyrillic(tmp_path):
    """Reviewer's real scenario: all-MiniLM-L6-v2 + Russian content → warn."""
    # Build a minimal SQLite with Russian content
    import sqlite3

    from pmb.health.multilingual_check import evaluate
    db = tmp_path / "events.sqlite"
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "CREATE TABLE events ("
            "  ulid TEXT, content TEXT, archived_at REAL"
            ")"
        )
        for c in ("Меня зовут Алексей", "Я живу в Киеве", "Мне нравятся игры"):
            conn.execute(
                "INSERT INTO events(ulid, content, archived_at) "
                "VALUES (?, ?, NULL)", ("u" + str(hash(c)), c),
            )
    r = evaluate(db, "all-MiniLM-L6-v2")
    assert r["severity"] == "warn"
    assert r["warning"] and "English-only" in r["warning"]
    assert r["recommendation"] and "multilingual" in r["recommendation"]


def test_multilingual_check_silent_on_correct_setup(tmp_path):
    """Multilingual model + multilingual data → no warning."""
    import sqlite3

    from pmb.health.multilingual_check import evaluate
    db = tmp_path / "events.sqlite"
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "CREATE TABLE events ("
            "  ulid TEXT, content TEXT, archived_at REAL"
            ")"
        )
        for c in ("Меня зовут Алексей", "I am a developer"):
            conn.execute(
                "INSERT INTO events(ulid, content, archived_at) "
                "VALUES (?, ?, NULL)", ("u" + str(hash(c)), c),
            )
    r = evaluate(db, "paraphrase-multilingual-MiniLM-L12-v2")
    assert r["severity"] == "ok"
    assert r["warning"] is None


# -------------------- P2: typed memory helpers --------------------


def test_record_preference_uses_preference_event_type(tmp_engine):
    eng = tmp_engine
    ulid = eng.record_preference("Я люблю спокойные игры")
    ev = eng.events.get_by_ulid(ulid)
    assert ev.event_type == "preference"
    assert (ev.metadata or {}).get("memory_type") == "preference"


def test_record_summary_uses_summary_event_type(tmp_engine):
    eng = tmp_engine
    ulid = eng.record_summary("User and assistant discussed project plans.")
    ev = eng.events.get_by_ulid(ulid)
    assert ev.event_type == "summary"
    assert (ev.metadata or {}).get("memory_type") == "summary"


def test_record_batch_accepts_keyed_fact_type(tmp_engine):
    eng = tmp_engine
    res = eng.record_batch([
        {"type": "keyed_fact", "subject": "user", "attribute": "residence",
         "value": "Kyiv"},
        {"type": "keyed_fact", "subject": "user", "attribute": "residence",
         "value": "Warsaw"},
    ])
    assert res["errors"] == [], f"errors: {res['errors']}"
    assert len(res["results"]) == 2
    second = res["results"][1]
    assert second["type"] == "keyed_fact"
    assert len(second["superseded_ulids"]) == 1
