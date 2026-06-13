"""S7: find_lessons / find_decisions are served by the idx_meta_kind expression
index (COALESCE of metadata.kind / metadata.activity_kind) instead of a
full-table metadata_json LIKE scan.

Two guarantees:
  * RESULTS unchanged — every lesson/decision the LIKE matched still matches
    (incl. the activity_kind route for decisions), plain facts don't.
  * The query PLAN uses the index, not a SCAN.
"""
from __future__ import annotations

import sqlite3

from pmb.core.engine import Engine


def _engine(ws, home):
    return Engine(cwd=ws, pmb_home=home, config_overrides={"recall.cache_size": 0})


def test_find_lessons_and_decisions_results(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    les = eng.record_fact("always run ruff before pushing",
                          metadata={"kind": "lesson", "source": "lesson"})
    dec_k = eng.record_fact("chose postgres over mysql for strong typing",
                            metadata={"kind": "decision"})
    dec_a = eng.record_fact("decided to ship the daemon first",
                            metadata={"activity_kind": "decision"})
    eng.record_fact("the api listens on port 5432", metadata={"kind": "fact"})

    lesson_ulids = {x["ulid"] for x in eng.find_lessons(limit=50)}
    assert les in lesson_ulids
    # a plain fact / a decision must NOT show up as a lesson
    assert dec_k not in lesson_ulids

    decision_ulids = {x["ulid"] for x in eng.find_decisions(limit=50)}
    assert dec_k in decision_ulids          # metadata.kind route
    assert dec_a in decision_ulids          # metadata.activity_kind route
    assert les not in decision_ulids


_COALESCE = ("COALESCE(json_extract(metadata_json, '$.kind'), "
            "json_extract(metadata_json, '$.activity_kind'))")


def test_find_lessons_predicate_is_indexable(tmp_pmb_home, tmp_workspace_dir):
    # Isolate the WHERE predicate from the ORDER BY: the kind lookup itself must
    # be served by idx_meta_kind (the ORDER BY's interaction with the timestamp
    # index is a separate, data-dependent cost decision tested below).
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.record_fact("prefer pnpm over npm", metadata={"kind": "lesson"})
    q = (f"SELECT ulid FROM events WHERE workspace_id = ? AND archived_at IS NULL "
         f"AND {_COALESCE} = 'lesson'")
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        plan = c.execute("EXPLAIN QUERY PLAN " + q, (eng.workspace.id,)).fetchall()
    text = " ".join(str(row[-1]) for row in plan)
    assert "idx_meta_kind" in text, f"expected idx_meta_kind, got: {text}"


def test_find_lessons_query_uses_index_on_real_data(tmp_pmb_home, tmp_workspace_dir):
    # The shipped query OMITS `ORDER BY timestamp` (it sorts in Python) so the
    # planner keeps the selective kind index even when stat1 has no per-value
    # histogram. Prove it holds on a realistic corpus where lessons are rare —
    # the pathological 2-distinct-value case that fooled the planner WITH an
    # ORDER BY. Insert rows directly (no embeddings needed for a plan check).
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    ws = eng.workspace.id
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        rows = []
        for i in range(400):
            kind = "lesson" if i % 200 == 0 else "fact"   # 2 lessons / 400 rows
            rows.append((f"u{i:04d}", ws, "fact", "x", f'{{"kind":"{kind}"}}',
                         float(i), 0.5, 0, float(i), None, None, "working"))
        c.executemany(
            "INSERT INTO events(ulid, workspace_id, event_type, content, "
            "metadata_json, timestamp, importance, access_count, last_accessed, "
            "archived_at, source_session_id, tier) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        c.execute("ANALYZE")
        q = (f"SELECT ulid FROM events WHERE workspace_id = ? AND archived_at IS NULL "
             f"AND {_COALESCE} = 'lesson'")   # no ORDER BY — matches the shipped query
        plan = c.execute("EXPLAIN QUERY PLAN " + q, (ws,)).fetchall()
    text = " ".join(str(row[-1]) for row in plan)
    assert "idx_meta_kind" in text, f"expected idx_meta_kind on real data, got: {text}"
