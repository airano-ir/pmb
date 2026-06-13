"""X5 — fault-injection + concurrency. A memory tool must DEGRADE, not crash,
on bad data, and must not lose writes under concurrent access.
"""
from __future__ import annotations

import sqlite3
import threading

from pmb.core.engine import Engine


def _engine(ws, home):
    return Engine(cwd=ws, pmb_home=home, config_overrides={"recall.cache_size": 0})


def test_recall_on_empty_workspace_returns_empty_not_crash(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    pack = eng.recall(query="anything at all", top_k=5)
    assert list(pack.results) == []
    assert pack.confidence == 0.0


def test_metadata_json_integrity_and_null_tolerance(tmp_pmb_home, tmp_workspace_dir):
    import time
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    good = eng.record_fact("we use postgres", metadata={"kind": "lesson"})

    cols = ("INSERT INTO events(ulid, workspace_id, event_type, content, "
            "metadata_json, timestamp, importance, access_count, last_accessed, "
            "archived_at, source_session_id, tier) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)")
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        # (a) the idx_meta_kind expression index ENFORCES valid JSON: a malformed
        # metadata_json row is REJECTED at write, so corrupt kind data can't enter.
        import pytest
        with pytest.raises(sqlite3.OperationalError):
            c.execute(cols, ("bad0", eng.workspace.id, "fact", "broken",
                             "{not valid json", time.time(), 0.5, 0, time.time(),
                             None, None, "working"))
        # (b) a NULL metadata row is fine and readers tolerate it
        c.execute(cols, ("nullmeta", eng.workspace.id, "fact", "no metadata",
                         None, time.time(), 0.5, 0, time.time(), None, None, "working"))
    # readers don't crash with the NULL-metadata row present
    assert any(x["ulid"] == good for x in eng.find_lessons(limit=50))
    assert isinstance(eng.project_overview("postgres"), dict)


def test_adversarial_content_does_not_crash(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    payloads = [
        "x" * 50000,                       # very long
        "null\x00byte and \x01 control",   # control chars
        "🧠💾 emoji + математика + 中文",     # mixed scripts
        "",                                # empty
    ]
    for p in payloads:
        eng.record_fact(p or "placeholder", metadata={"kind": "fact"})
    # recall over the mess must still answer without raising
    assert isinstance(eng.recall(query="математика", top_k=3).results, list)


def test_concurrent_writes_lose_nothing(tmp_pmb_home, tmp_workspace_dir):
    # dedup.enable=False isolates the CONCURRENCY guarantee (no race-induced
    # loss) from the semantic-dedup collapsing of near-identical content — which
    # is correct behaviour but not what this test measures.
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
                 config_overrides={"recall.cache_size": 0, "dedup.enable": False})
    n_threads, per = 6, 10
    errors: list[str] = []

    def worker(tid: int) -> None:
        try:
            for i in range(per):
                eng.record_fact(f"thread {tid} fact {i}", metadata={"kind": "fact"})
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"concurrent writes raised: {errors}"
    eng.wait_for_writes(timeout=30.0)   # async write path — drain before counting
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        n = c.execute(
            "SELECT COUNT(*) FROM events WHERE workspace_id=? AND event_type='fact'",
            (eng.workspace.id,)).fetchone()[0]
    assert n >= n_threads * per, f"lost writes: {n} < {n_threads * per}"


def test_background_spreading_activation_is_thread_safe(tmp_pmb_home, tmp_workspace_dir):
    # S9 deferred spreading activation to a BACKGROUND thread on the daemon path
    # (recall.touch_async=True). Drive concurrent recalls that each spawn one and
    # assert: no recall raises, and the DB is not corrupted by the concurrent
    # graph writes. (Priming is best-effort, so dropped edges are acceptable;
    # corruption / crashes are not.)
    import time
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
                 config_overrides={"recall.cache_size": 0,
                                   "recall.touch_async": True,        # the daemon path
                                   "recall.spreading_activation": True})
    try:
        eng.warmup()
    except Exception:
        pass
    for i in range(8):
        eng.record_fact(f"we deploy service {i} to AWS Fargate via GitHub Actions",
                        metadata={"kind": "fact"})
    eng.wait_for_writes(timeout=30.0)

    errors: list[str] = []

    def worker() -> None:
        try:
            for _ in range(5):
                eng.recall(query="aws fargate deploy github", top_k=5)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    time.sleep(1.5)   # let the detached spreading threads finish their writes
    eng.wait_for_writes(timeout=30.0)

    assert not errors, f"concurrent recall + bg spreading raised: {errors}"
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok", \
            "concurrent background graph writes corrupted the DB"
    # the engine is still usable afterwards
    assert isinstance(eng.recall(query="fargate", top_k=3).results, list)
