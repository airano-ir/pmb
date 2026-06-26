"""Tests for the exploration memo cache (incremental cognition).

The novel part is the hash-gated freshness check on recall, so we test it
against real files on disk: a memo is fresh while its sources are byte-identical
and goes stale the moment one changes.
"""
from __future__ import annotations

import hashlib

from pmb.memo.exploration import recall_exploration, record_exploration


def _seed_memo(eng, project_path, intent, conclusion, file_rel, sha,
               recorded_head=None):
    eng.record_batch([{
        "type": "fact",
        "content": f"Exploration memo: {intent}\nConclusion: {conclusion}",
        "importance": 0.7,
        "metadata": {
            "source": "exploration-memo",
            "intent": intent,
            "conclusion": conclusion,
            "project_path": str(project_path),
            "sources": [{"file": file_rel, "sha1": sha}],
            "recorded_at_head": recorded_head,
        },
    }])
    eng.wait_for_writes(timeout=120)


def test_recall_fresh_then_stale(isolated_engine, tmp_path):
    eng = isolated_engine
    f = tmp_path / "auth.py"
    f.write_text("def login(): ...\n", encoding="utf-8")
    sha = hashlib.sha1(f.read_bytes()).hexdigest()
    _seed_memo(eng, tmp_path, "where is auth handled",
               "Auth lives in auth.py login()", "auth.py", sha)

    r = recall_exploration(eng, "where is the auth logic", project_path=str(tmp_path))
    assert r["n"] == 1
    m = r["matches"][0]
    assert m["conclusion"] == "Auth lives in auth.py login()"
    assert m["fresh"] is True
    assert m["stale_files"] == []

    # Changing the source file invalidates the memo (hash gate).
    f.write_text("def login(): return True\n", encoding="utf-8")
    r2 = recall_exploration(eng, "where is the auth logic", project_path=str(tmp_path))
    m2 = r2["matches"][0]
    assert m2["fresh"] is False
    assert "auth.py" in m2["stale_files"]


def test_recall_no_match(isolated_engine, tmp_path):
    eng = isolated_engine
    f = tmp_path / "x.py"
    f.write_text("x=1\n", encoding="utf-8")
    sha = hashlib.sha1(f.read_bytes()).hexdigest()
    _seed_memo(eng, tmp_path, "database schema", "schema in models.py", "x.py", sha)
    r = recall_exploration(eng, "completely unrelated frontend css", project_path=str(tmp_path))
    assert r["n"] == 0


def test_record_exploration_hashes_sources(isolated_engine, tmp_path, monkeypatch):
    f = tmp_path / "b.py"
    f.write_text("y=1\n", encoding="utf-8")
    captured: dict = {}
    monkeypatch.setattr(
        isolated_engine, "record_batch_async",
        lambda items: captured.update(items=items) or {"ok": True},
    )
    res = record_exploration(
        isolated_engine, "how does b work", "b computes y",
        ["b.py"], project_path=str(tmp_path),
    )
    assert res["recorded"] is True
    assert res["n_sources"] == 1
    md = captured["items"][0]["metadata"]
    assert md["source"] == "exploration-memo"
    assert md["conclusion"] == "b computes y"
    assert md["sources"][0]["file"] == "b.py"
    assert md["sources"][0]["sha1"]  # hashed real file content


def test_record_requires_intent_and_conclusion(isolated_engine, tmp_path):
    res = record_exploration(isolated_engine, "", "something", [], project_path=str(tmp_path))
    assert res["recorded"] is False


def test_repo_changed_since_flags_external_churn(isolated_engine, tmp_path):
    """Source file unchanged (fresh) but the repo moved via an UNLISTED file ->
    repo_changed_since=True. Closes the false-fresh gap."""
    import subprocess

    def g(*a):
        r = subprocess.run(["git", *a], cwd=str(tmp_path),
                           capture_output=True, text=True)
        assert r.returncode == 0, f"git {a} failed ({r.returncode}): {r.stderr}"

    g("init")
    g("config", "user.email", "t@example.com")
    g("config", "user.name", "Test")
    f = tmp_path / "keep.py"
    f.write_text("v = 1\n", encoding="utf-8")
    # Add only the test's OWN files, never `git add -A`: the isolated_engine
    # workspace (pmb_home/, workspace/) lives UNDER tmp_path, so `-A` would sweep
    # in the engine's live SQLite/LanceDB files and race the async write thread
    # on a slow CI runner (git add exits 128 mid-rewrite).
    g("add", "keep.py")
    g("commit", "-m", "c1")
    sha = hashlib.sha1(f.read_bytes()).hexdigest()
    _seed_memo(isolated_engine, tmp_path, "where is v defined",
               "v is in keep.py", "keep.py", sha, recorded_head="0000000")

    # An unrelated commit moves HEAD; keep.py stays byte-identical.
    (tmp_path / "other.py").write_text("z = 2\n", encoding="utf-8")
    g("add", "other.py")
    g("commit", "-m", "c2")

    r = recall_exploration(isolated_engine, "where is v defined",
                           project_path=str(tmp_path))
    m = r["matches"][0]
    assert m["fresh"] is True             # the listed source is unchanged
    assert m["repo_changed_since"] is True  # but the repo moved -> verify
