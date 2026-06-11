"""
Phase 2 tests — signals: git, session, decay, files.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from pmb.core.engine import Engine
from pmb.core.workspace import detect_workspace
from pmb.signals.decay import boost_on_recall
from pmb.signals.files import FileCorrelation
from pmb.signals.git import GitSync, capture_recent_commits
from pmb.signals.session import SESSION_GAP_SECONDS, SessionTracker


@pytest.fixture
def git_repo():
    """Создаёт временный git репозиторий с несколькими commit'ами."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        # Init repo
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"],
                       cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"],
                       cwd=repo, check=True, capture_output=True)

        # Несколько commit'ов
        for i in range(3):
            f = repo / f"file_{i}.txt"
            f.write_text(f"content {i}\nline 2 of file {i}\n", encoding="utf-8")
            subprocess.run(["git", "add", f.name], cwd=repo, check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", f"feat: add file_{i} with details"],
                cwd=repo, check=True, capture_output=True,
            )

        yield repo


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def test_session_start_creates_state(tmp_pmb_home, tmp_workspace_dir):
    ws = detect_workspace(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ws.ensure_dirs()
    tracker = SessionTracker(ws)

    sess = tracker.start("test-session")
    assert sess.id
    assert sess.name == "test-session"
    assert (ws.storage_dir / "session.yaml").exists()


def test_session_persists_across_instances(tmp_pmb_home, tmp_workspace_dir):
    ws = detect_workspace(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ws.ensure_dirs()
    tracker1 = SessionTracker(ws)
    s1 = tracker1.start("persistent")

    tracker2 = SessionTracker(ws)
    s2 = tracker2.current(auto_create=False)
    assert s2 is not None
    assert s2.id == s1.id


def test_session_end_clears_state(tmp_pmb_home, tmp_workspace_dir):
    ws = detect_workspace(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ws.ensure_dirs()
    tracker = SessionTracker(ws)
    tracker.start()
    assert tracker.current(auto_create=False) is not None
    tracker.end()
    assert tracker.current(auto_create=False) is None


def test_session_auto_creates_after_gap(tmp_pmb_home, tmp_workspace_dir):
    ws = detect_workspace(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ws.ensure_dirs()
    tracker = SessionTracker(ws)
    s1 = tracker.start("first")
    # Manipulate state to simulate stale
    s1.last_activity = time.time() - SESSION_GAP_SECONDS - 100
    tracker._save(s1)

    s2 = tracker.touch()
    # Должна быть новая, потому что предыдущая stale
    assert s2.id != s1.id


def test_engine_remember_attaches_session(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ulid = eng.remember("Q", "A")
    ev = eng.events.get_by_ulid(ulid)
    assert ev.source_session_id is not None
    # Session id должен быть тем же что в engine.session_tracker.current
    sess = eng.session_tracker.current(auto_create=False)
    assert ev.source_session_id == sess.id


# ---------------------------------------------------------------------------
# Decay & Importance
# ---------------------------------------------------------------------------

def test_boost_on_recall_saturating():
    # High score boosts
    new_imp = boost_on_recall(0.5, 0.9)
    assert new_imp > 0.5
    assert new_imp <= 1.0

    # Low score does nothing
    same_imp = boost_on_recall(0.5, 0.1)
    assert same_imp == 0.5

    # Cannot exceed 1.0
    capped = boost_on_recall(0.99, 1.0)
    assert capped <= 1.0


def test_apply_decay_lowers_importance(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ulid = eng.remember("Q", "A", importance=0.7)

    # Backdate event so recent_boost не применяется
    import sqlite3
    with sqlite3.connect(eng.workspace.db_path) as conn:
        old_ts = time.time() - 30 * 86400
        conn.execute(
            "UPDATE events SET timestamp = ?, last_accessed = ? WHERE ulid = ?",
            (old_ts, old_ts, ulid),
        )

    result = eng.apply_daily_decay(days_since=30)
    assert result["n_decayed"] >= 1

    after = eng.events.get_by_ulid(ulid)
    assert after.importance < 0.7


def test_apply_decay_skips_pinned(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    ulid = eng.remember("Q", "A")
    eng.pin(ulid)

    eng.apply_daily_decay(days_since=10)
    after = eng.events.get_by_ulid(ulid)
    assert after.importance >= 0.99  # pinned stays


def test_recall_boosts_importance(tmp_pmb_home, tmp_workspace_dir):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    # Нужно несколько docs чтобы score normalization работал
    target_ulid = eng.remember(
        "What database?",
        "Postgres 17 on port 5432, deployed via docker-compose",
        importance=0.5,
    )
    eng.remember("Frontend?", "React with Tailwind CSS")
    eng.remember("Auth?", "JWT with refresh tokens in Redis")

    # Hit с похожим запросом
    eng.recall("postgres database setup")

    after = eng.events.get_by_ulid(target_ulid)
    assert after.importance > 0.5  # boosted by recall hit


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

def test_capture_recent_commits(git_repo):
    commits = capture_recent_commits(git_repo, since_timestamp=time.time() - 3600)
    assert len(commits) == 3
    # Most recent first
    assert commits[0].subject.startswith("feat: add file_2")
    assert commits[0].author == "Test"
    assert commits[0].insertions > 0
    assert commits[0].files_added >= 1


def test_capture_skips_when_not_git_repo(tmp_workspace_dir):
    commits = capture_recent_commits(tmp_workspace_dir)
    assert commits == []


def test_git_sync_imports_commits(tmp_pmb_home, git_repo):
    eng = Engine(cwd=git_repo, pmb_home=tmp_pmb_home)
    sync = GitSync(eng)

    result = sync.sync()
    assert result["captured"] == 3
    assert result["skipped_existing"] == 0
    assert result["branch"] in ("main", "master")

    # Re-run должен skip
    result2 = sync.sync()
    assert result2["captured"] == 0
    assert result2["skipped_existing"] == 3


def test_git_sync_recall(tmp_pmb_home, git_repo):
    eng = Engine(cwd=git_repo, pmb_home=tmp_pmb_home)
    eng.sync_git()

    pack = eng.recall("file_2")
    assert len(pack.results) > 0
    # Top hit should be the file_2 commit
    top = pack.results[0]
    assert top.event_type == "git"
    assert "file_2" in top.content


def test_git_sync_metadata_complete(tmp_pmb_home, git_repo):
    eng = Engine(cwd=git_repo, pmb_home=tmp_pmb_home)
    eng.sync_git()

    git_events = eng.events.list_active(eng.workspace.id, event_type="git")
    assert len(git_events) == 3
    for ev in git_events:
        assert "sha" in ev.metadata
        assert "short_sha" in ev.metadata
        assert "author" in ev.metadata
        assert "files_changed" in ev.metadata


# ---------------------------------------------------------------------------
# File correlation
# ---------------------------------------------------------------------------

def test_file_correlation_finds_co_changes(tmp_pmb_home):
    """Создаём fake git events с overlapping files и проверяем correlation."""
    with tempfile.TemporaryDirectory() as tmp:
        eng = Engine(cwd=Path(tmp), pmb_home=tmp_pmb_home)

        # Manually create 3 git events with overlapping files
        eng.record_event(
            event_type="git",
            content="commit 1",
            metadata={"files_changed": ["a.py", "b.py"], "sha": "111"},
        )
        eng.record_event(
            event_type="git",
            content="commit 2",
            metadata={"files_changed": ["a.py", "c.py"], "sha": "222"},
        )
        eng.record_event(
            event_type="git",
            content="commit 3",
            metadata={"files_changed": ["a.py", "b.py", "d.py"], "sha": "333"},
        )

        corr = FileCorrelation(eng)
        related = corr.correlations("a.py", top_k=10)
        # b.py появляется 2 раза с a.py, c.py 1 раз, d.py 1 раз
        related_dict = dict(related)
        assert related_dict.get("b.py") == 2
        assert related_dict.get("c.py") == 1
        assert related_dict.get("d.py") == 1


def test_file_history(tmp_pmb_home):
    with tempfile.TemporaryDirectory() as tmp:
        eng = Engine(cwd=Path(tmp), pmb_home=tmp_pmb_home)
        eng.record_event(
            event_type="git",
            content="commit X",
            metadata={"files_changed": ["target.py"], "sha": "abc",
                      "short_sha": "abc1234", "subject": "fix target",
                      "author": "alice"},
        )

        corr = FileCorrelation(eng)
        history = corr.file_history("target.py")
        assert len(history) == 1
        assert history[0]["sha"] == "abc1234"
        assert history[0]["author"] == "alice"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
