"""T1 (hardcode removal): the identity-marker boost and the router's
identity-intent detection must NOT hardcode a specific personal name
("alex"). Identity-by-name is driven by the mined user-name cache instead.

Covers:
  * router static identity regex still detects GENERIC self-words;
  * identity-by-NAME fires only for a KNOWN user name (the cache), never a
    baked-in literal;
  * the engine identity-marker boost lifts a fact that opens with the
    user's own learned name, and the generic "My ..." marker still works
    without any name on file;
  * a source guard: no "alex" literal survives in recall.py / router.py.
"""
from __future__ import annotations

from pathlib import Path

from pmb.core.engine import Engine
from pmb.reasoning.router import (
    QueryRouter,
    _identity_re_for_names,
)

# ── router: pure-function tests (fast, no engine) ──────────────────────────

def test_static_identity_regex_detects_generic_self():
    r = QueryRouter()
    assert "identity" in r.classify("who is the user").types
    assert "identity" in r.classify("what's my editor").types
    assert "identity" in r.classify("where do i live").types
    assert "identity" in r.classify("who am i working with").types


def test_identity_by_name_requires_known_user_name():
    r = QueryRouter()
    # Unknown name → NOT identity (we don't know the user is "bob").
    assert "identity" not in r.classify("where does bob live").types
    # Same query, bob is a known user name → identity fires.
    assert "identity" in r.classify(
        "where does bob live", user_names={"bob"}
    ).types
    # A different (non-user) name stays non-identity even with a name set.
    assert "identity" not in r.classify(
        "where does zorg live", user_names={"bob"}
    ).types


def test_identity_by_name_covers_the_old_alex_patterns():
    """The patterns that used to be hardcoded for "alex" now work for any
    known name."""
    r = QueryRouter()
    names = {"bob"}
    for q in (
        "who is bob",
        "what's bob's email",
        "where does bob work",
        "what languages does bob use",
        "what does bob prefer",
    ):
        assert "identity" in r.classify(q, user_names=names).types, q


def test_identity_re_for_names_empty_and_cyrillic():
    assert _identity_re_for_names(()) is None
    rx = _identity_re_for_names(("алексей",))
    assert rx is not None
    # Cyrillic name slots into the English identity frame (re.escape-safe).
    assert rx.search("who is алексей")


# ── source guard ───────────────────────────────────────────────────────────

def test_no_personal_name_literal_in_recall_and_router():
    src = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "src" / "pmb"
    for rel in ("core/engine/recall.py", "reasoning/router.py"):
        text = (src / rel).read_text(encoding="utf-8").lower()
        assert "alex" not in text, f"personal-name literal 'alex' found in {rel}"


# ── engine fixtures ─────────────────────────────────────────────────────────





def _engine(ws, home, **over):
    cfg = {"recall.cache_size": 0}
    cfg.update(over)
    return Engine(cwd=ws, pmb_home=home, config_overrides=cfg)


def _score_by_head(pack, n=4):
    return {(r.content or "")[:n].lower(): r.score for r in pack.results}


def test_name_prefixed_fact_boosted_when_name_known(tmp_pmb_home, tmp_workspace_dir):
    """A fact opening with the user's OWN (learned) name outranks an
    identical fact opening with a stranger's name, on an identity query."""
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.record_fact("My name is Bob")
    eng.record_fact("Bob's editor is vim and tmux")
    eng.record_fact("Zorg's editor is vim and tmux")

    pack = eng.recall("what's my editor", top_k=5)
    scores = _score_by_head(pack)
    assert "bob'" in scores and "zorg" in scores, [r.content for r in pack.results]
    # Bob (known user name) beats Zorg (unknown) purely on the
    # identity-marker boost — the texts are otherwise identical.
    assert scores["bob'"] > scores["zorg"]


def test_generic_my_marker_boost_without_any_name(tmp_pmb_home, tmp_workspace_dir):
    """No name on file: the generic first-person marker ("My ...") still
    earns the identity boost and ranks the personal fact first."""
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.record_fact("My terminal is wezterm")
    eng.record_fact("The api service listens on port 5433")

    pack = eng.recall("what's my terminal", top_k=5)
    assert pack.results, "no results"
    assert (pack.results[0].content or "").lower().startswith("my terminal")


# ── A8: a freshly recorded name takes effect on the NEXT recall ─────────────

def test_new_name_takes_effect_immediately(tmp_pmb_home, tmp_workspace_dir):
    """Recording "My name is Bob" must influence the very next recall — not
    after 25 more events. Warm the (empty) cache first, THEN record the name."""
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.record_fact("The api service listens on port 5433")
    # Warm the name cache while it is still empty (no name on file yet).
    eng.recall("anything", top_k=3)
    assert eng._get_user_names() == set()

    # Name arrives AFTER the cache was warmed.
    eng.record_fact("My name is Bob")
    # Picked up immediately — no 25-event wait.
    assert "bob" in eng._get_user_names()

    eng.record_fact("Bob's editor is vim and tmux")
    eng.record_fact("Zorg's editor is vim and tmux")
    scores = _score_by_head(eng.recall("what's my editor", top_k=5))
    assert scores.get("bob'", 0) > scores.get("zorg", 0)


def test_user_names_cache_not_remined_per_recall(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    """After a refresh, repeated _get_user_names() calls with no writes must NOT
    re-run the (SQL) miner — the common path is a cheap flag check."""
    import pmb.core.engine.recall as _rc
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    eng.record_fact("My name is Bob")
    eng._get_user_names()  # one real refresh (dirty after the name write)

    calls = {"n": 0}
    real = _rc._mine_user_names

    def _counting(db_path):
        calls["n"] += 1
        return real(db_path)
    monkeypatch.setattr(_rc, "_mine_user_names", _counting)

    for _ in range(10):
        eng._get_user_names()
    assert calls["n"] == 0  # served from cache, no re-mine within TTL
