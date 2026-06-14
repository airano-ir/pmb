"""Corpus-IDF surfacing gate (precision fix for auto-recall).

The plain `_ov >= 1` gate surfaced any lesson sharing ONE globally-distinctive
token with the message. In a workspace where most lessons are about the same
project, a word like "memory"/"recall"/"build" is globally distinctive but
LOCALLY generic (it recurs across a large fraction of lessons), so off-topic
lessons surfaced on a single such overlap - the main false-positive channel
observed live.

The fix (recall.lesson_idf_gate, default on): a lesson must share a token that
is RARE in THIS workspace's lesson corpus. Derived per-call from the candidate
lessons (document frequency), guarded by a minimum corpus size so tiny/new
workspaces keep the legacy behaviour exactly.
"""
from __future__ import annotations

from pmb.core.engine import Engine


def _engine(ws, home, **overrides):
    cfg = {"recall.cache_size": 0, "dedup.enable": False}  # deterministic corpus (no model-warmth-dependent semantic dedup)
    cfg.update(overrides)
    return Engine(cwd=ws, pmb_home=home, config_overrides=cfg)


def _rec(eng, content):
    eng.record_fact(content, metadata={"kind": "lesson", "source": "lesson"})


# A query about a SPECIFIC, rare topic. "pnpm"/"package"/"manager" are rare in
# the corpus below; "build" is made corpus-generic by the fillers.
QUERY = "how do I configure the pnpm package manager build"

# Shares the RARE tokens pnpm/package with the query -> must surface.
ON_TOPIC = "Build tooling rule: always use pnpm, never npm, for the package install."

# Shares ONLY the corpus-generic token "build" with the query -> must be
# dropped once the corpus is big enough to know "build" is generic here.
OFF_TOPIC = "PMB build pipeline memo: the recall daemon warms on a cold start."


def _seed_big_pmb_corpus(eng, n_fillers=14):
    # Every filler repeats the workspace-generic words build/recall/memory/
    # daemon/embedding, so their document frequency is high (>25%).
    for i in range(n_fillers):
        _rec(eng, f"PMB build pipeline note {i}: the recall memory daemon "
                  f"warms the embedding subsystem widget{i}.")
    _rec(eng, ON_TOPIC)
    _rec(eng, OFF_TOPIC)


def _contents(hits):
    return [(h.get("content") or "").lower() for h in hits]


def test_generic_token_lesson_is_dropped_in_big_corpus(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)  # gate on by default
    _seed_big_pmb_corpus(eng)

    hits = eng.find_lessons(query=QUERY, limit=10)
    joined = " || ".join(_contents(hits))

    # the rare-token match surfaces ...
    assert any("pnpm" in c for c in _contents(hits)), \
        f"on-topic pnpm lesson must surface; got: {joined}"
    # ... and the generic-only ("build") off-topic lesson does NOT
    assert not any("warms on a cold start" in c for c in _contents(hits)), \
        f"off-topic lesson sharing only the corpus-generic 'build' must be gated out; got: {joined}"
    # none of the fillers (overlap only on generic 'build') surface either
    assert not any("widget" in c for c in _contents(hits)), \
        f"generic-only filler lessons must not surface; got: {joined}"


def test_gate_off_restores_legacy_noise(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home,
                  **{"recall.lesson_idf_gate": False})
    _seed_big_pmb_corpus(eng)

    hits = eng.find_lessons(query=QUERY, limit=20)
    # with the gate OFF, the single-generic-token match comes back (legacy)
    assert any("warms on a cold start" in c for c in _contents(hits)), \
        "with the IDF gate off, the legacy overlap>=1 behaviour must still surface the noise"


def test_small_corpus_keeps_legacy_behaviour(tmp_pmb_home, tmp_workspace_dir):
    # Below recall.lesson_idf_min_corpus (default 12) there is not enough data
    # to call a token 'corpus-generic', so the legacy gate is used unchanged.
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    _rec(eng, ON_TOPIC)
    _rec(eng, OFF_TOPIC)  # only 2 lessons total

    hits = eng.find_lessons(query=QUERY, limit=10)
    # both surface (legacy overlap>=1) - the gate must NOT engage on a tiny corpus
    assert any("warms on a cold start" in c for c in _contents(hits)), \
        "on a tiny corpus the gate must stay inactive (legacy behaviour preserved)"


def test_rare_token_still_surfaces_under_gate(tmp_pmb_home, tmp_workspace_dir):
    # Regression guard: the gate must not suppress a genuine rare-token match.
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    for i in range(14):
        _rec(eng, f"PMB recall memory daemon note {i}: embedding widget{i}.")
    _rec(eng, "Deployment secret zircondeploytoken must be rotated weekly.")

    hits = eng.find_lessons(query="how to rotate zircondeploytoken", limit=5)
    assert any("zircondeploytoken" in c for c in _contents(hits)), \
        "a rare identifier match must still surface under the corpus gate"
