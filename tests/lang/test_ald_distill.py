"""Phase D — Anchor→Lexicon Distillation (ALD).

D1 (anchor-fire log) + D2 (distiller in the maintenance tick). The headline
property under test: a workspace that sees Polish goals-queries (which fired the
anchors on the warm path) compiles "zostało/zrobienia" into auto.yaml, after
which the COLD lexical regex classifies the Polish phrasing with no model and no
Polish anywhere in the codebase. All hermetic — no embedder is loaded.
"""
from __future__ import annotations

import re

import pmb.lang as _lang
from pmb.core.engine import Engine
from pmb.maintenance.distill import (
    distill_lexicon,
    load_anchor_log,
    log_anchor_fire,
    message_ngrams,
)


def _engine(ws, home):
    return Engine(cwd=ws, pmb_home=home, config_overrides={"recall.cache_size": 0})


def test_message_ngrams_unigram_to_trigram():
    g = message_ngrams("Co mi zostało")
    assert "co" in g and "zostało" in g            # unigrams, lowercased
    assert "co mi" in g and "mi zostało" in g      # bigrams
    assert "co mi zostało" in g                    # trigram


def test_log_and_load_roundtrip(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    log_anchor_fire(eng, "intent.goals_query", "co zostało zrobienia")
    log_anchor_fire(eng, "intent.goals_query", "co zostało zrobienia")
    log_anchor_fire(eng, "intent.past_query", "kiedy wybralismy baze")
    per, total = load_anchor_log(eng)
    assert per["intent.goals_query"]["zostało"] == 2
    assert per["intent.past_query"]["wybralismy"] == 1
    assert total["zostało"] == 2          # only goals fired it


def _seed_polish(eng, n_goals=6, n_past=6, n_shared=3):
    for _ in range(n_goals):
        log_anchor_fire(eng, "intent.goals_query", "co zostało zrobienia")
    for _ in range(n_past):
        log_anchor_fire(eng, "intent.past_query", "kiedy wybralismy baze")
    # A contentful word that fires TWO anchors equally → precision 0.5 → pruned.
    for _ in range(n_shared):
        log_anchor_fire(eng, "intent.goals_query", "wspolny projekt temat")
    for _ in range(n_shared):
        log_anchor_fire(eng, "intent.lessons_query", "wspolny projekt temat")


def test_distill_precision_support_and_contentfulness(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    _seed_polish(eng)
    out = distill_lexicon(eng, min_support=6, min_precision=0.95)
    assert out["entries"] > 0

    goals = [str(x) for x in _lang.merged_list("intent_goals_query")]
    past = [str(x) for x in _lang.merged_list("intent_past_query")]
    # high-precision, supported, contentful n-grams compiled into the right cat
    assert re.escape("zostało") in goals
    assert re.escape("zrobienia") in goals
    assert re.escape("wybralismy") in past
    # the ambiguous word (goals 3 / lessons 3 → precision 0.5) must be pruned
    everywhere = goals + past + [str(x) for x in _lang.merged_list("intent_lessons_query")]
    assert re.escape("projekt") not in everywhere
    assert re.escape("wspolny") not in everywhere


def test_low_support_ngram_excluded(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    for _ in range(3):                              # only 3 < min_support 6
        log_anchor_fire(eng, "intent.goals_query", "rzadkie slowo zadanie")
    distill_lexicon(eng, min_support=6, min_precision=0.95)
    goals = [str(x) for x in _lang.merged_list("intent_goals_query")]
    assert re.escape("rzadkie") not in goals
    assert re.escape("zadanie") not in goals


def test_cold_path_learns_polish_end_to_end(tmp_pmb_home, tmp_workspace_dir):
    """The proof line: after distillation, a regex built from the packs (exactly
    what detect_intents compiles) classifies the Polish goals-query — cold, no
    model. Built fresh here so we don't mutate the global auto_recall module."""
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    _seed_polish(eng)
    distill_lexicon(eng, min_support=6, min_precision=0.95)

    from pmb.hooks.auto_recall import _ialt
    body = _ialt([r"\bwhat'?s\s+left\b"], "intent_goals_query")
    pat = re.compile(body, re.IGNORECASE)
    assert pat.search("teraz co mi zostało do zrobienia"), \
        "cold lexical path did not learn the distilled Polish phrasing"
    assert pat.search("what's left to do"), "EN inline floor must still match"


def test_distill_step_runs_in_maintenance_tick(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    _seed_polish(eng)
    from pmb.maintenance.tick import run_maintenance_tick
    summary = run_maintenance_tick(eng, archive=False)
    assert "distill" in summary["steps"]
    step = summary["steps"]["distill"]
    assert "error" not in step
    assert step.get("entries", 0) > 0


def test_distill_noop_on_empty_log(tmp_pmb_home, tmp_workspace_dir):
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    out = distill_lexicon(eng)
    assert out["entries"] == 0
    assert not (tmp_pmb_home / "lang" / "auto.yaml").exists()


def test_d3_prune_ages_out_old_fires(tmp_pmb_home, tmp_workspace_dir):
    import sqlite3
    import time

    from pmb.maintenance.distill import prune_anchor_log
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    log_anchor_fire(eng, "intent.goals_query", "co zostalo zrobienia")   # recent
    # Inject an OLD fire (40 days ago) directly.
    with sqlite3.connect(str(eng.workspace.db_path)) as c:
        c.execute(
            "INSERT INTO anchor_fires (ts, workspace_id, anchor, text_hash, "
            "ngrams_json) VALUES (?,?,?,?,?)",
            (time.time() - 40 * 86400, eng.workspace.id, "intent.goals_query",
             "old", '["dawne", "slowo"]'))
    deleted = prune_anchor_log(eng, 30)
    assert deleted == 1
    _per, total = load_anchor_log(eng)
    assert "zostalo" in total and "dawne" not in total


# ── D3 shadow-T1: live-precision prune ────────────────────────────────────────


def test_shadow_precision_aggregates(tmp_pmb_home, tmp_workspace_dir):
    from pmb.maintenance.distill import load_shadow_precision, record_shadow_t1
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    record_shadow_t1(eng, "GOALS_QUERY", True)
    record_shadow_t1(eng, "GOALS_QUERY", False)
    record_shadow_t1(eng, "GOALS_QUERY", True)
    assert load_shadow_precision(eng)["GOALS_QUERY"] == (2, 3)


def test_shadow_drops_low_precision_category(tmp_pmb_home, tmp_workspace_dir):
    from pmb.maintenance.distill import record_shadow_t1
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    for _ in range(6):
        log_anchor_fire(eng, "intent.goals_query", "co zostalo zrobienia")
    for i in range(15):                       # 2/15 = 0.13 < 0.9 over >=10 samples
        record_shadow_t1(eng, "GOALS_QUERY", agreed=(i < 2))
    out = distill_lexicon(eng, min_support=6, min_precision=0.95)
    assert "intent_goals_query" in out.get("shadow_dropped", [])
    assert re.escape("zostalo") not in [str(x) for x in _lang.merged_list("intent_goals_query")]


def test_shadow_keeps_high_precision_category(tmp_pmb_home, tmp_workspace_dir):
    from pmb.maintenance.distill import record_shadow_t1
    eng = _engine(tmp_workspace_dir, tmp_pmb_home)
    for _ in range(6):
        log_anchor_fire(eng, "intent.goals_query", "co zostalo zrobienia")
    for i in range(15):                       # 14/15 = 0.93 >= 0.9 → kept
        record_shadow_t1(eng, "GOALS_QUERY", agreed=(i < 14))
    out = distill_lexicon(eng, min_support=6, min_precision=0.95)
    assert "intent_goals_query" not in out.get("shadow_dropped", [])
    assert re.escape("zostalo") in [str(x) for x in _lang.merged_list("intent_goals_query")]
