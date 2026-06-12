"""F3 — the X2 channel-weights learning loop: collect weak-labelled samples →
propose in the tick → surface (never auto-applied). Hermetic (no model)."""
from __future__ import annotations

from pmb.core.engine import Engine
from pmb.reasoning.weight_learning import (
    load_samples,
    load_suggestion,
    note_recall_useful,
    propose_from_samples,
    record_channel_sample,
)


def _eng(ws, home):
    return Engine(cwd=ws, pmb_home=home, config_overrides={"recall.cache_size": 0})


def test_propose_and_cache_suggestion(tmp_pmb_home, tmp_workspace_dir):
    eng = _eng(tmp_workspace_dir, tmp_pmb_home)
    # useful recalls coincided with HIGH importance; not-useful with low.
    for i in range(26):
        useful = (i % 2 == 0)
        record_channel_sample(
            eng, {"hit": 0.5, "importance": 0.9 if useful else 0.2},
            ulid=f"u{i}", useful=useful)
    out = propose_from_samples(eng, min_samples=20)
    assert "weights" in out and out["n_useful"] > 0
    # importance coincided with useful → its weight nudges up; never auto-applied.
    assert out["weights"]["importance"] >= 1.0
    s = load_suggestion(eng)
    assert s and s["weights"]["importance"] >= 1.0


def test_note_recall_useful_flips_label(tmp_pmb_home, tmp_workspace_dir):
    eng = _eng(tmp_workspace_dir, tmp_pmb_home)
    record_channel_sample(eng, {"hit": 0.5}, ulid="ux", useful=False)
    note_recall_useful(eng, "ux")
    assert any(ok for _c, ok in load_samples(eng))


def test_insufficient_samples_skips(tmp_pmb_home, tmp_workspace_dir):
    eng = _eng(tmp_workspace_dir, tmp_pmb_home)
    record_channel_sample(eng, {"hit": 0.5}, ulid="u", useful=True)
    out = propose_from_samples(eng, min_samples=20)
    assert "skipped" in out
    assert load_suggestion(eng) is None
