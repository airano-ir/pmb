"""G2 — packs-off ratchet.

With the built-in ru/uk packs DISABLED (PMB_DISABLE_DEFAULT_PACKS=1), the anchor
+ vector tier must STILL carry pack-free multilingual recall. This is the soak
that must hold for two releases before G3 deletes the packs — it proves the v0.9
claim ("no language packs required") on a configuration where the packs are gone.
"""
from __future__ import annotations

import pytest

import pmb.lang as _lang
from pmb.core.engine import Engine


def test_disable_default_packs_env_empties_active(tmp_pmb_home, monkeypatch):
    monkeypatch.setenv("PMB_DISABLE_DEFAULT_PACKS", "1")
    _lang.clear_cache()
    try:
        assert "ru" not in _lang.active_codes()
        assert "uk" not in _lang.active_codes()
    finally:
        monkeypatch.delenv("PMB_DISABLE_DEFAULT_PACKS", raising=False)
        _lang.clear_cache()


@pytest.mark.eval
def test_recall_holds_with_packs_off(tmp_pmb_home, tmp_workspace_dir, monkeypatch):
    monkeypatch.setenv("PMB_DISABLE_DEFAULT_PACKS", "1")
    _lang.clear_cache()
    try:
        eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
        try:
            eng.warmup()
        except Exception:
            pass
        facts = [("de", "Ich wohne in München."),
                 ("es", "Vivo en Madrid."),
                 ("fr", "Je travaille chez Datadog.")]
        ulids = [eng.record_fact(t, metadata={"lang": lang}) for lang, t in facts]
        for drain in ("wait_for_embed_queue", "_drain_embed_queue"):
            try:
                getattr(eng, drain)()
            except Exception:
                pass
        queries = [("wo wohne ich", 0), ("dónde vivo", 1),
                   ("où est-ce que je travaille", 2)]
        hits = 0
        for q, idx in queries:
            pack = eng.recall(query=q, top_k=5)
            if any(getattr(r, "ulid", None) == ulids[idx] for r in pack.results[:3]):
                hits += 1
        assert hits >= 2, f"packs-off multilingual recall regressed: {hits}/3"
    finally:
        monkeypatch.delenv("PMB_DISABLE_DEFAULT_PACKS", raising=False)
        _lang.clear_cache()
