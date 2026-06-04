"""A/B test on LoCoMo conv-30 + conv-41: pattern_split ON vs OFF.

Isolates whether pattern_split is responsible for the regression
post-hardening (94.4% -> 90.8%). Tests just 2 conversations to save time.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _bench_data import data_path

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parent.parent.parent / "src"))
sys.path.insert(0, str(_here.parent))

from benchmark_locomo import ingest_conversation, evaluate  # type: ignore

DEFAULT_LOCOMO_JSON = data_path("locomo10.json")

def _load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


CONVS_TO_TEST = ["conv-30", "conv-41"]


def run(pattern_split: bool, vocab_bridges: bool, label: str) -> dict:
    from pmb.core.engine import Engine
    data = _load(DEFAULT_LOCOMO_JSON)
    results = {}
    for conv in data:
        sample_id = conv.get("sample_id", "")
        if sample_id not in CONVS_TO_TEST:
            continue
        # benchmark_locomo expects {"conversation": {session_X: ...}, "qa": [...]}
        # but the raw dict already has the nested keys when iterated
        tmp_home = Path(tempfile.mkdtemp())
        tmp_ws = Path(tempfile.mkdtemp())
        os.environ["PMB_HOME"] = str(tmp_home)
        eng = Engine(
            cwd=tmp_ws, pmb_home=tmp_home, rerank_model=None,
            config_overrides={
                "recall.cache_size": 0,
                "recall.spreading_activation": False,
                "recall.pattern_split": pattern_split,
                "recall.auto_vocab_bridges": vocab_bridges,
            },
        )
        _ = eng.search.model
        conversation = conv.get("conversation") or {}
        qa_list = conv.get("qa") or []
        t0 = time.time()
        ingest_conversation(eng, conversation)
        t_ing = time.time() - t0
        t0 = time.time()
        stats = evaluate(eng, qa_list, top_k=10)
        t_eval = time.time() - t0
        print(f"  [{label}] {sample_id}: "
              f"recall@10 = {stats['overall']['evidence_recall_top_k']:.2%}, "
              f"ingest {t_ing:.1f}s eval {t_eval:.1f}s")
        results[sample_id] = stats["overall"]["evidence_recall_top_k"]
        try: eng.close()
        except: pass
    return results


if __name__ == "__main__":
    print("A/B: pattern_split + auto_vocab_bridges effect on conv-30/41\n")
    print("Run A: both OFF (baseline behaviour pre-hardening)")
    a = run(pattern_split=False, vocab_bridges=False, label="OFF/OFF")
    print(f"  mean: {sum(a.values())/len(a):.2%}\n")
    print("Run B: both ON (current hardened defaults)")
    b = run(pattern_split=True, vocab_bridges=True, label="ON/ON")
    print(f"  mean: {sum(b.values())/len(b):.2%}\n")
    print("Run C: pattern_split ON, auto_bridges OFF")
    c = run(pattern_split=True, vocab_bridges=False, label="ON/OFF")
    print(f"  mean: {sum(c.values())/len(c):.2%}\n")

    print("=" * 60)
    print(f"  baseline (off/off): {sum(a.values())/len(a):.2%}")
    print(f"  hardened (on/on):   {sum(b.values())/len(b):.2%}  delta = {(sum(b.values())-sum(a.values()))/len(a)*100:+.1f}pp")
    print(f"  split-only (on/off):{sum(c.values())/len(c):.2%}  delta = {(sum(c.values())-sum(a.values()))/len(a)*100:+.1f}pp")
