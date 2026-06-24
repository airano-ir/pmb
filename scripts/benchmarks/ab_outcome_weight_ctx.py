"""Controlled ablation: does CONTEXTUAL outcome-weighting beat baseline?

Global outcome-weighting failed (-1pp): boosting "useful in general" displaces
query-specific evidence. The fix to test: condition usefulness on the QUERY
TYPE. LoCoMo questions carry a category (1-5). We learn usefulness PER CATEGORY
from train questions, and when answering a held-out test question of category
C, boost ONLY the events that were useful for category C.

Per-question protocol (apples-to-apples, real recall pipeline):
  for each TEST question q (category C):
    - baseline:   recall(q) with default importance -> evidence hit?
    - contextual: boost importance of useful[C] events, recall(q), restore -> hit?
  aggregate evidence_recall@10 across all test questions, plus per-category.

Runs across N conversations for robustness. Theory holds iff contextual recall
> baseline by a real margin. If <= 0, the learning idea is dead on this data
even in its smartest (contextual) form.

Run:
    PYTHONPATH=src python scripts/benchmarks/ab_outcome_weight_ctx.py --n 3
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parent.parent.parent / "src"))
sys.path.insert(0, str(_here.parent))
sys.path.insert(0, str(_here.parent.parent))

from _bench_data import data_path  # noqa: E402
from benchmark_locomo import _evidence_hit, ingest_conversation  # noqa: E402

BOOST = 0.95
TOP_K = 10


def run_conv(conv, chunk_by):
    from pmb.core.engine import Engine
    qa = [q for q in (conv.get("qa") or []) if q.get("evidence")]
    if len(qa) < 8:
        return None
    half = len(qa) // 2
    train, test = qa[:half], qa[half:]

    tmp_home = Path(tempfile.mkdtemp(prefix="pmb-ctx-h-"))
    tmp_ws = Path(tempfile.mkdtemp(prefix="pmb-ctx-w-"))
    os.environ["PMB_HOME"] = str(tmp_home)
    eng = Engine(cwd=tmp_ws, pmb_home=tmp_home, rerank_model=None,
                 config_overrides={"recall.cache_size": 0,
                                   "recall.spreading_activation": False,
                                   "recall.graph_boost": 0.15,
                                   "dedup.enable": False})
    ingest_conversation(eng, conv.get("conversation") or {}, chunk_by=chunk_by)
    eng.wait_for_embed_queue(timeout_seconds=240.0)

    import sqlite3
    con = sqlite3.connect(eng.workspace.db_path); con.row_factory = sqlite3.Row
    dia2ulids = defaultdict(list); orig = {}
    for r in con.execute("SELECT ulid,importance,metadata_json FROM events WHERE archived_at IS NULL"):
        orig[r["ulid"]] = r["importance"]
        try:
            md = json.loads(r["metadata_json"] or "{}")
        except Exception:
            md = {}
        for d in ([md["dia_id"]] if md.get("dia_id") else []) + (md.get("dia_ids") or []):
            dia2ulids[d].append(r["ulid"])

    # Learn usefulness per category from TRAIN.
    useful_by_cat = defaultdict(lambda: defaultdict(int))
    for q in train:
        c = q.get("category")
        for d in (q.get("evidence") or []):
            for u in dia2ulids.get(d, []):
                useful_by_cat[c][u] += 1

    base_hits = ctx_hits = 0
    by_cat = defaultdict(lambda: [0, 0, 0])  # cat -> [n, base, ctx]
    for q in test:
        question = q.get("question", ""); c = q.get("category")
        gold = set(q.get("evidence", []) or [])
        # baseline
        pack = eng.recall(question, top_k=TOP_K)
        hb = int(any(_evidence_hit(r.metadata or {}, gold) for r in pack.results))
        # contextual boost (only this category's useful events)
        bset = list(useful_by_cat.get(c, {}).keys())
        for u in bset:
            eng.events.update_importance(u, BOOST)
        eng.recall_cache.bump_generation()
        pack2 = eng.recall(question, top_k=TOP_K)
        hc = int(any(_evidence_hit(r.metadata or {}, gold) for r in pack2.results))
        for u in bset:
            eng.events.update_importance(u, orig[u])
        eng.recall_cache.bump_generation()
        base_hits += hb; ctx_hits += hc
        by_cat[c][0] += 1; by_cat[c][1] += hb; by_cat[c][2] += hc
    return {"n_test": len(test), "base": base_hits, "ctx": ctx_hits,
            "by_cat": {k: v for k, v in by_cat.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=data_path("locomo10.json"))
    ap.add_argument("--n", type=int, default=3, help="conversations")
    ap.add_argument("--chunk-by", choices=("turn", "session"), default="turn")
    args = ap.parse_args()
    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"CONTEXTUAL outcome-weighting — {args.n} conv(s), chunk_by={args.chunk_by}, top_k={TOP_K}\n")
    tot_n = tot_b = tot_c = 0
    cat_agg = defaultdict(lambda: [0, 0, 0])
    for i in range(min(args.n, len(dataset))):
        t0 = time.time()
        r = run_conv(dataset[i], args.chunk_by)
        if not r:
            continue
        b = 100 * r["base"] / r["n_test"]; c = 100 * r["ctx"] / r["n_test"]
        print(f"  conv-{i}: test={r['n_test']:>3}  base={b:.1f}%  ctx={c:.1f}%  "
              f"delta={c-b:+.1f}pp  ({time.time()-t0:.0f}s)")
        tot_n += r["n_test"]; tot_b += r["base"]; tot_c += r["ctx"]
        for k, v in r["by_cat"].items():
            cat_agg[k][0] += v[0]; cat_agg[k][1] += v[1]; cat_agg[k][2] += v[2]

    print("\n" + "=" * 56)
    B = 100 * tot_b / max(tot_n, 1); C = 100 * tot_c / max(tot_n, 1)
    print(f"OVERALL  test={tot_n}  baseline={B:.1f}%  contextual={C:.1f}%  delta={C-B:+.1f}pp")
    print("-" * 56)
    print(f"{'cat':>4}{'n':>6}{'base':>9}{'ctx':>9}{'delta':>9}")
    for k in sorted(cat_agg, key=lambda x: (x is None, x)):
        n, b, c = cat_agg[k]
        if n:
            print(f"{str(k):>4}{n:>6}{100*b/n:>8.1f}%{100*c/n:>8.1f}%{100*(c-b)/n:>+8.1f}pp")
    print("=" * 56)
    if C - B > 1.0:
        print("VERDICT: contextual weighting HELPS -> the learning idea has a real, "
              "query-conditioned signal. This is the one that works.")
    elif C - B < -1.0:
        print("VERDICT: contextual weighting HURTS -> learning idea dead on this data.")
    else:
        print("VERDICT: flat (within noise) -> no usable signal even contextually.")


if __name__ == "__main__":
    main()
