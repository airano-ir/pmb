"""Full combined ablation — every proposed lever, measured fresh on ground
truth, in isolation AND all together. One ingested store per conversation,
config toggled live via Config._overrides (no rebuild, no shared-state lies).

Levers:
  - candidate-gen : recall.ppr_always=True (ungate the graph/PPR channel so it
                    pulls evidence BM25+vector miss). This is the only stage
                    that can add a missed candidate.
  - surprise-gate : archive events whose earlier near-twin cosine >= 0.90
                    (the safe compression point), excluding them from recall.
  - outcome-weight: per test-question, boost importance of events that were
                    gold evidence for TRAIN questions of the SAME category.
  - ALL           : the three combined.

Metric: evidence_recall@10 on held-out TEST questions (2nd half of QA).
Per-question loop so outcome-weight can condition on the question's category.

Run:
    PYTHONPATH=src python scripts/benchmarks/ab_all.py --n 3
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

import numpy as np  # noqa: E402

from _bench_data import data_path  # noqa: E402
from benchmark_locomo import _evidence_hit, ingest_conversation  # noqa: E402

GATE = 0.90
BOOST = 0.95
TOP_K = 10
ARMS = ["baseline", "cand_gen", "surprise", "outcome", "ALL"]


def run_conv(conv, chunk_by):
    from pmb.core.engine import Engine
    qa = [q for q in (conv.get("qa") or []) if q.get("evidence")]
    if len(qa) < 8:
        return None
    half = len(qa) // 2
    train, test = qa[:half], qa[half:]

    home = Path(tempfile.mkdtemp(prefix="pmb-all-h-"))
    ws = Path(tempfile.mkdtemp(prefix="pmb-all-w-"))
    os.environ["PMB_HOME"] = str(home)
    eng = Engine(cwd=ws, pmb_home=home, rerank_model=None,
                 config_overrides={"recall.cache_size": 0,
                                   "recall.spreading_activation": False,
                                   "recall.graph_boost": 0.15,
                                   "dedup.enable": False})
    ingest_conversation(eng, conv.get("conversation") or {}, chunk_by=chunk_by)
    eng.wait_for_embed_queue(timeout_seconds=240.0)

    import sqlite3
    con = sqlite3.connect(eng.workspace.db_path); con.row_factory = sqlite3.Row
    dia2ulids = defaultdict(list); orig = {}
    rows = con.execute("SELECT ulid,importance,timestamp,metadata_json "
                       "FROM events WHERE archived_at IS NULL ORDER BY timestamp").fetchall()
    for r in rows:
        orig[r["ulid"]] = r["importance"]
        try:
            md = json.loads(r["metadata_json"] or "{}")
        except Exception:
            md = {}
        for d in ([md["dia_id"]] if md.get("dia_id") else []) + (md.get("dia_ids") or []):
            dia2ulids[d].append(r["ulid"])

    # surprise-gate set: events with an earlier twin >= GATE.
    import lancedb
    tbl = lancedb.connect(str(Path(eng.workspace.db_path).parent / "vectors.lance")).open_table("events").to_arrow()
    vmap = {u: i for i, u in enumerate(tbl.column("ulid").to_pylist())}
    vecs = np.array(tbl.column("vector").to_pylist(), dtype=np.float32)
    order = [r["ulid"] for r in rows if r["ulid"] in vmap]
    V = vecs[[vmap[u] for u in order]]; V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    S = V @ V.T
    gate_set = [order[i] for i in range(1, len(order)) if S[i, :i].max() >= GATE]

    # outcome-weight: useful events per category, learned from TRAIN.
    useful_by_cat = defaultdict(lambda: defaultdict(int))
    for q in train:
        for d in (q.get("evidence") or []):
            for u in dia2ulids.get(d, []):
                useful_by_cat[q.get("category")][u] += 1

    def restore():
        eng.config._overrides["recall.ppr_always"] = False
        for u in gate_set:
            eng.events.unarchive(u)
        for u, imp in orig.items():
            eng.events.update_importance(u, imp)
        eng.recall_cache.bump_generation()

    def eval_arm(arm):
        restore()
        if arm in ("cand_gen", "ALL"):
            eng.config._overrides["recall.ppr_always"] = True
            eng.config._overrides["recall.ppr_weight"] = 1.0
        if arm in ("surprise", "ALL"):
            for u in gate_set:
                eng.events.archive(u)
        eng.recall_cache.bump_generation()
        hits = 0
        for q in test:
            bset = []
            if arm in ("outcome", "ALL"):
                bset = list(useful_by_cat.get(q.get("category"), {}).keys())
                for u in bset:
                    eng.events.update_importance(u, BOOST)
                eng.recall_cache.bump_generation()
            pack = eng.recall(q.get("question", ""), top_k=TOP_K)
            gold = set(q.get("evidence", []) or [])
            hits += int(any(_evidence_hit(r.metadata or {}, gold) for r in pack.results))
            if bset:
                for u in bset:
                    eng.events.update_importance(u, orig.get(u, 0.5))
                eng.recall_cache.bump_generation()
        return hits

    out = {"n_test": len(test), "gate_n": len(gate_set), "n_events": len(order)}
    for arm in ARMS:
        out[arm] = eval_arm(arm)
    restore()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=data_path("locomo10.json"))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--chunk-by", choices=("turn", "session"), default="turn")
    args = ap.parse_args()
    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"FULL ablation — {args.n} conv(s), chunk_by={args.chunk_by}, top_k={TOP_K}, "
          f"gate={GATE}\n")
    agg = {a: 0 for a in ARMS}; tot = 0
    for i in range(min(args.n, len(dataset))):
        t0 = time.time()
        r = run_conv(dataset[i], args.chunk_by)
        if not r:
            continue
        tot += r["n_test"]
        line = f"  conv-{i}: test={r['n_test']:>3} events={r['n_events']:>3} gate={r['gate_n']:>2}  "
        base = r["baseline"]
        for a in ARMS:
            agg[a] += r[a]
            pct = 100 * r[a] / r["n_test"]
            d = "" if a == "baseline" else f"({100*(r[a]-base)/r['n_test']:+.1f})"
            line += f"{a}={pct:.0f}%{d} "
        line += f"[{time.time()-t0:.0f}s]"
        print(line)

    print("\n" + "=" * 60)
    b = 100 * agg["baseline"] / max(tot, 1)
    print(f"OVERALL (test={tot})")
    for a in ARMS:
        p = 100 * agg[a] / max(tot, 1)
        d = "" if a == "baseline" else f"   delta={p-b:+.1f}pp"
        print(f"  {a:<10} recall@{TOP_K} = {p:.1f}%{d}")
    print("=" * 60)


if __name__ == "__main__":
    main()
