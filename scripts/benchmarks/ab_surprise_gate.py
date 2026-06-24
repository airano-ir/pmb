"""Controlled ablation: does "surprise-gate" (don't store what's predictable
from existing memory) compress the store WITHOUT hurting evidence_recall@10?

WHY POST-HOC (not live dedup): PMB defers embeddings during bulk ingest, and
write-time semantic dedup needs the vector index populated to find a near-twin.
During fast ingest that index is still building, so live dedup never engages
(measured: identical store size at every threshold — an invalid test). So we
do it correctly post-hoc, where every vector exists:

  1. Ingest the whole conversation (dedup OFF), drain the embed queue.
  2. Measure baseline evidence_recall@10 on the FULL store.
  3. For each threshold T: an event is "predictable" if it has an EARLIER
     event with cosine >= T. Archive those (simulating the gate dropping them),
     re-run evaluate() through the real recall pipeline, then un-archive.
  4. Report stored-after-gate (compression) AND evidence_recall@10 (quality).

Theory holds iff a lower T removes a meaningful share of events while
evidence_recall@10 stays ~baseline. The lowest T with ~0 recall loss is the
honest sweet spot. Real ground-truth data, real recall pipeline, one knob.

Run:
    PYTHONPATH=src python scripts/benchmarks/ab_surprise_gate.py --conv 0
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")  # Δ-safe

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parent.parent.parent / "src"))
sys.path.insert(0, str(_here.parent))
sys.path.insert(0, str(_here.parent.parent))  # scripts/ — for _bench_data

import numpy as np  # noqa: E402

from _bench_data import data_path  # noqa: E402
from benchmark_locomo import evaluate, ingest_conversation  # noqa: E402

THRESHOLDS = [0.95, 0.90, 0.85, 0.80]


def _load_vectors(eng):
    import lancedb
    lance_dir = Path(eng.workspace.db_path).parent / "vectors.lance"
    tbl = lancedb.connect(str(lance_dir)).open_table("events").to_arrow()
    ulids = tbl.column("ulid").to_pylist()
    vecs = np.array(tbl.column("vector").to_pylist(), dtype=np.float32)
    return ulids, vecs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=data_path("locomo10.json"))
    ap.add_argument("--conv", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--chunk-by", choices=("turn", "session"), default="turn")
    args = ap.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    conv = dataset[args.conv]
    qa = conv.get("qa") or []
    sid = conv.get("sample_id", f"conv-{args.conv}")
    print(f"surprise-gate POST-HOC ablation on {sid} — {len(qa)} QA, "
          f"chunk_by={args.chunk_by}, top_k={args.top_k}\n")

    from pmb.core.engine import Engine
    tmp_home = Path(tempfile.mkdtemp(prefix="pmb-sg-home-"))
    tmp_ws = Path(tempfile.mkdtemp(prefix="pmb-sg-ws-"))
    os.environ["PMB_HOME"] = str(tmp_home)
    eng = Engine(cwd=tmp_ws, pmb_home=tmp_home, rerank_model=None,
                 config_overrides={"recall.cache_size": 0,
                                   "recall.spreading_activation": False,
                                   "recall.graph_boost": 0.15,
                                   "dedup.enable": False})
    t0 = time.time()
    ingest_conversation(eng, conv.get("conversation") or {}, chunk_by=args.chunk_by)
    eng.wait_for_embed_queue(timeout_seconds=240.0)
    print(f"  ingested {eng.events.count(eng.workspace.id)} events in {time.time()-t0:.0f}s")

    # Baseline recall on the full store.
    base = evaluate(eng, qa, top_k=args.top_k)["overall"]
    n_full = eng.events.count(eng.workspace.id)
    print(f"  baseline: stored={n_full}  recall@{args.top_k}={base['evidence_recall_top_k']:.1%}\n")

    # Vectors + chronological order (earlier = smaller timestamp).
    ulids, vecs = _load_vectors(eng)
    vec_of = {u: i for i, u in enumerate(ulids)}
    import sqlite3
    con = sqlite3.connect(eng.workspace.db_path)
    order = [(r[0], r[1]) for r in con.execute(
        "SELECT ulid,timestamp FROM events WHERE archived_at IS NULL ORDER BY timestamp").fetchall()
        if r[0] in vec_of]
    idx = np.array([vec_of[u] for u, _ in order])
    V = vecs[idx]; V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    S = V @ V.T
    nn = np.full(len(order), -1.0)
    for i in range(1, len(order)):
        nn[i] = S[i, :i].max()

    rows = []
    base_recall = base["evidence_recall_top_k"]
    for T in THRESHOLDS:
        suppressed = [order[i][0] for i in range(len(order)) if nn[i] >= T]
        for u in suppressed:
            eng.events.archive(u)
        eng.recall_cache.bump_generation()
        r = evaluate(eng, qa, top_k=args.top_k)["overall"]
        for u in suppressed:
            eng.events.unarchive(u)
        eng.recall_cache.bump_generation()
        rows.append((T, len(suppressed), r["evidence_recall_top_k"]))
        print(f"  T={T:.2f}  suppressed={len(suppressed):>3} "
              f"({100*len(suppressed)/n_full:.0f}%)  "
              f"stored={n_full-len(suppressed):>3}  "
              f"recall@{args.top_k}={r['evidence_recall_top_k']:.1%}  "
              f"delta={100*(r['evidence_recall_top_k']-base_recall):+.1f}pp")

    print("\n" + "=" * 60)
    print(f"baseline stored={n_full}  recall@{args.top_k}={base_recall:.1%}")
    print(f"{'thresh':>7}{'suppressed':>12}{'stored':>8}{'recall':>9}{'delta':>9}")
    for T, sup, rec in rows:
        print(f"{T:>7.2f}{sup:>9} ({100*sup/n_full:>2.0f}%){n_full-sup:>8}"
              f"{rec:>8.1%}{100*(rec-base_recall):>+8.1f}pp")
    print("=" * 60)
    print("Theory holds iff a lower T cuts 'suppressed' up meaningfully while "
          "delta stays ~0pp.")


if __name__ == "__main__":
    main()
