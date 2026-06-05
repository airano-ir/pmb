"""
A/B test PAMVR on the full LoCoMo benchmark (~1990 queries).

We run TWO complete passes over the same 10 conversations:
  Run A: PAMVR OFF (recall.pamvr_enabled=False)
  Run B: PAMVR ON  (default, recall.pamvr_enabled=True)

Each pass measures evidence_recall@1, @3, @5, @10 — same retrieval method
that LoCoMo papers report, just at smaller top-K cuts to see the top-1
effect that the qualitative bench surfaced.

We use session-chunked ingestion (matches our previous LoCoMo runs at
94.1% recall@10 baseline). Total runtime: ~25-40 minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
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

from benchmark_locomo import (  # noqa: E402
    sessions_in_order, ingest_conversation, _evidence_hit,
)


def evaluate_at_k(eng, qa_list, ks=(1, 3, 5, 10)) -> dict:
    """Evidence recall at multiple cut-off ranks."""
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    n = 0
    latencies = []
    for q in qa_list:
        question = q.get("question", "")
        gold = set(q.get("evidence", []) or [])
        if not gold:
            continue
        n += 1
        t0 = time.perf_counter()
        pack = eng.recall(question, top_k=max_k)
        latencies.append((time.perf_counter() - t0) * 1000)
        results = pack.results
        # For each rank cutoff, count if ANY result <= rank matches gold
        for k in ks:
            for r in results[:k]:
                if _evidence_hit(r.metadata or {}, gold):
                    hits[k] += 1
                    break
    rec = {f"recall@{k}": round(hits[k] / max(1, n), 4) for k in ks}
    rec["n"] = n
    if latencies:
        latencies.sort()
        rec["p50_ms"] = round(latencies[len(latencies) // 2], 1)
        rec["p95_ms"] = round(latencies[min(len(latencies) - 1,
                                            int(0.95 * len(latencies)))], 1)
    return rec


def run_pass(dataset_path: str, n_conv: int, pamvr: bool) -> dict:
    """Run LoCoMo over `n_conv` conversations with PAMVR on/off."""
    from pmb.core.engine import Engine

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    convs = dataset[: n_conv]

    overrides = {"recall.pamvr_enabled": pamvr}
    per_conv_results = []
    total_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    total_n = 0
    all_lats: list[float] = []
    for ci, conv in enumerate(convs):
        sample_id = conv.get("sample_id", f"conv-{ci}")
        qa = conv.get("qa") or []
        conversation = conv.get("conversation") or {}
        tmp_home = Path(tempfile.mkdtemp(prefix=f"pmb-ab-{ci}-"))
        tmp_ws = Path(tempfile.mkdtemp(prefix=f"pmb-ab-ws-{ci}-"))
        os.environ["PMB_HOME"] = str(tmp_home)
        eng = Engine(cwd=tmp_ws, pmb_home=tmp_home, rerank_model=None,
                     config_overrides=overrides)
        ingest_conversation(eng, conversation, chunk_by="session")
        time.sleep(2.0)
        # Force-drain async embed queue before eval
        try:
            from pmb.health.consolidate import _drain_pending_embeds
            _drain_pending_embeds(eng, timeout_s=30.0)
        except Exception:
            pass
        r = evaluate_at_k(eng, qa)
        r["sample_id"] = sample_id
        per_conv_results.append(r)
        for k in (1, 3, 5, 10):
            total_hits[k] += int(r[f"recall@{k}"] * r["n"])
        total_n += r["n"]
        if "p50_ms" in r:
            all_lats.append(r["p50_ms"])
        print(f"    {sample_id}: n={r['n']}, "
              f"r@1={r['recall@1']*100:>5.1f}%, "
              f"r@3={r['recall@3']*100:>5.1f}%, "
              f"r@5={r['recall@5']*100:>5.1f}%, "
              f"r@10={r['recall@10']*100:>5.1f}%, "
              f"p50={r.get('p50_ms', '?')}ms")
        try:
            eng.close()
        except Exception:
            pass
    out = {
        "pamvr": pamvr,
        "n_conversations": len(convs),
        "total_questions": total_n,
        "recall@1":  round(total_hits[1] / max(1, total_n), 4),
        "recall@3":  round(total_hits[3] / max(1, total_n), 4),
        "recall@5":  round(total_hits[5] / max(1, total_n), 4),
        "recall@10": round(total_hits[10] / max(1, total_n), 4),
        "mean_p50_ms": round(statistics.mean(all_lats), 1) if all_lats else 0,
        "per_conv": per_conv_results,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",
                    default=data_path("locomo10.json"))
    ap.add_argument("--n-conversations", type=int, default=10)
    ap.add_argument("--out",
                    default=data_path("pmb_ab_pamvr.json"))
    args = ap.parse_args()

    print(f"A/B PAMVR test on LoCoMo ({args.n_conversations} conversations)")
    print(f"  dataset: {args.dataset}\n")

    print("=== Run A: PAMVR OFF (baseline) ===")
    t0 = time.time()
    run_off = run_pass(args.dataset, args.n_conversations, pamvr=False)
    print(f"  wall: {time.time() - t0:.1f}s\n")

    print("=== Run B: PAMVR ON ===")
    t0 = time.time()
    run_on = run_pass(args.dataset, args.n_conversations, pamvr=True)
    print(f"  wall: {time.time() - t0:.1f}s\n")

    print("=" * 72)
    print(f"AGGREGATE over {run_on['total_questions']} questions "
          f"({args.n_conversations} conversations)")
    print("=" * 72)
    print(f"  metric        PAMVR off   PAMVR on    Δ")
    print(f"  ---------------------------------------------------")
    for k in (1, 3, 5, 10):
        off = run_off[f"recall@{k}"] * 100
        on = run_on[f"recall@{k}"] * 100
        delta = on - off
        sign = "+" if delta >= 0 else ""
        print(f"  recall@{k:<3}     {off:>5.2f}%     {on:>5.2f}%   "
              f"{sign}{delta:>5.2f}pp")
    print(f"  ---------------------------------------------------")
    print(f"  p50 latency   {run_off['mean_p50_ms']:>5}ms      "
          f"{run_on['mean_p50_ms']:>5}ms     "
          f"{run_on['mean_p50_ms'] - run_off['mean_p50_ms']:+}ms")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"off": run_off, "on": run_on}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
