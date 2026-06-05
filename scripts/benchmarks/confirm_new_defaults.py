"""
Confirm the new defaults beat the old ones.

We changed two SCHEMA defaults in config.py based on ablation findings:
  - recall.bm25_weight     0.5  -> 0.7   (BM25-heavy fusion)
  - recall.typo_correction True -> False (typo "fixes" hurt recall)

This script reruns the same 3 LoCoMo conversations under both configs,
side by side, so we can confirm the improvement holds.
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

from benchmark_locomo import ingest_conversation, evaluate  # noqa


OLD_DEFAULTS = {
    "recall.bm25_weight": 0.5,
    "recall.typo_correction": True,
}
NEW_DEFAULTS = {
    "recall.bm25_weight": 0.7,
    "recall.typo_correction": False,
}


def run_one(convs, overrides, top_k):
    from pmb.core.engine import Engine
    out = []
    for ci, conv in enumerate(convs):
        sample_id = conv.get("sample_id", f"conv-{ci}")
        qa = conv.get("qa") or []
        conversation = conv.get("conversation") or {}
        tmp_home = Path(tempfile.mkdtemp(prefix=f"pmb-conf-{ci}-"))
        tmp_ws = Path(tempfile.mkdtemp(prefix=f"pmb-conf-ws-{ci}-"))
        os.environ["PMB_HOME"] = str(tmp_home)
        eng = Engine(cwd=tmp_ws, pmb_home=tmp_home,
                     rerank_model=None, config_overrides=overrides)
        ingest_conversation(eng, conversation, chunk_by="session")
        r = evaluate(eng, qa, top_k=top_k, rerank=False)
        out.append({
            "sample_id": sample_id,
            "evidence_recall_at_10": r["overall"]["evidence_recall_top_k"],
            "mean_f1": r["overall"]["mean_f1"],
            "p50_ms": r["overall"]["p50_latency_ms"],
            "p95_ms": r["overall"]["p95_latency_ms"],
        })
        try:
            eng.close()
        except Exception:
            pass
    recalls = [c["evidence_recall_at_10"] for c in out]
    return {
        "per_conv": out,
        "mean_recall_at_10": round(statistics.mean(recalls), 4),
        "mean_p50_ms": round(statistics.mean([c["p50_ms"] for c in out]), 1),
        "mean_p95_ms": round(statistics.mean([c["p95_ms"] for c in out]), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=data_path("locomo10.json"))
    ap.add_argument("--n-conversations", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    convs = dataset[: args.n_conversations]
    print(f"convs: {[c.get('sample_id') for c in convs]}\n")

    print(">>> Run A: OLD defaults  (bm25=0.5, typo=True)")
    t0 = time.time()
    a = run_one(convs, OLD_DEFAULTS, args.top_k)
    print(f"    recall@10 = {a['mean_recall_at_10']:.2%}, "
          f"p50 = {a['mean_p50_ms']}ms, "
          f"wall = {time.time() - t0:.1f}s")
    for c in a["per_conv"]:
        print(f"    {c['sample_id']}: {c['evidence_recall_at_10']:.2%}")
    print()

    print(">>> Run B: NEW defaults  (bm25=0.7, typo=False)")
    t0 = time.time()
    b = run_one(convs, NEW_DEFAULTS, args.top_k)
    print(f"    recall@10 = {b['mean_recall_at_10']:.2%}, "
          f"p50 = {b['mean_p50_ms']}ms, "
          f"wall = {time.time() - t0:.1f}s")
    for c in b["per_conv"]:
        print(f"    {c['sample_id']}: {c['evidence_recall_at_10']:.2%}")
    print()

    print("=" * 60)
    print(f"OLD recall@10:  {a['mean_recall_at_10']:.2%}   p50 {a['mean_p50_ms']}ms")
    print(f"NEW recall@10:  {b['mean_recall_at_10']:.2%}   p50 {b['mean_p50_ms']}ms")
    d_rec = b["mean_recall_at_10"] - a["mean_recall_at_10"]
    d_p50 = b["mean_p50_ms"] - a["mean_p50_ms"]
    print(f"Delta:          {d_rec:+.4f}   {d_p50:+.1f}ms")
    print("=" * 60)
    print("Per-conv deltas:")
    for ca, cb in zip(a["per_conv"], b["per_conv"]):
        d = cb["evidence_recall_at_10"] - ca["evidence_recall_at_10"]
        print(f"  {ca['sample_id']}: {ca['evidence_recall_at_10']:.2%} "
              f"-> {cb['evidence_recall_at_10']:.2%}  ({d:+.3f})")

    out = {"old": a, "new": b, "delta_recall_at_10": d_rec, "delta_p50_ms": d_p50}
    with open(data_path("pmb_confirm_defaults.json"),
              "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
