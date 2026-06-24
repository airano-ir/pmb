"""Controlled ablation: does OUTCOME-WEIGHTING (rank memories higher if they
proved useful before) improve evidence_recall@10 on HELD-OUT questions?

This is the honest ground-truth test of "memory that learns from outcomes".
Earlier I mistakenly measured the `followed` column (degenerate: ~all 1).
Here the "outcome" is the GOLD evidence label, which is real:

  1. Ingest the conversation (dedup off), drain embeddings.
  2. Split QA: TRAIN = first half, TEST = second half (held out).
  3. LEARN usefulness from TRAIN only: every event that was gold evidence for
     a train question gets a usefulness count (it "helped before").
  4. PRECONDITION CHECK: of the events that are evidence for TEST questions,
     how many were ALSO useful in TRAIN? If ~0, nothing transfers and
     outcome-weighting CANNOT help — an honest negative before we even score.
  5. Score TEST recall@10 two ways through the real recall pipeline:
       - baseline (default importance)
       - boosted (useful events' importance raised)
     then restore importance.

Theory holds iff boosted recall@10 on held-out questions > baseline, and that
is only possible to the extent evidence is REUSED across train->test.

Run:
    PYTHONPATH=src python scripts/benchmarks/ab_outcome_weight.py --conv 0
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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parent.parent.parent / "src"))
sys.path.insert(0, str(_here.parent))
sys.path.insert(0, str(_here.parent.parent))

from _bench_data import data_path  # noqa: E402
from benchmark_locomo import evaluate, ingest_conversation  # noqa: E402

BOOST_IMPORTANCE = 0.95   # useful events get pushed up


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
    qa = [q for q in (conv.get("qa") or []) if q.get("evidence")]
    sid = conv.get("sample_id", f"conv-{args.conv}")
    half = len(qa) // 2
    train_qa, test_qa = qa[:half], qa[half:]
    print(f"outcome-weighting ablation on {sid} — {len(qa)} QA w/ evidence "
          f"(train={len(train_qa)}, test={len(test_qa)}), chunk_by={args.chunk_by}\n")

    from pmb.core.engine import Engine
    tmp_home = Path(tempfile.mkdtemp(prefix="pmb-ow-home-"))
    tmp_ws = Path(tempfile.mkdtemp(prefix="pmb-ow-ws-"))
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

    # Map gold dia_id -> event ulid(s).
    import sqlite3
    con = sqlite3.connect(eng.workspace.db_path); con.row_factory = sqlite3.Row
    dia2ulids: dict[str, list[str]] = {}
    orig_imp: dict[str, float] = {}
    for r in con.execute("SELECT ulid,importance,metadata_json FROM events WHERE archived_at IS NULL"):
        orig_imp[r["ulid"]] = r["importance"]
        try:
            md = json.loads(r["metadata_json"] or "{}")
        except Exception:
            md = {}
        ids = []
        if md.get("dia_id"): ids.append(md["dia_id"])
        ids += (md.get("dia_ids") or [])
        for d in ids:
            dia2ulids.setdefault(d, []).append(r["ulid"])

    # LEARN usefulness from TRAIN evidence.
    useful: dict[str, int] = {}
    for q in train_qa:
        for d in (q.get("evidence") or []):
            for u in dia2ulids.get(d, []):
                useful[u] = useful.get(u, 0) + 1
    useful_set = set(useful)

    # PRECONDITION: does test evidence reuse train-useful events?
    test_ev_ulids = set()
    for q in test_qa:
        for d in (q.get("evidence") or []):
            test_ev_ulids.update(dia2ulids.get(d, []))
    reused = test_ev_ulids & useful_set
    reuse_pct = 100 * len(reused) / max(len(test_ev_ulids), 1)
    print(f"  learned-useful events (from train): {len(useful_set)}")
    print(f"  test-evidence events: {len(test_ev_ulids)}, "
          f"of which already useful in train: {len(reused)} ({reuse_pct:.0f}%)")
    print(f"  >>> PRECONDITION: outcome-weighting can only help via this {reuse_pct:.0f}% reuse\n")

    # Baseline recall on test.
    base = evaluate(eng, test_qa, top_k=args.top_k)["overall"]["evidence_recall_top_k"]

    # Boost useful events' importance, re-eval, restore.
    for u in useful_set:
        eng.events.update_importance(u, BOOST_IMPORTANCE)
    eng.recall_cache.bump_generation()
    boosted = evaluate(eng, test_qa, top_k=args.top_k)["overall"]["evidence_recall_top_k"]
    for u, imp in orig_imp.items():
        eng.events.update_importance(u, imp)
    eng.recall_cache.bump_generation()

    print("=" * 56)
    print(f"  baseline recall@{args.top_k} (test): {base:.1%}")
    print(f"  outcome-weighted recall@{args.top_k}: {boosted:.1%}")
    print(f"  delta: {100*(boosted-base):+.1f}pp")
    print("=" * 56)
    if reuse_pct < 5:
        print("VERDICT: ~no evidence reuse -> outcome-weighting has nothing to "
              "transfer on this data. Idea is structurally inert here.")
    elif boosted > base:
        print("VERDICT: outcome-weighting IMPROVED held-out recall. Signal real.")
    else:
        print("VERDICT: no improvement despite reuse -> importance lever too weak "
              "or recall already saturated on reused events.")


if __name__ == "__main__":
    main()
