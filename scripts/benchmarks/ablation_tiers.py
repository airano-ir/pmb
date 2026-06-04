"""
Ablation study: does the memory-tier system actually help recall on LoCoMo?

Hypothesis (pre-registered):
    Tiers affect runtime behavior through two paths -
      (1) per-tier importance decay multipliers (signals/decay.py)
      (2) access-count promotion (engine.py recall loop)
    Decay (1) does not run during a fresh-ingest benchmark (no days pass),
    and promotion (2) is a *side effect* of recall - it never enters the
    scoring formula. Therefore the prediction is: disabling tiers should
    have ~0 effect on evidence_recall@10 on LoCoMo.

    If the experiment confirms this, the tier abstraction adds code
    complexity without measurable benefit on this benchmark and should be
    treated as a long-term-only feature, not a recall-quality feature.

How the ablation works:
    Two runs over the same N conversations with the same seed and config.
    Run A: tiers ON (default).
    Run B: tiers OFF - we monkey-patch the four entry points where the
           tier system enters runtime behavior:
             - default_tier_for_event_type   → always 'working'
             - PROMOTE_WORKING_TO_EPISODIC_ACCESS / EPISODIC_TO_SEMANTIC → infinity
             - TIER_DECAY_FACTORS            → all 1.0  (no decay diff)

Run:
    python scripts/benchmarks/ablation_tiers.py --n-conversations 2

Output:
    Side-by-side table of evidence_recall@10, mean_f1, latency.
    Writes JSON to %TMP%/pmb_ablation_tiers.json.
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

# Import the existing benchmark machinery so we don't duplicate ingestion +
# evaluation logic.
from benchmark_locomo import (  # noqa: E402
    ingest_conversation,
    evaluate,
)


def disable_tiers():
    """Neuter every runtime path tiers reach into."""
    from pmb.core import events as ev_mod
    # 1) All freshly recorded events start in 'working' regardless of type.
    ev_mod.default_tier_for_event_type = lambda event_type: ev_mod.TIER_WORKING
    # 2) Never promote: thresholds set to 10**9 (effectively infinity).
    ev_mod.PROMOTE_WORKING_TO_EPISODIC_ACCESS = 10**9
    ev_mod.PROMOTE_EPISODIC_TO_SEMANTIC_ACCESS = 10**9
    # 3) Flatten decay factors. (No effect during benchmark since no days
    #    pass, but keeps the model symmetric just in case.)
    for k in list(ev_mod.TIER_DECAY_FACTORS.keys()):
        ev_mod.TIER_DECAY_FACTORS[k] = 1.0
    # Also patch the imported names inside engine.py, which captured them
    # at import time.
    from pmb.core import engine as eng_mod
    eng_mod.PROMOTE_WORKING_TO_EPISODIC_ACCESS = 10**9
    eng_mod.PROMOTE_EPISODIC_TO_SEMANTIC_ACCESS = 10**9


def run_one(dataset_path: str, n_conv: int, top_k: int, mode: str) -> dict:
    """Run the LoCoMo benchmark once in the given mode.

    mode ∈ {"tiers_on", "tiers_off"}.
    """
    if mode == "tiers_off":
        disable_tiers()

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    n_conv = min(n_conv, len(dataset))

    per_conv = []
    for ci, conv in enumerate(dataset[:n_conv]):
        sample_id = conv.get("sample_id", f"conv-{ci}")
        qa = conv.get("qa") or []
        conversation = conv.get("conversation") or {}
        print(f"  [{mode}] {sample_id} - {len(qa)} QA")

        from pmb.core.engine import Engine
        tmp_home = Path(tempfile.mkdtemp(prefix=f"pmb-abl-{mode}-{ci}-"))
        tmp_ws = Path(tempfile.mkdtemp(prefix=f"pmb-abl-ws-{mode}-{ci}-"))
        os.environ["PMB_HOME"] = str(tmp_home)
        eng = Engine(
            cwd=tmp_ws, pmb_home=tmp_home,
            rerank_model=None,
            config_overrides={
                "recall.cache_size": 0,            # honest eval
                "recall.spreading_activation": False,
                "recall.graph_boost": 0.15,
                "recall.adaptive_decompose": False,
            },
        )

        t_ing = time.time()
        n_turns = ingest_conversation(eng, conversation, chunk_by="session")
        ing_seconds = time.time() - t_ing

        t_eval = time.time()
        result = evaluate(eng, qa, top_k=top_k, rerank=False)
        eval_seconds = time.time() - t_eval

        ov = result["overall"]
        print(f"        ingested {n_turns} events in {ing_seconds:.1f}s, "
              f"evaluated in {eval_seconds:.1f}s")
        print(f"        evidence_recall@{ov['top_k']} = {ov['evidence_recall_top_k']:.2%}, "
              f"mean_f1 = {ov['mean_f1']:.3f}, "
              f"p50 = {ov['p50_latency_ms']}ms")

        # Inspect final tier distribution to confirm the patch worked.
        tier_counts: dict[str, int] = {}
        with eng.events._conn() as conn:
            for row in conn.execute(
                "SELECT tier, COUNT(*) AS c FROM events WHERE workspace_id = ? "
                "AND archived_at IS NULL GROUP BY tier",
                (eng.workspace.id,),
            ):
                tier_counts[row["tier"]] = row["c"]
        result["sample_id"] = sample_id
        result["ingest_seconds"] = round(ing_seconds, 1)
        result["eval_seconds"] = round(eval_seconds, 1)
        result["tier_counts_after_eval"] = tier_counts
        per_conv.append(result)

    # Aggregate
    all_ev = [r["overall"]["evidence_recall_top_k"] for r in per_conv]
    all_f1 = [r["overall"]["mean_f1"] for r in per_conv]
    all_p50 = [r["overall"]["p50_latency_ms"] for r in per_conv]
    agg = {
        "mode": mode,
        "n_conversations": len(per_conv),
        "mean_evidence_recall_at_10": round(statistics.mean(all_ev), 4) if all_ev else 0.0,
        "mean_f1": round(statistics.mean(all_f1), 4) if all_f1 else 0.0,
        "median_p50_latency_ms": round(statistics.median(all_p50), 1) if all_p50 else 0.0,
        "per_conversation": [
            {
                "sample_id": r["sample_id"],
                "evidence_recall_at_10": r["overall"]["evidence_recall_top_k"],
                "mean_f1": r["overall"]["mean_f1"],
                "p50_ms": r["overall"]["p50_latency_ms"],
                "p95_ms": r["overall"]["p95_latency_ms"],
                "tier_counts_after_eval": r["tier_counts_after_eval"],
            }
            for r in per_conv
        ],
    }
    return agg


def print_side_by_side(a: dict, b: dict) -> None:
    print()
    print("=" * 70)
    print(f"{'metric':<35}{'tiers_on':>15}{'tiers_off':>15}{'Δ':>6}")
    print("-" * 70)

    def fmt_pct(x): return f"{x:.2%}"
    def fmt_f(x): return f"{x:.3f}"
    def fmt_ms(x): return f"{x:.1f}ms"

    rows = [
        ("mean evidence_recall@10", a["mean_evidence_recall_at_10"],
         b["mean_evidence_recall_at_10"], fmt_pct),
        ("mean f1",                a["mean_f1"], b["mean_f1"], fmt_f),
        ("median p50 latency",     a["median_p50_latency_ms"],
         b["median_p50_latency_ms"], fmt_ms),
    ]
    for label, va, vb, fmt in rows:
        delta = vb - va
        print(f"{label:<35}{fmt(va):>15}{fmt(vb):>15}{delta:+.3f}")
    print("=" * 70)

    print("\nper-conversation evidence_recall@10:")
    print(f"{'conv':<14}{'tiers_on':>15}{'tiers_off':>15}{'Δ':>10}")
    print("-" * 54)
    for ra, rb in zip(a["per_conversation"], b["per_conversation"]):
        d = rb["evidence_recall_at_10"] - ra["evidence_recall_at_10"]
        print(f"{ra['sample_id']:<14}"
              f"{ra['evidence_recall_at_10']:>15.2%}"
              f"{rb['evidence_recall_at_10']:>15.2%}"
              f"{d:>+10.3f}")

    print("\nfinal tier distribution (tiers_on run):")
    for r in a["per_conversation"]:
        print(f"  {r['sample_id']}: {r['tier_counts_after_eval']}")
    print("final tier distribution (tiers_off run):")
    for r in b["per_conversation"]:
        print(f"  {r['sample_id']}: {r['tier_counts_after_eval']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=data_path("locomo10.json"))
    ap.add_argument("--n-conversations", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", default=data_path("pmb_ablation_tiers.json"))
    args = ap.parse_args()

    print("Ablation: memory tiers ON vs OFF (LoCoMo)")
    print(f"  dataset: {args.dataset}")
    print(f"  n_conversations: {args.n_conversations}, top_k: {args.top_k}")
    print()

    print(">>> Run A: tiers ON (default)")
    t0 = time.time()
    a = run_one(args.dataset, args.n_conversations, args.top_k, "tiers_on")
    print(f"  Run A wall: {time.time() - t0:.1f}s\n")

    # Must restart Python state cleanly between runs since we monkey-patched
    # globals. We are running both in the same process so the patches stick;
    # apply them only inside run_one when mode=='tiers_off'. To ensure
    # ordering effects do not bias, we always do tiers_on first.
    print(">>> Run B: tiers OFF (ablation)")
    t0 = time.time()
    b = run_one(args.dataset, args.n_conversations, args.top_k, "tiers_off")
    print(f"  Run B wall: {time.time() - t0:.1f}s")

    print_side_by_side(a, b)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"tiers_on": a, "tiers_off": b}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
