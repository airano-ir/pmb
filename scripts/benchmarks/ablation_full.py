"""
Full system ablation across every recall subsystem.

For each component we either disable it (default-ON → OFF) or enable
it (default-OFF → ON) and measure the delta from baseline on the
*same* N LoCoMo conversations.

Output: a sorted table of (component, ΔRecall@10, ΔLatency_ms, notes),
making it obvious which components actually carry the 91.6% number and
which are dead weight.

Run:
    python scripts/benchmarks/ablation_full.py --n-conversations 3

Time:
    First conv warms up models (~50s). After that each run ≈ 10-15s.
    19 runs × ~15s = ~5-8 min for n_conversations=3.
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

from benchmark_locomo import ingest_conversation, evaluate  # noqa: E402


# ----------------------------------------------------------------------
# Ablation registry. Each row: (key, overrides_dict, kind, note)
# - overrides_dict is applied on TOP of baseline defaults.
# - kind ∈ {"disable", "enable", "extreme"} for table grouping.
# ----------------------------------------------------------------------

ABLATIONS: list[tuple[str, dict, str, str]] = [
    # Baseline. Must be first.
    ("baseline", {}, "baseline", "all defaults"),

    # === Always-ON features → flip to OFF ===
    ("no_graph_boost",
     {"recall.graph_boost": 0.0},
     "disable", "no entity-adjacency boost"),
    ("no_causation_walk",
     {"recall.causation_walk": False},
     "disable", "no event_edges traversal"),
    ("no_arc_expansion",
     {"recall.arc_expansion": False},
     "disable", "no narrative arc injection"),
    ("no_collapse_reflections",
     {"recall.collapse_reflections": False},
     "disable", "reflections returned as separate hits"),
    ("no_reflection_to_edges",
     {"recall.reflection_to_edges": False},
     "disable", "reflection terms not linked to source"),
    ("no_temporal",
     {"recall.temporal_enabled": False},
     "disable", "no event_time parsing / boost"),
    ("no_adaptive_routing",
     {"recall.adaptive_routing": False},
     "disable", "no intent-based weight tuning"),
    ("no_predictive_cache",
     {"recall.predictive_enabled": False},
     "disable", "skip pre-computed answer lookup"),
    ("no_person_extraction",
     {"recall.person_extraction": False},
     "disable", "no person-entity boost"),
    ("no_code_ast",
     {"recall.code_ast_extraction": False},
     "disable", "no AST entity extraction"),
    ("no_typo_correction",
     {"recall.typo_correction": False},
     "disable", "no fuzzy query token fix"),
    ("no_spreading",
     {"recall.spreading_activation": False},
     "disable", "no graph-neighbour priming"),
    ("no_cache",
     {"recall.cache_size": 0},
     "disable", "LRU cache off"),
    ("no_multi_entity_bonus",
     {"recall.multi_entity_bonus": 0.0},
     "disable", "no multi-hop entity bonus"),

    # === Always-OFF features → flip to ON ===
    ("with_rerank",
     {"recall.rerank": True, "recall.rerank_top_n": 25},
     "enable", "cross-encoder reranks top-25"),
    ("with_ppr_always",
     {"recall.ppr_enabled": True, "recall.ppr_always": True},
     "enable", "PPR on every query (not just multi-hop)"),

    # === Fusion extremes ===
    ("bm25_only",
     {"recall.bm25_weight": 1.0},
     "extreme", "vector channel zeroed"),
    ("vector_only",
     {"recall.bm25_weight": 0.0},
     "extreme", "BM25 channel zeroed"),
]


# Tier ablation needs a monkey-patch, not a config knob.
def _apply_tier_state(disabled: bool) -> None:
    from pmb.core import events as ev_mod
    if disabled:
        ev_mod.default_tier_for_event_type = lambda et: ev_mod.TIER_WORKING
        ev_mod.PROMOTE_WORKING_TO_EPISODIC_ACCESS = 10**9
        ev_mod.PROMOTE_EPISODIC_TO_SEMANTIC_ACCESS = 10**9
        for k in list(ev_mod.TIER_DECAY_FACTORS.keys()):
            ev_mod.TIER_DECAY_FACTORS[k] = 1.0
    else:
        # Restore originals.
        ev_mod.default_tier_for_event_type = _ORIG_default_tier
        ev_mod.PROMOTE_WORKING_TO_EPISODIC_ACCESS = _ORIG_promote_we
        ev_mod.PROMOTE_EPISODIC_TO_SEMANTIC_ACCESS = _ORIG_promote_es
        for k, v in _ORIG_decay.items():
            ev_mod.TIER_DECAY_FACTORS[k] = v


def _capture_tier_originals():
    global _ORIG_default_tier, _ORIG_promote_we, _ORIG_promote_es, _ORIG_decay
    from pmb.core import events as ev_mod
    _ORIG_default_tier = ev_mod.default_tier_for_event_type
    _ORIG_promote_we = ev_mod.PROMOTE_WORKING_TO_EPISODIC_ACCESS
    _ORIG_promote_es = ev_mod.PROMOTE_EPISODIC_TO_SEMANTIC_ACCESS
    _ORIG_decay = dict(ev_mod.TIER_DECAY_FACTORS)


# ----------------------------------------------------------------------
# Single-run driver
# ----------------------------------------------------------------------


def run_one(conversations: list[dict], overrides: dict, top_k: int,
            tiers_disabled: bool = False) -> dict:
    """Run ingest + evaluate for the given config overrides.
    Returns aggregated metrics across all conversations.
    """
    _apply_tier_state(tiers_disabled)

    from pmb.core.engine import Engine

    per_conv = []
    for ci, conv in enumerate(conversations):
        sample_id = conv.get("sample_id", f"conv-{ci}")
        qa = conv.get("qa") or []
        conversation = conv.get("conversation") or {}

        tmp_home = Path(tempfile.mkdtemp(prefix=f"pmb-abl-{ci}-"))
        tmp_ws = Path(tempfile.mkdtemp(prefix=f"pmb-abl-ws-{ci}-"))
        os.environ["PMB_HOME"] = str(tmp_home)
        eng = Engine(
            cwd=tmp_ws, pmb_home=tmp_home,
            rerank_model=None,
            config_overrides=overrides,
        )

        t_ing = time.time()
        n_turns = ingest_conversation(eng, conversation, chunk_by="session")
        ing_seconds = time.time() - t_ing

        rerank = bool(overrides.get("recall.rerank", False))
        t_eval = time.time()
        result = evaluate(eng, qa, top_k=top_k, rerank=rerank)
        eval_seconds = time.time() - t_eval

        per_conv.append({
            "sample_id": sample_id,
            "evidence_recall_at_10": result["overall"]["evidence_recall_top_k"],
            "mean_f1": result["overall"]["mean_f1"],
            "p50_ms": result["overall"]["p50_latency_ms"],
            "p95_ms": result["overall"]["p95_latency_ms"],
            "ingest_s": round(ing_seconds, 1),
            "eval_s": round(eval_seconds, 1),
            "n_turns": n_turns,
        })

        # Cleanup so we don't leak GB of temp dirs
        try:
            eng.close()
        except Exception:
            pass

    # Restore tier originals if we touched them
    if tiers_disabled:
        _apply_tier_state(False)

    recalls = [c["evidence_recall_at_10"] for c in per_conv]
    p50s = [c["p50_ms"] for c in per_conv]
    p95s = [c["p95_ms"] for c in per_conv]
    f1s = [c["mean_f1"] for c in per_conv]
    return {
        "per_conversation": per_conv,
        "mean_recall_at_10": round(statistics.mean(recalls), 4) if recalls else 0.0,
        "mean_p50_ms": round(statistics.mean(p50s), 1) if p50s else 0.0,
        "mean_p95_ms": round(statistics.mean(p95s), 1) if p95s else 0.0,
        "mean_f1": round(statistics.mean(f1s), 4) if f1s else 0.0,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=data_path("locomo10.json"))
    ap.add_argument("--n-conversations", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", default=data_path("pmb_ablation_full.json"))
    args = ap.parse_args()

    _capture_tier_originals()

    print("Full-system ablation on LoCoMo")
    print(f"  dataset: {args.dataset}")
    print(f"  n_conversations: {args.n_conversations}, top_k: {args.top_k}")
    print()

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    convs = dataset[: args.n_conversations]
    print(f"  conversations selected:",
          [c.get("sample_id", f"conv-{i}") for i, c in enumerate(convs)])
    print()

    # All standard ablations from the registry
    runs: list[dict] = []
    t_total = time.time()
    for label, overrides, kind, note in ABLATIONS:
        print(f">>> {label:<25} {note}")
        t0 = time.time()
        try:
            r = run_one(convs, overrides, args.top_k, tiers_disabled=False)
            r["label"] = label
            r["kind"] = kind
            r["note"] = note
            r["overrides"] = overrides
            r["wall_s"] = round(time.time() - t0, 1)
            runs.append(r)
            print(f"    recall@10 = {r['mean_recall_at_10']:.2%}, "
                  f"p50 = {r['mean_p50_ms']:.1f}ms, "
                  f"p95 = {r['mean_p95_ms']:.1f}ms, "
                  f"wall = {r['wall_s']}s")
        except Exception as e:
            print(f"    FAILED: {e}")
            runs.append({"label": label, "kind": kind, "note": note,
                         "error": str(e), "wall_s": round(time.time() - t0, 1)})

    # Tier ablation (monkey-patch)
    print(">>> no_tiers                 tier system fully disabled (patch)")
    t0 = time.time()
    try:
        r = run_one(convs, {}, args.top_k, tiers_disabled=True)
        r["label"] = "no_tiers"
        r["kind"] = "disable"
        r["note"] = "tier system fully disabled (patch)"
        r["overrides"] = {"_patched_tiers": True}
        r["wall_s"] = round(time.time() - t0, 1)
        runs.append(r)
        print(f"    recall@10 = {r['mean_recall_at_10']:.2%}, "
              f"p50 = {r['mean_p50_ms']:.1f}ms, "
              f"p95 = {r['mean_p95_ms']:.1f}ms, "
              f"wall = {r['wall_s']}s")
    except Exception as e:
        print(f"    FAILED: {e}")

    total_wall = time.time() - t_total

    # ----------------------------------------------------------------
    # Final table sorted by |Δ recall@10| descending
    # ----------------------------------------------------------------
    baseline = next(r for r in runs if r.get("label") == "baseline")
    base_recall = baseline["mean_recall_at_10"]
    base_p50 = baseline["mean_p50_ms"]

    print()
    print("=" * 100)
    print(f"BASELINE   recall@10 = {base_recall:.2%}   p50 = {base_p50:.1f}ms   "
          f"(n={args.n_conversations} convs)")
    print("=" * 100)
    print(f"{'component':<26}{'kind':<10}{'recall@10':>12}{'Δrecall':>10}"
          f"{'p50_ms':>10}{'Δp50':>9}  note")
    print("-" * 100)

    def _sort_key(r):
        if r.get("label") == "baseline":
            return -float("inf")
        if "error" in r:
            return -float("inf")
        return -abs(r["mean_recall_at_10"] - base_recall)

    runs_sorted = sorted(runs, key=_sort_key)
    for r in runs_sorted:
        if "error" in r:
            print(f"{r['label']:<26}{r['kind']:<10}{'ERROR':>12}{'':>10}{'':>10}{'':>9}  "
                  f"{r['note']} :: {r['error'][:60]}")
            continue
        d_recall = r["mean_recall_at_10"] - base_recall
        d_p50 = r["mean_p50_ms"] - base_p50
        marker = ""
        if r["label"] != "baseline":
            if d_recall <= -0.02:
                marker = " <- carries baseline"
            elif d_recall >= 0.02:
                marker = " <- hurts baseline"
            elif abs(d_recall) <= 0.005:
                marker = " <- no effect"
        print(f"{r['label']:<26}{r['kind']:<10}"
              f"{r['mean_recall_at_10']:>12.2%}"
              f"{d_recall:>+10.3f}"
              f"{r['mean_p50_ms']:>10.1f}"
              f"{d_p50:>+9.1f}  "
              f"{r['note']}{marker}")
    print("=" * 100)
    print(f"Total wall time: {total_wall:.1f}s")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"baseline": baseline, "runs": runs,
                   "n_conversations": args.n_conversations,
                   "top_k": args.top_k,
                   "total_wall_s": round(total_wall, 1)}, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
