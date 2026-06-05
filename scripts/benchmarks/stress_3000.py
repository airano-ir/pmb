"""
3000-query stress benchmark on the 200-event quality_demo workspace.

For each of 30 base queries we programmatically generate ~100 paraphrases
(template substitution + casing + punctuation noise). Expected-substring
labels stay the same as the base.

A/B compares PAMVR off vs on across 3000 queries. We measure top-1,
top-3, top-10 accuracy and latency distribution.

Run:
    python scripts/benchmarks/stress_3000.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
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

from quality_demo import CHATS                          # noqa: E402
from research_top1 import EXPECTED, matches_expected    # noqa: E402


# ----------------------------------------------------------------------
# Paraphrase generator
# ----------------------------------------------------------------------


_PARAPHRASE_TEMPLATES = {
    # "What's the X?" → many forms
    r"^what(?:'?s|s)?\s+(?:the|our)?\s*(.+)\?$": [
        "What is the {x}?",
        "Tell me the {x}",
        "I'd like to know the {x}",
        "Can you tell me about the {x}?",
        "What's our {x}?",
        "Could you share the {x}?",
        "{x} — what is it?",
        "Remind me of the {x}.",
        "Do you remember the {x}?",
    ],
    r"^(?:where|in what place)\s+(?:does|do|did)\s+(.+)\?$": [
        "Where does {x}?",
        "In what location {x}?",
        "Where exactly {x}?",
        "Tell me where {x}",
        "Which place {x}?",
    ],
    r"^who\s+(.+)\?$": [
        "Who {x}?",
        "Which person {x}?",
        "Tell me who {x}",
        "Can you remind me who {x}?",
        "I forgot who {x}",
    ],
    r"^why\s+(.+)\?$": [
        "Why {x}?",
        "What's the reason {x}?",
        "What was the reason {x}?",
        "Explain why {x}",
        "Tell me why {x}",
        "For what reason {x}?",
    ],
    r"^how\s+(.+)\?$": [
        "How {x}?",
        "By what method {x}?",
        "What's the way {x}?",
        "Tell me how {x}",
    ],
    r"^when\s+(.+)\?$": [
        "When {x}?",
        "At what time {x}?",
        "What's the date {x}?",
        "Tell me when {x}",
    ],
}


_PREFIXES = [
    "", "Quickly — ", "Hey, ", "OK so ", "Could you remind me: ",
    "Real fast: ", "Random question — ", "BTW ",
]
_SUFFIXES = ["", " please", "?", " thanks", " — quick question",
             " (if you remember)", ""]


def _gen_paraphrases(base: str, n: int = 100, rng: random.Random | None = None) -> list[str]:
    """Generate ~n paraphrases of base via template substitution."""
    rng = rng or random.Random(hash(base) & 0xFFFFFFFF)
    out: list[str] = [base]
    base_norm = base.strip().lower()

    # Find matching templates
    matched: list[str] = []
    for pat, templates in _PARAPHRASE_TEMPLATES.items():
        m = re.match(pat, base_norm, re.IGNORECASE)
        if m:
            x = m.group(1)
            for t in templates:
                if "{x}" in t:
                    matched.append(t.format(x=x))
                else:
                    matched.append(t)

    # Always add: casing variants
    matched.extend([
        base.upper(),
        base.lower(),
        base.replace("?", "."),
        base.replace("'", ""),
        " ".join(base.split()),   # whitespace normalisation
        base.replace(" ", "  "),   # double-space stress
    ])

    # Add prefix/suffix combinations
    while len(out) < n:
        b = rng.choice(matched) if matched else base
        prefix = rng.choice(_PREFIXES)
        suffix = rng.choice(_SUFFIXES)
        q = (prefix + b + suffix).strip()
        # Light dedup
        if q not in out:
            out.append(q)
        if len(out) >= n:
            break
        if len(out) > 5 * n:
            break  # safety net
    return out[:n]


# ----------------------------------------------------------------------
# Engine setup
# ----------------------------------------------------------------------


def fresh_engine(overrides: dict | None = None):
    from pmb.core.engine import Engine
    tmp_home = Path(tempfile.mkdtemp())
    tmp_ws = Path(tempfile.mkdtemp())
    os.environ["PMB_HOME"] = str(tmp_home)
    eng = Engine(cwd=tmp_ws, pmb_home=tmp_home, rerank_model=None,
                 config_overrides=overrides or {})
    _ = eng.search.model
    return eng


def ingest(eng):
    for chat in CHATS:
        eng.record_batch(chat["items"])
        time.sleep(0.3)
    time.sleep(3.0)


# ----------------------------------------------------------------------
# Main evaluation
# ----------------------------------------------------------------------


def evaluate(eng, queries: list[tuple[str, list[str]]]) -> dict:
    top1 = top3 = top10 = 0
    lats: list[float] = []
    for q, exp in queries:
        t0 = time.perf_counter()
        pack = eng.recall(q, top_k=10)
        lats.append((time.perf_counter() - t0) * 1000)
        if not pack.results:
            continue
        if matches_expected(pack.results[0].content, exp):
            top1 += 1
        if any(matches_expected(r.content, exp) for r in pack.results[:3]):
            top3 += 1
        if any(matches_expected(r.content, exp) for r in pack.results):
            top10 += 1
    n = len(queries)
    lats.sort()
    return {
        "n": n,
        "top1": top1, "top1_pct": round(100 * top1 / n, 2),
        "top3": top3, "top3_pct": round(100 * top3 / n, 2),
        "top10": top10, "top10_pct": round(100 * top10 / n, 2),
        "p50": round(lats[len(lats) // 2], 1) if lats else 0,
        "p95": round(lats[min(len(lats) - 1, int(0.95 * len(lats)))], 1) if lats else 0,
        "p99": round(lats[min(len(lats) - 1, int(0.99 * len(lats)))], 1) if lats else 0,
        "mean": round(statistics.mean(lats), 1) if lats else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-query", type=int, default=100,
                    help="Paraphrases per base query (default 100 -> ~3000 total)")
    ap.add_argument("--out", default=data_path("pmb_stress_3000.json"))
    args = ap.parse_args()

    print(f"Stress test: 30 base queries × {args.n_per_query} paraphrases = "
          f"{30 * args.n_per_query} queries")

    # Build labeled paraphrase set
    rng = random.Random(42)
    paraphrased: list[tuple[str, list[str]]] = []
    for base, exp in EXPECTED.items():
        for p in _gen_paraphrases(base, n=args.n_per_query, rng=rng):
            paraphrased.append((p, exp))
    print(f"  generated {len(paraphrased)} total queries\n")

    # Run A: PAMVR off
    print("=== Run A: PAMVR OFF ===")
    eng = fresh_engine({"recall.pamvr_enabled": False})
    t0 = time.time()
    ingest(eng)
    print(f"  ingest: {time.time() - t0:.1f}s")
    t0 = time.time()
    off = evaluate(eng, paraphrased)
    print(f"  eval:   {time.time() - t0:.1f}s")
    print(f"  top-1:  {off['top1_pct']:.2f}%  top-3: {off['top3_pct']:.2f}%  "
          f"top-10: {off['top10_pct']:.2f}%")
    print(f"  p50/p95/p99: {off['p50']}/{off['p95']}/{off['p99']}ms\n")
    try: eng.close()
    except Exception: pass

    # Run B: PAMVR on
    print("=== Run B: PAMVR ON ===")
    eng = fresh_engine({"recall.pamvr_enabled": True})
    t0 = time.time()
    ingest(eng)
    print(f"  ingest: {time.time() - t0:.1f}s")
    t0 = time.time()
    on = evaluate(eng, paraphrased)
    print(f"  eval:   {time.time() - t0:.1f}s")
    print(f"  top-1:  {on['top1_pct']:.2f}%  top-3: {on['top3_pct']:.2f}%  "
          f"top-10: {on['top10_pct']:.2f}%")
    print(f"  p50/p95/p99: {on['p50']}/{on['p95']}/{on['p99']}ms\n")
    try: eng.close()
    except Exception: pass

    print("=" * 70)
    print(f"AGGREGATE over {len(paraphrased)} queries")
    print("=" * 70)
    print(f"  metric        PAMVR off   PAMVR on    Δ")
    print(f"  --------------------------------------------------")
    for k in ("top1", "top3", "top10"):
        a = off[f"{k}_pct"]; b = on[f"{k}_pct"]; d = b - a
        sign = "+" if d >= 0 else ""
        print(f"  {k:<10}    {a:>6.2f}%    {b:>6.2f}%   {sign}{d:>5.2f}pp")
    print(f"  --------------------------------------------------")
    for k in ("p50", "p95", "p99"):
        a = off[k]; b = on[k]; d = b - a
        sign = "+" if d >= 0 else ""
        print(f"  {k:<10}    {a:>5}ms     {b:>5}ms    {sign}{d:.1f}ms")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"off": off, "on": on, "n_queries": len(paraphrased)},
                  f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
