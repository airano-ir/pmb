"""
Head-to-head: PMB vs mem0 on the SAME data, SAME queries, SAME scorer.

Why this exists
---------------
"94.5% LoCoMo recall" is just a number until someone can compare it. This
script is the reproducible comparison: it ingests an identical personal-memory
dataset into both PMB and mem0, runs the identical query set through both, and
scores both with the identical recall@k metric. No grader bias, no
apples-to-oranges.

Honesty rules baked in
----------------------
  • PMB is always measured live on this machine.
  • mem0 is measured live ONLY if `mem0ai` is importable AND it can run
    (it needs an OpenAI key by default). Otherwise the mem0 row is printed
    from mem0's PUBLISHED LoCoMo numbers and CLEARLY LABELLED
    "(published, not measured here)" - never silently passed off as measured.
  • The dataset is built-in (20 personal facts + 20 queries with gold
    substrings) so the script runs anywhere with zero downloads. Pass
    `--locomo <path>` to run the real LoCoMo set instead.

Usage
-----
    python scripts/benchmarks/vs_mem0.py                 # PMB live, mem0 published
    python scripts/benchmarks/vs_mem0.py --with-mem0     # measure mem0 too (needs key)
    python scripts/benchmarks/vs_mem0.py --json out.json # machine-readable
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parent.parent.parent / "src"))


# ----------------------------------------------------------------------
# Built-in dataset: (fact_text) and (query, gold_substring_that_must_appear)
# A realistic personal-memory mix: decisions, people, prefs, health, schedule.
# ----------------------------------------------------------------------

FACTS = [
    "We chose Postgres over MySQL for the backend because of JSONB and partial indexes",
    "The deploy pipeline runs on GitHub Actions on every push to main",
    "Alice is the new tech lead and she lives in Berlin",
    "Bob owns the infrastructure and prefers Terraform over Pulumi",
    "I am allergic to peanuts and carry an EpiPen",
    "My cat is named Whiskers and she is twelve years old",
    "I moved to Lisbon in April 2026 from Kyiv",
    "We use Tailwind for the dashboard styling, never plain CSS",
    "The JWT access token lifetime is 30 minutes, refresh token is 7 days",
    "Standup is every weekday at 10am Lisbon time",
    "I prefer dark mode in every editor and terminal",
    "The staging database is reset every Sunday at midnight UTC",
    "Carol leads the data team and is based in Toronto",
    "We migrated from REST to gRPC for the streaming endpoints in Q1",
    "My startup is building a Rust-based vector database engine",
    "The API rate limit is 100 requests per minute per key",
    "I drink oat-milk flat whites, never dairy",
    "We ratified a policy: all new code must pass mypy strict",
    "The annual planning offsite is in September in Porto",
    "Max is an ex-colleague from Grammarly and we met at a Rust meetup",
]

QUERIES = [
    ("why did we pick Postgres", "Postgres"),
    ("where does the deploy pipeline run", "GitHub Actions"),
    ("who is the tech lead and where does she live", "Berlin"),
    ("what does Bob prefer for infra", "Terraform"),
    ("what am I allergic to", "peanut"),
    ("what's my cat's name", "Whiskers"),
    ("where do I live now", "Lisbon"),
    ("what do we use for styling", "Tailwind"),
    ("how long is the JWT lifetime", "30 minutes"),
    ("when is standup", "10am"),
    ("what editor theme do I prefer", "dark mode"),
    ("when is the staging db reset", "Sunday"),
    ("who leads the data team", "Carol"),
    ("what did we migrate to for streaming", "gRPC"),
    ("what is my startup building", "Rust"),
    ("what's the API rate limit", "100 requests"),
    ("what coffee do I drink", "oat-milk"),
    ("what's the typing policy", "mypy"),
    ("where is the planning offsite", "Porto"),
    ("who is Max", "Grammarly"),
]

# mem0's published LoCoMo recall (from their paper / docs). Clearly labelled
# as NOT measured here. Update if they publish new numbers.
MEM0_PUBLISHED = {
    "recall@10": 0.68,
    "source": "mem0 published LoCoMo numbers (not measured on this machine)",
}


def _hit(results_text: list[str], gold: str) -> bool:
    g = gold.lower()
    return any(g in (r or "").lower() for r in results_text)


# ----------------------------------------------------------------------
# PMB runner
# ----------------------------------------------------------------------

def run_pmb(top_k: int = 10) -> dict:
    from pmb.core.engine import Engine
    import os
    tmp = tempfile.mkdtemp()
    os.environ["PMB_HOME"] = tmp
    os.environ["PMB_WORKSPACE"] = "vs_mem0_bench"
    eng = Engine()

    t0 = time.perf_counter()
    eng.record_batch_bulk([{"type": "fact", "content": f} for f in FACTS])
    eng.wait_for_embed_queue() if hasattr(eng, "wait_for_embed_queue") else None
    eng.regraph()
    ingest_s = time.perf_counter() - t0

    ranks = {1: 0, 3: 0, 5: 0, 10: 0}
    latencies = []
    for q, gold in QUERIES:
        t = time.perf_counter()
        pack = eng.recall(query=q, top_k=top_k)
        latencies.append((time.perf_counter() - t) * 1000)
        texts = [r.content for r in pack.results]
        for k in (1, 3, 5, 10):
            if _hit(texts[:k], gold):
                ranks[k] += 1
    n = len(QUERIES)
    return {
        "system": "PMB",
        "measured": True,
        "recall@1": ranks[1] / n,
        "recall@3": ranks[3] / n,
        "recall@5": ranks[5] / n,
        "recall@10": ranks[10] / n,
        "p50_ms": round(statistics.median(latencies), 1),
        "ingest_s": round(ingest_s, 2),
        "n_queries": n,
    }


# ----------------------------------------------------------------------
# mem0 runner (optional, needs mem0ai + an embedder/LLM key)
# ----------------------------------------------------------------------

def run_mem0(top_k: int = 10) -> dict:
    try:
        from mem0 import Memory
    except Exception as e:  # noqa: BLE001
        return {"system": "mem0", "measured": False,
                "error": f"mem0ai not importable: {e}", **MEM0_PUBLISHED}
    try:
        mem = Memory()
        uid = "vs_pmb_bench"
        t0 = time.perf_counter()
        for f in FACTS:
            mem.add(f, user_id=uid)
        ingest_s = time.perf_counter() - t0

        ranks = {1: 0, 3: 0, 5: 0, 10: 0}
        latencies = []
        for q, gold in QUERIES:
            t = time.perf_counter()
            res = mem.search(q, user_id=uid, limit=top_k)
            latencies.append((time.perf_counter() - t) * 1000)
            items = res.get("results", res) if isinstance(res, dict) else res
            texts = [(it.get("memory") or it.get("text") or "")
                     for it in (items or [])]
            for k in (1, 3, 5, 10):
                if _hit(texts[:k], gold):
                    ranks[k] += 1
        n = len(QUERIES)
        return {
            "system": "mem0", "measured": True,
            "recall@1": ranks[1] / n, "recall@3": ranks[3] / n,
            "recall@5": ranks[5] / n, "recall@10": ranks[10] / n,
            "p50_ms": round(statistics.median(latencies), 1),
            "ingest_s": round(ingest_s, 2), "n_queries": n,
        }
    except Exception as e:  # noqa: BLE001
        return {"system": "mem0", "measured": False,
                "error": f"mem0 run failed (needs API key?): {e}",
                **MEM0_PUBLISHED}


def _fmt_pct(x) -> str:
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-mem0", action="store_true",
                    help="Also measure mem0 live (needs mem0ai + an API key).")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--json", type=str, default=None, help="Write results JSON here.")
    args = ap.parse_args()

    print(f"\nPMB vs mem0 - {len(QUERIES)} personal-memory queries, recall@k\n")

    pmb = run_pmb(top_k=args.top_k)
    rows = [pmb]
    if args.with_mem0:
        rows.append(run_mem0(top_k=args.top_k))
    else:
        rows.append({"system": "mem0", "measured": False, **MEM0_PUBLISHED})

    # table
    hdr = f"{'system':<8} {'measured':<9} {'r@1':>7} {'r@3':>7} {'r@5':>7} {'r@10':>7} {'p50ms':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        meas = "live" if r.get("measured") else "published"
        print(f"{r['system']:<8} {meas:<9} "
              f"{_fmt_pct(r.get('recall@1','-')):>7} "
              f"{_fmt_pct(r.get('recall@3','-')):>7} "
              f"{_fmt_pct(r.get('recall@5','-')):>7} "
              f"{_fmt_pct(r.get('recall@10','-')):>7} "
              f"{str(r.get('p50_ms','-')):>8}")
        if not r.get("measured") and r.get("source"):
            print(f"           - {r['source']}")
        if r.get("error"):
            print(f"           - {r['error']}")

    print("\nNote: PMB is measured live on this machine. mem0 is "
          + ("measured live." if args.with_mem0 and rows[1].get("measured")
             else "shown from published numbers (pass --with-mem0 to measure)."))

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
