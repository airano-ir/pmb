"""
Performance benchmark - latency, throughput, scaling, footprint.

Complements ablation_full.py (which measures recall *quality*). This one
measures the operational characteristics that show up in real use:

  1. Cold-start latency      - Engine() instantiation -> first recall return
  2. Write throughput        - record_batch p50/p95/p99 at batch sizes 1/10/100
  3. Recall scaling          - recall p50/p95 at workspace sizes 10/100/500/2000
  4. Cache hit latency       - second recall of the same query (LRU)
  5. Concurrent throughput   - 8 parallel recalls, queries/sec
  6. Disk footprint          - bytes_per_event for SQLite + LanceDB
  7. Prewarm benefit         - first recall with vs without prewarm

Run:
    python scripts/benchmarks/perf_bench.py
    python scripts/benchmarks/perf_bench.py --no-concurrent  # skip step 5
    python scripts/benchmarks/perf_bench.py --max-size 1000  # smaller scale

Output: a multi-section text report + JSON dump.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _bench_data import data_path

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parent.parent.parent / "src"))


# ----------------------------------------------------------------------
# Fixture: deterministic synthetic content so runs are reproducible.
# ----------------------------------------------------------------------


def synth_content(i: int) -> str:
    topics = [
        ("Postgres", "JSONB columns are indexable with GIN"),
        ("Redis", "use redis-cli to inspect keys"),
        ("Python", "f-strings support inline expressions"),
        ("Docker", "compose v2 uses pyproject syntax"),
        ("Git", "rebase rewrites history, merge preserves it"),
        ("FastAPI", "dependency injection happens at request time"),
        ("Rust", "borrow checker enforces aliasing rules"),
        ("React", "useMemo caches expensive computations"),
        ("AWS", "S3 supports presigned URLs for upload"),
        ("Kubernetes", "pods are ephemeral, services are stable"),
    ]
    topic, fact = topics[i % len(topics)]
    return f"Event {i} about {topic}: {fact}. Detail tag #{i // 10}."


def synth_queries(n: int = 20) -> list[str]:
    qs = [
        "Postgres JSONB indexing", "Redis CLI usage", "Python string formatting",
        "Docker compose syntax", "Git rebase vs merge", "FastAPI dependencies",
        "Rust borrow rules", "React memoization", "S3 upload URLs",
        "Kubernetes pod lifecycle", "JSONB GIN index", "key inspection redis",
        "f-string expression", "compose v2", "rewriting history",
        "request scoped deps", "borrow checker", "expensive memo",
        "presigned upload", "ephemeral pods",
    ]
    return qs[:n]


def percentiles(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    xs = sorted(xs)
    n = len(xs)
    return {
        "n": n,
        "p50": round(xs[n // 2], 2),
        "p95": round(xs[min(n - 1, int(0.95 * n))], 2),
        "p99": round(xs[min(n - 1, int(0.99 * n))], 2),
        "mean": round(statistics.mean(xs), 2),
    }


def dir_size_bytes(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


# ----------------------------------------------------------------------
# Engine factory
# ----------------------------------------------------------------------


def fresh_engine(disable_cache: bool = False):
    """Make an Engine in fresh tmp dirs and return (engine, pmb_home, workspace).

    `disable_cache=True` turns off the LRU recall cache - use this whenever you
    measure scaling or per-query latency so cache hits don't masquerade as
    "recall is sub-millisecond". The realistic-workload number is measured
    elsewhere in `bench_cache_hit()` which deliberately replays one query.
    """
    from pmb.core.engine import Engine
    tmp_home = Path(tempfile.mkdtemp(prefix="pmb-perf-"))
    tmp_ws = Path(tempfile.mkdtemp(prefix="pmb-perf-ws-"))
    os.environ["PMB_HOME"] = str(tmp_home)
    overrides = {"recall.cache_size": 0} if disable_cache else None
    eng = Engine(cwd=tmp_ws, pmb_home=tmp_home, rerank_model=None,
                 config_overrides=overrides)
    return eng, tmp_home, tmp_ws


# ----------------------------------------------------------------------
# Benchmark sections
# ----------------------------------------------------------------------


def bench_cold_start() -> dict:
    """Engine construction time + first recall time."""
    print(">>> 1. Cold-start latency")
    from pmb.core.engine import Engine

    tmp_home = Path(tempfile.mkdtemp(prefix="pmb-cold-"))
    tmp_ws = Path(tempfile.mkdtemp(prefix="pmb-cold-ws-"))
    os.environ["PMB_HOME"] = str(tmp_home)

    t0 = time.perf_counter()
    eng = Engine(cwd=tmp_ws, pmb_home=tmp_home, rerank_model=None)
    t_construct = (time.perf_counter() - t0) * 1000

    # Seed with one item so recall has anything to return
    eng.record_batch([{"type": "fact", "content": "seed for cold start"}])

    t0 = time.perf_counter()
    eng.recall("seed query")
    t_first_recall = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    eng.recall("seed query")  # warm
    t_warm_recall = (time.perf_counter() - t0) * 1000

    try:
        eng.close()
    except Exception:
        pass

    result = {
        "engine_construct_ms": round(t_construct, 1),
        "first_recall_ms": round(t_first_recall, 1),
        "warm_recall_ms": round(t_warm_recall, 1),
    }
    print(f"    Engine() construction: {result['engine_construct_ms']:.1f}ms")
    print(f"    First recall (cold):   {result['first_recall_ms']:.1f}ms")
    print(f"    Same recall (warm):    {result['warm_recall_ms']:.1f}ms")
    return result


def bench_write_throughput() -> dict:
    """record_batch latency at batch sizes 1, 10, 100."""
    print(">>> 2. Write throughput")
    eng, _, _ = fresh_engine()
    # Warm up so first call doesn't dominate
    eng.record_batch([{"type": "fact", "content": "warmup"}])

    results = {}
    counter = 0
    for batch_size in (1, 10, 100):
        latencies = []
        for trial in range(20):
            items = [
                {"type": "fact", "content": synth_content(counter + i)}
                for i in range(batch_size)
            ]
            counter += batch_size
            t0 = time.perf_counter()
            eng.record_batch(items)
            latencies.append((time.perf_counter() - t0) * 1000)
        stats = percentiles(latencies)
        stats["batch_size"] = batch_size
        stats["events_per_sec"] = round(
            batch_size / (stats["p50"] / 1000.0), 1
        ) if stats["p50"] > 0 else 0
        results[f"batch_{batch_size}"] = stats
        print(f"    batch={batch_size:>3} (n=20 trials): "
              f"p50={stats['p50']:.1f}ms p95={stats['p95']:.1f}ms "
              f"p99={stats['p99']:.1f}ms  -> {stats['events_per_sec']:.0f} ev/s")

    try:
        eng.close()
    except Exception:
        pass
    return results


def bench_recall_scaling(sizes: list[int]) -> dict:
    """Ingest progressively larger corpora, measure recall latency at each.

    LRU cache is OFF here - every measurement is a true cache miss, so the
    p50 you see is the actual recall pipeline cost. The cache-hit speedup
    is measured separately in `bench_cache_hit()`.
    """
    print(">>> 3. Recall scaling (cache disabled - true cold-query latency)")
    eng, home, ws = fresh_engine(disable_cache=True)
    queries = synth_queries(20)
    results = {}

    counter = 0
    for target_size in sizes:
        # Top up to target_size events
        while counter < target_size:
            batch = [
                {"type": "fact", "content": synth_content(counter + i)}
                for i in range(min(50, target_size - counter))
            ]
            eng.record_batch(batch)
            counter += len(batch)

        # Wait for async batch processing to drain
        time.sleep(1.5)

        # Warm-up recall + 50 measured
        eng.recall(queries[0])
        latencies = []
        for i in range(50):
            q = queries[i % len(queries)]
            t0 = time.perf_counter()
            eng.recall(q)
            latencies.append((time.perf_counter() - t0) * 1000)
        stats = percentiles(latencies)
        stats["workspace_size"] = target_size
        results[f"size_{target_size}"] = stats

        # Disk footprint at this scale
        disk = dir_size_bytes(home) + dir_size_bytes(ws)
        results[f"size_{target_size}"]["disk_bytes"] = disk
        results[f"size_{target_size}"]["bytes_per_event"] = round(disk / max(1, counter), 1)

        print(f"    n={target_size:>5} events: "
              f"p50={stats['p50']:.1f}ms p95={stats['p95']:.1f}ms "
              f"p99={stats['p99']:.1f}ms  disk={disk / 1024 / 1024:.1f}MB "
              f"({disk / max(1, counter):.0f} B/ev)")

    try:
        eng.close()
    except Exception:
        pass
    return results


def bench_cache_hit() -> dict:
    """How fast is the LRU cache vs fresh recall?"""
    print(">>> 4. Cache hit latency")
    eng, _, _ = fresh_engine()
    eng.record_batch([
        {"type": "fact", "content": synth_content(i)} for i in range(100)
    ])
    time.sleep(1.5)
    q = "Postgres JSONB indexing"

    # First (cold + miss)
    t0 = time.perf_counter()
    eng.recall(q)
    t_first = (time.perf_counter() - t0) * 1000
    # Second (warm + cache hit)
    t0 = time.perf_counter()
    eng.recall(q)
    t_second = (time.perf_counter() - t0) * 1000
    # Third (still in cache)
    t0 = time.perf_counter()
    eng.recall(q)
    t_third = (time.perf_counter() - t0) * 1000

    try:
        eng.close()
    except Exception:
        pass

    result = {
        "first_miss_ms": round(t_first, 2),
        "second_hit_ms": round(t_second, 2),
        "third_hit_ms": round(t_third, 2),
        "speedup_x": round(t_first / max(0.001, t_second), 1),
    }
    print(f"    first call (miss):  {result['first_miss_ms']:.2f}ms")
    print(f"    second call (hit):  {result['second_hit_ms']:.2f}ms")
    print(f"    third call (hit):   {result['third_hit_ms']:.2f}ms")
    print(f"    speedup:            {result['speedup_x']}x")
    return result


def bench_concurrent(n_workers: int = 8, n_queries: int = 80) -> dict:
    """Parallel recalls - measure aggregate throughput.

    Cache is OFF so each query genuinely hits the engine. Repeating a query
    from the warmup loop with the cache on would just measure the LRU
    dict lookup, not contention.
    """
    print(f">>> 5. Concurrent throughput ({n_workers} workers, cache disabled)")
    eng, _, _ = fresh_engine(disable_cache=True)
    eng.record_batch([
        {"type": "fact", "content": synth_content(i)} for i in range(200)
    ])
    time.sleep(2.0)
    queries = synth_queries(20)

    # Warmup
    eng.recall(queries[0])

    def one_recall(i):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        eng.recall(q)
        return (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    latencies = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(one_recall, i) for i in range(n_queries)]
        for f in as_completed(futures):
            latencies.append(f.result())
    wall = time.perf_counter() - t0

    try:
        eng.close()
    except Exception:
        pass

    stats = percentiles(latencies)
    stats["wall_s"] = round(wall, 2)
    stats["queries_per_sec"] = round(n_queries / wall, 1)
    stats["n_workers"] = n_workers
    print(f"    wall: {wall:.2f}s for {n_queries} queries -> "
          f"{stats['queries_per_sec']:.1f} q/s")
    print(f"    per-query: p50={stats['p50']:.1f}ms p95={stats['p95']:.1f}ms "
          f"p99={stats['p99']:.1f}ms")
    return stats


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=data_path("pmb_perf.json"))
    ap.add_argument("--max-size", type=int, default=2000,
                    help="Top scaling tier (smaller = faster run)")
    ap.add_argument("--no-concurrent", action="store_true",
                    help="Skip the concurrent throughput section")
    args = ap.parse_args()

    print("PMB performance benchmark")
    print(f"  output: {args.out}")
    print(f"  max scaling size: {args.max_size}")
    print()

    t_total = time.perf_counter()
    report = {}

    report["cold_start"] = bench_cold_start()
    print()
    report["write_throughput"] = bench_write_throughput()
    print()
    scale_sizes = [s for s in (10, 100, 500, 1000, 2000) if s <= args.max_size]
    report["recall_scaling"] = bench_recall_scaling(scale_sizes)
    print()
    report["cache_hit"] = bench_cache_hit()
    print()
    if not args.no_concurrent:
        report["concurrent"] = bench_concurrent()
        print()

    total_wall = time.perf_counter() - t_total
    report["_meta"] = {
        "total_wall_s": round(total_wall, 1),
        "platform": sys.platform,
        "python": sys.version.split()[0],
    }

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    cs = report["cold_start"]
    print(f"Cold start:        {cs['engine_construct_ms']}ms construct + "
          f"{cs['first_recall_ms']}ms first recall = "
          f"{cs['engine_construct_ms'] + cs['first_recall_ms']:.1f}ms total")
    print(f"Warm recall:       {cs['warm_recall_ms']}ms")
    wt = report["write_throughput"]
    print(f"Write (batch=1):   p50={wt['batch_1']['p50']}ms  "
          f"({wt['batch_1']['events_per_sec']:.0f} ev/s)")
    print(f"Write (batch=100): p50={wt['batch_100']['p50']}ms  "
          f"({wt['batch_100']['events_per_sec']:.0f} ev/s)")
    rs = report["recall_scaling"]
    largest = f"size_{scale_sizes[-1]}"
    print(f"Recall @{scale_sizes[-1]} ev:    p50={rs[largest]['p50']}ms  "
          f"p95={rs[largest]['p95']}ms  "
          f"({rs[largest]['bytes_per_event']:.0f} B/event on disk)")
    ch = report["cache_hit"]
    print(f"Cache speedup:     {ch['speedup_x']}x ({ch['first_miss_ms']}ms -> "
          f"{ch['second_hit_ms']}ms)")
    if "concurrent" in report:
        c = report["concurrent"]
        print(f"Concurrent:        {c['queries_per_sec']:.0f} q/s with {c['n_workers']} workers")
    print(f"Total wall time:   {total_wall:.1f}s")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
