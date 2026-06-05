"""
Comprehensive validation suite: performance + every E2E feature + stress.

Runs ~30-60 minutes. Output:
  - structured JSON dump (for trend tracking / CI history)
  - human-readable summary at the end

Sections:
  1. Performance scaling  (cold start, recall latency, write throughput, cache, concurrency, footprint)
  2. E2E feature coverage (one curated scenario per designed feature)
  3. Stress tests         (sustained load, concurrent writes+reads, large content)
  4. Quality summary      (pass rate, latency budget compliance, feature coverage)

Run:
    python scripts/benchmarks/full_validation.py
    python scripts/benchmarks/full_validation.py --quick     # smaller scales
    python scripts/benchmarks/full_validation.py --skip-stress
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _bench_data import data_path

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parent.parent.parent / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def percentiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
    xs = sorted(xs)
    n = len(xs)
    return {
        "n": n,
        "p50": round(xs[n // 2], 2),
        "p95": round(xs[min(n - 1, int(0.95 * n))], 2),
        "p99": round(xs[min(n - 1, int(0.99 * n))], 2),
        "mean": round(statistics.mean(xs), 2),
        "max": round(max(xs), 2),
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


def fresh_engine(disable_cache: bool = False, extra_overrides: Optional[dict] = None):
    from pmb.core.engine import Engine
    tmp_home = Path(tempfile.mkdtemp(prefix="pmb-val-"))
    tmp_ws = Path(tempfile.mkdtemp(prefix="pmb-val-ws-"))
    os.environ["PMB_HOME"] = str(tmp_home)
    overrides = dict(extra_overrides or {})
    if disable_cache:
        overrides["recall.cache_size"] = 0
    eng = Engine(cwd=tmp_ws, pmb_home=tmp_home, rerank_model=None,
                 config_overrides=overrides)
    return eng, tmp_home, tmp_ws


def synth_content(i: int) -> str:
    topics = [
        ("Postgres",   "JSONB columns indexable with GIN"),
        ("Redis",      "use redis-cli to inspect keys"),
        ("Python",     "f-strings support inline expressions"),
        ("Docker",     "compose v2 uses pyproject syntax"),
        ("Git",        "rebase rewrites history, merge preserves"),
        ("FastAPI",    "dependency injection happens at request time"),
        ("Rust",       "borrow checker enforces aliasing rules"),
        ("React",      "useMemo caches expensive computations"),
        ("AWS",        "S3 supports presigned URLs for upload"),
        ("Kubernetes", "pods are ephemeral, services are stable"),
    ]
    t, fact = topics[i % len(topics)]
    return f"Event {i} about {t}: {fact}. Tag #{i // 10}."


def safe(label: str, fn: Callable, *args, **kwargs):
    """Run a section, capture exceptions so the suite continues."""
    print(f"\n{'='*72}\n  {label}\n{'='*72}")
    t0 = time.perf_counter()
    try:
        out = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"  -> {label} done in {elapsed:.1f}s")
        return {"ok": True, "result": out, "wall_s": round(elapsed, 1)}
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "wall_s": round(time.perf_counter() - t0, 1)}


# ---------------------------------------------------------------------------
# 1. PERFORMANCE
# ---------------------------------------------------------------------------


def perf_cold_start(trials: int = 3) -> dict:
    """Median of N cold-start measurements (fresh process per trial)."""
    print(f"  Cold start ({trials} trials, fresh subprocess each)...")
    import subprocess
    times_ms = []
    for i in range(trials):
        code = (
            "import time, os, tempfile, sys\n"
            "from pathlib import Path\n"
            "tmp_home = Path(tempfile.mkdtemp())\n"
            "tmp_ws = Path(tempfile.mkdtemp())\n"
            "os.environ['PMB_HOME'] = str(tmp_home)\n"
            "t0 = time.perf_counter()\n"
            "from pmb.core.engine import Engine\n"
            "eng = Engine(cwd=tmp_ws, pmb_home=tmp_home, rerank_model=None)\n"
            "print(f'CONSTRUCT_MS={(time.perf_counter()-t0)*1000:.1f}')\n"
            "t0 = time.perf_counter()\n"
            "eng.record_batch([{'type':'fact','content':'seed'}])\n"
            "print(f'FIRST_WRITE_MS={(time.perf_counter()-t0)*1000:.1f}')\n"
            "import time as _t; _t.sleep(1.0)\n"  # let async embed start
        )
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, check=False)
        construct = None
        for line in out.stdout.splitlines():
            if line.startswith("CONSTRUCT_MS="):
                construct = float(line.split("=")[1])
                break
        if construct is not None:
            times_ms.append(construct)
        print(f"    trial {i+1}: construct = {construct} ms")
    return {"trials": trials, "construct_ms": percentiles(times_ms)}


def perf_recall_scaling(sizes: list[int]) -> dict:
    """Recall latency at progressively larger workspace sizes."""
    print(f"  Recall scaling at sizes {sizes}...")
    eng, home, ws = fresh_engine(disable_cache=True)
    # Force model load so warm path is measured
    _ = eng.search.model
    results = {}
    counter = 0
    for target in sizes:
        # Top up to target
        while counter < target:
            batch = [{"type": "fact", "content": synth_content(counter + i)}
                     for i in range(min(50, target - counter))]
            eng.record_batch(batch)
            counter += len(batch)
        # Let async drain
        time.sleep(2.0)

        # 30 measured recalls, unique queries each
        latencies = []
        queries = [f"about {topic} {i//3}" for i, topic in enumerate(
            ["Postgres", "Redis", "Python", "Docker", "Git", "FastAPI",
             "Rust", "React", "AWS", "Kubernetes"] * 3)][:30]
        for q in queries:
            t0 = time.perf_counter()
            eng.recall(q)
            latencies.append((time.perf_counter() - t0) * 1000)

        stats = percentiles(latencies)
        stats["workspace_size"] = target
        stats["disk_mb"] = round((dir_size_bytes(home) + dir_size_bytes(ws)) / 1024 / 1024, 1)
        stats["bytes_per_event"] = round((dir_size_bytes(home) + dir_size_bytes(ws)) / counter, 0)
        results[f"size_{target}"] = stats
        print(f"    n={target:>5}: p50={stats['p50']:>6.1f}ms  "
              f"p95={stats['p95']:>6.1f}ms  p99={stats['p99']:>6.1f}ms  "
              f"disk={stats['disk_mb']:>5.1f}MB  ({stats['bytes_per_event']:.0f} B/ev)")

    try:
        eng.close()
    except Exception:
        pass
    return results


def perf_write_throughput(batch_sizes: list[int]) -> dict:
    """Throughput at various batch sizes."""
    print(f"  Write throughput at batch sizes {batch_sizes}...")
    eng, _, _ = fresh_engine()
    # Warm up model so embedding cost isn't blocked
    _ = eng.search.model
    eng.record_batch([{"type": "fact", "content": "warmup"}])
    time.sleep(2.0)

    results = {}
    counter = 1000  # offset so we don't re-trigger dedup
    for b in batch_sizes:
        latencies = []
        for trial in range(10 if b <= 10 else 5):
            items = [{"type": "fact", "content": synth_content(counter + i)}
                     for i in range(b)]
            counter += b
            t0 = time.perf_counter()
            eng.record_batch(items)
            latencies.append((time.perf_counter() - t0) * 1000)
        stats = percentiles(latencies)
        stats["batch_size"] = b
        stats["per_item_ms"] = round(stats["p50"] / b, 2) if b > 0 else 0
        stats["events_per_sec"] = round(1000 / stats["per_item_ms"], 0) if stats["per_item_ms"] > 0 else 0
        results[f"batch_{b}"] = stats
        print(f"    batch={b:>3}: p50={stats['p50']:>7.1f}ms  "
              f"per-item={stats['per_item_ms']:>5.1f}ms  "
              f"={stats['events_per_sec']:.0f} ev/s")

    try:
        eng.close()
    except Exception:
        pass
    return results


def perf_cache_speedup() -> dict:
    """LRU cache: repeated identical query."""
    print("  Cache speedup...")
    eng, _, _ = fresh_engine()
    eng.record_batch([{"type": "fact", "content": synth_content(i)} for i in range(50)])
    time.sleep(2.0)
    _ = eng.search.model
    q = "Postgres JSONB GIN indexing"

    t0 = time.perf_counter()
    eng.recall(q)
    miss = (time.perf_counter() - t0) * 1000
    hits = []
    for _ in range(5):
        t0 = time.perf_counter()
        eng.recall(q)
        hits.append((time.perf_counter() - t0) * 1000)

    try:
        eng.close()
    except Exception:
        pass

    return {
        "miss_ms": round(miss, 2),
        "hit_ms_mean": round(statistics.mean(hits), 4),
        "hit_ms_min": round(min(hits), 4),
        "speedup_x": round(miss / max(0.001, statistics.mean(hits)), 1),
    }


def perf_concurrent(worker_counts: list[int], n_per_worker: int = 20) -> dict:
    """Throughput at varying parallelism."""
    print(f"  Concurrent recall, workers in {worker_counts}...")
    eng, _, _ = fresh_engine(disable_cache=True)
    eng.record_batch([{"type": "fact", "content": synth_content(i)} for i in range(500)])
    time.sleep(2.5)
    _ = eng.search.model
    eng.recall("warmup")

    queries = [f"event about {t}" for t in
               ["Postgres", "Redis", "Python", "Docker", "Git", "FastAPI",
                "Rust", "React", "AWS", "Kubernetes"]]

    def one(i):
        t0 = time.perf_counter()
        eng.recall(queries[i % len(queries)])
        return (time.perf_counter() - t0) * 1000

    results = {}
    for w in worker_counts:
        n_total = w * n_per_worker
        t0 = time.perf_counter()
        lats = []
        with ThreadPoolExecutor(max_workers=w) as ex:
            futs = [ex.submit(one, i) for i in range(n_total)]
            for f in as_completed(futs):
                lats.append(f.result())
        wall = time.perf_counter() - t0
        stats = percentiles(lats)
        stats["workers"] = w
        stats["wall_s"] = round(wall, 2)
        stats["qps"] = round(n_total / wall, 1)
        results[f"workers_{w}"] = stats
        print(f"    {w:>2} workers: {stats['qps']:>5.1f} q/s, "
              f"p50={stats['p50']:>6.1f}ms, p95={stats['p95']:>6.1f}ms")

    try:
        eng.close()
    except Exception:
        pass
    return results


# ---------------------------------------------------------------------------
# 2. E2E FEATURE COVERAGE
# ---------------------------------------------------------------------------

@dataclass
class FeatureCheck:
    name: str
    description: str
    setup: list[dict] = field(default_factory=list)
    queries: list[dict] = field(default_factory=list)
    custom_check: Optional[Callable] = None


def _contains_any(text: str, needles: list[str]) -> bool:
    t = text.lower()
    return any(n.lower() in t for n in needles)


def _contains_all(text: str, needles: list[str]) -> bool:
    t = text.lower()
    return all(n.lower() in t for n in needles)


def _all_features() -> list[FeatureCheck]:
    return [
        FeatureCheck(
            name="F1. Personal facts",
            description="Identity, languages, project, preferences",
            setup=[
                {"type": "fact", "content": "User's name is Alex Smith.", "pin": True},
                {"type": "fact", "content": "Alex uses Python and Rust as primary languages."},
                {"type": "fact", "content": "Alex prefers vim keybindings."},
            ],
            queries=[
                {"q": "Who am I working with?", "any": ["Alex", "Smith"]},
                {"q": "What languages does the user use?", "any": ["Python", "Rust"]},
            ],
        ),

        FeatureCheck(
            name="F2. Multi-hop bridging",
            description="A -> B -> C, ask about C surfaces A or chain",
            setup=[
                {"type": "fact", "content": "We chose Postgres for the database."},
                {"type": "fact", "content": "Postgres supports JSONB columns natively."},
                {"type": "fact", "content": "JSONB lets us index nested metadata."},
            ],
            queries=[
                {"q": "Why does our metadata indexing work?", "any": ["JSONB", "Postgres"]},
                {"q": "What database did we pick?", "any": ["Postgres"]},
            ],
        ),

        FeatureCheck(
            name="F3. Cross-language (RU -> EN)",
            description="Russian query against English memory",
            setup=[
                {"type": "fact", "content": "Alex's cat is named Murka and is allergic to chicken."},
                {"type": "fact", "content": "Alex lives in Kyiv."},
                {"type": "fact", "content": "Alex drinks flat white coffee."},
            ],
            queries=[
                {"q": "Как зовут кошку?", "any": ["Murka", "cat"]},
                {"q": "Где живёт Alex?", "any": ["Kyiv"]},
                {"q": "Какой кофе?", "any": ["flat white", "coffee"]},
            ],
        ),

        FeatureCheck(
            name="F4. Recency over older equivalent",
            description="Latest fact ranks above older equivalent",
            setup=[
                {"type": "fact", "content": "We use Anthropic API for consolidation."},
                {"type": "fact", "content": "We switched to local Ollama for consolidation."},
            ],
            queries=[
                {"q": "What LLM backend for consolidation?", "any": ["Ollama"], "top_k": 3},
            ],
        ),

        FeatureCheck(
            name="F5. Pin priority",
            description="Pinned critical fact stays at top regardless",
            setup=[
                {"type": "fact", "content": "User generally enjoys coffee in the morning."},
                {"type": "fact", "content": "CRITICAL: User has a SEVERE caffeine allergy.",
                 "pin": True, "importance": 0.99},
                {"type": "fact", "content": "User drank espresso last week at the office."},
            ],
            queries=[
                {"q": "What should I know about caffeine for this user?",
                 "any": ["allergy", "caffeine"], "top_k": 3},
            ],
        ),

        FeatureCheck(
            name="F6. Goal tracking",
            description="record_goal + recall via status query",
            setup=[
                {"type": "goal", "title": "Ship PMB v0.1 to GitHub",
                 "status": "in_progress"},
                {"type": "goal", "title": "Build long-term ablation benchmark",
                 "status": "pending"},
            ],
            queries=[
                {"q": "What goals am I working on?", "any": ["v0.1", "GitHub"]},
            ],
        ),

        FeatureCheck(
            name="F7. Milestone chain",
            description="Milestones in a named chain surface together",
            setup=[
                {"type": "milestone", "chain_name": "release-v0.1",
                 "title": "Ablation found typo bug"},
                {"type": "milestone", "chain_name": "release-v0.1",
                 "title": "Defaults flipped, recall improved"},
                {"type": "milestone", "chain_name": "release-v0.1",
                 "title": "README rewrite"},
            ],
            queries=[
                {"q": "What milestones in release-v0.1?",
                 "any": ["typo", "Defaults", "README"], "top_k": 10},
            ],
        ),

        FeatureCheck(
            name="F8. Activity log",
            description="Recent actions accessible by 'what did agent do'",
            setup=[
                {"type": "activity", "content": "Fixed lazy LanceDB import bug.",
                 "kind": "fix"},
                {"type": "activity", "content": "Ran full ablation across components.",
                 "kind": "experiment"},
                {"type": "activity", "content": "Flipped typo_correction default to False.",
                 "kind": "decision"},
            ],
            queries=[
                {"q": "What did the agent do recently?",
                 "any": ["LanceDB", "ablation", "typo"], "top_k": 10},
            ],
        ),

        FeatureCheck(
            name="F9. Concept-to-instance (top-3)",
            description="'database' should surface 'PostgreSQL' fact in top-3",
            setup=[
                {"type": "fact", "content": "Alex prefers PostgreSQL over MySQL for the database."},
                {"type": "fact", "content": "Alex prefers tea over coffee."},
                {"type": "fact", "content": "Alex prefers vim over emacs."},
                {"type": "fact", "content": "Alex prefers dark mode."},
            ],
            queries=[
                {"q": "What database does Alex prefer?",
                 "any": ["PostgreSQL", "MySQL"], "top_k": 3},
            ],
        ),

        FeatureCheck(
            name="F10. Fact tree (hierarchical)",
            description="Main fact + subfacts retrievable as connected unit",
            setup=[
                {"type": "fact_tree",
                 "main": "Project PMB is open-source under Apache 2.0.",
                 "subfacts": [
                     "License changed from MIT in publication pass.",
                     "Apache 2.0 has explicit patent grant.",
                     "Same as mem0, Letta, Zep community editions.",
                 ],
                 "pin": True},
            ],
            queries=[
                {"q": "What license does PMB use?",
                 "any": ["Apache", "2.0"]},
                {"q": "Why did we pick this license?",
                 "any": ["patent", "grant", "Apache"]},
            ],
        ),

        FeatureCheck(
            name="F11. Person extraction",
            description="Speaker / named entity persons indexed in graph",
            setup=[
                {"type": "fact", "content": "Alice said we should use type hints everywhere."},
                {"type": "fact", "content": "Bob disagreed and prefers gradual typing."},
                {"type": "fact", "content": "Alice and Bob agreed to a compromise: mypy in CI only."},
            ],
            queries=[
                {"q": "What did Alice think about typing?",
                 "any": ["type hints", "Alice"], "top_k": 3},
                {"q": "What was Bob's view?",
                 "any": ["gradual", "Bob"], "top_k": 3},
            ],
        ),

        FeatureCheck(
            name="F12. Temporal proximity (date-anchored)",
            description="Date references parsed into event_time",
            setup=[
                {"type": "fact", "content": "On April 1, 2026 we launched the staging environment."},
                {"type": "fact", "content": "On May 15, 2026 we found the cold start bug."},
                {"type": "fact", "content": "On May 25, 2026 we fixed the LanceDB lazy load."},
            ],
            queries=[
                {"q": "What did we do in May 2026?",
                 "any": ["May", "cold start", "LanceDB"], "top_k": 5},
            ],
        ),

        FeatureCheck(
            name="F13. Conflict / supersession",
            description="Later statement marked or surfaced over old one",
            setup=[
                {"type": "fact", "content": "We will deploy to AWS."},
                {"type": "fact", "content": "After review we will deploy to GCP instead of AWS."},
            ],
            queries=[
                {"q": "Where are we deploying?",
                 "any": ["GCP"], "top_k": 3},
            ],
        ),

        FeatureCheck(
            name="F14. Causation graph (temporal_next)",
            description="Events linked by temporal-next edges traverse correctly",
            setup=[
                {"type": "fact", "content": "First we identified that BM25 was 86% of the signal."},
                {"type": "fact", "content": "After that we flipped typo_correction off."},
                {"type": "fact", "content": "Then we measured 94.1% on full LoCoMo."},
            ],
            queries=[
                {"q": "What was the sequence of changes that led to 94.1%?",
                 "all": ["BM25", "typo"], "top_k": 5},
            ],
        ),

        FeatureCheck(
            name="F15. Large content (5KB cap)",
            description="Content near the 5000-char cap still indexes and recalls",
            setup=[
                {"type": "fact",
                 "content": "Mega fact: " + ("Postgres JSONB " * 300) +
                            "and the GIN index makes nested queries fast."},
            ],
            queries=[
                {"q": "What about GIN index?", "any": ["GIN", "JSONB", "Postgres"]},
            ],
        ),

        FeatureCheck(
            name="F16. Repeated identical content (dedup L1)",
            description="Exact duplicate of an existing fact returns same ULID",
            setup=[
                {"type": "fact", "content": "User favourite editor is Helix."},
            ],
            queries=[],
            custom_check=lambda eng: _dedup_check(eng),
        ),
    ]


def _dedup_check(eng) -> dict:
    """Custom check: write the same fact twice, expect dedup to return same ULID."""
    r1 = eng.record_batch([{"type": "fact", "content": "User favourite editor is Helix."}])
    time.sleep(1.0)
    r2 = eng.record_batch([{"type": "fact", "content": "User favourite editor is Helix."}])
    ulid_1 = r1.get("results", [{}])[0].get("ulid")
    ulid_2 = r2.get("results", [{}])[0].get("ulid")
    return {
        "ulid_1": ulid_1,
        "ulid_2": ulid_2,
        "matched": ulid_1 == ulid_2,
    }


def run_feature_coverage() -> dict:
    """Run every feature check on a fresh engine, collect results."""
    features = _all_features()
    print(f"  Running {len(features)} feature coverage checks...")
    out = {}
    for fc in features:
        eng, _, _ = fresh_engine()
        _ = eng.search.model
        try:
            # Run setup
            if fc.setup:
                eng.record_batch(fc.setup)
                time.sleep(1.5)
            # Run queries
            query_results = []
            for q in fc.queries:
                pack = eng.recall(q["q"], top_k=q.get("top_k", 3))
                ctx = " ".join(r.content for r in pack.results).lower()
                ok = True
                if "any" in q and q["any"]:
                    ok = ok and _contains_any(ctx, q["any"])
                if "all" in q and q["all"]:
                    ok = ok and _contains_all(ctx, q["all"])
                query_results.append({
                    "query": q["q"],
                    "passed": ok,
                    "top_k_returned": len(pack.results),
                })
            # Custom check
            custom = fc.custom_check(eng) if fc.custom_check else None
            passed = (all(qr["passed"] for qr in query_results)
                      and (custom is None or custom.get("matched", True)))
            out[fc.name] = {
                "description": fc.description,
                "query_results": query_results,
                "custom": custom,
                "passed": passed,
            }
            mark = "✅" if passed else "❌"
            print(f"    {mark} {fc.name}")
        except Exception as e:
            out[fc.name] = {
                "description": fc.description,
                "error": f"{type(e).__name__}: {e}",
                "passed": False,
            }
            print(f"    ❌ {fc.name} ERRORED: {e}")
        finally:
            try:
                eng.close()
            except Exception:
                pass
    return out


# ---------------------------------------------------------------------------
# 3. STRESS
# ---------------------------------------------------------------------------


def stress_sustained_load(n_events: int = 1000, n_queries: int = 200) -> dict:
    """1000 events, then 200 unique queries back-to-back."""
    print(f"  Sustained load: {n_events} events, {n_queries} queries...")
    eng, _, _ = fresh_engine(disable_cache=True)
    _ = eng.search.model
    # Ingest
    counter = 0
    t_ing = time.perf_counter()
    while counter < n_events:
        batch = [{"type": "fact", "content": synth_content(counter + i)}
                 for i in range(min(50, n_events - counter))]
        eng.record_batch(batch)
        counter += len(batch)
    ingest_s = time.perf_counter() - t_ing
    time.sleep(3.0)
    # Sustained queries
    lats = []
    queries = [f"about {t} variant {i}" for i, t in enumerate(
        ["Postgres", "Redis", "Python", "Docker", "Git", "FastAPI",
         "Rust", "React", "AWS", "Kubernetes"] * 20)][:n_queries]
    t_q = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter()
        eng.recall(q)
        lats.append((time.perf_counter() - t0) * 1000)
    query_wall = time.perf_counter() - t_q
    stats = percentiles(lats)
    stats["n_events"] = n_events
    stats["n_queries"] = n_queries
    stats["ingest_s"] = round(ingest_s, 1)
    stats["query_wall_s"] = round(query_wall, 1)
    stats["sustained_qps"] = round(n_queries / query_wall, 1)
    print(f"    p50={stats['p50']:.1f}ms p95={stats['p95']:.1f}ms p99={stats['p99']:.1f}ms "
          f"max={stats['max']:.1f}ms  -> {stats['sustained_qps']:.1f} q/s")
    try:
        eng.close()
    except Exception:
        pass
    return stats


def stress_concurrent_rw(n_workers: int = 4, duration_s: float = 10.0) -> dict:
    """Concurrent writes + reads for N seconds."""
    print(f"  Concurrent R/W stress: {n_workers} workers for {duration_s}s...")
    eng, _, _ = fresh_engine(disable_cache=True)
    eng.record_batch([{"type": "fact", "content": synth_content(i)} for i in range(100)])
    time.sleep(2.0)
    _ = eng.search.model
    eng.recall("warmup")

    import threading
    stop = threading.Event()
    write_count = {"n": 0, "errors": 0}
    read_count = {"n": 0, "errors": 0}
    read_lats: list[float] = []

    def writer():
        i = 100
        while not stop.is_set():
            try:
                eng.record_batch([{"type": "fact", "content": synth_content(i)}])
                write_count["n"] += 1
                i += 1
            except Exception:
                write_count["errors"] += 1

    def reader():
        i = 0
        topics = ["Postgres", "Redis", "Python", "Docker", "Git"]
        while not stop.is_set():
            try:
                t0 = time.perf_counter()
                eng.recall(f"about {topics[i % 5]}")
                read_lats.append((time.perf_counter() - t0) * 1000)
                read_count["n"] += 1
                i += 1
            except Exception:
                read_count["errors"] += 1

    threads = []
    for _ in range(n_workers // 2):
        t = threading.Thread(target=writer, daemon=True)
        t.start()
        threads.append(t)
    for _ in range(n_workers // 2):
        t = threading.Thread(target=reader, daemon=True)
        t.start()
        threads.append(t)

    time.sleep(duration_s)
    stop.set()
    for t in threads:
        t.join(timeout=2.0)

    stats = {
        "duration_s": duration_s,
        "writes": write_count["n"], "write_errors": write_count["errors"],
        "reads": read_count["n"],   "read_errors": read_count["errors"],
        "read_latency": percentiles(read_lats),
    }
    print(f"    {stats['writes']} writes ({stats['write_errors']} err), "
          f"{stats['reads']} reads ({stats['read_errors']} err), "
          f"read p50={stats['read_latency']['p50']:.1f}ms p95={stats['read_latency']['p95']:.1f}ms")
    try:
        eng.close()
    except Exception:
        pass
    return stats


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Smaller scales (5x faster, less coverage)")
    ap.add_argument("--skip-stress", action="store_true",
                    help="Skip stress section")
    ap.add_argument("--out",
                    default=data_path("pmb_full_validation.json"))
    args = ap.parse_args()

    quick = args.quick

    print("=" * 72)
    print("PMB FULL VALIDATION SUITE")
    print("=" * 72)
    print(f"  Mode: {'QUICK' if quick else 'FULL'}")
    print(f"  Skip stress: {args.skip_stress}")
    print()

    t_total = time.perf_counter()
    report: dict[str, Any] = {"mode": "quick" if quick else "full"}

    # Section 1: Performance
    print("\n" + "#" * 72)
    print("# SECTION 1 - PERFORMANCE")
    print("#" * 72)

    report["cold_start"] = safe(
        "1.1 Cold start (median of N trials)",
        perf_cold_start, trials=2 if quick else 3
    )
    report["recall_scaling"] = safe(
        "1.2 Recall scaling vs workspace size",
        perf_recall_scaling,
        sizes=[10, 100, 500] if quick else [10, 100, 500, 1000, 2000]
    )
    report["write_throughput"] = safe(
        "1.3 Write throughput vs batch size",
        perf_write_throughput,
        batch_sizes=[1, 10, 50] if quick else [1, 5, 10, 50, 100]
    )
    report["cache_speedup"] = safe(
        "1.4 Cache hit speedup",
        perf_cache_speedup
    )
    report["concurrent"] = safe(
        "1.5 Concurrent throughput vs worker count",
        perf_concurrent,
        worker_counts=[1, 4, 8] if quick else [1, 2, 4, 8, 16]
    )

    # Section 2: Feature coverage
    print("\n" + "#" * 72)
    print("# SECTION 2 - E2E FEATURE COVERAGE")
    print("#" * 72)

    report["features"] = safe(
        "2.1 All feature checks",
        run_feature_coverage
    )

    # Section 3: Stress
    if not args.skip_stress:
        print("\n" + "#" * 72)
        print("# SECTION 3 - STRESS")
        print("#" * 72)
        report["sustained_load"] = safe(
            "3.1 Sustained load (N events + N queries)",
            stress_sustained_load,
            n_events=500 if quick else 1000,
            n_queries=100 if quick else 200,
        )
        report["concurrent_rw"] = safe(
            "3.2 Concurrent R/W stress",
            stress_concurrent_rw,
            n_workers=4,
            duration_s=5.0 if quick else 10.0,
        )

    total_wall = time.perf_counter() - t_total
    report["_meta"] = {
        "total_wall_s": round(total_wall, 1),
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "mode": "quick" if quick else "full",
    }

    # Section 4: Summary
    print("\n" + "#" * 72)
    print("# SECTION 4 - QUALITY SUMMARY")
    print("#" * 72)

    summary = _build_summary(report)
    print(summary)
    report["summary"] = {"text": summary}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {args.out}")
    print(f"Total wall: {total_wall:.1f}s")


def _build_summary(report: dict) -> str:
    lines = []
    lines.append("")
    lines.append("┌" + "─" * 70 + "┐")
    lines.append("│" + "  Performance".ljust(70) + "│")
    lines.append("├" + "─" * 70 + "┤")

    if report.get("cold_start", {}).get("ok"):
        cs = report["cold_start"]["result"]["construct_ms"]
        lines.append(f"│  Engine() cold construct (p50):  {cs['p50']:>8.1f} ms".ljust(71) + "│")

    if report.get("recall_scaling", {}).get("ok"):
        rs = report["recall_scaling"]["result"]
        sizes_sorted = sorted(rs.keys(), key=lambda k: int(k.split("_")[1]))
        last = sizes_sorted[-1]
        first = sizes_sorted[0]
        lines.append(f"│  Recall p50 at n={rs[first]['workspace_size']:>5}:        "
                     f"{rs[first]['p50']:>8.1f} ms".ljust(71) + "│")
        lines.append(f"│  Recall p50 at n={rs[last]['workspace_size']:>5}:        "
                     f"{rs[last]['p50']:>8.1f} ms".ljust(71) + "│")
        lines.append(f"│  Disk per event @ scale:         "
                     f"{rs[last]['bytes_per_event']:>8.0f} B".ljust(71) + "│")

    if report.get("write_throughput", {}).get("ok"):
        wt = report["write_throughput"]["result"]
        for k in sorted(wt.keys(), key=lambda k: int(k.split("_")[1])):
            stats = wt[k]
            lines.append(f"│  Write batch={stats['batch_size']:>3}: "
                         f"{stats['per_item_ms']:>5.1f} ms/item  "
                         f"({stats['events_per_sec']:.0f} ev/s)".ljust(71) + "│")

    if report.get("cache_speedup", {}).get("ok"):
        cs = report["cache_speedup"]["result"]
        lines.append(f"│  Cache miss / hit / speedup:  "
                     f"{cs['miss_ms']:>5.1f} / {cs['hit_ms_mean']:>5.3f} ms / "
                     f"{cs['speedup_x']:.0f}×".ljust(71) + "│")

    if report.get("concurrent", {}).get("ok"):
        co = report["concurrent"]["result"]
        for k in sorted(co.keys(), key=lambda k: int(k.split("_")[1])):
            st = co[k]
            lines.append(f"│  Concurrent w={st['workers']:>2}: "
                         f"{st['qps']:>5.1f} q/s, p95={st['p95']:>6.1f}ms".ljust(71) + "│")

    lines.append("├" + "─" * 70 + "┤")
    lines.append("│" + "  Features (E2E)".ljust(70) + "│")
    lines.append("├" + "─" * 70 + "┤")
    if report.get("features", {}).get("ok"):
        fr = report["features"]["result"]
        passed = sum(1 for f in fr.values() if f.get("passed"))
        total = len(fr)
        lines.append(f"│  Passed: {passed}/{total}  "
                     f"({100 * passed / max(1, total):.0f}%)".ljust(71) + "│")
        for name, data in fr.items():
            mark = "✓" if data.get("passed") else "✗"
            lines.append(f"│    {mark} {name}".ljust(71) + "│")

    if not report.get("sustained_load", {}).get("ok") is False:
        lines.append("├" + "─" * 70 + "┤")
        lines.append("│" + "  Stress".ljust(70) + "│")
        lines.append("├" + "─" * 70 + "┤")
        if report.get("sustained_load", {}).get("ok"):
            sl = report["sustained_load"]["result"]
            lines.append(f"│  Sustained {sl['n_queries']} q/{sl['n_events']} events: "
                         f"{sl['sustained_qps']:.1f} q/s, "
                         f"p95={sl['p95']:.1f}ms".ljust(71) + "│")
        if report.get("concurrent_rw", {}).get("ok"):
            cr = report["concurrent_rw"]["result"]
            lines.append(f"│  Concurrent R/W:  "
                         f"{cr['writes']} writes, {cr['reads']} reads, "
                         f"{cr['write_errors'] + cr['read_errors']} errors".ljust(71) + "│")

    lines.append("└" + "─" * 70 + "┘")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
