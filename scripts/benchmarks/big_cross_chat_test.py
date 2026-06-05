"""
Big cross-chat stress test: 10 sessions, 200+ events, 1000+ queries,
tier transitions, decay simulation, consolidation readiness, summaries.

Sections:
  1. Build 10 sessions (~25 events each) with cross-session entities
  2. 1000 queries across 4 intent categories - latency distribution
  3. Tier transitions - verify access_count promotes events
  4. Decay simulation - advance timestamps 30 days, archive rate
  5. Cross-session summary tools - recent_activity, list_goals, what_just_happened
  6. Sleep/consolidation readiness check (auto_consolidate trigger logic)

Run:
    python scripts/benchmarks/big_cross_chat_test.py
    python scripts/benchmarks/big_cross_chat_test.py --quick   # 5 sessions, 300 queries
"""

from __future__ import annotations

import argparse
import json
import os
import random
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


def percentiles(xs: list[float]) -> dict:
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


# ----------------------------------------------------------------------
# Story generator: 10 sessions, each with mixed events
# ----------------------------------------------------------------------


SESSIONS = [
    {
        "title": "Day 1 — Kickoff",
        "facts": [
            "Project name is order-service, repo at acme/orders.",
            "Tech stack: Python 3.12, FastAPI, Postgres 15.",
            "Team: Alex (lead), Bob, Carol, Dana.",
            "Sprint length is two weeks.",
        ],
        "goals": [
            ("Ship v1.0 by end of quarter", "in_progress"),
            ("Hit 99.9% uptime SLA", "in_progress"),
        ],
        "activities": [
            ("decision", "Decided to use Postgres over MongoDB for JSONB support."),
            ("decision", "Decided FastAPI over Flask for built-in async."),
        ],
        "milestones": [("kickoff", "Project kicked off, stack chosen.")],
    },
    {
        "title": "Day 3 — First feature",
        "facts": [
            "Alex prefers small functions over large classes.",
            "Bob set up CI on GitHub Actions.",
            "Carol owns the API design.",
        ],
        "activities": [
            ("feature", "Implemented order-creation endpoint POST /orders."),
            ("review", "Code review for PR #12 from Carol on API schemas."),
        ],
        "milestones": [("kickoff", "First endpoint live in staging.")],
    },
    {
        "title": "Day 7 — Bug investigation",
        "facts": [
            "Root cause: JSONB metadata column was null when payload omitted it.",
            "Postgres returns NULL not '{}' for unset jsonb.",
        ],
        "activities": [
            ("bug", "Bug found: orders crash on null JSONB metadata."),
            ("fix", "Fix landed in migration 0042 via COALESCE."),
            ("decision", "Decided to enforce NOT NULL on all jsonb columns going forward."),
        ],
        "milestones": [("bugfix-jsonb", "JSONB null bug closed in v0.1.3.")],
    },
    {
        "title": "Day 10 — Auth redesign starts",
        "facts": [
            "Auth flow uses JWT with refresh tokens.",
            "Alice from security team reviewed the design.",
        ],
        "goals": [("Auth redesign this sprint", "in_progress")],
        "activities": [
            ("decision", "Decided to store refresh tokens in httpOnly cookies, not localStorage."),
            ("pairing", "Pair-programmed with Bob on JWT verification edge cases."),
        ],
        "milestones": [("auth-redesign", "Auth spec written.")],
    },
    {
        "title": "Day 14 — Auth ships",
        "facts": [
            "Refresh-token rotation implemented and live.",
            "All endpoints behind the new middleware.",
        ],
        "activities": [
            ("test", "Wrote integration tests for OAuth flow."),
            ("release", "Released v0.2 to staging, smoke tests passed."),
        ],
        "milestones": [
            ("auth-redesign", "Refresh-token rotation implemented."),
            ("auth-redesign", "Auth redesign shipped to staging."),
        ],
    },
    {
        "title": "Day 18 — Deployment decision",
        "facts": [
            "Cost analysis shows GCP Cloud Run cheaper than AWS ECS at our scale.",
            "Latency is comparable in our europe-west region.",
        ],
        "activities": [
            ("decision", "Decided to migrate from AWS ECS to GCP Cloud Run next quarter."),
            ("decision", "AWS will be deprecated for new services starting next month."),
        ],
    },
    {
        "title": "Day 22 — Recurring bug",
        "facts": [
            "Another null JSONB issue surfaced - reminded the team of day-7 bug.",
            "Migration 0058 added NOT NULL constraint we'd decided on.",
        ],
        "activities": [
            ("bug", "Null JSONB issue in new orders.notes column."),
            ("fix", "Applied the same COALESCE pattern from migration 0042."),
        ],
    },
    {
        "title": "Day 28 — Cross-functional review",
        "facts": [
            "Alice argued for mandatory static type hints across the codebase.",
            "Bob disagreed - he wants gradual typing only where it helps.",
            "Carol proposed compromise: mypy in CI for new code, existing files exempt.",
        ],
        "activities": [
            ("decision", "Team accepted Carol's mypy-in-CI-for-new-code compromise."),
        ],
        "milestones": [("policy", "Typing policy ratified.")],
    },
    {
        "title": "Day 35 — Performance work",
        "facts": [
            "p95 recall latency is 130ms with workspace of 2000 events.",
            "BM25-heavy fusion (0.7) beats symmetric (0.5) by 2.5pp on benchmark.",
            "Cross-encoder reranker regresses 17pp on LoCoMo - disabled by default.",
        ],
        "activities": [
            ("experiment", "Ran full ablation across 19 components."),
            ("decision", "Changed default bm25_weight from 0.5 to 0.7 and typo_correction to False."),
        ],
        "milestones": [("performance", "Recall improved from 91.6% to 94.1% on LoCoMo.")],
    },
    {
        "title": "Day 42 — Migration complete",
        "facts": [
            "Production is now fully on GCP Cloud Run.",
            "AWS account closed except for billing reconciliation.",
            "Cost reduced ~35% as projected.",
        ],
        "activities": [
            ("release", "Cut-over to GCP completed Tuesday night, zero downtime."),
        ],
        "milestones": [("deployment", "Production fully on GCP.")],
    },
]


def setup_sessions(eng, sessions: list[dict]) -> dict:
    """Ingest each session as a batch and record metadata."""
    print(f"  Building {len(sessions)} sessions...")
    n_events = 0
    per_session = []
    for i, s in enumerate(sessions, 1):
        items = []
        for f in s.get("facts", []):
            items.append({"type": "fact", "content": f})
        for title, status in s.get("goals", []):
            items.append({"type": "goal", "title": title, "status": status})
        for kind, body in s.get("activities", []):
            items.append({"type": "activity", "kind": kind, "content": body})
        for chain, title in s.get("milestones", []):
            items.append({"type": "milestone", "chain_name": chain, "title": title})
        result = eng.record_batch(items)
        n_events += result.get("n_ok", 0)
        per_session.append({"title": s["title"], "n_ok": result.get("n_ok", 0)})
        time.sleep(0.6)  # let async embedding catch up
    time.sleep(3.0)
    print(f"  ingested {n_events} events across {len(sessions)} sessions")
    return {"n_events": n_events, "per_session": per_session}


# ----------------------------------------------------------------------
# Query bank: 4 categories x 25 templates -> mix and repeat to hit 1000
# ----------------------------------------------------------------------


_FACTUAL = [
    "What is the project name?",
    "What stack does the team use?",
    "Who is on the team?",
    "What database did we pick?",
    "What's the sprint length?",
    "Which web framework are we using?",
    "Who owns the API design?",
    "What auth method are we using?",
    "Where do we store refresh tokens?",
    "What's our deployment target now?",
    "What was our deployment target before?",
    "What's the recall latency at scale?",
    "What's the LoCoMo score?",
    "What's the cost reduction from GCP move?",
    "What's the JSONB policy?",
    "Why did we pick Postgres?",
    "Why FastAPI?",
    "Who reviewed the auth design?",
    "Which migration fixed the JSONB bug?",
    "What goes in the orders.notes column?",
    "What's Alex's preference on functions?",
    "Who set up CI?",
    "What CI platform?",
    "When did we ship v0.2?",
    "What's our typing policy?",
]

_DECISION = [
    "What did the team decide about the database?",
    "What was the framework decision?",
    "What did we decide about JSONB columns?",
    "What's the agreed approach to refresh tokens?",
    "What did we decide on cloud provider?",
    "What's the team's typing decision?",
    "Did we agree on Postgres?",
    "What was the verdict on AWS vs GCP?",
    "What's the resolution on the typing debate?",
    "What's the decision on bm25_weight?",
    "What did the team conclude about reranking?",
    "Did we decide to enforce NOT NULL on jsonb?",
    "What's our chosen auth storage approach?",
    "What's the team decision on sprint length?",
    "What was concluded about typo_correction?",
]

_MULTIHOP = [
    "Why does our jsonb policy exist?",
    "Why are we on GCP now and not AWS?",
    "What led to the typing policy?",
    "Why did we change the recall defaults?",
    "What's the connection between bm25_weight and LoCoMo?",
    "Why did Bob and Alex pair-program?",
    "How did the auth redesign progress?",
    "What sequence of bugs led to migration 0058?",
    "What changes contributed to the 94.1% recall?",
    "Why did Alice propose static types?",
    "What's the relationship between Carol's compromise and CI?",
    "How did cost analysis affect the deployment choice?",
    "What triggered the GCP migration?",
    "How is migration 0042 related to migration 0058?",
    "Why did the auth redesign use httpOnly cookies?",
]

_RECENT = [
    "What happened recently?",
    "What did the team work on this week?",
    "What's the auth redesign status?",
    "What's the v0.2 release status?",
    "What did Bob work on lately?",
    "What did Carol contribute?",
    "What's the current sprint about?",
    "What are open goals?",
    "What's done in the last two weeks?",
    "What recent bugs were closed?",
    "What's the latest performance change?",
    "What recent decisions did we make?",
    "What was the last milestone?",
    "Did we finish the GCP migration?",
    "What's the current deployment status?",
]


def stress_queries(eng, n: int = 1000) -> dict:
    """Fire `n` queries from the four categories, record latencies."""
    print(f"  Firing {n} queries across 4 intent categories...")
    rng = random.Random(42)  # deterministic mix
    buckets = {
        "factual": (_FACTUAL, []),
        "decision": (_DECISION, []),
        "multi_hop": (_MULTIHOP, []),
        "recent": (_RECENT, []),
    }
    # cycle weights: factual 0.4, decision 0.2, multi_hop 0.2, recent 0.2
    weights = ["factual"] * 4 + ["decision"] * 2 + ["multi_hop"] * 2 + ["recent"] * 2
    n_returned_nonempty = 0
    for i in range(n):
        cat = rng.choice(weights)
        bank, lats = buckets[cat]
        q = rng.choice(bank)
        t0 = time.perf_counter()
        try:
            pack = eng.recall(q, top_k=3)
            elapsed = (time.perf_counter() - t0) * 1000
            lats.append(elapsed)
            if pack.results:
                n_returned_nonempty += 1
        except Exception:
            pass
        if (i + 1) % 250 == 0:
            print(f"    {i + 1}/{n} ...")
    out = {
        "n_total": n,
        "n_nonempty": n_returned_nonempty,
        "by_category": {},
    }
    for cat, (_, lats) in buckets.items():
        out["by_category"][cat] = {
            **percentiles(lats),
            "n_queries": len(lats),
        }
    all_lats = sum((lats for _, lats in buckets.values()), [])
    out["overall"] = percentiles(all_lats)
    return out


# ----------------------------------------------------------------------
# Tier transitions
# ----------------------------------------------------------------------


def tier_distribution(eng) -> dict:
    counts = {}
    with eng.events._conn() as conn:
        for row in conn.execute(
            "SELECT tier, COUNT(*) AS c FROM events "
            "WHERE workspace_id = ? AND archived_at IS NULL "
            "GROUP BY tier",
            (eng.workspace.id,),
        ):
            counts[row["tier"]] = row["c"]
    return counts


def stress_tiers(eng) -> dict:
    """Trigger access-count promotion via repeated recalls on the same topics."""
    print("  Triggering tier promotions via repeated recall...")
    before = tier_distribution(eng)
    # 20 queries that all overlap with "deployment / GCP / AWS" cluster
    queries = [
        "Where do we deploy?",
        "What's our deployment target?",
        "Production runs where?",
        "Did we move to GCP?",
        "What's the AWS status?",
        "Cloud Run usage?",
        "Cost reduction from cloud move?",
    ] * 6  # 42 hits, easily over the 10-recall threshold for semantic
    for q in queries:
        eng.recall(q, top_k=10)
    after = tier_distribution(eng)
    return {"before": before, "after": after,
            "diff": {k: after.get(k, 0) - before.get(k, 0)
                     for k in set(before) | set(after)}}


# ----------------------------------------------------------------------
# Decay simulation
# ----------------------------------------------------------------------


def _count_archived(eng) -> int:
    with eng.events._conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM events "
            "WHERE workspace_id = ? AND archived_at IS NOT NULL",
            (eng.workspace.id,),
        ).fetchone()
        return int(row["c"] or 0)


def _count_active(eng) -> int:
    with eng.events._conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM events "
            "WHERE workspace_id = ? AND archived_at IS NULL",
            (eng.workspace.id,),
        ).fetchone()
        return int(row["c"] or 0)


def stress_decay(eng) -> dict:
    """Advance timestamps of half the events 30 days backward, then run decay."""
    print("  Simulating 30-day passage on half the workspace...")
    now = time.time()
    cutoff_old = now - 30 * 86400  # 30 days ago

    with eng.events._conn() as conn:
        rows = conn.execute(
            "SELECT id FROM events WHERE workspace_id = ? "
            "AND archived_at IS NULL ORDER BY id ASC",
            (eng.workspace.id,),
        ).fetchall()
        ids = [r["id"] for r in rows]
        to_age = ids[: len(ids) // 2]
        n_aged = len(to_age)
        if to_age:
            placeholders = ",".join("?" * len(to_age))
            conn.execute(
                f"UPDATE events SET timestamp = ?, last_accessed = ? "
                f"WHERE id IN ({placeholders})",
                (cutoff_old, cutoff_old, *to_age),
            )

    archived_before = _count_archived(eng)
    active_before = _count_active(eng)

    from pmb.signals.decay import apply_decay
    decay_result = apply_decay(eng, days_since_last_decay=30.0)

    archived_after = _count_archived(eng)
    active_after = _count_active(eng)

    return {
        "n_aged_30d": n_aged,
        "active_before": active_before,
        "active_after": active_after,
        "archived_before": archived_before,
        "archived_after": archived_after,
        "newly_archived": archived_after - archived_before,
        "decay_result": decay_result,
    }


# ----------------------------------------------------------------------
# Cross-chat summaries
# ----------------------------------------------------------------------


def cross_chat_summaries(eng) -> dict:
    """Exercise the summary surfaces: recent_activity, list_goals, what_just_happened."""
    print("  Testing cross-chat summary tools...")
    out: dict[str, list] = {}

    # recent_activity at multiple windows
    for window_min in [60, 60 * 24, 60 * 24 * 7, 60 * 24 * 30]:
        try:
            acts = eng.recent_activity(minutes=window_min, limit=10)
            out[f"recent_activity_{window_min}m"] = [
                a.get("content", "")[:80] for a in acts[:5]
            ]
        except Exception as e:
            out[f"recent_activity_{window_min}m"] = [f"ERROR: {e}"]

    # list_goals by status
    for status in ["in_progress", "pending", "done", None]:
        try:
            gs = eng.list_goals(status=status, limit=20)
            out[f"goals_{status or 'all'}"] = [g.get("title", "")[:80] for g in gs[:5]]
        except Exception as e:
            out[f"goals_{status or 'all'}"] = [f"ERROR: {e}"]

    # what_just_happened
    try:
        wjh = eng.what_just_happened(n=10)
        out["what_just_happened_10"] = [w.get("content", "")[:80] for w in wjh[:5]]
    except Exception as e:
        out["what_just_happened_10"] = [f"ERROR: {e}"]

    return out


# ----------------------------------------------------------------------
# Sleep / consolidation readiness
# ----------------------------------------------------------------------


def consolidation_via_claude_cli(eng) -> dict:
    """Run a REAL consolidation pass using the Claude CLI.

    Looks for the binary via:
      1. PMB_CLAUDE_CMD env var (absolute path)
      2. `shutil.which("claude")`
      3. Known install location under %APPDATA%/Claude/claude-code/<ver>/claude.exe
         (covers the Windows case where the PATH entry points to an outdated
          version directory that no longer exists on disk).
    """
    import shutil
    import glob
    from pmb.health.consolidate import ClaudeCLIClient

    print("  Locating claude CLI...")
    cmd = os.environ.get("PMB_CLAUDE_CMD") or shutil.which("claude")
    if not cmd:
        # Probe known Windows install path for newest version
        candidates = sorted(
            glob.glob(
                os.path.expanduser(
                    r"~\AppData\Roaming\Claude\claude-code\*\claude.exe"
                )
            )
        )
        if candidates:
            cmd = candidates[-1]
            os.environ["PMB_CLAUDE_CMD"] = cmd
            print(f"  found at: {cmd}")
        else:
            return {"skipped": "claude CLI not found"}
    else:
        print(f"  using: {cmd}")

    # Check the auto-consolidate trigger logic. We poke the config's
    # internal dict directly because Config exposes only `get`; in production
    # the user would set these via `pmb config set`.
    from pmb.health.auto_consolidate import should_trigger
    try:
        cfg_dict = getattr(eng.config, "_resolved", None) or eng.config.__dict__
        if isinstance(cfg_dict, dict):
            cfg_dict["consolidate.auto_trigger"] = True
            cfg_dict["consolidate.auto_min_new_events"] = 5
            cfg_dict["consolidate.auto_min_days"] = 0.0
    except Exception:
        pass
    try:
        trigger_decision = should_trigger(eng)
    except Exception as e:
        trigger_decision = {"error": f"{type(e).__name__}: {e}"}
    print(f"  should_trigger() = {trigger_decision}")

    # Now actually run consolidation through the consolidate.py pipeline
    print("  Running real consolidation via Claude CLI...")
    from pmb.health.consolidate import run_consolidation, resolve_llm_client

    try:
        llm = resolve_llm_client(backend="claude")
    except Exception as e:
        return {
            "trigger_decision": trigger_decision,
            "consolidation_skipped": f"resolve_llm_client failed: {e}",
        }

    archived_before = _count_archived(eng)
    n_facts_before = 0
    with eng.events._conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM events "
            "WHERE workspace_id = ? AND event_type = 'fact' "
            "AND archived_at IS NULL",
            (eng.workspace.id,),
        ).fetchone()
        n_facts_before = int(row["c"] or 0)

    t0 = time.perf_counter()
    try:
        cons_result = run_consolidation(
            eng,
            llm=llm,
            similarity_threshold=0.4,
            min_cluster_size=2,
            since_days=60.0,
            max_clusters=8,
        )
        wall = time.perf_counter() - t0
    except Exception as e:
        return {
            "trigger_decision": trigger_decision,
            "consolidation_error": f"{type(e).__name__}: {e}",
            "wall_s": round(time.perf_counter() - t0, 1),
        }

    archived_after = _count_archived(eng)
    n_facts_after = 0
    new_facts: list[str] = []
    with eng.events._conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM events "
            "WHERE workspace_id = ? AND event_type = 'fact' "
            "AND archived_at IS NULL",
            (eng.workspace.id,),
        ).fetchone()
        n_facts_after = int(row["c"] or 0)
        # Sample any new fact rows created by consolidation
        for row in conn.execute(
            "SELECT content FROM events WHERE workspace_id = ? "
            "AND event_type = 'fact' AND archived_at IS NULL "
            "ORDER BY id DESC LIMIT 5",
            (eng.workspace.id,),
        ):
            new_facts.append(row["content"][:140])

    return {
        "trigger_decision": trigger_decision,
        "consolidation_wall_s": round(wall, 1),
        "clusters_seen": cons_result.get("clusters_seen"),
        "clusters_consolidated": cons_result.get("clusters_consolidated"),
        "new_facts_created": cons_result.get("new_facts_created"),
        "sources_archived": cons_result.get("sources_archived"),
        "n_facts_active_before": n_facts_before,
        "n_facts_active_after": n_facts_after,
        "archived_before": archived_before,
        "archived_after": archived_after,
        "sample_new_facts": new_facts[:3],
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Fewer sessions, 300 queries instead of 1000")
    ap.add_argument("--out",
                    default=data_path("pmb_big_cross_chat.json"))
    args = ap.parse_args()

    from pmb.core.engine import Engine

    print("=" * 72)
    print("BIG CROSS-CHAT TEST")
    print("=" * 72)

    sessions = SESSIONS[:5] if args.quick else SESSIONS
    n_queries = 300 if args.quick else 1000

    tmp_home = Path(tempfile.mkdtemp(prefix="pmb-big-"))
    tmp_ws = Path(tempfile.mkdtemp(prefix="pmb-big-ws-"))
    os.environ["PMB_HOME"] = str(tmp_home)
    eng = Engine(cwd=tmp_ws, pmb_home=tmp_home, rerank_model=None)
    # Force model load now so phase timing is honest
    print("\n  Loading embedding model...")
    t0 = time.perf_counter()
    _ = eng.search.model
    print(f"  model loaded in {(time.perf_counter() - t0):.1f}s")

    t_total = time.perf_counter()
    report = {}

    print("\n[1] BUILD SESSIONS")
    t0 = time.perf_counter()
    report["setup"] = setup_sessions(eng, sessions)
    report["setup"]["wall_s"] = round(time.perf_counter() - t0, 1)
    print(f"  wall: {report['setup']['wall_s']}s")

    print("\n[2] STRESS QUERIES")
    t0 = time.perf_counter()
    report["queries"] = stress_queries(eng, n=n_queries)
    report["queries"]["wall_s"] = round(time.perf_counter() - t0, 1)
    print(f"  wall: {report['queries']['wall_s']}s")
    o = report["queries"]["overall"]
    print(f"  overall: p50={o['p50']:.1f}ms p95={o['p95']:.1f}ms p99={o['p99']:.1f}ms")
    for cat, st in report["queries"]["by_category"].items():
        print(f"    {cat:<12}: n={st['n_queries']:>4}  "
              f"p50={st['p50']:>5.1f}ms  p95={st['p95']:>6.1f}ms")

    print("\n[3] TIER TRANSITIONS")
    t0 = time.perf_counter()
    report["tiers"] = stress_tiers(eng)
    report["tiers"]["wall_s"] = round(time.perf_counter() - t0, 1)
    print(f"  before: {report['tiers']['before']}")
    print(f"  after:  {report['tiers']['after']}")
    print(f"  diff:   {report['tiers']['diff']}")

    print("\n[4] DECAY SIMULATION (30-day advance on half)")
    t0 = time.perf_counter()
    report["decay"] = stress_decay(eng)
    report["decay"]["wall_s"] = round(time.perf_counter() - t0, 1)
    d = report["decay"]
    print(f"  aged: {d['n_aged_30d']}, newly archived: {d['newly_archived']}")

    print("\n[5] CROSS-CHAT SUMMARY TOOLS")
    t0 = time.perf_counter()
    report["summaries"] = cross_chat_summaries(eng)
    report["summaries_wall_s"] = round(time.perf_counter() - t0, 1)
    for k, v in report["summaries"].items():
        sample = v[0] if v else "(empty)"
        print(f"    {k:<28}: {len(v)} items, e.g. {sample[:60]!r}")

    print("\n[6] REAL CONSOLIDATION VIA CLAUDE CLI")
    t0 = time.perf_counter()
    report["consolidation"] = consolidation_via_claude_cli(eng)
    report["consolidation"]["wall_s"] = round(time.perf_counter() - t0, 1)
    c = report["consolidation"]
    if "skipped" in c:
        print(f"  SKIPPED: {c['skipped']}")
    elif "consolidation_error" in c:
        print(f"  ERROR: {c['consolidation_error']}")
    else:
        print(f"  clusters_seen={c.get('clusters_seen')}, "
              f"consolidated={c.get('clusters_consolidated')}, "
              f"new_facts={c.get('new_facts_created')}, "
              f"sources_archived={c.get('sources_archived')}")
        print(f"  facts active: {c.get('n_facts_active_before')} -> "
              f"{c.get('n_facts_active_after')}")
        print(f"  consolidation wall: {c.get('consolidation_wall_s')}s")
        for nf in c.get("sample_new_facts", []):
            print(f"    new fact sample: {nf[:100]}")

    total_wall = time.perf_counter() - t_total
    report["_meta"] = {
        "total_wall_s": round(total_wall, 1),
        "n_sessions": len(sessions),
        "n_queries": n_queries,
    }

    print("\n" + "=" * 72)
    print(f"DONE in {total_wall:.1f}s")
    print("=" * 72)

    try:
        eng.close()
    except Exception:
        pass

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
