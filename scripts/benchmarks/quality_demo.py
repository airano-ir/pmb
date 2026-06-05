"""
Quality demo: 200 messages across 10 chats, then look at what PMB actually
remembered and how useful the answers are.

Output is verbose by design - we print full content of what got recorded,
full text of what gets recalled, and side-by-side comparisons. Speed is
not the point here (already measured elsewhere); we want to read the
output and judge whether the system is genuinely useful.

Sections:
  1. Build 10 chat sessions with 20 messages each (200 total).
     Each "message" is a realistic user statement + agent record_batch.
  2. Per-session inventory: what got recorded in each chat.
  3. 30 cross-session queries with FULL top-3 answer text.
  4. Summary tools - recent_activity / list_goals / what_just_happened.
     Print the actual content returned.
  5. Tier composition + sample events from each tier.
  6. Decay simulation: aged half the workspace 30 days, show what was
     archived (with content).
  7. Claude CLI consolidation: cluster events, ask LLM for generalisations,
     print the actual new facts the LLM produced.
  8. Quality judgment: did the answers actually help?

Run:
    python scripts/benchmarks/quality_demo.py
"""

from __future__ import annotations

import argparse
import json
import os
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


# ----------------------------------------------------------------------
# 10 chat sessions, 20 messages each, mix of types.
# Each item becomes one record in PMB.
# ----------------------------------------------------------------------

CHATS = [
    # ---------------------------------------- Chat 1: onboarding
    {
        "title": "Chat 1 — onboarding (Mon morning)",
        "items": [
            {"type": "fact", "content": "User's name is Alex, senior backend engineer.", "pin": True, "importance": 0.95},
            {"type": "fact", "content": "Alex joined acme-corp two weeks ago."},
            {"type": "fact", "content": "Alex previously worked at startup-xyz on payments."},
            {"type": "fact", "content": "Alex's primary language is Python 3.12."},
            {"type": "fact", "content": "Alex also writes Rust for low-level code."},
            {"type": "fact", "content": "Alex is allergic to Java verbosity."},
            {"type": "fact", "content": "Alex prefers small functions to large classes."},
            {"type": "fact", "content": "Alex's editor of choice is Helix, switched from Neovim recently."},
            {"type": "fact", "content": "Alex's terminal is Wezterm with the Catppuccin theme."},
            {"type": "fact", "content": "Alex lives in Kyiv and works mostly remote."},
            {"type": "goal", "title": "Ship payment processor v1 by end of Q2", "status": "in_progress", "importance": 0.85},
            {"type": "goal", "title": "Achieve 99.9% uptime SLA", "status": "in_progress"},
            {"type": "activity", "kind": "onboarding", "content": "Read the architecture documents."},
            {"type": "activity", "kind": "onboarding", "content": "Set up local dev environment with Docker compose."},
            {"type": "activity", "kind": "onboarding", "content": "Cloned acme/orders, acme/payments, acme/notifications repos."},
            {"type": "fact", "content": "Team consists of Alex (lead), Bob (backend), Carol (API design), Dana (infra)."},
            {"type": "fact", "content": "Sprint length is two weeks; standup at 10am UTC."},
            {"type": "fact", "content": "PR review SLA is 24 hours."},
            {"type": "fact", "content": "Code coverage target is 80% minimum for new modules."},
            {"type": "activity", "kind": "decision", "content": "Decided to focus first sprint on observability."},
        ],
    },
    # ---------------------------------------- Chat 2: stack picks
    {
        "title": "Chat 2 — stack decisions (Mon afternoon)",
        "items": [
            {"type": "fact", "content": "After analysis the team picked Postgres 15 over MongoDB."},
            {"type": "fact", "content": "JSONB columns in Postgres support GIN indexes for nested-field queries."},
            {"type": "activity", "kind": "decision", "content": "Decided Postgres 15 over MongoDB for JSONB support and ACID guarantees."},
            {"type": "activity", "kind": "decision", "content": "Decided FastAPI over Flask for built-in async."},
            {"type": "activity", "kind": "decision", "content": "Decided pytest over unittest for the test runner."},
            {"type": "fact", "content": "Redis is used for session store and rate limiting."},
            {"type": "fact", "content": "Background jobs run on Celery with Redis broker."},
            {"type": "fact", "content": "Observability: OpenTelemetry SDK in code, Honeycomb as backend."},
            {"type": "activity", "kind": "decision", "content": "Decided OpenTelemetry over Datadog APM for vendor neutrality."},
            {"type": "fact", "content": "We deploy with Docker images, multi-stage builds for minimal size."},
            {"type": "fact", "content": "CI runs on GitHub Actions, deployment via Cloud Build to GCP Cloud Run."},
            {"type": "activity", "kind": "decision", "content": "Decided GCP Cloud Run over AWS ECS for cost reasons."},
            {"type": "fact", "content": "Cost analysis: GCP is ~35% cheaper at our scale."},
            {"type": "fact", "content": "Latency is comparable in europe-west1 region."},
            {"type": "milestone", "chain_name": "stack-selection", "title": "All major stack decisions made."},
            {"type": "fact", "content": "Bob owns Postgres and Redis infrastructure."},
            {"type": "fact", "content": "Dana owns deployment and observability."},
            {"type": "fact", "content": "Carol owns API design and OpenAPI spec."},
            {"type": "fact", "content": "Alex owns business logic and integrations."},
            {"type": "activity", "kind": "fact", "content": "Documented stack decisions in ARCHITECTURE.md."},
        ],
    },
    # ---------------------------------------- Chat 3: first feature
    {
        "title": "Chat 3 — first feature (Tue)",
        "items": [
            {"type": "fact", "content": "First feature: POST /orders endpoint to create new orders."},
            {"type": "activity", "kind": "feature", "content": "Implemented POST /orders with request validation via Pydantic."},
            {"type": "activity", "kind": "feature", "content": "Added order status enum: pending, paid, shipped, delivered, cancelled."},
            {"type": "fact", "content": "Order has metadata JSONB column for flexible attributes."},
            {"type": "activity", "kind": "test", "content": "Wrote 12 unit tests covering happy path and edge cases."},
            {"type": "activity", "kind": "review", "content": "Code review with Carol on PR #12 - API schema."},
            {"type": "activity", "kind": "fix", "content": "Fixed validation error response format per Carol's feedback."},
            {"type": "activity", "kind": "review", "content": "Carol approved PR #12, merged to main."},
            {"type": "milestone", "chain_name": "orders-api", "title": "POST /orders endpoint live in staging."},
            {"type": "fact", "content": "Orders endpoint handles 50 RPS in load test with p95 25ms."},
            {"type": "fact", "content": "Bob added a database migration 0001_create_orders for the schema."},
            {"type": "activity", "kind": "decision", "content": "Decided to use UUID v7 for order IDs (sortable + opaque)."},
            {"type": "fact", "content": "UUIDv7 gives us time-ordered IDs without exposing wall-clock."},
            {"type": "activity", "kind": "experiment", "content": "Benchmarked uuid7 generation: ~0.5μs per call."},
            {"type": "activity", "kind": "feature", "content": "Added GET /orders/{id} endpoint to retrieve a single order."},
            {"type": "activity", "kind": "feature", "content": "Added GET /orders with cursor pagination."},
            {"type": "fact", "content": "Cursor pagination uses opaque base64-encoded UUIDv7 of last item."},
            {"type": "activity", "kind": "release", "content": "Released v0.1.0 of orders-service to staging environment."},
            {"type": "milestone", "chain_name": "release", "title": "v0.1.0 shipped to staging."},
            {"type": "fact", "content": "Smoke tests on staging all green."},
        ],
    },
    # ---------------------------------------- Chat 4: bug investigation
    {
        "title": "Chat 4 — JSONB null bug (Wed)",
        "items": [
            {"type": "activity", "kind": "bug", "content": "Bug found: orders-service crashes when JSONB metadata is null."},
            {"type": "activity", "kind": "bug", "content": "Stack trace shows KeyError when reading order.metadata.tags."},
            {"type": "fact", "content": "Root cause: Postgres returns NULL not '{}' when JSONB column is unset."},
            {"type": "fact", "content": "Our handler called dict.get('tags', []) but NULL bypassed that."},
            {"type": "activity", "kind": "experiment", "content": "Reproduced bug on staging at 14:32 UTC by sending request without metadata."},
            {"type": "activity", "kind": "fix", "content": "Fix: added COALESCE(metadata, '{}'::jsonb) to migration 0042."},
            {"type": "activity", "kind": "fix", "content": "Backfilled NULL metadata to '{}' in production via SET metadata='{}'::jsonb WHERE metadata IS NULL."},
            {"type": "activity", "kind": "test", "content": "Added regression test for NULL JSONB in tests/test_orders.py::test_null_metadata."},
            {"type": "activity", "kind": "decision", "content": "Decided to enforce NOT NULL on all JSONB columns going forward."},
            {"type": "fact", "content": "Migration 0058 (planned) will add NOT NULL constraint on metadata."},
            {"type": "activity", "kind": "review", "content": "Bob reviewed the COALESCE fix and approved."},
            {"type": "milestone", "chain_name": "bugfix-jsonb", "title": "JSONB null bug fixed in v0.1.3."},
            {"type": "fact", "content": "Hotfix v0.1.3 deployed to production at 16:00 UTC."},
            {"type": "activity", "kind": "incident", "content": "Customer impact: ~12 failed requests over 90 minutes."},
            {"type": "activity", "kind": "incident", "content": "Posted incident review in #engineering channel."},
            {"type": "fact", "content": "Postmortem: missing schema validation at write time is the deeper issue."},
            {"type": "activity", "kind": "decision", "content": "Decided to add Pydantic validation BEFORE database insert as a layer of defense."},
            {"type": "activity", "kind": "feature", "content": "Added OrderMetadata pydantic model with explicit fields."},
            {"type": "milestone", "chain_name": "policy", "title": "Schema validation policy ratified."},
            {"type": "fact", "content": "All new JSONB columns must have a Pydantic schema in code."},
        ],
    },
    # ---------------------------------------- Chat 5: auth redesign
    {
        "title": "Chat 5 — auth redesign begins (Thu)",
        "items": [
            {"type": "goal", "title": "Redesign auth flow this sprint", "status": "in_progress", "importance": 0.8},
            {"type": "fact", "content": "Current auth uses HTTP basic auth - good for service-to-service, bad for users."},
            {"type": "fact", "content": "Alice from the security team reviewed our auth design."},
            {"type": "activity", "kind": "decision", "content": "Decided to migrate to JWT-based auth with refresh tokens."},
            {"type": "fact", "content": "JWT access tokens valid 15 minutes; refresh tokens valid 7 days."},
            {"type": "activity", "kind": "decision", "content": "Decided to store refresh tokens in httpOnly cookies, not localStorage."},
            {"type": "fact", "content": "httpOnly prevents XSS theft; SameSite=strict prevents CSRF."},
            {"type": "activity", "kind": "feature", "content": "Designed token rotation scheme: each refresh creates a new refresh token."},
            {"type": "fact", "content": "Token rotation invalidates the previous refresh token on each use."},
            {"type": "activity", "kind": "feature", "content": "Implemented JWT signing with RS256, keys in Secret Manager."},
            {"type": "activity", "kind": "pairing", "content": "Pair-programmed with Bob on JWT verification edge cases."},
            {"type": "activity", "kind": "fix", "content": "Fixed clock-skew tolerance: allow 60s skew on token nbf/exp checks."},
            {"type": "milestone", "chain_name": "auth-redesign", "title": "Spec written and reviewed by security."},
            {"type": "milestone", "chain_name": "auth-redesign", "title": "API endpoints scaffolded."},
            {"type": "activity", "kind": "test", "content": "Wrote integration tests for the OAuth flow including refresh-token rotation."},
            {"type": "fact", "content": "Tests cover: login, refresh, logout, token reuse detection, clock skew."},
            {"type": "activity", "kind": "decision", "content": "Decided to revoke refresh tokens on password change."},
            {"type": "fact", "content": "Revoked tokens are stored in Redis with TTL matching token lifetime."},
            {"type": "milestone", "chain_name": "auth-redesign", "title": "Refresh-token rotation implemented."},
            {"type": "milestone", "chain_name": "auth-redesign", "title": "Integration tests green on staging."},
        ],
    },
    # ---------------------------------------- Chat 6: performance work
    {
        "title": "Chat 6 — performance investigation (Fri)",
        "items": [
            {"type": "fact", "content": "p95 latency for GET /orders/{id} crept up to 250ms in production."},
            {"type": "activity", "kind": "bug", "content": "Performance regression: p95 went from 25ms to 250ms over a week."},
            {"type": "activity", "kind": "experiment", "content": "Profiled the endpoint with py-spy under load."},
            {"type": "fact", "content": "Profile showed 80% of time in SQLAlchemy ORM session setup."},
            {"type": "activity", "kind": "fix", "content": "Fix: switched to raw asyncpg for hot-path read queries."},
            {"type": "fact", "content": "Raw asyncpg path uses prepared statements cached per connection."},
            {"type": "activity", "kind": "experiment", "content": "Load test after fix: p95 back to 22ms."},
            {"type": "milestone", "chain_name": "performance", "title": "Read-path latency back to baseline."},
            {"type": "fact", "content": "p95 latency budget for any single endpoint is 50ms."},
            {"type": "activity", "kind": "decision", "content": "Decided to use raw asyncpg for any read endpoint expected over 10 RPS."},
            {"type": "activity", "kind": "decision", "content": "Decided to keep SQLAlchemy for writes and complex transactions."},
            {"type": "fact", "content": "Trade-off: less ergonomic for reads, much faster."},
            {"type": "activity", "kind": "feature", "content": "Added connection pool sizing config (default 20)."},
            {"type": "fact", "content": "Connection pool size = 2 * num_cores + 5 per industry guidance."},
            {"type": "activity", "kind": "experiment", "content": "Benchmarked pool sizes 10/20/50: 20 is the sweet spot."},
            {"type": "fact", "content": "Pool too small = queueing; pool too large = pg connection exhaustion."},
            {"type": "milestone", "chain_name": "performance", "title": "Pool sizing tuned."},
            {"type": "activity", "kind": "decision", "content": "Decided to add Prometheus metric for pool utilisation."},
            {"type": "fact", "content": "Pool-saturation alert fires at 80% utilisation for 5 minutes."},
            {"type": "milestone", "chain_name": "performance", "title": "Pool observability ready."},
        ],
    },
    # ---------------------------------------- Chat 7: team debate
    {
        "title": "Chat 7 — typing policy debate (Mon week 2)",
        "items": [
            {"type": "fact", "content": "Discussion: should we mandate static type hints across the codebase?"},
            {"type": "fact", "content": "Alice argued strongly for mandatory static type hints everywhere."},
            {"type": "fact", "content": "Alice cited reduced bug rate at her previous company after adopting mypy."},
            {"type": "fact", "content": "Bob disagreed and pushed for gradual typing - hints only where they help."},
            {"type": "fact", "content": "Bob worried about churn and migration cost on the existing codebase."},
            {"type": "fact", "content": "Carol proposed compromise: enforce mypy in CI for new code only, grandfather existing files."},
            {"type": "fact", "content": "Carol's reasoning: typing pays off on new code, existing code is risky to touch."},
            {"type": "fact", "content": "Dana suggested also enforcing types on public API boundaries even in old files."},
            {"type": "activity", "kind": "decision", "content": "Team accepted Carol's mypy-in-CI-for-new-code proposal."},
            {"type": "activity", "kind": "decision", "content": "Public API boundaries get types regardless of file age (Dana's add-on)."},
            {"type": "activity", "kind": "fact", "content": "mypy --strict added to CI for files under src/orders/api/."},
            {"type": "fact", "content": "Old files exempted via mypy.cfg [ignore-paths]."},
            {"type": "milestone", "chain_name": "policy", "title": "Typing policy ratified."},
            {"type": "activity", "kind": "feature", "content": "Updated CONTRIBUTING.md with the new typing rules."},
            {"type": "fact", "content": "Decided to revisit the policy after one month."},
            {"type": "goal", "title": "Add type hints to all public API modules", "status": "in_progress"},
            {"type": "activity", "kind": "feature", "content": "Added types to src/orders/api/v1.py - PR #45 merged."},
            {"type": "activity", "kind": "feature", "content": "Added types to src/orders/api/v2.py - PR #46 merged."},
            {"type": "fact", "content": "mypy CI run now takes 25 seconds, acceptable."},
            {"type": "fact", "content": "Found 3 real bugs while adding types - off-by-one in pagination, wrong dict key, missing None check."},
        ],
    },
    # ---------------------------------------- Chat 8: deployment switch
    {
        "title": "Chat 8 — GCP migration plan (Wed week 2)",
        "items": [
            {"type": "fact", "content": "AWS ECS cost analysis: spending ~$8k/month at current scale."},
            {"type": "fact", "content": "GCP Cloud Run cost projection: ~$5k/month, same workload."},
            {"type": "activity", "kind": "decision", "content": "Decided to migrate production from AWS ECS to GCP Cloud Run next quarter."},
            {"type": "activity", "kind": "decision", "content": "AWS will be deprecated for new services starting next month."},
            {"type": "activity", "kind": "feature", "content": "Created GCP project acme-prod and acme-staging."},
            {"type": "activity", "kind": "feature", "content": "Configured workload identity federation for CI to deploy to GCP."},
            {"type": "fact", "content": "Cloud Run gives us automatic HTTPS, auto-scaling, and no idle cost."},
            {"type": "fact", "content": "Cloud Run cold start is ~800ms - acceptable for our traffic shape."},
            {"type": "activity", "kind": "experiment", "content": "Migrated orders-service to GCP staging for a soak test."},
            {"type": "fact", "content": "Soak test passed: 72 hours, zero errors, p95 stable at 25ms."},
            {"type": "activity", "kind": "decision", "content": "Decided to use Cloud SQL for Postgres (managed) over self-hosted on GKE."},
            {"type": "activity", "kind": "feature", "content": "Set up Cloud SQL Postgres 15 with HA replicas in europe-west1."},
            {"type": "fact", "content": "Database backups: daily automated snapshots, 7-day retention."},
            {"type": "activity", "kind": "feature", "content": "Migrated staging traffic to GCP-hosted orders-service."},
            {"type": "milestone", "chain_name": "gcp-migration", "title": "Staging fully on GCP."},
            {"type": "activity", "kind": "feature", "content": "Set up Cloud Logging with 30-day retention policy."},
            {"type": "activity", "kind": "feature", "content": "Secrets migrated to Secret Manager from AWS Secrets Manager."},
            {"type": "milestone", "chain_name": "gcp-migration", "title": "Secrets migrated to Secret Manager."},
            {"type": "fact", "content": "Cost reduction visible: GCP staging running at ~$300/month vs AWS $700/month."},
            {"type": "activity", "kind": "decision", "content": "Decided production cut-over scheduled for next Tuesday night, low-traffic window."},
        ],
    },
    # ---------------------------------------- Chat 9: production cut-over
    {
        "title": "Chat 9 — production cut-over (Tue night week 3)",
        "items": [
            {"type": "activity", "kind": "release", "content": "Cut-over started 22:00 UTC, traffic gradually routed to GCP via Cloud Load Balancer."},
            {"type": "activity", "kind": "release", "content": "10% traffic on GCP for 10 minutes - no errors, latency normal."},
            {"type": "activity", "kind": "release", "content": "50% traffic on GCP for 30 minutes - all green."},
            {"type": "activity", "kind": "release", "content": "100% traffic on GCP at 23:15 UTC - zero customer-visible errors."},
            {"type": "milestone", "chain_name": "gcp-migration", "title": "Production cut-over completed, zero downtime."},
            {"type": "fact", "content": "Production is now fully on GCP Cloud Run.", "pin": True, "importance": 0.9},
            {"type": "activity", "kind": "fact", "content": "AWS account scheduled for closure after billing reconciliation."},
            {"type": "fact", "content": "Cost reduction confirmed: first 24h on GCP shows ~38% lower hourly cost."},
            {"type": "activity", "kind": "incident", "content": "Minor incident: 4 requests to old AWS endpoint returned 502 during DNS cutover."},
            {"type": "activity", "kind": "fix", "content": "Updated CNAME TTLs to 60s for future migrations."},
            {"type": "activity", "kind": "fact", "content": "Posted migration retrospective in #engineering channel."},
            {"type": "activity", "kind": "decision", "content": "Decided to schedule cut-overs only on Tuesday/Wednesday nights, never Friday."},
            {"type": "fact", "content": "Lesson learned: pre-warming Cloud Run instances before cut-over avoids first-request latency."},
            {"type": "activity", "kind": "feature", "content": "Added pre-warm step to deployment runbook."},
            {"type": "milestone", "chain_name": "gcp-migration", "title": "Post-migration polish done."},
            {"type": "fact", "content": "AWS will be deprecated for billing reconciliation only - no new workloads."},
            {"type": "fact", "content": "Team is officially GCP-first for any new service."},
            {"type": "activity", "kind": "decision", "content": "Decided to write internal playbook 'GCP from day one' for future services."},
            {"type": "milestone", "chain_name": "gcp-migration", "title": "GCP migration complete."},
            {"type": "fact", "content": "All targets hit: 0 downtime, 38% cost reduction, latency unchanged."},
        ],
    },
    # ---------------------------------------- Chat 10: planning
    {
        "title": "Chat 10 — next-quarter planning (Mon week 4)",
        "items": [
            {"type": "goal", "title": "Build payment processor v2 (Stripe + Adyen)", "status": "pending"},
            {"type": "goal", "title": "Add fraud detection layer", "status": "pending"},
            {"type": "goal", "title": "Achieve SOC 2 readiness", "status": "pending"},
            {"type": "fact", "content": "Q3 OKR: ship payment processor v2 with two backends."},
            {"type": "fact", "content": "Stripe is the primary, Adyen is failover for high-risk regions."},
            {"type": "fact", "content": "Fraud detection will use both rule-based and ML scoring."},
            {"type": "activity", "kind": "decision", "content": "Decided to use Stripe Radar as the first-pass fraud filter."},
            {"type": "activity", "kind": "decision", "content": "Decided to build our own ML model in-house for second-pass scoring."},
            {"type": "fact", "content": "ML model trained on historical chargebacks - need 100K+ labeled samples."},
            {"type": "fact", "content": "Carol will lead the payments API design."},
            {"type": "fact", "content": "Alex will integrate Stripe SDK and write the abstraction layer."},
            {"type": "fact", "content": "Bob will set up Adyen sandbox and integration tests."},
            {"type": "fact", "content": "Dana will handle PCI compliance audits and observability."},
            {"type": "activity", "kind": "decision", "content": "Decided to defer 3DS Secure 2 until Q4 due to scope."},
            {"type": "milestone", "chain_name": "planning", "title": "Q3 roadmap drafted."},
            {"type": "activity", "kind": "feature", "content": "Added payments-v2 milestones to project tracker."},
            {"type": "activity", "kind": "decision", "content": "Decided weekly demo every Friday afternoon."},
            {"type": "fact", "content": "First demo will cover Stripe integration end-to-end."},
            {"type": "milestone", "chain_name": "planning", "title": "Roadmap shared with stakeholders."},
            {"type": "activity", "kind": "decision", "content": "Decided next sprint kick-off Monday morning."},
        ],
    },
]


# ----------------------------------------------------------------------
# 30 cross-session queries (mix of intents)
# ----------------------------------------------------------------------

QUERIES = [
    # Identity / personal facts
    ("Who is the user?", "Identity"),
    ("What languages does Alex use?", "Identity"),
    ("Where does Alex live?", "Identity"),

    # Project structure
    ("Who is on the team?", "Team"),
    ("What's the sprint length?", "Process"),
    ("Who owns the database?", "Roles"),
    ("Who owns the API design?", "Roles"),

    # Stack decisions
    ("Why did we pick Postgres?", "Stack decision"),
    ("Why FastAPI?", "Stack decision"),
    ("What did we decide about observability?", "Stack decision"),

    # Bug + fix chain
    ("Have we hit a JSONB null bug?", "Bug history"),
    ("What was the fix for the JSONB issue?", "Bug history"),
    ("What's our JSONB policy now?", "Policy"),

    # Auth redesign
    ("Where do we store refresh tokens?", "Auth"),
    ("Why httpOnly cookies?", "Auth"),
    ("What's the JWT lifetime?", "Auth"),

    # Performance
    ("Why are we using raw asyncpg?", "Performance"),
    ("What's our latency budget per endpoint?", "Performance"),
    ("How is the connection pool sized?", "Performance"),

    # Typing debate (decision-intent)
    ("What did Alice think about typing?", "Argument"),
    ("What was Bob's position on type hints?", "Argument"),
    ("What did the team decide about typing?", "Decision"),

    # Deployment (conflict resolution)
    ("Where do we deploy now?", "Current state"),
    ("What did we use before?", "Historical state"),
    ("Why did we migrate to GCP?", "Reasoning"),
    ("What was the cost reduction from GCP?", "Outcome"),

    # Planning / goals
    ("What's the Q3 roadmap?", "Planning"),
    ("Who's leading payments?", "Roles"),
    ("What's the fraud detection plan?", "Planning"),
    ("When is the production cut-over scheduled?", "Planning"),
]


def fresh_engine():
    from pmb.core.engine import Engine
    tmp_home = Path(tempfile.mkdtemp(prefix="pmb-qual-"))
    tmp_ws = Path(tempfile.mkdtemp(prefix="pmb-qual-ws-"))
    os.environ["PMB_HOME"] = str(tmp_home)
    eng = Engine(cwd=tmp_ws, pmb_home=tmp_home, rerank_model=None)
    _ = eng.search.model  # force model load now so latency is honest
    return eng


def truncate(s: str, n: int = 110) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-consolidation", action="store_true",
                    help="Skip the Claude CLI consolidation phase")
    ap.add_argument("--out",
                    default=data_path("pmb_quality_demo.json"))
    args = ap.parse_args()

    eng = fresh_engine()
    report = {}

    # =================================================================
    print("\n" + "=" * 78)
    print("  PMB QUALITY DEMO - 200 messages across 10 chats, full transcripts")
    print("=" * 78)

    # ---- 1. Ingestion ----
    print("\n[1] INGEST 10 CHATS × 20 MESSAGES = 200 EVENTS")
    print("-" * 78)
    total_items = 0
    per_chat = []
    for i, chat in enumerate(CHATS, 1):
        items = chat["items"]
        result = eng.record_batch(items)
        per_chat.append({"title": chat["title"], "n_ok": result.get("n_ok", 0)})
        total_items += result.get("n_ok", 0)
        time.sleep(0.6)  # let async embed catch up
        print(f"  {i:>2}. {chat['title']:<55} {result.get('n_ok', 0):>2} events stored")
    time.sleep(3.0)
    print(f"\n  total events stored: {total_items}")
    report["ingest"] = {"total": total_items, "per_chat": per_chat}

    # ---- 2. Cross-session queries with full text ----
    print("\n[2] 30 CROSS-CHAT QUERIES - FULL TOP-3 ANSWER TEXT")
    print("-" * 78)
    queries_report = []
    for i, (q, cat) in enumerate(QUERIES, 1):
        t0 = time.perf_counter()
        pack = eng.recall(q, top_k=3)
        ms = (time.perf_counter() - t0) * 1000
        results_text = [truncate(r.content, 110) for r in pack.results]
        print(f"\n  Q{i:>2} [{cat}]: {q}")
        for j, r in enumerate(results_text, 1):
            print(f"        [{j}] {r}")
        print(f"        ({ms:.0f} ms)")
        queries_report.append({
            "n": i, "category": cat, "query": q,
            "top": results_text, "latency_ms": round(ms, 1)
        })
    report["queries"] = queries_report

    # ---- 3. Summary tools with full content ----
    print("\n[3] SUMMARY TOOLS - WHAT THEY RETURN")
    print("-" * 78)
    summary_report = {}

    print("\n  recent_activity(minutes=60):")
    try:
        recent = eng.recent_activity(minutes=60, limit=10)
        for r in recent[:8]:
            print(f"    - {truncate(r.get('content',''), 100)}")
        summary_report["recent_activity_60"] = [r.get("content", "")[:120] for r in recent]
    except Exception as e:
        print(f"    ERROR: {e}")

    print("\n  list_goals(status='in_progress'):")
    try:
        goals = eng.list_goals(status="in_progress", limit=20)
        for g in goals:
            print(f"    - {g.get('title','')}")
        summary_report["goals_in_progress"] = [g.get("title", "") for g in goals]
    except Exception as e:
        print(f"    ERROR: {e}")

    print("\n  list_goals(status='pending'):")
    try:
        goals = eng.list_goals(status="pending", limit=20)
        for g in goals:
            print(f"    - {g.get('title','')}")
        summary_report["goals_pending"] = [g.get("title", "") for g in goals]
    except Exception as e:
        print(f"    ERROR: {e}")

    print("\n  what_just_happened(n=10):")
    try:
        wjh = eng.what_just_happened(n=10)
        for w in wjh[:8]:
            print(f"    - {truncate(w.get('content',''), 100)}")
        summary_report["what_just_happened"] = [w.get("content", "")[:120] for w in wjh]
    except Exception as e:
        print(f"    ERROR: {e}")

    report["summaries"] = summary_report

    # ---- 4. Tier composition + samples ----
    print("\n[4] TIER COMPOSITION + SAMPLE EVENTS PER TIER")
    print("-" * 78)
    tier_report = {}
    with eng.events._conn() as conn:
        # Count by tier
        for row in conn.execute(
            "SELECT tier, COUNT(*) AS c FROM events "
            "WHERE workspace_id = ? AND archived_at IS NULL "
            "GROUP BY tier",
            (eng.workspace.id,),
        ):
            tier_report[row["tier"]] = {"count": row["c"], "samples": []}
        # Sample 3 per tier
        for tier in tier_report:
            for row in conn.execute(
                "SELECT content, importance, access_count FROM events "
                "WHERE workspace_id = ? AND tier = ? AND archived_at IS NULL "
                "ORDER BY importance DESC LIMIT 3",
                (eng.workspace.id, tier),
            ):
                tier_report[tier]["samples"].append({
                    "content": row["content"][:120],
                    "importance": round(row["importance"], 3),
                    "access_count": row["access_count"],
                })
    for tier, data in tier_report.items():
        print(f"\n  Tier '{tier}': {data['count']} events")
        for s in data["samples"]:
            print(f"    [imp={s['importance']:.2f} acc={s['access_count']}] "
                  f"{truncate(s['content'], 100)}")
    report["tiers"] = tier_report

    # ---- 5. Decay simulation with content ----
    print("\n[5] DECAY SIMULATION - SHOW ARCHIVED CONTENT")
    print("-" * 78)
    now = time.time()
    cutoff_old = now - 30 * 86400
    with eng.events._conn() as conn:
        rows = conn.execute(
            "SELECT id FROM events WHERE workspace_id = ? "
            "AND archived_at IS NULL ORDER BY id ASC",
            (eng.workspace.id,),
        ).fetchall()
        ids = [r["id"] for r in rows]
        to_age = ids[: len(ids) // 2]
        if to_age:
            placeholders = ",".join("?" * len(to_age))
            conn.execute(
                f"UPDATE events SET timestamp = ?, last_accessed = ? "
                f"WHERE id IN ({placeholders})",
                (cutoff_old, cutoff_old, *to_age),
            )
        archived_before = conn.execute(
            "SELECT COUNT(*) AS c FROM events WHERE workspace_id = ? "
            "AND archived_at IS NOT NULL",
            (eng.workspace.id,),
        ).fetchone()["c"]

    from pmb.signals.decay import apply_decay
    decay_result = apply_decay(eng, days_since_last_decay=30.0)

    with eng.events._conn() as conn:
        archived_after = conn.execute(
            "SELECT COUNT(*) AS c FROM events WHERE workspace_id = ? "
            "AND archived_at IS NOT NULL",
            (eng.workspace.id,),
        ).fetchone()["c"]
        # Sample the newly archived
        newly_archived = []
        for row in conn.execute(
            "SELECT content, importance, tier FROM events WHERE workspace_id = ? "
            "AND archived_at IS NOT NULL ORDER BY archived_at DESC LIMIT 5",
            (eng.workspace.id,),
        ):
            newly_archived.append({
                "content": row["content"][:120],
                "importance": round(row["importance"], 3),
                "tier": row["tier"],
            })

    print(f"  aged {len(to_age)} events by 30 days, archived: "
          f"{archived_before} -> {archived_after} (+{archived_after - archived_before})")
    print(f"  decay factor applied: {decay_result.get('decay_factor')}")
    print("\n  Sample newly archived events (low importance after decay):")
    for a in newly_archived:
        print(f"    [imp={a['importance']:.3f} tier={a['tier']}] {truncate(a['content'], 100)}")
    report["decay"] = {
        "aged": len(to_age),
        "archived_before": archived_before,
        "archived_after": archived_after,
        "samples": newly_archived,
        "decay_factor": decay_result.get("decay_factor"),
    }

    # ---- 6. Claude CLI consolidation ----
    if not args.skip_consolidation:
        print("\n[6] REAL CONSOLIDATION VIA CLAUDE CLI - LLM GENERALISATIONS")
        print("-" * 78)
        import shutil, glob
        cmd = os.environ.get("PMB_CLAUDE_CMD") or shutil.which("claude")
        if not cmd:
            candidates = sorted(glob.glob(os.path.expanduser(
                r"~\AppData\Roaming\Claude\claude-code\*\claude.exe")))
            if candidates:
                cmd = candidates[-1]
                os.environ["PMB_CLAUDE_CMD"] = cmd
        if not cmd:
            print("  SKIPPED: claude CLI not found")
            report["consolidation"] = {"skipped": "claude CLI not found"}
        else:
            print(f"  using: {cmd}")
            from pmb.health.consolidate import resolve_llm_client, run_consolidation
            try:
                llm = resolve_llm_client(backend="claude")
                t0 = time.perf_counter()
                cons_result = run_consolidation(
                    eng, llm=llm,
                    similarity_threshold=0.4,
                    min_cluster_size=2,
                    since_days=60.0,
                    max_clusters=6,
                )
                wall = time.perf_counter() - t0
                print(f"  Consolidation finished in {wall:.1f}s")
                print(f"  clusters_seen={cons_result.get('clusters_seen')}, "
                      f"consolidated={cons_result.get('clusters_consolidated')}, "
                      f"new_facts={cons_result.get('new_facts_created')}")
                # Print per-cluster outcomes
                for i, cl in enumerate(cons_result.get("clusters", []), 1):
                    print(f"\n  Cluster {i}: {cl.get('size', '?')} source events")
                    print(f"    LLM decided: consolidate={cl.get('consolidate')} "
                          f"confidence={cl.get('confidence')}")
                    if cl.get("summary"):
                        print(f"    Generalisation: {truncate(cl['summary'], 150)}")
                    if cl.get("reasoning"):
                        print(f"    Reasoning: {truncate(cl['reasoning'], 150)}")
                    for s in (cl.get("source_contents") or [])[:3]:
                        print(f"      - {truncate(s, 100)}")
                report["consolidation"] = {
                    "wall_s": round(wall, 1),
                    "summary": cons_result,
                }
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
                report["consolidation"] = {"error": f"{type(e).__name__}: {e}"}

    # ---- Wrap-up ----
    print("\n" + "=" * 78)
    print("  Demo complete. Quality is in the transcripts above - read and judge.")
    print("=" * 78)

    try:
        eng.close()
    except Exception:
        pass

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
