"""
Console-configurable settings layer.

Two YAML files:
  - per-workspace: <workspace_storage>/config.yaml
  - global default: <PMB_HOME>/config.yaml

Resolution order (highest wins):
  1. explicit kwarg passed to Engine() or recall()
  2. per-workspace config.yaml
  3. global config.yaml
  4. hard-coded default in `DEFAULTS`

Schema is intentionally flat with dotted keys (`recall.bm25_weight`)
so the CLI can do  `pmb config set recall.bm25_weight 0.7` without
parsing nested YAML. Internally we mirror it as a nested dict for
readability when the YAML is opened in an editor.

Validation lives in `_TYPES`. Bad input prints a clear error and
refuses the write.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


# ----------------------------------------------------------------------
# Schema - every knob the user can tune, with type and human help text
# ----------------------------------------------------------------------


@dataclass
class _Setting:
    type: type
    default: Any
    help: str
    choices: Optional[tuple] = None
    min: Optional[float] = None
    max: Optional[float] = None


# ----------------------------------------------------------------------
# Tier curation — keep `pmb config list` short and signal-dense
# ----------------------------------------------------------------------
#
# PMB has ~100 tunables. Most users only need to know the 25 below; the
# rest are quality-tuned defaults that "just work" on a fresh install.
#
# When you run `pmb config list` (no flags) you only see the DEFAULT-tier
# keys. `pmb config list --pro` shows every tunable; `pmb config list
# --all` is its alias. Same for `pmb config <key>` lookups — every key
# is still readable, just hidden from the default list.
DEFAULT_TIER_KEYS: frozenset[str] = frozenset({
    # ── Recall (the 7 you'll actually want to touch) ─────────────
    "recall.top_k",
    "recall.bm25_weight",
    "recall.recency_half_life_days",
    "recall.rerank",
    "recall.rerank_when_close",
    "recall.ppr_enabled",
    "recall.keyed_fact_boost",

    # ── MCP transport ─────────────────────────────────────────────
    "mcp.record_batch_async",

    # ── Auto-recall hook (zero-cooperation memory injection) ──────
    "auto_recall.enabled",
    "auto_recall.budget_chars",

    # ── Ambient auto-write (memory journals the agent) ────────────
    "autowrite.enabled",
    "autowrite.synthesizer",
    "autowrite.min_importance",

    # ── Agent behaviour (rules in CLAUDE.md / AGENTS.md) ──────────
    "agent.active_mode",
    "agent.apply_lessons",
    "agent.context_continuity",
    "agent.log_decisions",
    "agent.log_completed",
    "agent.log_lessons",
    "agent.log_failures",
    "agent.log_goals",

    # ── Embeddings + graph extraction ─────────────────────────────
    "embedding.backend",
    "embedding.model",
    "graph.extractor",

    # ── Lifecycle ─────────────────────────────────────────────────
    "dedup.enable",
    "decay.factor_per_day",
    "consolidate.auto_trigger",
    "lessons.auto_distill_on_session_end",

    # ── Chat (pmb-chat) ───────────────────────────────────────────
    "chat.model",
    "chat.window",
})


def is_default_tier(key: str) -> bool:
    """True if KEY is in the curated default-tier (visible in plain `pmb
    config list`). Pro / experimental knobs are hidden behind --pro."""
    return key in DEFAULT_TIER_KEYS


def tier_of(key: str) -> str:
    return "default" if key in DEFAULT_TIER_KEYS else "pro"


def keys_by_tier(tier: str) -> list[str]:
    """List config keys at TIER ('default' | 'pro' | 'all')."""
    if tier == "all":
        return sorted(SCHEMA.keys())
    if tier == "default":
        return sorted(DEFAULT_TIER_KEYS & SCHEMA.keys())
    if tier == "pro":
        return sorted(SCHEMA.keys() - DEFAULT_TIER_KEYS)
    raise ValueError(f"unknown tier {tier!r}")


# A single source of truth - every knob in PMB lives here. Adding a key
# here is the only step needed to expose it as `pmb config <key>`.
SCHEMA: dict[str, _Setting] = {
    # Recall / search
    "recall.bm25_weight": _Setting(
        float, 0.7,
        "Weight of BM25 in BM25+vector fusion (vec_weight = 1 - this). "
        "Tuned via ablation_full.py on LoCoMo conv-26/30/41: BM25-heavy "
        "fusion (0.7) beat the original symmetric (0.5) by ~2.5 points. "
        "Vector-only (0.0) loses 18 points; the embedding channel adds "
        "marginal signal at best.",
        min=0.0, max=1.0,
    ),
    "recall.top_k": _Setting(int, 5, "Default top-K returned by recall", min=1, max=100),
    # Agent proactive-logging (consumed by `pmb connect --active` to build the
    # rules the agent follows). Toggle WHAT the agent records about its own
    # work, and whether it applies past lessons. Pro users tweak these, then
    # re-run `pmb connect <agent> --active` to regenerate the agent's rules.
    "agent.active_mode": _Setting(
        bool, False,
        "Auto-logging switch: when True, `pmb connect` / `pmb setup` install "
        "the proactive-logging rules by default (no --active flag needed)."),
    "agent.log_decisions": _Setting(bool, True, "Active mode: log design/code decisions"),
    "agent.log_completed": _Setting(bool, True, "Active mode: log what was done (features/fixes)"),
    "agent.log_lessons": _Setting(bool, True, "Active mode: log lessons / project conventions"),
    "agent.log_failures": _Setting(bool, True, "Active mode: log failures (don't-repeat)"),
    "agent.log_goals": _Setting(bool, True, "Active mode: log user goals / intents"),
    "agent.apply_lessons": _Setting(
        bool, True,
        "Active mode: recall + apply past lessons/failures before a task "
        "(the self-improvement loop - agent gets better at the project over time)",
    ),
    "agent.context_continuity": _Setting(
        bool, True,
        "Active mode: tell the agent to call session_brief to re-orient after "
        "its OWN context compacts in a long session (PMB as durable session memory)"),

    # ── Auto-recall (the hook that bypasses agent cooperation) ─────
    # The UserPromptSubmit hook (`pmb hooks install claude-code`) runs
    # `pmb prepare-context --stdin --auto`. The classifier in
    # pmb.hooks.auto_recall decides — without asking the model — which
    # PMB calls to pre-execute and inject as context. Sub-100ms on a
    # warm workspace. Knobs here control the dispatcher.
    "auto_recall.enabled": _Setting(
        bool, True,
        "Master switch for the auto-recall hook. When True (default), "
        "`pmb prepare-context` classifies the user message and dispatches "
        "matching PMB queries. When False, the hook falls back to the "
        "legacy always-on bundle (project_overview + find_lessons + "
        "recent_activity + list_goals).",
    ),
    "auto_recall.min_message_chars": _Setting(
        int, 5,
        "Messages shorter than this are treated as trivial (greeting / "
        "ack / single emoji) and the hook injects nothing. Lower it for "
        "languages with many short questions; raise it to suppress more "
        "noise.",
        min=1, max=200,
    ),
    "auto_recall.budget_chars": _Setting(
        int, 4000,
        "Hard cap on the formatted context block size. Hosts truncate "
        "long hook output anyway; this gives us a controlled cut with a "
        "marker instead of a silent chop.",
        min=200, max=20000,
    ),
    "auto_recall.recall_top_k": _Setting(
        int, 5,
        "When PAST_QUERY or GENERIC_FACTUAL fires, ask recall for this "
        "many hits. Lower = faster + less prompt bloat; higher = better "
        "coverage on multi-fact questions.",
        min=1, max=25,
    ),
    "auto_recall.recall_min_score": _Setting(
        float, 0.30,
        "For GENERIC_FACTUAL fallback (no explicit pattern, just a '?'), "
        "surface a recall hit only if its top score clears this floor. "
        "Set to 0.0 to always surface; raise to 0.5 to suppress noisy hits.",
        min=0.0, max=1.0,
    ),
    "auto_recall.include_trace": _Setting(
        bool, True,
        "Include `[intents=...,latency=...ms]` in the injected context "
        "header. Useful for adherence debugging; turn off to make the "
        "block look more natural in transcripts.",
    ),
    "auto_recall.surface_decisions": _Setting(
        bool, True,
        "Also surface past DECISIONS (the 'why we did X' rationale) next to "
        "lessons in the auto-recall block. Lets the agent see settled calls "
        "before re-deciding them. Turn off if your workspace doesn't record "
        "decisions and the extra query is wasted.",
    ),

    # ── Ambient memory (auto-write) — memory journals the agent itself ──
    # PostToolUse logs the agent's actions; the Stop hook synthesizes an
    # activity entry ONLY if the agent didn't journal its own work this turn.
    # ON by default: a memory that captures work the agent forgot to record
    # is PMB's signature. Enabling writes-on-its-own is a trust decision, so
    # every entry is tagged source=autowrite, shown as auto, and removable in
    # one command (`pmb forget-auto`).
    "autowrite.enabled": _Setting(
        bool, True,
        "Ambient auto-write: the Stop hook journals what the agent did this "
        "turn — but ONLY if the agent didn't call record_* itself (so it "
        "never duplicates the agent's own, richer summary). Observes actions "
        "via the PostToolUse hook. ON by default: this is PMB's signature — "
        "memory that captures work even when the agent stays silent. Every "
        "entry is tagged source=autowrite, shown as auto, and removable in "
        "one command (`pmb forget-auto`). Turn off with "
        "`pmb config set autowrite.enabled false` if you'd rather record "
        "everything explicitly.",
    ),
    "autowrite.min_actions": _Setting(
        int, 2,
        "Minimum number of SIGNIFICANT observed actions (edits / tests / "
        "commits — not reads or ls) before ambient memory bothers to write. "
        "2 is the sweet spot: catches a real unit of work (an edit + a test, "
        "or a couple of edits) without journaling trivial one-offs. Higher = "
        "fewer, chunkier entries.",
        min=1, max=50,
    ),
    "autowrite.min_importance": _Setting(
        float, 0.45,
        "Quality bar, not just a count: a turn is journaled only if its "
        "estimated importance clears this. Importance comes from OUTCOME "
        "signals — tests passed, an error got fixed, a deploy/migrate ran, "
        "the breadth of edits — NOT from how many files were touched alone. "
        "So 'edited two files and nothing else' (score ~0.30) is skipped, "
        "while 'edited + tests green' (~0.55) is kept and ranks higher in "
        "recall. Lower it to capture more; raise it to keep only clear "
        "milestones. 0 disables the bar (count gate only).",
        min=0.0, max=1.0,
    ),
    "autowrite.window_minutes": _Setting(
        float, 30.0,
        "How far back the Stop hook looks for this turn's actions and for "
        "whether the agent already journaled. Roughly 'one turn'.",
        min=1.0, max=240.0,
    ),
    "autowrite.synthesizer": _Setting(
        str, "template",
        "How the journal line is written. 'template' = instant, "
        "deterministic, no model ('edited 3 files; ran tests; committed'). "
        "'llm:ollama' / 'llm:claude' / 'llm:codex' = a nicer human summary "
        "via the local/CLI model, with a timeout and automatic fallback to "
        "the template so it never blocks the turn.",
        choices=["template", "llm:ollama", "llm:claude", "llm:codex"],
    ),
    "autowrite.llm_model": _Setting(
        str, "",
        "Model id for autowrite.synthesizer when it's an LLM backend. "
        "Empty = backend default (ollama → qwen2.5:3b, claude → haiku).",
    ),
    "autowrite.llm_timeout_s": _Setting(
        float, 20.0,
        "Timeout for the LLM synthesizer. On timeout we fall back to the "
        "template — the turn is never blocked.",
        min=3.0, max=120.0,
    ),
    "overview.max_events": _Setting(
        int, 40,
        "How many memories `pmb overview` / the MCP overview tool synthesize "
        "for a topic", min=5, max=500),
    "session.brief_minutes": _Setting(
        int, 180,
        "Fallback window (minutes) for `pmb session brief` / the session_brief "
        "MCP tool when no session is active", min=5, max=10080),
    "recall.recency_half_life_days": _Setting(
        float, 30.0, "Half-life for recency boost in days", min=0.5, max=3650.0,
    ),
    "recall.graph_boost": _Setting(
        float, 0.15, "Additive bonus from graph traversal (0 disables)",
        min=0.0, max=1.0,
    ),
    "recall.multi_entity_bonus": _Setting(
        float, 0.5,
        "Multi-hop bonus: events matching N query entities get graph weight "
        "× (1 + bonus*(N-1)). 0 disables. Helps multi-hop questions where "
        "answer event mentions multiple query entities.",
        min=0.0, max=2.0,
    ),
    "recall.causation_walk": _Setting(
        bool, True,
        "PMB v2: walk causation graph (event_edges) when query looks "
        "multi-hop (after/before/because/...). Surfaces bridge events that "
        "lexical search misses. Free if no edges exist.",
    ),
    "recall.causation_boost": _Setting(
        float, 0.10,
        "Additive bonus to events surfaced by causation walk (multiplied by "
        "importance and recency). Lower than graph_boost because causation "
        "edges are LLM-extracted and noisier.",
        min=0.0, max=1.0,
    ),
    "recall.arc_expansion": _Setting(
        bool, True,
        "PMB v2: search narrative arc summaries and inject member events "
        "into recall candidates. Especially helps 'tell me about X' and "
        "'history of Y' style queries.",
    ),
    "recall.arc_boost": _Setting(
        float, 0.08,
        "Additive bonus for events that belong to an arc matching the query.",
        min=0.0, max=1.0,
    ),
    "recall.collapse_reflections": _Setting(
        bool, True,
        "PMB v2: after scoring, collapse reflection events onto their source. "
        "Reflections are bridges (boost source's score) but the final result "
        "list returns the actual source events with their dia_ids/metadata.",
    ),
    "recall.ppr_enabled": _Setting(
        bool, True,
        "Personalized PageRank over the entity graph — diffuses relevance "
        "for multi-hop questions. Gated by intent detection (`ppr_always` "
        "off), so single-entity lookups skip it. Default ON since the gate "
        "keeps cost ~0 for non-multi-hop queries and the upside on "
        "narrative questions is real.",
    ),
    "recall.ppr_weight": _Setting(
        float, 0.5,
        "Weight of PPR contribution to final event score. Higher = trust graph more.",
        min=0.0, max=3.0,
    ),
    "recall.ppr_alpha": _Setting(
        float, 0.5,
        "PPR teleportation probability. 0.5 balances depth (low) vs locality "
        "(high). Lower = walks further out (more multi-hop), higher = stays "
        "near query entities.",
        min=0.05, max=0.95,
    ),
    "recall.ppr_iters": _Setting(
        int, 20,
        "PPR power iterations. 20 is plenty for graphs under 100k nodes.",
        min=5, max=200,
    ),
    "recall.ppr_always": _Setting(
        bool, False,
        "Run PPR even on single-entity / non-multi-hop queries. Off by default "
        "because PPR adds noise to exact-match lookups. Useful for benchmark "
        "comparisons or recall-heavy workloads.",
    ),
    "recall.adaptive_decompose": _Setting(
        bool, False,
        "PMB v2.2: when query looks multi-hop, LLM splits it into 2-3 atomic "
        "sub-queries, runs each, fuses via Reciprocal Rank Fusion. Costs 1 "
        "LLM call per multi-hop query (cached on disk). Off by default - "
        "enable for multi-hop heavy workloads. Single-hop queries unaffected.",
    ),
    "recall.smart_deadline_ms": _Setting(
        int, 15000,
        "Overall wall-clock budget (ms) for recall_smart escalation. The "
        "interactive path returns its best-so-far result the moment this is "
        "exceeded — no stage may start past the deadline. This is a safety "
        "ceiling, not a target: the fast path still returns in milliseconds "
        "when confidence is high. Set 30000 to allow up to 30s.",
        min=200, max=120000,
    ),
    "recall.smart_allow_llm": _Setting(
        bool, False,
        "Allow recall_smart to escalate into LLM query-decomposition INSIDE "
        "the deadline (each LLM call is bounded to the remaining budget). Off "
        "by default: the interactive path stays local-only "
        "(BM25+vec+graph+local rerank) and never spawns the Claude CLI / "
        "Ollama, so a hung backend can't stall recall. Deep LLM recall is "
        "always available explicitly via recall_deep() / the recall_deep tool.",
    ),
    "keyed.auto_detect_current_state": _Setting(
        bool, True,
        "When a user fact plainly states a CURRENT personal attribute "
        "(\"I now live in Tampa\", \"my current employer is X\", \"сейчас живу "
        "в …\"), also upsert the matching keyed fact so the live value "
        "supersedes any stale one; the original fact is kept as history. "
        "Conservative: fires only on explicit present-state phrasing from "
        "user-origin facts — never reflections / project index / autowrite.",
    ),
    "keyed.archive_obsolete_negations": _Setting(
        bool, True,
        "When a positive keyed value is set (e.g. user::city = Tampa), archive "
        "older active facts that NEGATE or mark-unknown that same attribute "
        "(\"the user does not currently live in Warsaw; current city is "
        "unknown\") — they assert ignorance about a now-known attribute. "
        "Archive-only (reversible, tagged superseded_by); skips pinned events "
        "and lessons. This is a bug-class fix, so it defaults ON.",
    ),
    "write.quality_gate": _Setting(
        bool, False,
        "Write-time junk gate (opt-in, default OFF). When ON, a fact that "
        "looks like junk (placeholder / test patterns / pure stopwords) is NOT "
        "rejected — it is flagged metadata.quality_flag=suspect_junk, its "
        "importance is capped at 0.2, and it is excluded from keyed-fact "
        "promotion. `pmb declutter` then treats it as a first-class candidate.",
    ),
    "daemon.idle_exit_min": _Setting(
        float, 120.0,
        "Persistent memory daemon: exit after this many minutes with no "
        "request, so a forgotten daemon doesn't hold ~400MB forever. 0 = never "
        "exit. The hook autostarts a new one on the next message.",
        min=0.0, max=10080.0,
    ),
    "daemon.autostart": _Setting(
        bool, True,
        "When the cold hook path runs and no daemon is live, spawn one in the "
        "background (rate-limited) so the NEXT message hits a warm daemon with "
        "real semantic recall. The current message still answers cold. Set "
        "false to require an explicit `pmb daemon start`.",
    ),
    "mcp.compact_responses": _Setting(
        bool, True,
        "Trim MCP tool responses before sending: drop null/empty fields and "
        "cap each recall result's content (see mcp.max_item_chars). Saves "
        "tokens per call without losing information the agent needs. Off "
        "restores verbose responses.",
    ),
    "mcp.max_item_chars": _Setting(
        int, 600,
        "When mcp.compact_responses is on, cap each recall result's content at "
        "this many chars (… suffix when trimmed). Generous by default so normal "
        "facts are untouched; only pathologically long items shrink. 0 = no cap.",
        min=0, max=100000,
    ),
    "write.outbox": _Setting(
        bool, True,
        "Durable write outbox for record_batch_async. When ON (default), an "
        "async batch is persisted to the write_outbox table synchronously "
        "before returning, then replayed by a background drainer — so a crash "
        "between accept and write loses nothing. OFF restores the old "
        "fire-and-forget daemon-thread path (only for bisecting).",
    ),
    "llm.offline_max_calls": _Setting(
        int, 40,
        "Hard cap on the number of LLM calls a SINGLE offline pass (keyed "
        "suggestions / declutter judge / consolidation) may make. Bounds the "
        "whole pass, not just one call. Lower = cheaper, less thorough.",
        min=1, max=1000,
    ),
    "llm.offline_budget_s": _Setting(
        float, 120.0,
        "Hard wall-clock cap (seconds) on a SINGLE offline LLM pass. The pass "
        "stops launching new calls once exceeded (a call already in flight "
        "finishes). Prevents a slow local model from holding `pmb consolidate` "
        "for many minutes.",
        min=1.0, max=3600.0,
    ),
    "recall.breaker_threshold": _Setting(
        int, 2,
        "Circuit breaker: consecutive backend failures (timeouts/errors) "
        "before that backend (LLM / reranker / …) is temporarily disabled for "
        "the interactive path. Lower = trip sooner.",
        min=1, max=20,
    ),
    "recall.breaker_cooldown_s": _Setting(
        float, 60.0,
        "Circuit breaker: how long (seconds) a tripped backend stays disabled "
        "before PMB tries it again. A single success closes it early.",
        min=1.0, max=3600.0,
    ),
    "recall.reflection_to_edges": _Setting(
        bool, True,
        "Improvement B (HippoRAG 2 inspired): during reflection, link the "
        "LLM-extracted entities/people/themes BACK to the source event in "
        "the graph (not just to the reflection chunk). Source becomes "
        "findable via reflection vocabulary without a separate index hit. "
        "On by default - pure win, no cost.",
    ),
    "recall.temporal_enabled": _Setting(
        bool, True,
        "Improvement C (Zep/Graphiti inspired): parse explicit date "
        "references from events (regex, no LLM) into event_time metadata, "
        "and boost candidates by temporal proximity when the query has "
        "time markers. Cheap.",
    ),
    "recall.temporal_boost": _Setting(
        float, 0.20,
        "Weight of temporal-proximity contribution to final score.",
        min=0.0, max=2.0,
    ),
    "recall.keyed_fact_boost": _Setting(
        float, 0.35,
        "Additive boost for facts recorded via record_keyed_fact. Raises the "
        "current canonical personal-attribute value (e.g. 'user.city=Warsaw') "
        "above generic facts that share vocabulary on queries like 'where do "
        "I live'. Set 0 to disable.",
        min=0.0, max=2.0,
    ),
    "recall.temporal_half_life_days": _Setting(
        float, 14.0,
        "Days at which temporal proximity drops to 0.5. Lower = stricter "
        "time matching; higher = wider window.",
        min=0.5, max=3650.0,
    ),
    "recall.adaptive_routing": _Setting(
        bool, True,
        "Improvement E: classify query intent (direct/temporal/multi-hop/"
        "narrative/inferential) and re-weight layer boosts accordingly. "
        "Cheap (<0.1ms, no LLM). On by default - pure win.",
    ),
    "recall.predictive_enabled": _Setting(
        bool, True,
        "Improvement F: check predictive cache first. If pre-computed "
        "answer matches current query (cosine ≥ threshold), return cached "
        "top-K in ~3ms instead of running full recall (~80ms). "
        "Cache populated during sleep via `precompute_predictive_cache()`.",
    ),
    "recall.predictive_threshold": _Setting(
        float, 0.85,
        "Cosine similarity needed for a cached query to count as a hit.",
        min=0.5, max=1.0,
    ),
    "recall.predictive_ttl_days": _Setting(
        float, 7.0,
        "Days a cached entry stays valid. After this it's ignored on read "
        "(but kept until cleanup).",
        min=0.1, max=3650.0,
    ),
    "recall.person_extraction": _Setting(
        bool, True,
        "Improvement H: extract person entities (no ML) via speaker "
        "metadata + capitalized-word regex + verb-context + self-reinforcing "
        "dict. Boosts person-heavy queries (cat 1/3 in LoCoMo).",
    ),
    "recall.code_ast_extraction": _Setting(
        bool, True,
        "Improvement J (code half): when content looks like Python source, "
        "extract function/class/import entities via stdlib `ast`. Lets "
        "the graph layer answer code-structure queries.",
    ),
    "recall.typo_correction": _Setting(
        bool, False,
        "Improvement K: at recall start, fuzzy-match query tokens against "
        "known entity names (Levenshtein ≤ 2). 'Aliceee'→'alice', "
        "'Postgers'→'postgres'. Cheap, no ML. "
        "DISABLED by default after ablation showed -6.2 pts recall on "
        "LoCoMo (the 'corrections' often rewrite a correctly-typed query "
        "token into a similar-looking but wrong entity name). Enable per "
        "workspace if you have a known typo-prone use case.",
    ),
    "recall.graph_expansion_llm": _Setting(
        bool, False,
        "Use an LLM to extract concrete entities from abstract queries before "
        "graph traversal. Adds one LLM call per recall - off by default.",
    ),
    "recall.cache_size": _Setting(
        int, 128, "LRU cache size for recall queries (0 disables)",
        min=0, max=10000,
    ),
    "recall.touch_async": _Setting(
        bool, True,
        "Apply recall's reinforcement side-effects (access_count, importance "
        "boost, last_accessed) via a deferred ~250ms background flusher instead "
        "of synchronously. ON by default: under concurrent recalls this turns "
        "~16 SQLite write-lock acquisitions per second into ~4. Set False when "
        "recall's side-effects must be visible IMMEDIATELY after the call "
        "(deterministic tests, single-shot CLI scripts) — recall then drains "
        "the touch buffer inline before returning.",
    ),
    "recall.lesson_min_overlap": _Setting(
        int, 1,
        "Min DISTINCTIVE shared tokens (stopword-filtered, length>=4, "
        "identifiers kept) between a message and a lesson for that lesson to "
        "surface (find_lessons / auto-recall). 1 is the sweet spot: the "
        "stopword set already strips generic/path noise, so a single "
        "distinctive overlap (pnpm, numpy, lancedb) is a real signal. Raise to "
        "2 for stricter precision — fewer, surer lessons, which is what makes "
        "the adherence follow-rate meaningful instead of drowning in noise.",
        min=1, max=5,
    ),
    "recall.lesson_semantic": _Setting(
        bool, False,
        "EXPERIMENTAL semantic tier for lesson surfacing: alongside the lexical "
        "token gate, score lessons by cosine on the embeddings recall already "
        "uses, to catch paraphrase / synonym / cross-lingual matches. Not an "
        "LLM call, just a vector cosine. OFF by default — and a real-workspace "
        "eval (June 8 2026) with the default MiniLM embedder showed it did NOT "
        "reliably beat the lexical tier: it MISSED obvious matches (e.g. a "
        "LanceDB lesson for an 'apple-silicon vector store' query) and added "
        "off-topic noise, with no clean cosine threshold. Revisit only with a "
        "stronger embedder (bge-m3) or the cross-encoder reranker; until then "
        "the lexical gate + stopwords are the reliable noise fix. Enabling also "
        "loads the embedding model on the per-turn hook path.",
    ),
    "recall.lesson_semantic_min": _Setting(
        float, 0.45,
        "Cosine-similarity threshold for the semantic lesson tier "
        "(recall.lesson_semantic). Higher = stricter (fewer paraphrase "
        "matches). 0.45 is a sane start for bge-m3 / MiniLM; tune against your "
        "own lessons.",
        min=0.0, max=1.0,
    ),
    "recall.cache_ttl_seconds": _Setting(
        float, 300.0,
        "How long a cached recall stays valid before re-running (5min default)",
        min=0.0, max=86400.0,
    ),
    "recall.spreading_activation": _Setting(
        bool, True,
        "Boost importance of graph neighbours of recall hits (priming). "
        "Decays over hours; mimics human spreading-activation.",
    ),
    "recall.spreading_boost": _Setting(
        float, 0.05,
        "Magnitude of priming boost added to each hit's graph neighbour",
        min=0.0, max=0.5,
    ),
    "recall.spreading_half_life_hours": _Setting(
        float, 2.0,
        "How fast the priming boost itself decays (hours)",
        min=0.1, max=168.0,
    ),
    "recall.rerank": _Setting(
        bool, False,
        "Use cross-encoder reranker on top-N hits ALWAYS. WARNING: on "
        "single-session retrieval benchmarks (LoCoMo) the reranker "
        "regresses evidence-recall by ~17 points. Prefer "
        "`recall.rerank_when_close` instead, which only invokes the "
        "reranker when the hybrid ranker can't decide between the top "
        "candidates.",
    ),
    "recall.pamvr_enabled": _Setting(
        bool, False,
        "PAMVR (Predicate-Aware Multi-View Reranking). Post-scoring set of "
        "small content-based boosts: verb match, entity strict, vocab "
        "bridges (typing↔mypy, database↔Postgres), topic constraint, "
        "policy intent, time-duration. "
        "DOMAIN-SPECIFIC: lifts top-1 by ~30pp on coding-agent / dev "
        "memory queries (60% → 90% on 30-query bench) but slightly "
        "regresses LoCoMo (-1pp) because vocab bridges + verb syns are "
        "dev-lexicon. Enable per workspace if your usage is coding/dev "
        "focused: `pmb config set recall.pamvr_enabled true`. "
        "See pmb.reasoning.pamvr for the rule set; extend VOCAB_BRIDGES "
        "/ VERB_SYNS for your domain.",
    ),
    "recall.llm_rerank": _Setting(
        bool, False,
        "Improvement XX: optional LLM-as-judge rerank over the current "
        "top-N candidates. Uses a small local Ollama model (default "
        "qwen2.5:1.5b, ~900MB) to pick the single best candidate. "
        "Adds ~100-300ms per query but lifts top-1 by 5-15pp on hard "
        "queries where hybrid + cross-encoder all give close scores. "
        "OFF by default - opt-in: `pmb config set recall.llm_rerank true`. "
        "Requires `ollama serve` running and `ollama pull qwen2.5:1.5b`. "
        "Degrades gracefully: any LLM error keeps the previous order.",
    ),
    "recall.llm_rerank_model": _Setting(
        str, "qwen2.5:1.5b",
        "Ollama model tag to use for LLM rerank. Good defaults: "
        "qwen2.5:0.5b (~400MB), qwen2.5:1.5b (default), llama3.2:1b, "
        "phi3:mini.",
    ),
    "recall.llm_rerank_top_n": _Setting(
        int, 10, "How many top candidates to show the LLM judge.",
        min=2, max=50,
    ),
    "recall.llm_rerank_timeout": _Setting(
        float, 5.0, "HTTP timeout (seconds) for LLM rerank request.",
        min=0.5, max=120.0,
    ),
    "write.atomic_fact_extract": _Setting(
        bool, False,
        "Improvement WW: on each `fact` recorded via record_batch, run "
        "pattern-based atomic-fact extraction (mem0-style, no LLM) and "
        "record each extracted atom as a child event with metadata."
        "parent_ulid pointing to the source. Patterns: 'X lives in Y', "
        "'X is the Z at W', 'We use X', 'X leads Y', etc. Lifts recall "
        "on questions targeting one fact inside a long paragraph. "
        "OFF by default: adds ~1-5ms per record (regex pass) + ~N extra "
        "events. Enable when the typical message is multi-sentence "
        "(meeting notes, project log) and per-message size matters less "
        "than question-precision recall.",
    ),
    "recall.pattern_split": _Setting(
        bool, True,
        "Improvement UU: split compound queries on natural markers "
        "('X and why Y' / 'X потому что Y' / 'X, also Y') and fuse "
        "sub-query results via Reciprocal Rank Fusion. No LLM needed - "
        "patterns cover ~80%% of compound queries on EN+RU. "
        "Default ON: regression-safe (single-clause queries skip split "
        "entirely; only fires when both halves carry content tokens).",
    ),
    "recall.auto_vocab_bridges": _Setting(
        bool, True,
        "Improvement TT: auto-mine VOCAB_BRIDGES from this workspace's "
        "own events via PMI co-occurrence. Makes PAMVR domain-agnostic - "
        "instead of hand-curated coding-lexicon bridges (typing↔mypy, "
        "database↔Postgres), the engine learns the user's actual vocabulary "
        "(e.g. on a personal workspace it might learn ['recipe','onion'] or "
        "['workout','squat']). Default ON: harmless when no patterns are "
        "strong (empty bridges just fall back to hand-curated defaults). "
        "Cache: ~/.pmb/workspaces/<id>/vocab_bridges.json, refreshed every "
        "`recall.auto_vocab_refresh_after` new events.",
    ),
    "recall.auto_vocab_window": _Setting(
        int, 6, "Auto-bridges: ±N token window for co-occurrence counting.",
        min=2, max=30,
    ),
    "recall.auto_vocab_min_count": _Setting(
        int, 3, "Auto-bridges: minimum pair count to be considered.",
        min=1, max=100,
    ),
    "recall.auto_vocab_min_pmi": _Setting(
        float, 2.0,
        "Auto-bridges: minimum PMI score for a pair to enter the bridge "
        "table. Higher = fewer, stronger pairs.",
        min=0.0, max=20.0,
    ),
    "recall.auto_vocab_max_per_key": _Setting(
        int, 8, "Auto-bridges: max bridge terms kept per key, sorted by PMI.",
        min=1, max=64,
    ),
    "recall.auto_vocab_refresh_after": _Setting(
        int, 50,
        "Auto-bridges: re-mine the bridge table after this many new events "
        "have landed since the last mine.",
        min=10, max=10000,
    ),
    "recall.rerank_when_close": _Setting(
        bool, False,
        "Improvement #1: GATED reranker. Only run the cross-encoder when "
        "the gap between top-1 and top-3 scores is below "
        "`recall.rerank_close_epsilon`. The intuition: when BM25+vector "
        "have a clear winner, trust them; when they don't, ask the "
        "reranker to break the tie. "
        "OFF by default: gating helps on the specific 'Alex prefers X' "
        "type ambiguity but measurably regresses LoCoMo evidence-recall "
        "(~0.5-1pp). Enable per workspace if your queries lean conceptual "
        "and you have already validated against your own data.",
    ),
    "recall.rerank_swap_margin": _Setting(
        float, 0.20,
        "Improvement VV: confidence margin required to commit a top-1 swap "
        "during GATED rerank. If the cross-encoder's new-top-1 score doesn't "
        "beat the previous-top-1 by at least this margin, keep the hybrid "
        "order. This prevents the LoCoMo-style regression where the CE "
        "reorders confidently-correct hits with low confidence.",
        min=0.0, max=5.0,
    ),
    "recall.rerank_close_epsilon": _Setting(
        float, 0.05,
        "Score-gap threshold for gated reranking. If "
        "(top1_score - top3_score) < epsilon, fire the reranker. Smaller "
        "= fires more often; larger = only on very-tight ties.",
        min=0.0, max=1.0,
    ),
    "recall.rerank_top_n": _Setting(
        int, 25, "Candidates fed into the reranker", min=5, max=200,
    ),
    "recall.rerank_model": _Setting(
        str, "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "HuggingFace cross-encoder model name",
    ),
    # Embedding
    "embedding.model": _Setting(
        str, "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "Embedding model name (any sentence-transformers id). Default is "
        "multilingual (50+ langs incl. RU/EN/ES/ZH). After changing run "
        "`pmb reindex` to re-embed existing events.",
    ),
    "embedding.backend": _Setting(
        str, "sentence-transformers",
        "Embedding inference runtime. 'ollama'/'openai' use a server (different "
        "vector dim → use a FRESH workspace or `pmb reindex`).",
        choices=("sentence-transformers", "fastembed", "ollama", "openai"),
    ),
    "embedding.fastembed_model": _Setting(
        str, "sentence-transformers/all-MiniLM-L6-v2",
        "fastembed-compatible model id (used only when backend=fastembed)",
    ),
    "embedding.ollama_model": _Setting(
        str, "nomic-embed-text",
        "Ollama embedding model (used only when backend=ollama). "
        "Run `ollama pull nomic-embed-text` first. 768-dim.",
    ),
    "embedding.ollama_url": _Setting(
        str, "http://localhost:11434",
        "Ollama server base URL (used only when backend=ollama).",
    ),
    "embedding.openai_model": _Setting(
        str, "text-embedding-3-small",
        "OpenAI embedding model (used only when backend=openai). Needs "
        "OPENAI_API_KEY in the environment. 1536-dim.",
    ),
    "lessons.auto_distill_on_session_end": _Setting(
        bool, False,
        "On `pmb session end`, auto-distill durable lessons/failures from the "
        "session via an LLM (zero-command memory growth). Off by default - "
        "needs an LLM backend (claude CLI / Anthropic key / Ollama).",
    ),
    "recall.lesson_boost": _Setting(
        bool, True,
        "On how-to/convention queries, gently boost lesson & failure memories "
        "so they surface. Only affects events with kind=lesson/failure, so it "
        "cannot change recall on datasets without them (e.g. LoCoMo).",
    ),
    "recall.lesson_boost_factor": _Setting(
        float, 1.3,
        "Score multiplier applied to lesson/failure events on lesson-intent "
        "queries (when recall.lesson_boost is on).",
        min=1.0, max=3.0,
    ),
    # Decay / forgetting
    "decay.factor_per_day": _Setting(
        float, 0.985, "Daily importance decay multiplier (0..1)",
        min=0.5, max=1.0,
    ),
    "decay.archive_threshold": _Setting(
        float, 0.05, "Importance below this triggers archive (if also old)",
        min=0.0, max=1.0,
    ),
    "decay.archive_min_age_days": _Setting(
        float, 90.0, "Don't auto-archive events younger than this",
        min=0.0, max=3650.0,
    ),
    "decay.archive_cold_days": _Setting(
        int, 90,
        "`pmb decay --archive-cold`: minimum age (days) for a cold low-value "
        "fact/activity to be eligible for time-based archival.",
        min=1, max=3650,
    ),
    "decay.archive_cold_max_importance": _Setting(
        float, 0.25,
        "`pmb decay --archive-cold`: only archive cold events at/below this "
        "importance (never pinned / keyed / lessons / goals).",
        min=0.0, max=1.0,
    ),
    # Reinforcement
    "feedback.useful_boost_rate": _Setting(
        float, 0.08, "Per-call boost when feedback=useful (saturating)",
        min=0.0, max=1.0,
    ),
    "feedback.wrong_demote": _Setting(
        float, 0.05, "Per-call demote when feedback=wrong/irrelevant",
        min=0.0, max=1.0,
    ),
    # Consolidation
    "consolidate.backend": _Setting(
        str, "auto", "LLM backend for consolidation",
        choices=("auto", "claude", "anthropic", "ollama"),
    ),
    "consolidate.model": _Setting(
        str, "", "Override model name; empty = backend default",
    ),
    "consolidate.suggest_keyed": _Setting(
        bool, True,
        "Offline LLM tier: during consolidation, ask the LLM to extract "
        "current-state keyed facts (city/employer/…) from plain facts the "
        "cheap regex missed. confidence>=0.8 positives are upserted via the "
        "canonical keyed-fact path (+ negation-tombstone cleanup); weaker ones "
        "are tagged metadata.suggested_key. Offline only — never on recall; "
        "timeout-clamped + circuit-broken. Default ON (it runs offline anyway).",
    ),
    "consolidate.similarity_threshold": _Setting(
        float, 0.5, "Cosine similarity threshold for clustering",
        min=0.0, max=1.0,
    ),
    "consolidate.min_cluster_size": _Setting(
        int, 3, "Min events in a cluster", min=2, max=50,
    ),
    "consolidate.min_confidence": _Setting(
        float, 0.6, "LLM confidence threshold to store",
        min=0.0, max=1.0,
    ),
    "consolidate.since_days": _Setting(
        float, 14.0, "Look back N days for clustering",
        min=0.5, max=3650.0,
    ),
    "consolidate.auto_trigger": _Setting(
        bool, False,
        "Auto-run consolidation when thresholds are met (writes or time). "
        "Off by default - LLM calls cost time/money so explicit is safer. "
        "To enable: `pmb config set consolidate.auto_trigger true`. The "
        "scheduler then fires `pmb consolidate` automatically once "
        "consolidate.auto_min_new_events AND consolidate.auto_min_days are "
        "both satisfied.",
    ),
    "consolidate.auto_min_new_events": _Setting(
        int, 50,
        "Trigger threshold: N new events since last consolidation",
        min=1, max=100000,
    ),
    "consolidate.auto_min_days": _Setting(
        float, 7.0,
        "Trigger threshold: M days since last consolidation",
        min=0.5, max=365.0,
    ),
    # Ollama
    "ollama.url": _Setting(str, "", "Ollama URL (empty -> localhost:11434)"),
    "ollama.model": _Setting(str, "llama3.1:8b", "Ollama model id for consolidation"),
    # Agent wrapper / pmb-chat
    "chat.transport": _Setting(
        str, "auto", "pmb-chat transport",
        choices=("auto", "claude", "anthropic", "ollama"),
    ),
    "chat.model": _Setting(str, "haiku", "Model alias for pmb-chat"),
    "chat.window": _Setting(int, 200_000, "Token window", min=1024, max=10_000_000),
    "chat.target_max": _Setting(
        float, 0.75, "Fraction of window before compaction triggers",
        min=0.1, max=0.99,
    ),
    "chat.selective_compression": _Setting(
        bool, True, "Use SelectivePolicy (vs DropOldestNarrative)",
    ),

    # ------------------------------------------------------------------
    # Improvement U: Multi-layer dedup
    # ------------------------------------------------------------------
    "dedup.enable": _Setting(
        bool, True,
        "Master switch for write-time dedup (L1 exact + L2 semantic). "
        "Off → all writes go through unchanged (legacy behavior).",
    ),
    "dedup.enable_semantic": _Setting(
        bool, True,
        "L2: cosine-similarity dedup at write time. Off keeps only L1 "
        "(exact-text match). L2 adds ~50ms per write (embedding+search).",
    ),
    "dedup.cosine_high": _Setting(
        float, 0.92,
        "L2 high threshold - at or above this, the new write is silently "
        "merged into the existing canonical event. Conservative default; "
        "tighter = fewer false merges, looser = catches more dups.",
        min=0.5, max=0.999,
    ),
    "dedup.cosine_mid": _Setting(
        float, 0.80,
        "L2 mid threshold - pairs in [mid, high) are written as borderline "
        "candidates into the dedup queue for async LLM verification (L2.5).",
        min=0.5, max=0.99,
    ),
    "dedup.lookback_days": _Setting(
        float, 90.0,
        "How far back to search for dedup candidates. Older items are too "
        "stale to be likely duplicates; bounds the search for speed.",
        min=1.0, max=3650.0,
    ),
    "dedup.async_verify": _Setting(
        bool, True,
        "L2.5: enqueue borderline pairs for async LLM verify. Workers "
        "(Ollama or Anthropic) drain the queue via `pmb dedupe --run-pending`.",
    ),

    # ------------------------------------------------------------------
    # Improvement AA: fire-and-forget MCP record_batch
    # ------------------------------------------------------------------
    "mcp.record_batch_async": _Setting(
        bool, True,
        "MCP `record_batch` tool returns IMMEDIATELY after spawning "
        "background processing - no waiting for embedding/graph/LanceDB. "
        "Trade-off: ULIDs not returned synchronously, and recall called "
        "within ~1s of the write may miss the new events. Set False for "
        "synchronous semantics (testing/debugging).",
    ),
    # ------------------------------------------------------------------
    # Graph extractor backend (the thing that turns event text into
    # entities + edges). Default is fast offline regex; opt-in to spaCy
    # POS-filter / NER, or a local LLM CLI for cleaner concept nodes.
    # ------------------------------------------------------------------
    "graph.extractor": _Setting(
        str, "regex",
        "Entity-extraction backend. 'regex' = fast, offline, no deps "
        "(default). 'spacy' = POS-filter + NER (needs spacy + a model). "
        "'llm:claude' = Claude Code CLI extracts concepts at write time. "
        "'llm:ollama' = local Ollama model. 'llm:codex' = OpenAI Codex CLI. "
        "LLM backends give the cleanest 'knowledge graph' but add a CLI "
        "round-trip per write; off by default to preserve the no-LLM hot path.",
        choices=["regex", "spacy", "llm:claude", "llm:ollama", "llm:codex"],
    ),
    "graph.llm_max_concepts": _Setting(
        int, 5,
        "Max concept entities the LLM extractor may return per event. "
        "Higher = denser graph, longer prompt. Only used when "
        "graph.extractor starts with 'llm:'.",
        min=1, max=20,
    ),
    "graph.llm_timeout_s": _Setting(
        float, 30.0,
        "Per-event timeout (seconds) for the LLM extractor CLI. On timeout "
        "we silently fall back to the regex extractor — never block a write.",
        min=3.0, max=300.0,
    ),
    "graph.async_llm": _Setting(
        bool, True,
        "When the extractor is an LLM backend (graph.extractor starts with "
        "'llm:'), run entity extraction in a BACKGROUND worker so the write "
        "returns immediately instead of blocking on the CLI round-trip "
        "(up to graph.llm_timeout_s PER event). The graph becomes "
        "eventually-consistent; `pmb regraph` rebuilds it if the process "
        "dies before the worker drains. Regex/spaCy backends are fast and "
        "always run inline regardless of this flag. Set False only for "
        "deterministic tests that need the graph populated synchronously.",
    ),
    "graph.llm_model": _Setting(
        str, "haiku",
        "Model identifier passed to the LLM CLI. For graph.extractor=llm:claude "
        "use 'haiku' (cheap+fast, default), 'sonnet' (better quality), or a "
        "full Anthropic model id. For llm:ollama use the local model name like "
        "'qwen2.5:3b'. For llm:codex leave empty.",
    ),
    "graph.viz_min_mentions": _Setting(
        int, 1,
        "In the dashboard memory graph, hide entities with fewer than N "
        "mentions. Set to 2 to skip one-off concepts; 1 (default) shows all.",
        min=1, max=20,
    ),
}


# ----------------------------------------------------------------------
# Conversion + validation
# ----------------------------------------------------------------------


def _coerce(value: Any, setting: _Setting) -> Any:
    """Convert YAML/CLI string into the right Python type, validate range/choices."""
    t = setting.type
    if value is None:
        return setting.default
    # Booleans from "true"/"false"/"1"/"0"
    if t is bool:
        if isinstance(value, bool):
            v = value
        elif isinstance(value, (int, float)):
            v = bool(value)
        else:
            s = str(value).strip().lower()
            if s in ("true", "1", "yes", "on"):
                v = True
            elif s in ("false", "0", "no", "off"):
                v = False
            else:
                raise ValueError(f"expected boolean, got {value!r}")
        return v
    if t is int:
        v = int(value)
    elif t is float:
        v = float(value)
    elif t is str:
        v = str(value)
    else:
        v = value
    if setting.choices and v not in setting.choices:
        raise ValueError(f"value {v!r} not in {setting.choices}")
    if setting.min is not None and v < setting.min:
        raise ValueError(f"value {v} below min {setting.min}")
    if setting.max is not None and v > setting.max:
        raise ValueError(f"value {v} above max {setting.max}")
    return v


# ----------------------------------------------------------------------
# YAML <-> flat dict helpers
# ----------------------------------------------------------------------


def _flatten(nested: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in nested.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, path))
        else:
            out[path] = v
    return out


def _unflatten(flat: dict[str, Any]) -> dict:
    out: dict = {}
    for key, value in flat.items():
        parts = key.split(".")
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
    return out


# ----------------------------------------------------------------------
# Config holder
# ----------------------------------------------------------------------


class Config:
    """
    Layered config. Reads global + workspace YAMLs once, applies overrides
    on top. `get(key)` always returns a validated, typed value.
    """

    def __init__(
        self,
        workspace_dir: Optional[Path] = None,
        pmb_home: Optional[Path] = None,
        overrides: Optional[dict[str, Any]] = None,
    ):
        self.workspace_dir = workspace_dir
        self.pmb_home = pmb_home
        self._global = self._load(self.global_path) if pmb_home else {}
        self._workspace = self._load(self.workspace_path) if workspace_dir else {}
        self._overrides = dict(overrides or {})

    # -- paths --
    @property
    def global_path(self) -> Path:
        assert self.pmb_home is not None
        return self.pmb_home / "config.yaml"

    @property
    def workspace_path(self) -> Path:
        assert self.workspace_dir is not None
        return self.workspace_dir / "config.yaml"

    # -- I/O --
    @staticmethod
    def _load(p: Path) -> dict[str, Any]:
        if not p.exists():
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                return {}
            return _flatten(data)
        except Exception:
            return {}

    @staticmethod
    def _save(p: Path, flat: dict[str, Any]) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(_unflatten(flat), f, sort_keys=False, allow_unicode=True)

    # -- lookup --
    def get(self, key: str) -> Any:
        if key not in SCHEMA:
            raise KeyError(f"unknown config key: {key!r}")
        setting = SCHEMA[key]
        for source in (self._overrides, self._workspace, self._global):
            if key in source and source[key] is not None and source[key] != "":
                try:
                    return _coerce(source[key], setting)
                except Exception:
                    continue
        return setting.default

    def effective(self) -> dict[str, Any]:
        """All keys with resolved values."""
        return {k: self.get(k) for k in SCHEMA}

    def source_of(self, key: str) -> str:
        """Where the current value comes from: override|workspace|global|default."""
        if key in self._overrides:
            return "override"
        if key in self._workspace:
            return "workspace"
        if key in self._global:
            return "global"
        return "default"

    # -- mutation --
    def set_workspace(self, key: str, value: Any) -> Any:
        """Set in the per-workspace file. Returns the typed value stored."""
        if key not in SCHEMA:
            raise KeyError(f"unknown config key: {key!r}")
        typed = _coerce(value, SCHEMA[key])
        self._workspace[key] = typed
        if self.workspace_dir:
            self._save(self.workspace_path, self._workspace)
        return typed

    def set_global(self, key: str, value: Any) -> Any:
        if key not in SCHEMA:
            raise KeyError(f"unknown config key: {key!r}")
        typed = _coerce(value, SCHEMA[key])
        self._global[key] = typed
        if self.pmb_home:
            self._save(self.global_path, self._global)
        return typed

    def reset_workspace(self, key: Optional[str] = None) -> None:
        """Remove key (or all keys) from the per-workspace file."""
        if key is None:
            self._workspace.clear()
        elif key in self._workspace:
            del self._workspace[key]
        if self.workspace_dir:
            self._save(self.workspace_path, self._workspace)
