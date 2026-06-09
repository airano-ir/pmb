# Changelog

All notable changes to PMB are documented here.

## [Unreleased]

### Fixed

- **No personal-name or test-name literals leak into ranking.** The recall
  identity-marker boost no longer hardcodes a name — it is driven by the mined
  user-name cache, and the router's identity-intent detection matches the
  user's OWN learned names dynamically instead of a baked-in literal.
  `DEFAULT_NAMED_ENTITIES` (which leaked benchmark names like alice/stripe/adyen
  into every production query) is now empty; real entities come from the dynamic
  proper-noun extractor + the user-name cache.
- **Negation / "unknown" facts no longer linger as stale noise.** Once a
  positive keyed value exists (e.g. `user::city = Tampa`), older facts that
  negate or mark-unknown that attribute ("the user does not currently live in
  Warsaw; current city is unknown") are archived (`superseded_by`,
  archive-only). Runs at write time and as a `pmb repair-keyed` pass. Gated by
  `keyed.archive_obsolete_negations` (default ON). Lessons and pinned events are
  never touched.
- **The CLI is English-only.** Russian docstrings/examples that leaked into
  `--help` are translated; functional multilingual trigger templates
  (`pmb connect`) are unchanged.
- **Recall never crashes on an empty/degenerate BM25 corpus.** `rank_bm25`
  divides by zero when the corpus has no terms (a fresh-workspace recall before
  the embed/index queue drained); search now guards it and falls back to
  vector-only ranking instead of raising. Also stabilized 4 long-flaky tests
  (temporal proximity, rehearse cap, auto-consolidate, MCP e2e) so the full
  suite is deterministic.

### Added

- **Status dashboard.** Bare `pmb` (no subcommand) prints a workspace status
  panel — active workspace + how it resolved, storage sizes, event counts,
  running MCP servers, embedding warm/cold. `pmb --help` is unchanged. Slow
  paths (`recall`/`remember` first run, `index`, `migrate-workspaces`,
  `compact`, `declutter`) show a loading spinner.
- **Workspace switching.** `pmb workspace use <name>` persists a default
  workspace (resolution: env → project `.pmb/workspace.yaml` → saved default →
  git/cwd), `pmb workspace current` shows the active one + which rule won, and
  `pmb workspaces` marks the active one. Fully additive — setups that never run
  `use` resolve exactly as before.
- **Time-based forgetting.** `pmb decay --archive-cold` archives facts/activities
  that are old AND never recalled AND low-value (never pinned/keyed/lessons/
  goals). Dry-run by default; config `decay.archive_cold_days` (90),
  `decay.archive_cold_max_importance` (0.25).
- **Junk sweeper.** `pmb declutter` archives test artifacts, near-empty/stopword
  content, exact duplicates, and obsoleted negation tombstones; optional `--llm`
  judge (bounded, capped, circuit-broken) reviews borderline low-value facts.
  Dry-run + archive-only.
- **Plans become goals.** A future-intent fact ("remember we'll do X next") is
  routed to a goal: MCP docstrings + the `pmb connect` template carry the rule,
  `record_batch` accepts `{"type": "plan"}` (a goal with `kind=plan`), and
  `record_fact` flags forward-looking phrasing with `metadata.suggest_goal`
  (a hint — never an auto-convert). New `pmb goals` / `pmb goals done <ulid>`.
- **Offline LLM keyed-state tier.** During consolidation, an offline bounded LLM
  proposes keyed current-state (`{attribute, value, negation, confidence}`) for
  facts the cheap regex missed; confidence≥0.8 positives upsert via the
  canonical keyed path, weaker ones are tagged `metadata.suggested_key`. Gated by
  `consolidate.suggest_keyed`; never on the recall hot path.
- **Per-deployment reference data.** Optional `PMB_HOME/reference.yaml` extends
  attribute aliases / known techs / stopwords / function-words (extend-only) and
  overrides kind priorities — no Python edits. Missing file = identical
  behaviour.

### Changed

- **Write-time quality gate (opt-in, `write.quality_gate`, default OFF).** When
  on, suspected-junk facts are flagged (`quality_flag=suspect_junk`) and capped
  at importance 0.2 and excluded from keyed promotion — never rejected.

## [0.5.0]

### Fixed

- **recall_smart no longer hangs the interactive path.** It is bounded by an
  overall wall-clock deadline (`recall.smart_deadline_ms`, default 15s) with a
  local-only fast path — it never resolves an LLM / spawns the Claude CLI on the
  foreground path (the cause of 120s timeouts). LLM query-decomposition is opt-in
  (`recall.smart_allow_llm`) and, when on, is clamped to the remaining budget. A
  new explicit `recall_deep` tool/method runs the slow, thorough pass on demand.
  Each pack carries an `escalation` field (stages run + why it stopped) so callers
  don't fan out redundant recalls.
- **Stale personal attributes no longer out-rank the current value.** Keyed-fact
  attribute names are canonicalized (`city` / `current_city` / `current_city_2026`
  / `lives_in` / `город` → one key), so an update supersedes the old value instead
  of creating a competing key. Old values are kept as history.
- **Current-state facts become keyed facts.** A plain "I now live in X" / "сейчас
  живу в X" is detected (conservatively, with a negation guard so a "do not say X"
  instruction is never promoted) and upserts the matching keyed attribute.

### Added

- **`pmb repair-keyed`** — two-pass keyed-fact repair: promote current-state facts
  buried in plain text into keyed facts, then collapse alias/duplicate keys onto
  one canonical value. Archive-only, dry-run by default.
- **`pmb migrate-workspaces`** — merge a per-project workspace into a unified
  memory, tagged `project=<name>`; the source is left intact (reversible). The
  `recall` tool gains an optional `project` filter over the one memory.
- **`pmb mcp status`** + a running-server registry — see how many MCP servers are
  live and their memory; an HTTP `pmb mcp serve` refuses to start a second instance
  on a live host:port (per-session servers each load the model + LanceDB).
- **Backend circuit breaker** — a repeatedly-failing/slow deep backend is
  temporarily disabled for the interactive path (`recall.breaker_threshold`,
  `recall.breaker_cooldown_s`); state exposed via `breaker_status`.
- **Performance dashboard** now records & shows per-call recall_smart stages,
  client-timeout (vs server completion), backend, and cache hit/miss.

### Changed

- New (additive) config keys: `recall.smart_deadline_ms`, `recall.smart_allow_llm`,
  `keyed.auto_detect_current_state`, `recall.breaker_threshold`,
  `recall.breaker_cooldown_s`. Defaults preserve prior behaviour.
- MCP tool profiles: default now 30 tools, lean 26 (added `recall_deep`).

## [0.4.0]

### Added

- **Ambient memory (auto-write).** PMB journals the agent's work even when it
  forgets to call `record_batch`. A PostToolUse hook logs significant actions
  (`pmb track-action`); a Stop hook synthesizes one activity entry per turn
  *only if* the agent didn't record its own (`pmb autowrite`), gated by a
  git-free, outcome-based importance score (tests passed / failure fixed /
  deploy ran — not raw file count). Every entry is tagged `source=autowrite`,
  shown as auto, and removable with `pmb forget-auto`. Works on Claude Code
  (hooks), OpenAI Codex (rollout parse, `pmb codex-notify`), and MCP-only hosts
  (git observer, `pmb ambient-watch`). `pmb hooks capabilities` reports
  per-agent support. ON by default (`autowrite.enabled`).
- **`lean` MCP tool profile** (25 tools): the default set minus the read-status
  browse tools a host hook already covers, set automatically by
  `pmb connect claude-code` so the agent isn't offered slow MCP versions of what
  auto-recall / session-restore already inject for free.
- **Lesson follow-through `not_applicable` state.** The Stop-hook followcheck
  classifies a surfaced lesson with zero overlap with the turn's work as
  not-applicable and excludes it from the adherence denominator, instead of
  counting it as a phantom "not followed". Follow-rate is now over *applicable*
  surfaces — it reflects relevant lessons, not surfacing volume.
- **Opt-in semantic lesson tier** (`recall.lesson_semantic`, experimental, off
  by default): cosine over the existing embeddings to catch paraphrase /
  cross-lingual lesson matches the lexical gate can't.

### Changed

- **Lesson-surfacing precision.** Surfacing uses a shared tokenizer + stopword
  set (`pmb.core.text_match`, symmetric with followcheck): generic and
  filesystem-path noise is filtered, and a relevance gate
  (`recall.lesson_min_overlap`, default 1) replaces the old "any shared 3-char
  word" match. Auto-recall also skips non-message blocks (task-notifications /
  system reminders). Materially cuts irrelevant surfacing on real workspaces.
- **`graph.async_llm`** (default on): LLM entity extraction runs in a background
  worker off the write hot path, so records return instantly; `pmb regraph` is
  the backstop.

### Fixed

- **`lean` profile was silently ignored** — the post-registration tool filter
  fell back to the full default set; it now selects the lean set correctly.
- **`Engine.close()` drains the async batch-write, deferred-graph, and touch
  queues** (each bounded) so in-flight work isn't dropped on shutdown.
- **Persistent WAL no longer forced on non-PMB databases** — the global
  `sqlite3.connect` pragma patch applies `journal_mode=WAL` only to PMB-owned
  DBs (under `PMB_HOME` / named `events.sqlite` / `:memory:`); third-party
  connections in the same process get only the ephemeral pragmas.
- Hyphenated compound words (`lesson-surfacing`) are matched by their parts.
- `SCHEMA_VERSION` bumped to 6 to match the shipped `lesson_surfaces` table.

## [0.3.0]

### Added

- Auto-recall, session-restore, and lesson follow-through lifecycle hooks;
  optional HTTP transport with bearer-token auth for shared/team workspaces.

## [0.2.1]

### Added

- **`pmb learn "..."` + `pmb lessons`** - procedural memory: teach PMB durable
  lessons ("this repo uses pnpm, never npm") that surface via the hybrid +
  predicate-aware ranker, and review them.
- **`pmb learn --failed` / negative memory** - record failures ("numpy 2.x
  broke lancedb") that recall flags with a warning so they're not repeated.
- **`pmb distill` + auto-distill on session end** - LLM extracts durable
  lessons/failures from a session's events automatically (zero-command when
  `lessons.auto_distill_on_session_end` is on). Off the recall hot path.
- **Lesson-intent boost** - on how-to/convention queries, lesson & failure
  memories are gently boosted so the agent applies them. Scoped to
  lesson/failure events only, so it cannot affect recall on datasets without
  them (LoCoMo stays exactly 94.5%). Config: `recall.lesson_boost`.
- **Trust signals in recall** - source attribution, source-derived confidence
  (high/med/low), and a freshness/staleness flag, all display-only.
- **`pmb audit` memory-health** - counts of lessons, failures, stale, low-
  confidence, and conflicting memories.
- **`pmb note "..."`** - instant memory capture from the terminal, no agent.
- **`pmb audit`** - "what does PMB know about me?": a grouped, read-only view
  of everything stored, by type and by source.
- **Source attribution on recall** - every hit shows where it came from
  ("from: chatgpt · Project planning", "from: note (cli)", etc.). Trust feature.
- **`pmb watch <file|dir>`** - auto-capture: new paragraphs in a notes file or
  folder (e.g. `~/journal.md`) get ingested as memory; content-hash dedup.
- **6 more agent integrations (9 total).** `pmb connect` now wires
  windsurf, gemini, vscode, zed, opencode and continue in addition to
  claude-code / cursor / codex. `pmb connect --list` shows every agent and
  its config path; `--config-path` overrides the location.
- **Git-backed workspace sync.** `pmb workspace init|push|pull|status|clone`
  versions and syncs a workspace to any git remote - cross-device, team
  memory, and backups with no server.
- **Encrypted workspace bundles.** `pmb workspace export|import` packs a
  workspace into a single authenticated-encrypted file (scrypt + AES/HMAC),
  safe to store even on a public remote. Needs `pip install 'pmb-ai[crypto]'`.
- **Memory import.** `pmb import chatgpt|claude|mem0|markdown <path>` brings
  existing history into a fresh workspace; the entity graph rebuilds after.
- **`pmb why "<query>"`.** Explains recall ranking with a full trace of which
  PAMVR rules fired and each multiplier - no more black box.
- **Pluggable embedders.** `embedding.backend` now also accepts `ollama` and
  `openai`. A dimension guard refuses to mix embedders of different vector
  sizes in one workspace (which would corrupt recall).
- **`scripts/benchmarks/vs_mem0.py`.** Reproducible head-to-head: same data,
  same queries, same scorer. PMB measured live; mem0 measured with
  `--with-mem0` or shown from published numbers (clearly labelled).

### Fixed

- PyPI page now renders the README logo + screenshots (absolute image URLs).
- LanceDB table is created with the active embedder's real vector dimension
  instead of a hardcoded 384, enabling non-default embedders on fresh
  workspaces.

### Notes

- 57 new regression tests. The headline 94.5% LoCoMo recall@10 and 70ms p50
  are unchanged - the hardening suite verifies no recall regression.

## [0.2.5]

### Onboarding, agent logging, cross-platform & docs

- **`pmb setup`** - guided first-time setup: detects your agent, asks active vs
  conservative logging, and wires PMB in one go (`--yes` for non-interactive).
- **`pmb connect <agent> --active`** - proactive-logging rules: the agent records
  its own decisions / done / lessons / failures / goals during coding, not just
  on "remember". Recall stays lazy; the conservative default is unchanged.
  - **Pro config:** `agent.log_decisions` / `.log_completed` / `.log_lessons` /
    `.log_failures` / `.log_goals` toggle exactly what gets logged, and
    `agent.apply_lessons` enables the **self-improvement loop** - recall + apply
    past lessons/failures before a task so the agent gets better at the project
    over time. Set via `pmb config set` / `pmb tune`, then re-run
    `pmb connect <agent> --active` to regenerate the rules.
- **`pmb overview "<topic>"` + MCP `overview` tool** - a structured "what do I
  know about X?" synthesis (key facts & decisions, lessons, failures, goals, a
  timeline, related topics) from memory, no LLM, fully local. The MCP tool lets
  an agent get up to speed on a project/feature in ONE call instead of several
  recalls. Config: `overview.max_events`.
- **`agent.active_mode`** - auto-logging switch: when on, `pmb connect` /
  `pmb setup` install the proactive rules by default (no `--active` needed).
- **Session continuity (`pmb session brief` + MCP `session_brief`)** - a digest
  of what was decided / done / learned **this session**, so an agent can
  re-orient after its OWN context window compacts in a long session instead of
  re-asking the user what it already did. Active-mode rules tell the agent to
  use it (`agent.context_continuity`); fallback window `session.brief_minutes`.
- **`docs/COMMANDS.md`** - full command reference grouped by task, with examples
  and a clear mark of which commands run fully offline vs need an LLM backend.
- **macOS in CI** - a macOS (arm64) job added to the test matrix so the
  "runs on macOS" claim is tested, not just asserted.
- New tests: `tests/test_connect_active.py` (10: active rules + agent detection).

### MCP correctness: new tools exposed + `session_brief` fixes

End-to-end MCP tests (`tests/test_mcp_e2e.py`, via the fastmcp in-memory client)
drive the real tool routing for the new features on a long-chat scenario, and
caught three bugs that made `session_brief` / `overview` unusable from an agent:

- **`session_brief` and `overview` are now actually exposed.** The post-
  registration tool-profile filter (minimal / default whitelists) was silently
  dropping both new tools, so an agent that called them got *"Unknown tool"*.
  Added `session_brief` to the minimal profile and `overview` to the default.
- **`session_brief` now covers the whole session, not just activities.** Only
  `record_activity` auto-binds an event to the session; facts / goals / lessons
  do not, so the old tag-only scope dropped them once a session existed. It now
  scopes as a union: tagged with the session **or** recorded since the session
  began. (This is why a long chat's lessons/goals went missing from the brief.)
- **`session_brief` classifies decisions / done correctly.** It keyed off
  `metadata.kind`, but `record_activity` stores the kind under `activity_kind`,
  so every decision / completed item fell into "other". It now reads both.
- **Tool-profile filter is event-loop-safe.** It used `asyncio.run(list_tools())`,
  which raised (and left an un-awaited coroutine) when the server was built
  inside a running loop (in-memory client / embedded host). It now detects a
  running loop and skips the introspection cleanly; the stdio server path —
  where gating actually matters — is unchanged.

Tests: `tests/test_mcp_e2e.py` (3: tool exposure, long-chat `session_brief`,
recall answer-quality + lessons + `overview`).

### Answer-ready recall output + temporal validity windows

Three product-code levers targeting end-to-end answer quality (the gap between
strong retrieval and the final answer), informed by per-question J-score
failure analysis. All additive / out of the recall ranking hot path - retrieval
recall@10 and latency are unchanged.

- **Resolved date in every recall result.** `RecallResult.to_dict()` now
  includes a human-readable `date`, resolved as event_time (the date the
  content refers to) -> session date -> creation time. An agent can answer
  "when …?" without epoch math and anchors relative dates to the EVENT, not to
  "today". `to_text()` (prompt injection) uses it too.
- **More write-time atomic-fact patterns** (no-LLM regex): relationship status
  ("X is single") and origin ("X moved from Y"). Still opt-in via
  `consolidate.write_atomic_facts` (default off); turning it on by default is
  gated on a LoCoMo regression run.
- **Temporal validity windows + as-of query.** `record_keyed_fact` now stamps
  `valid_from` on the new value and `valid_to` on each superseded value, and
  `Engine.keyed_fact_as_of(subject, attribute, at_time)` returns the value that
  was current at a past time (Zep-style "what was true in March") - prior
  values stay queryable instead of just being archived.

Tests: `tests/test_jscore_levers.py` (10).

## [0.2.4]

### LLM-as-judge (J-score) eval harness improvements

The `--judge` mode of `scripts/benchmarks/benchmark_locomo.py` reports a
J-score (LLM-as-judge) - the end-to-end metric mem0 / Zep / Letta publish,
where a reader LLM answers from retrieved context and a judge LLM grades the
answer against gold. This release makes that harness fair and debuggable:

- **Full reader context.** Per-chunk cap raised 1500 -> 6000 and total
  6000 -> 20000. Previously the answer turn could be truncated out of the
  reader's view, capping the J-score far below retrieval recall@10.
- **Reader prompt** now scans every numbered chunk (not just the first) and
  resolves relative dates step by step.
- **Per-question failure logging** (question / gold / prediction / verdict)
  in the output JSON, so a miss can be attributed to reader vs retrieval vs
  judge.

EVAL-harness only - no product / recall code is touched, so recall@10 and
latency are unchanged. The J figure is still measured on small samples; a
full multi-conversation run (with a fast LLM backend) is needed for a
publishable number.

### Local-use & own-your-data commands

New CLI surface for using PMB as a personal memory you fully own and organize
offline. All of these are CLI + display + write-layer only - **none touch the
recall hot path**, so the 94.5% LoCoMo recall@10 and ~70ms p50 are unchanged
(verified: full hardening regression suite green).

- **`pmb timeline`** - chronological, day-grouped view of your memory
  (`--days`, `--type`, `--newest-first`).
- **`pmb insights`** - personal analytics: totals, type breakdown, growth per
  week, top topics (entity graph), and lessons/failures/goals/pinned counts.
- **`pmb digest [today|week|month]`** - quick recap of recent memories
  (`--days N`).
- **`pmb export [--format markdown|json] [--out FILE]`** - dump all memory to
  readable Markdown or JSON (`--include-archived`). Plain/unencrypted; for an
  encrypted portable bundle use `pmb workspace export`.
- **`pmb forget-topic <topic>`** - archive every memory about a topic in one
  command (`--dry-run`, `--yes`, `--in content|tag|source`). Reversible
  (archived, not deleted).
- **TTL / expiry** - `pmb ttl <ulid> 30d` (or `clear`), a `--ttl` option on
  `note` / `learn` / `fact`, and `pmb prune-expired` to sweep. Enforced only by
  the explicit sweep, never inside recall.
- **Tags / collections** - `pmb tag`, `pmb untag`, `pmb tags`, and
  `pmb tagged <tag>` for local organization.
- **`pmb reminders`** - surfaces open goals that are overdue or due soon
  (`--within N`, `--all`).
- **`pmb snapshot create|list|restore`** - local, offline, timestamped
  workspace snapshots (WAL-checkpointed copy; restore auto-backs-up current
  state first).

New store helpers: `EventStore.set_metadata` (annotate an event without
touching its content or embeddings) and `EventStore.list_all` (export /
analytics, optionally including archived rows). 34 new tests in
`tests/test_local_features.py`.

### Hardening pass 2 - lazy LanceDB + use-case clarity

- **`pmb stats` and friends now run in ~1 s (was ~14 s).** The single biggest user-facing cost was `import lancedb` itself: on Windows it pulls in pyarrow + torch and takes ~22 s to import. `HybridSearch.__init__` triggered that import for every CLI command, even ones that never touch vectors. Fix: `HybridSearch` defers `lancedb.connect(...)` and the table-open to a lazy `_table` property; `Engine.__init__` no longer eagerly calls `reload_bm25()`. The cost is now paid on the first `recall()` or write that actually needs vector search.
- **README "What gets stored, when" gets a `Which features help which use case` table.** Honest mapping of feature flags to workloads (multi-hop, narrative, temporal, code, multilingual, etc.) - so contributors don't assume every feature must boost LoCoMo recall.
- **All 88 core tests + 9 smoke tests still pass.**

### Hardening pass 1 - import hygiene + test foundation

- **Lazy top-level imports.** `import pmb` no longer pulls in LanceDB, sentence-transformers, numpy, rank_bm25, fastmcp, yaml, torch or transformers. Heavy attributes (`Engine`, `Workspace`, `detect_workspace`) are exposed via PEP 562 `__getattr__` and loaded on first access. Measured: bare `import pmb` is now **2.4 ms** (was ~14 s when the engine was eagerly imported). Same treatment applied to `pmb.core`.
- **`tests/conftest.py`** added with shared fixtures (`tmp_pmb_home`, `tmp_workspace_dir`, `isolated_engine`) and a `sys.path` fallback so new tests don't need the boilerplate.
- **`tests/test_lightweight_imports.py`** added - 9 smoke tests that spawn fresh interpreters and assert which heavy modules are loaded after specific imports. If a future change re-introduces an eager heavy import at `pmb/__init__.py` it will fail this suite.
- **No behaviour change.** All 88 core tests (the `make test` set used in CI) still pass; `from pmb import Engine` still works.

### Recall ablation findings + default changes

Ran a full-system ablation across 19 retrieval components on LoCoMo conv-26/30/41 (see `scripts/benchmarks/ablation_full.py`). Key findings:

- **`recall.typo_correction = True` was hurting recall by ~6.2 points.** The Levenshtein-≤2 fuzzy rewriter was "correcting" correctly-spelled tokens into similar-looking but wrong entity names. **Default flipped to `False`.** Existing per-workspace overrides are unaffected.
- **BM25-heavy fusion outperformed the symmetric default.** `recall.bm25_weight` raised `0.5 -> 0.7`. Vector-only retrieval loses 18 points on this dataset; the embedding channel adds only marginal signal.
- **Net effect on the full 10-conversation LoCoMo benchmark: 91.6 % -> 94.1 % evidence-recall@10** (+2.5 pp, every conversation improved, range now 91.2-96.2 %).
- **Twelve of nineteen ablated features show 0.000 delta on LoCoMo:** tiers, causation walk, arc expansion, collapse-reflections, reflection-to-edges, predictive cache, code-AST, LRU cache, multi-entity bonus, PPR, adaptive routing, spreading activation, temporal. These are not removed - some are designed for long-term dynamics that LoCoMo doesn't probe - but the README and docs no longer claim they carry the benchmark number.
- **Cross-encoder reranker (`recall.rerank = True`) regresses 17 points and adds 845 ms p50.** Recommendation removed from default setup; flag is still available as experimental.

See `scripts/benchmarks/ablation_full.py` and `docs/HARDENING_NOTES.md` for the raw data and methodology.

## [0.1.0] - Initial public release

### Highlights

- Single-file install: `pip install -e .` exposes `pmb` on `$PATH`.
- 91.6 % evidence-recall@10 on LoCoMo full 10-conversation run (vs published mem0 / Letta / Zep numbers of 70-80 %).
- 90-140 ms p50 recall latency, 2 ms async writes.
- 13 semantic layers, 5 access paths, 3 tiers - every layer optional and configurable.
- Web dashboard + 5-tab terminal TUI for inspection.
- MCP server for Claude Code, Codex CLI, Cursor.
- Optional Ollama integration for fully-local LLM ops (consolidation, dedup verify, pmb-chat).

### Architecture decisions worth knowing

- **Local-first.** SQLite + LanceDB on disk; no daemon, no service, no network.
- **Lazy-by-default agent prompt.** PMB is OFF until an explicit trigger ("remember", personal fact, "what did I…"). General Q&A bypasses PMB entirely.
- **Async writes (fire-and-forget MCP).** `record_batch` returns in ~2 ms; embedding + LanceDB indexing happen in a background thread.
- **BM25 fallback for cold reads.** First `recall` after process start returns text-match results in ~100 ms while the sentence-transformers model finishes loading.
- **Multi-layer dedup.** L1 exact (5 ms) + L2 cosine ≥ 0.92 (50 ms) at write time; L2.5 borderline queue for LLM-verified merges; L3 dashboard review tab for ambiguous pairs.
- **Conservative defaults.** False-merge risk minimised: never merges across event types, threshold 0.92 leaves a safe gap.

### Tools and surfaces

- `pmb tui` - five tabs: Memory, Recall, Stats, Dedup, Tune
- `pmb dashboard` - HTTP dashboard on :8765 with Graph, Events, Performance, Duplicates, Recall Debug
- `pmb tune` - settings-only TUI (67 knobs)
- `pmb ollama` - health, install/use models, smoke test
- `pmb connect codex | claude | cursor` - wires MCP into the agent's config and rules file
- `pmb dedupe` / `pmb regraph` / `pmb prune-graph` - maintenance ops

### Settings (67 across 9 categories)

`recall` (36), `consolidate` (9), `dedup` (6), `chat` (5), `decay` (3), `embedding` (3), `feedback` (2), `ollama` (2), `mcp` (1). All exposed in TUI Tune tab, CLI `pmb config`, and dashboard.

### Known limits

- Single-machine. No cross-device sync.
- Cold first recall after process start blocks on model load if the BM25 fallback path is disabled.
- Code-AST extraction is Python-only (regex fallback for broken / partial code).
- Dashboard and TUI assume one workspace at a time.
