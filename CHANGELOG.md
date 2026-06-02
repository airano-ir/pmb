# Changelog

All notable changes to PMB are documented here.

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

## [Unreleased]

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
