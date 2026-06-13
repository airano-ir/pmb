# PMB command reference

Every command supports `--help` for its full flag list:

```bash
pmb <command> --help          # e.g. pmb recall --help
pmb connect --help
```

**Legend:** 🟢 fully offline (no network) · 🧠 needs an LLM backend
(Claude CLI / `ANTHROPIC_API_KEY` / Ollama - see [LLM-powered commands](#llm-powered-commands)).

Nothing leaves your machine except the workspace-sync commands you run on purpose.

---

## Setup & connect

| Command | What it does |
|---|---|
| `pmb connect <agent>` 🟢 | Wire PMB into an agent's MCP config + install its rules. Agents: `claude-code`, `cursor`, `codex`, `windsurf`, `gemini`, `vscode`, `zed`, `opencode`, `continue`. |
| `pmb connect <agent> --active` 🟢 | Same, but installs **proactive-logging** rules: the agent records its own decisions / lessons / what it did during coding, without waiting for "remember". Recall stays lazy. |
| `pmb connect --list` 🟢 | Show every supported agent and where its config lives. |
| `pmb connect <agent> --workspace NAME` 🟢 | Point several agents at **one shared** workspace. |
| `pmb connect <agent> --probe` 🟢 | After wiring, spawn `pmb-mcp` briefly to confirm it starts. |
| `pmb doctor` 🟢 | Diagnose the install + runtime state. `--remote user@host:/path` prints an SSH-tunneled MCP snippet. |
| `pmb warmup` 🟢 | Pre-load the model + BM25 + LanceDB so the next recall is fast (avoids the ~1-2 s cold start). |
| `pmb init [--name NAME]` 🟢 | Initialize a workspace in the current directory (optional - a workspace auto-detects from cwd). |

```bash
pip install pmb-ai
pmb connect claude-code            # wire Claude Code (conservative rules)
pmb connect codex --active         # wire Codex, agent logs its own work
# restart the agent
```

---

## Capture (write memory)

| Command | What it does |
|---|---|
| `pmb note "..."` 🟢 | Jot a memory from the terminal. `--pin` to keep it forever, `--ttl 30d` to auto-expire. |
| `pmb learn "..."` 🟢 | Record a durable **lesson** ("this repo uses pnpm, never npm"). `--failed` records a failure to avoid repeating. |
| `pmb fact "..."` 🟢 | Record a standalone fact. `--ttl 30d` optional. |
| `pmb remember "query" "response"` 🟢 | Store a Q/A pair. |
| `pmb import <source> <path>` 🟢 | Import existing history: `chatgpt`, `claude`, `mem0`, `markdown` (Obsidian vault). Rebuilds the graph after. |
| `pmb watch <file\|dir>` 🟢 | Auto-capture new paragraphs from a notes file/folder. `--once` for cron. |

```bash
pmb note "decided to use Postgres for JSONB" --pin
pmb learn "always run make fmt before committing"
pmb import chatgpt ~/Downloads/conversations.json
```

---

## Recall & explore (read memory)

| Command | What it does |
|---|---|
| `pmb recall "query" [-k 5]` 🟢 | Search memory. Each hit shows source, confidence, freshness, ★lesson / ⚠failure markers. |
| `pmb why "query"` 🟢 | Explain the ranking - which of the 14 PAMVR rules fired and each multiplier. |
| `pmb overview "<topic>"` 🟢 | Structured "what do I know about X?" - key facts & decisions, lessons, failures, goals, timeline, related topics. Also an **MCP tool** so the agent can get up to speed on a topic in one call. |
| `pmb timeline` 🟢 | Chronological, day-grouped view (`--days`, `--type`, `--newest-first`). |
| `pmb insights` 🟢 | Analytics: totals, type breakdown, weekly growth, top topics, lessons/goals counts. |
| `pmb digest [today\|week\|month]` 🟢 | Recap of recent memories (`--days N`). |
| `pmb audit` 🟢 | "What does PMB know about me?" - grouped, read-only view + memory-health signals. |
| `pmb lessons` 🟢 | List the durable lessons & failures (procedural memory). |
| `pmb reminders` 🟢 | Goals that are overdue or due soon (`--within N`, `--all`). |
| `pmb tags` / `pmb tagged <tag>` 🟢 | List tags / list memories in a tag "collection". |
| `pmb list [-n 20] [--type T]` 🟢 | Recent events. |
| `pmb stats` 🟢 | Workspace stats. |
| `pmb graph stats\|top\|neighbors\|rebuild` 🟢 | Inspect the entity/association graph. |

```bash
pmb recall "where do I live now" -k 5
pmb why "what port did we choose"
pmb timeline --days 7
pmb insights
```

---

## Own-your-data & lifecycle

| Command | What it does |
|---|---|
| `pmb export [--format markdown\|json] [--out FILE]` 🟢 | Dump all memory to readable text (`--include-archived`). Plain/unencrypted. |
| `pmb tag <ulid> <tags...>` / `pmb untag` 🟢 | Tag / untag a memory for local organization. |
| `pmb ttl <ulid> 30d\|clear` 🟢 | Set / clear an expiry on a memory. |
| `pmb prune-expired` 🟢 | Archive memories past their TTL (run from cron). |
| `pmb forget <ulid>` 🟢 | Archive one memory (reversible). |
| `pmb forget-topic <topic>` 🟢 | Archive everything about a topic in one command (`--dry-run`, `--yes`, `--in content\|tag\|source`). |
| `pmb pin <ulid>` 🟢 | Pin a memory (max importance, never auto-archived). |
| `pmb snapshot create\|list\|restore` 🟢 | Local, offline, timestamped workspace snapshots. |
| `pmb decay` 🟢 | Apply the forgetting curve (lowers importance, archives stale). |
| `pmb dedupe` 🟢/🧠 | Deduplicate. Offline cosine sweep by default; `--run-pending` uses an LLM for borderline pairs. |
| `pmb compact` 🟢 | Move old archived events to cold storage + VACUUM. |

```bash
pmb export --format json --out memory.json
pmb forget-topic project-x --dry-run
pmb snapshot create --note "before refactor"
```

---

## Sync & backup (only these touch the network - and only when you run them)

| Command | What it does |
|---|---|
| `pmb workspace init [--remote URL]` 🟢 | Turn the workspace into a git repo. |
| `pmb workspace push` / `pull` / `status` | Sync memory to/from a git remote (cross-device, team, backup). |
| `pmb workspace clone <url> <name>` | Clone a remote workspace locally. |
| `pmb workspace export <out> [--key-file]` 🟢 | Encrypt the workspace into one portable bundle (scrypt + AES/HMAC) - safe even on a public remote. Needs `pip install 'pmb-ai[crypto]'`. |
| `pmb workspace import <bundle> <name>` 🟢 | Decrypt a bundle into a local workspace. |
| `pmb workspaces` 🟢 | List all known workspaces. |

---

## Config & UI

| Command | What it does |
|---|---|
| `pmb config list` 🟢 | Every setting, its value, and where the value comes from (`--only-overridden`). |
| `pmb config get\|set\|reset <key> [value]` 🟢 | Read / change / reset a setting (workspace or `--global`). |
| `pmb tune` 🟢 | TUI to browse + edit all 67 settings live (needs `textual`). |
| `pmb tui` 🟢 | Full 5-tab terminal workspace (Memory / Recall / Stats / Dedup / Tune). |
| `pmb dashboard` 🟢 | Local web dashboard on `127.0.0.1:8765` (graph, events, perf, recall debugger). |

```bash
pmb config set recall.top_k 8
pmb config list --only-overridden
```

### Pro: tune what `--active` logs

`pmb connect --active` builds the agent's rules from these toggles (all default
on). Change them, then re-run `pmb connect <agent> --active` to regenerate.

| Key | Effect |
|---|---|
| `agent.active_mode` | When on, `pmb connect` / `pmb setup` install the proactive rules **by default** (no `--active` flag needed) |
| `agent.log_decisions` | Log design/code decisions |
| `agent.log_completed` | Log what was done (features / fixes) |
| `agent.log_lessons` | Log project conventions / corrections |
| `agent.log_failures` | Log failures (don't-repeat) |
| `agent.log_goals` | Log the user's goals / intents |
| `agent.apply_lessons` | **Self-improvement loop** - recall + apply past lessons/failures before a task, so the agent gets better at this project over time |
| `agent.context_continuity` | Tell the agent to call `session_brief` to re-orient after its own context compacts in a long session |

```bash
pmb config set agent.log_goals false       # don't log goals
pmb connect codex --active                 # regenerate rules
```

### Pro: choose the entity-graph extractor

The dashboard's memory graph and `recall`'s graph-boost step both lean on the
**entity extractor** - the thing that turns event text into nodes
(`file` / `tech` / `person` / `concept`). PMB ships three backends; swap one
at runtime, no code changes:

| Backend | What it does | Cost | Deps |
|---|---|---|---|
| `regex` (default) 🟢 | Fast file/tech regex + improved stop-list + multi-word phrase detection ("Claude Code" → one node). Fully offline. | ~0 ms | none |
| `spacy` 🟢 | Adds POS-filter (noun/proper-noun only) and real NER (PERSON / ORG / GPE / PRODUCT). Cleanest no-LLM option. | ~3-10 ms | `pip install spacy` + `python -m spacy download en_core_web_sm` |
| `llm:claude` 🧠 | One Claude Code CLI call per event - returns clean named-entity JSON. Same idea as graphify / Penpax. Falls back to regex on timeout / error. | ~1-3 s/event | `claude` CLI on PATH |
| `llm:ollama` 🧠 | Same, but via a local Ollama model (default `qwen2.5:3b`). Fully offline if you have a model pulled. | ~1-4 s/event | `ollama` CLI + a model |
| `llm:codex` 🧠 | OpenAI Codex CLI. | ~1-3 s/event | `codex` CLI on PATH |

```bash
# default - leave it at regex unless the noise bugs you
pmb config set graph.extractor regex

# nicer no-LLM extraction
pip install spacy && python -m spacy download en_core_web_sm
pmb config set graph.extractor spacy

# cleanest knowledge graph (LLM at write time)
pmb config set graph.extractor llm:claude       # uses your Claude Code login
pmb config set graph.llm_max_concepts 5
pmb config set graph.llm_timeout_s 30

# hide one-off noise in the dashboard graph (DB stays intact)
pmb config set graph.viz_min_mentions 2
```

LLM backends never block the write path: if the CLI times out or returns
malformed output, the record falls back to the regex extractor for that one
event. Recall still works exactly the same - the choice only changes WHICH
entities end up as graph nodes, not the recall pipeline.

---

## Ollama (optional fully-local LLM)

| Command | What it does |
|---|---|
| `pmb ollama status` 🟢 | Health check + list installed models. |
| `pmb ollama use <profile>` 🟢 | Select a model profile (tiny / balanced / quality). |
| `pmb ollama test` 🟢 | Smoke-test the local model. |

Ollama is **optional** - PMB works fully offline without any LLM. It's only used
by the LLM-powered commands below (and only if you choose it over Claude
CLI / Anthropic).

---

## LLM-powered commands

These are the **only** commands that need an LLM backend (Claude CLI in PATH /
`ANTHROPIC_API_KEY` / Ollama). They run **off the recall hot path** - recall
itself never calls an LLM.

| Command | What it does |
|---|---|
| `pmb consolidate` 🧠 | Sleep-stage consolidation: cluster related memories, extract one rule per cluster. |
| `pmb reflect` 🧠 | For each event, an LLM asks "why does this matter?" and stores searchable bridges for multi-hop queries. |
| `pmb distill` 🧠 | Extract durable lessons/failures from a session's events automatically. |
| `pmb arcs cluster` 🧠 | Cluster events into narrative threads ("Postgres adoption journey"). |
| `pmb dedupe --run-pending` 🧠 | Resolve borderline duplicate pairs via LLM. |

```bash
pmb consolidate --backend auto        # auto = Claude CLI > Anthropic > Ollama
pmb distill                           # turn a session into lessons
```

---

## Maintenance

| Command | What it does |
|---|---|
| `pmb regraph` 🟢 | Rebuild the entity/edge graph from active events. |
| `pmb prune-graph` 🟢 | Drop weak co-occurrence edges to keep recall fast on large workspaces. |
| `pmb reindex` 🟢 | Re-embed all events (after switching the embedding model). |
| `pmb rehearse` 🟢 | Spaced-repetition refresh of important-but-idle memories. |
| `pmb session start\|end\|current\|brief` 🟢 | Session management. `brief` = digest of what was decided/done/learned **this session** (re-orient after a long session or context loss). Also an **MCP tool** `session_brief` the agent calls when it loses the thread. End can auto-distill lessons if enabled. |
| `pmb sync [--days N]` 🟢 | Capture git commits into memory. |
| `pmb schedule` 🟢 | Print OS scheduler config (cron / Windows schtasks) for background jobs. |

---

## Hooks & ambient memory

Hooks force-feed PMB at the protocol level, so memory works without the
model remembering to call anything. `pmb hooks install claude-code` wires
all four lifecycle hooks; Codex and MCP-only hosts get the equivalent via
their own mechanisms (check `pmb hooks capabilities`).

| Command | What it does |
|---|---|
| `pmb hooks install <agent>` 🟢 | Wire the lifecycle hooks. Claude Code: UserPromptSubmit + PostToolUse + SessionStart + Stop. Codex: a `notify` → `pmb codex-notify`. |
| `pmb hooks list` 🟢 | Show which hooks are installed. |
| `pmb hooks capabilities` 🟢 | What ambient mechanism each agent supports: `hooks` (Claude Code) / `rollout` (Codex) / `mcp-only` (git observer). |
| `pmb hooks uninstall <agent>` 🟢 | Remove the hooks. |
| `pmb auto-context "..."` 🟢 | Preview the per-turn memory a UserPromptSubmit hook would inject. |
| `pmb session-restore [-m MIN]` 🟢 | Preview the "where you left off" digest a SessionStart hook injects after a compaction. |
| `pmb lesson-followcheck --dry-run` 🟢 | Preview deterministic follow-through scoring for surfaced lessons. |
| `pmb autowrite [--dry-run]` 🟢🧠 | Ambient auto-write for the current turn: if the agent didn't call a `record_*` tool, synthesize ONE activity entry from observed actions. No-op unless `autowrite.enabled`. 🧠 only if `autowrite.synthesizer` is an LLM backend - the default template needs no model. |
| `pmb track-action` 🟢 | (Hook-invoked.) Append one observed action to the ambient journal - the PostToolUse hot path (single SQLite INSERT, no model). |
| `pmb ambient-watch <dir>` 🟢🧠 | Ambient auto-write for MCP-only hosts (Cursor/Zed/VS Code): poll git for changes, auto-write once the project goes idle. |
| `pmb codex-notify` 🟢🧠 | (Hook-invoked by Codex on `agent-turn-complete`.) Parse the session rollout, then run ambient auto-write. |
| `pmb forget-auto [--minutes N]` 🟢 | Archive memory the ambient layer wrote itself (`source=autowrite`). Reversible - archived, not hard-deleted. |

Ambient auto-write is **ON by default** and never duplicates the agent's
own `record_*` calls - it only fills the gap when the agent stays silent.
A turn is journaled only if it clears an outcome-based quality bar (tests
passed, a failure fixed, a deploy ran), so mechanical churn is dropped.
Tune everything via `autowrite.*` (`pmb config list`).

---

## How it embeds into your workflow (no lock-in)

- **Lazy by default.** After `pmb connect`, the agent ignores PMB for general
  and coding questions and only touches it on explicit triggers (or, with
  `--active`, when it finishes a meaningful unit of work). You can talk to the
  agent normally and never trigger PMB.
- **Nothing is forced.** Use the agent's native abilities freely; PMB only
  chimes in when asked.
- **Remove it anytime.** Delete the `pmb` entry from the agent's MCP config to
  disconnect - your stored memory stays on disk and is still readable via the
  CLI.
