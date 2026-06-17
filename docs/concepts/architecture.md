# Architecture

This page explains what PMB is made of and how a request flows through it. For
the implementation-level Engine and storage schema, see
[Core engine](core-engine.md). For the why behind the choices, see
[Design and technology](design-and-tech.md).

## The one idea

Reading memory should cost nothing. PMB injects the relevant memory into the
agent before the model thinks, with no LLM call on the read path. Recall is
therefore free in tokens and fast, and it works the same in any language.

## Big picture

<div class="pmb-signal-grid" markdown>
<div markdown>
<strong>Read path</strong>
<span>No LLM call. The Engine ranks local memory before the model answers.</span>
</div>
<div markdown>
<strong>Runtime</strong>
<span>One warm daemon keeps the embedder and indexes loaded for all agents.</span>
</div>
<div markdown>
<strong>Storage</strong>
<span>SQLite is the source of truth; LanceDB and BM25 are searchable side indexes.</span>
</div>
<div markdown>
<strong>Lifecycle</strong>
<span>Soft delete by default, durable queues for async writes, hard purge only on request.</span>
</div>
</div>

``` mermaid
flowchart LR
  Agent["Agents"] --> Bridge["MCP bridge"]
  Bridge --> Engine["Engine"]
  Engine --> Events[("SQLite")]
  Engine --> Search["Indexes"]
  Engine --> Graph[("Graph")]
  Events --> Context["Context"]
  Search --> Context
  Graph --> Context
  Context --> Agent
```

Everything runs on your machine. Nothing is sent to the cloud, and the only
commands that touch the network are the optional sync commands, and only when
you run them.

## Components

- **CLI (`pmb`)**, built on Typer. Commands for setup, connect, recall, delete,
  the dashboard, the daemon, and maintenance. Source: `src/pmb/cli/`.
- **MCP server (`pmb-mcp`)**, built on FastMCP. It exposes the memory tools over
  stdio (one process per agent) or streamable HTTP (one shared process). Tool
  profiles (minimal, lean, default, full) trim what the agent sees so the tool
  list stays small. Source: `src/pmb/mcp/server.py`, `_toolspec.py`.
- **Warm daemon (`pmb daemon`)**. One process that holds a warm Engine, the
  embedding model, and the vector store, served over local HTTP with a bearer
  token plus a few internal hook routes. It can idle-exit, and it registers
  itself in a small JSON registry so hooks and the proxy can find it. Source:
  `src/pmb/mcp/daemon.py`.
- **Proxy (`pmb mcp proxy`)**. A light stdio to daemon bridge for stdio only
  hosts such as Codex. It holds no model and forwards to the one daemon, which
  removes the per session cold start. Source: `src/pmb/cli/commands/hooks.py`.
- **Lifecycle hooks**. The host runs these at fixed points and folds their
  output into the model context. Source: `src/pmb/cli/hooks.py`,
  `src/pmb/hooks/`.
- **Dashboard**. A stdlib HTTP server plus a single page of vanilla HTML, CSS,
  and JavaScript (the entity map uses vis-network). It binds to `127.0.0.1`
  only. Source: `src/pmb/dashboard/`.

## Lifecycle hooks

For Claude Code, PMB installs the full set:

| Event | What runs | Purpose |
|---|---|---|
| UserPromptSubmit | `pmb-hook prepare-context` | inject relevant memory before the model thinks (auto recall) |
| SessionStart | `pmb-hook session-restore` | rebuild where you left off after a compaction or resume |
| PreToolUse | `pmb-hook pretool` | surface a matching rule before a tool call (advisory, never blocks) |
| PostToolUse | `pmb-hook track-action` | observe the action for ambient memory |
| Stop | `pmb-hook lesson-followcheck` and `autowrite` | score lesson follow through, and journal the turn if the agent did not |

Codex has no per turn or session shell hook. Its one extension point is
`notify`, fired when a turn completes, wired to `pmb codex-notify`. That gives
ambient auto write by reading the Codex rollout log. Read first and recall for
Codex come from the AGENTS.md rules that `pmb connect codex` installs, so the
agent calls `prepare` and `recall` itself.

## The read path, per message

``` mermaid
sequenceDiagram
  autonumber
  participant User
  participant Host as Agent host
  participant Hook as pmb-hook
  participant Daemon as Warm daemon
  participant Engine

  User->>Host: Send message
  Host->>Hook: UserPromptSubmit
  Hook->>Daemon: prepare-context
  Daemon->>Engine: classify + hybrid recall + gates
  Engine-->>Daemon: compact context block
  Daemon-->>Hook: relevant memory
  Hook-->>Host: print injected context
  Host-->>User: model answers with memory
```

1. The host fires UserPromptSubmit with the user message.
2. `pmb-hook` posts it to the daemon route `/internal/hook/prepare-context`.
3. The Engine classifies the message, runs hybrid recall, gathers lessons,
   decisions, and project context, applies the precision gates, and returns one
   compact block of text.
4. The hook prints that block, and the host folds it into the model context
   before the model thinks.

No model call is made to decide what to recall. If no daemon is running, the
hook answers from a cold per process Engine (which skips heavy semantic recall)
and asks a daemon to start, so the next message is warm.

## The write path

``` mermaid
flowchart LR
  Record["record_*"] --> Event[("event")]
  Record --> Embed["embed"]
  Record --> Lexical["BM25"]
  Record --> Entities["entities"]
  Embed --> LanceDB[("vectors")]
  Entities --> Graph[("graph")]
  Event --> Recall["recall-ready"]
  LanceDB --> Recall
  Lexical --> Recall
  Graph --> Recall
```

The `record_*` tools, and ambient auto write, append an event to SQLite, embed
it into LanceDB, update the BM25 index, and link it into the entity graph.
Ambient auto write only fills the gap when the agent did not record anything
itself, and only when the turn clears a quality bar, so routine churn is
dropped.

## Retrieval

Recall is hybrid. A lexical channel (BM25) and a semantic channel (vector
similarity) are combined with min-max normalization, then ranked by signals such
as importance, recency, and access count. For a query whose content words are
out of vocabulary for the lexical index, for example a question in another
language over an English corpus, the lexical channel is down weighted so the
semantic channel decides. Precision gates keep the read path quiet when nothing
is genuinely relevant, which is what stops small talk from pulling in noise.

## Storage layout

- `~/.pmb/` holds the global config, the daemon token, the server registry, and
  the per workspace data.
- Each workspace has a SQLite database (events, the entity graph, event edges,
  tool call metrics, dedup candidates, and schema migrations) and a LanceDB
  table for vectors.
- Compacted, archived events move to a cold store so the active database stays
  small.
- See [Core engine](core-engine.md) for the table-by-table schema map and the
  queue files that make async writes recoverable.

## Where to look in the code

| Area | Path |
|---|---|
| Engine (write, recall, health, ambient) | `src/pmb/core/engine/` |
| Event and vector stores | `src/pmb/core/events.py`, `src/pmb/core/search.py` |
| MCP server, daemon, tools, registry | `src/pmb/mcp/` |
| Hooks (installer and logic) | `src/pmb/cli/hooks.py`, `src/pmb/hooks/` |
| Dashboard | `src/pmb/dashboard/` |
| CLI commands | `src/pmb/cli/` |
