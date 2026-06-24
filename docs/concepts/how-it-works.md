# How it works

This page follows real requests end to end. For the component map and where
things live in the code, see [Architecture](architecture.md). For the Engine
schema and queue internals, see [Core engine](core-engine.md). For the
reasons behind the design, see [Design and technology](design-and-tech.md).

## Follow one message (the read path)

This is the path that makes memory feel free.

``` mermaid
flowchart LR
  Message["Message"] --> Prepare["Prepare"]
  Prepare --> Recall["Recall"]
  Recall --> Gate{"Relevant?"}
  Gate -->|Yes| Context["Context"]
  Gate -->|No| Silent["Quiet"]
  Context --> Answer["Answer"]
  Silent --> Answer
```

1. You send a message to your agent. The host fires the UserPromptSubmit hook
   with your text.
2. `pmb-hook`, a small stdlib only client, posts the message to the warm daemon
   at `/internal/hook/prepare-context`. If no daemon is running, it answers from
   a cold in process Engine and asks a daemon to start in the background, so the
   next message is warm.
3. The Engine does the work, with no LLM call:
   - it classifies the message, for example small talk versus a real question,
     so it can stay quiet when there is nothing to add;
   - it runs hybrid recall (lexical plus semantic, described below);
   - it gathers any matching lessons and decisions, plus a little project
     context;
   - it applies the precision gates so only genuinely relevant memory survives;
   - it formats one compact block of text.
4. The hook prints that block. The host folds it into the model context before
   the model starts thinking.
5. The model answers, now aware of the relevant memory. No agent tokens were
   spent deciding what to recall.

## Follow one record (the write path)

``` mermaid
sequenceDiagram
  autonumber
  participant Agent
  participant MCP as PMB MCP tool
  participant Engine
  participant Store as SQLite + LanceDB + BM25
  participant Graph as Entity graph

  Agent->>MCP: record_batch / record_fact / record_goal
  MCP->>Engine: validate and normalize event
  Engine->>Store: write event and indexes
  Engine->>Graph: extract and link entities
  Store-->>Engine: committed
  Graph-->>Engine: linked
  Engine-->>Agent: memory is durable and searchable
```

1. The agent calls a `record_*` tool (or ambient auto write fires at the end of
   a turn).
2. The event is written to SQLite with its content, type, importance, metadata,
   and timestamp.
3. The text is embedded and stored in LanceDB, and the BM25 index is updated, so
   the new memory is immediately recallable.
4. Entities are extracted from the text and linked into the graph, so topics and
   relationships build up over time.

Ambient auto write only fills the gap when the agent did not record anything
itself, and only when the turn clears a quality bar, so routine churn is
dropped rather than journaled.

## What the daemon does

The daemon is the one warm process behind fast recall.

``` mermaid
flowchart LR
  Start["start"] --> Warm["warm"]
  Warm --> Register["register"]
  Register --> Serve["serve"]
  Serve --> Idle{"idle?"}
  Idle -->|No| Serve
  Idle -->|Yes| Exit["exit"]
  Serve --> Tick["maintenance"]
  Tick --> Serve
```

- On start it builds the MCP server, begins warming the embedding model in a
  background thread (so it can serve immediately while the model finishes
  loading), and registers itself in a small JSON registry.
- It serves the MCP tools over local HTTP with a bearer token, plus a few
  internal routes that the hooks call (`prepare-context`, `session-restore`,
  `pretool`).
- It can idle exit after a configured time so a forgotten daemon does not hold
  memory forever. The hooks restart it on the next message.
- A maintenance tick runs in the background while the daemon is idle, for tasks
  like archiving cold events. It never runs on the per message path.
- Stdio only hosts such as Codex reach the daemon through `pmb mcp proxy`, a
  light bridge that holds no model.

## What each hook does

For Claude Code, PMB installs the full set:

- **UserPromptSubmit** runs `prepare-context`: the read path above. This is the
  one that injects memory before the model thinks.
- **SessionStart** runs `session-restore`: after a compaction or a resume, it
  rebuilds a short digest of what this session already decided and did, so the
  agent does not re-ask you.
- **PreToolUse** runs `pretool`: before a tool call it can surface a matching
  rule, for example a lesson that names the command about to run. It also fires
  the "do NOT repeat this" corpus - past failures and captured corrections - so
  if the action walks into something you were corrected on before, you see
  "⛔ STOP - you were corrected on this before" at the moment it matters, even
  inside an autonomous tool loop with no user message. It is advisory and never
  blocks the action.
- **PostToolUse** runs `track-action`: it records the action for ambient memory,
  as a single fast write with no model.
- **Stop** runs `lesson-followcheck` and `autowrite`: it scores which surfaced
  lessons were actually followed, and journals the turn if the agent recorded
  nothing itself.

Codex has no per turn or session shell hook. Its one extension point is
`notify`, fired when a turn completes, wired to `pmb codex-notify`, which reads
the Codex rollout log for ambient memory. Read first and recall for Codex come
from the AGENTS.md rules that `pmb connect codex` installs, so the agent calls
`prepare` and `recall` itself.

## Recall internals

Recall combines two channels and then ranks:

- **Lexical (BM25)** matches the actual words. It is strong for exact terms,
  identifiers, and rare tokens.
- **Semantic (vectors)** matches meaning. It is strong for paraphrase and for
  cross language queries.
- The two are combined with min-max normalization so neither channel dominates
  just because its raw scores live on a different scale.
- Candidates are then ranked by signals such as importance, recency, and how
  often a memory has been useful before.

On top of that:

- **Precision gates** keep the read path quiet. A question the workspace knows
  nothing about, or pure small talk, surfaces nothing rather than the closest
  weak match.
- **Cross language handling.** When a query's content words are out of
  vocabulary for the lexical index, for example a question in one language over
  notes in another, the lexical channel is down weighted so the semantic channel
  decides.
- **Lessons render as rules.** Procedural memory (lessons, corrections) is shown
  as rules to follow, separate from ordinary background context, so the agent
  treats it with the right weight.
