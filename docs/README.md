<div class="hero" markdown>
<div class="hero__eyebrow">Local-first memory for coding agents</div>

# PMB gives coding agents local memory

PMB gives Claude Code, Codex, Cursor, and other MCP agents a shared private
memory. It captures project decisions and lessons as you work, ranks relevant
context locally, and injects it before the next answer.

<div class="hero__actions" markdown>
[Start with setup](guide/getting-started.md){ .md-button .md-button--primary }
[See how it works](concepts/how-it-works.md){ .md-button }
</div>
</div>

<div class="hero__metrics" markdown>
<div class="hero__metric" markdown>
<strong>Local by default</strong>
<span>SQLite, LanceDB, and config live on your machine.</span>
</div>
<div class="hero__metric" markdown>
<strong>Fast recall</strong>
<span>One warm daemon serves every connected agent.</span>
</div>
<div class="hero__metric" markdown>
<strong>Agent-native</strong>
<span>MCP tools plus hooks fit into existing workflows.</span>
</div>
</div>

## How PMB fits into your workflow

``` mermaid
flowchart LR
  Work["Work"] --> Capture["Capture"]
  Capture --> Store["Store"]
  Store --> Recall["Recall"]
  Recall --> Context["Context"]
  Context --> Answer["Answer"]
  Answer --> Work
```

## Choose your path

<div class="grid cards" markdown>

-   **Getting started**

    Go from zero to a wired agent with `pip`, `uv`, `pipx`, or `npx`.

    [Install PMB →](guide/getting-started.md)

-   **Usage**

    Connect Claude Code, Codex, Cursor, Windsurf, Gemini, Zed, and more.

    [Connect an agent →](guide/usage.md)

-   **Architecture**

    See the daemon, hook, MCP, storage, and retrieval paths as diagrams.

    [Understand the system →](concepts/architecture.md)

-   **Core engine**

    Read the Engine map, schema, queues, and code paths behind recall.

    [Inspect the core →](concepts/core-engine.md)

-   **Commands**

    Look up setup, recall, dashboard, delete, sync, and maintenance commands.

    [Open the reference →](reference/COMMANDS.md)

</div>

## Why teams use it

<div class="pmb-feature-grid" markdown>
<div markdown>
<strong>Durable memory</strong>
<span>Facts, decisions, goals, lessons, and session summaries persist between sessions.</span>
</div>
<div markdown>
<strong>Free reads</strong>
<span>Local pre-message recall through hooks and MCP, so remembering costs no tokens.</span>
</div>
<div markdown>
<strong>One shared brain</strong>
<span>Several agents on one project share a warm daemon and one workspace memory.</span>
</div>
<div markdown>
<strong>Private by default</strong>
<span>Local storage; the network is touched only by the explicit sync commands you run.</span>
</div>
<div markdown>
<strong>Learns from drift</strong>
<span>Lessons and failures surface before the work they apply to, so mistakes do not repeat.</span>
</div>
</div>

## The core loop

``` mermaid
flowchart TB
  User["Request"] --> Prepare["Prepare"]
  Prepare --> Context{"Found?"}
  Context -->|Yes| Relevant["Relevant memory"]
  Context -->|No| Quiet["Stay quiet"]
  Relevant --> Agent["Act"]
  Quiet --> Agent
  Agent --> Record{"Worth saving?"}
  Record -->|Yes| Memory["Record"]
  Record -->|No| Skip["Skip"]
  Memory --> Next["Better next turn"]
  Skip --> Next
```

## Common tasks

| I want to... | Do this |
|---|---|
| Set up one agent | `pmb setup` |
| Set up every detected agent | `pmb setup --all` |
| Install from npm in one step | `npx pmb-ai setup` |
| See whether memory is warm | `pmb daemon status` |
| Change the embedding model | `pmb model` |
| Archive a memory | `pmb delete <ulid>` |
| Delete a memory permanently | `pmb delete <ulid> --hard` |
| Bring an archived memory back | `pmb restore <ulid>` |
| Open the local dashboard | `pmb dashboard` |
| Reset stray PMB processes | `pmb daemon kill-all` |

## What to read next

- [Guide](guide/index.md): install PMB, connect agents, and operate the dashboard.
- [Concepts](concepts/index.md): understand the architecture and retrieval model.
- [Reference](reference/index.md): look up commands and configuration.
- [Contributing](contributing/index.md): extend language packs or improve the docs.
