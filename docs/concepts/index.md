# Concepts

These pages explain what PMB is made of, how memory moves through the system,
and why the design is local-first.

``` mermaid
flowchart TB
  Agent["Agent host"] --> Integration["MCP tools and lifecycle hooks"]
  Integration --> Daemon["Warm daemon"]
  Daemon --> Engine["Engine"]
  Engine --> Retrieval["Hybrid recall"]
  Engine --> Storage["SQLite, LanceDB, BM25"]
  Engine --> Core["Core queues, caches, and schema"]
  Retrieval --> Context["Relevant context block"]
  Context --> Agent
```

<div class="grid cards" markdown>

-   **Architecture**

    Component map, storage layout, hooks, daemon, and where to look in code.

    [Read architecture →](architecture.md)

-   **How it works**

    Step-by-step read path, write path, daemon behavior, and recall internals.

    [Follow the flows →](how-it-works.md)

-   **Memory model**

    The kinds of memory PMB keeps, keyed facts, importance, tiers, and decay.

    [See the memory model →](memory-model.md)

-   **Core engine**

    Implementation-level map of the Engine, storage schema, queues, and code paths.

    [Inspect the core →](core-engine.md)

-   **Design and technology**

    The design patterns, tradeoffs, and stack choices behind PMB.

    [See the decisions →](design-and-tech.md)

-   **Privacy and security**

    Local-first guarantees, secret redaction, bearer-token team mode, and
    encrypted export.

    [Review privacy →](privacy-and-security.md)

</div>
