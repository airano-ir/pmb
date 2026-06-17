# Reference

Use reference pages when you already know the task and need the exact command,
flag, or configuration key.

``` mermaid
flowchart LR
  Task["Task"] --> Command["Command reference"]
  Task --> Config["Configuration"]
  Command --> Run["Run pmb command"]
  Config --> Tune["Tune behavior"]
  Run --> Verify["doctor, stats, dashboard"]
  Tune --> Verify
```

<div class="grid cards" markdown>

-   **Commands**

    Every PMB command grouped by setup, capture, recall, lifecycle, sync, and UI.

    [Open the command reference →](COMMANDS.md)

-   **MCP tools**

    The tools an agent calls (prepare, recall, record_batch, and more), plus the
    tool profiles.

    [See the MCP tools →](mcp-tools.md)

-   **Configuration**

    Settings for recall, models, hooks, graph extraction, and active logging.

    [Open configuration →](configuration.md)

</div>

## Fast lookup

| Task | Start with |
|---|---|
| Wire PMB into an agent | `pmb setup` or `pmb connect <agent>` |
| Check install health | `pmb doctor` |
| Search memory manually | `pmb recall "query"` |
| Inspect memory visually | `pmb dashboard` |
| Change model profile | `pmb model` |
| Tune behavior | `pmb config list` |
