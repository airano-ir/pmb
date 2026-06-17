# Guide

Use this section when you want to get PMB running, connect an agent, inspect
memory, or manage data.

``` mermaid
flowchart LR
  Install["Install PMB"] --> Setup["Run pmb setup"]
  Setup --> Restart["Restart your agent"]
  Restart --> Work["Work normally"]
  Work --> Inspect["Open dashboard or recall"]
  Inspect --> Tune["Tune models, hooks, and config"]
```

<div class="grid cards" markdown>

-   **Getting started**

    Install PMB and wire your first agent in a few minutes.

    [Start here →](getting-started.md)

-   **Usage**

    Connect specific agents, verify the install, and share one workspace.

    [Connect an agent →](usage.md)

-   **Dashboard**

    Browse memories, inspect graphs, and delete records from the local UI.

    [Open the dashboard →](dashboard.md)

-   **Deleting memories**

    Archive, restore, or permanently remove memories.

    [Manage deletion →](deleting-memories.md)

</div>

## Recommended first pass

1. Read [Getting started](getting-started.md).
2. Connect each agent you use with [Usage](usage.md).
3. Open the [Dashboard](dashboard.md) once so you know where your data lives.
4. Keep [Commands](../reference/COMMANDS.md) nearby for lookup.
5. If something is off, run `pmb doctor` and check [Troubleshooting](troubleshooting.md).
