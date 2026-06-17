# Getting started

PMB is local-first persistent memory for AI coding agents. It runs fully on your
machine, with no API keys and nothing sent to the cloud. This guide takes you
from install to a working agent in a few minutes.

## 1. Install

You can install with pip or with npm. Both end with a working `pmb` command.

### pip (or pipx / uv)

```bash
pip install pmb-ai
pmb setup
```

`pipx install pmb-ai` and `uv tool install pmb-ai` work the same way and keep
PMB isolated from your other Python packages.

### npm

```bash
npx pmb-ai setup
```

The npm package is a thin launcher. On first use it runs the full install
cycle: it finds or installs the Python `pmb-ai` package (it prefers `uv`, then
`pipx`, then `pip`), and then runs the command you asked for. So
`npx pmb-ai setup` both installs PMB and walks you through setup. If you instead
run `npm install -g pmb-ai`, the postinstall step installs the Python package
for you, and you then run `pmb setup`.

To skip the automatic Python bootstrap in CI or Docker, set
`PMB_SKIP_POSTINSTALL=1` before installing.

## 2. Run setup

`pmb setup` is the guided, one command path. It detects your agent, lets you
pick the memory model, wires the MCP config, installs the agent rules, adds
supported lifecycle hooks, warms the engine, and starts the shared daemon.

```bash
pmb setup            # detect one agent and wire it
pmb setup --all      # wire every detected agent at once
```

Use `pmb setup --all` when you work with more than one agent (for example Claude
Code and Codex and Cursor together). They all share a single warm daemon.

During setup you can choose:

- The memory model (embedder). Light is tiny and fast, Balanced is more
  accurate, Best is the strongest for cross language and rare scripts. You can
  change it later with `pmb model`.
- The offline brain (the background LLM tier). This only runs during
  consolidation and never affects how fast memory is injected.

The menus redraw in place: use the arrow keys to move, Enter to choose, a digit
to jump straight to an option, and Esc to keep the current value.

## 3. The shared warm daemon

The slow part of memory is loading the embedding model. PMB solves this with one
warm daemon: a single background process that holds the model and the index, and
that every connected agent reuses. N agents then cost about 400 MB in total
rather than 400 MB each, and recall is instant.

- JSON hosts (Claude Code, Cursor) point at the daemon over local HTTP.
- Codex is stdio only, so it launches a light bridge (`pmb mcp proxy`) that
  forwards to the same daemon. This removes the Codex cold start lag.

Useful daemon commands:

```bash
pmb daemon status      # is the daemon up and warm?
pmb daemon start       # start it (setup does this for you)
pmb daemon restart     # restart it, for example after changing the model
pmb daemon kill-all    # stop the daemon and every stray PMB process, then reset
```

`pmb daemon kill-all` is the reset button. If duplicate warm processes ever pile
up and slow your machine, this stops all of them and clears the registry so you
can start fresh with `pmb daemon start`.

## 4. Use it

1. Restart your agent so it picks up the new configuration.
2. Just talk to it. PMB captures and recalls memory through the installed
   rules, MCP tools, and hooks supported by that host.

There are no commands to memorize for everyday use. The agent reads relevant
memory before it answers and records new memory as you work.

## 5. Verify and explore

```bash
pmb doctor          # check the install and runtime state
pmb stats           # how much memory you have, by type
pmb dashboard       # open the local web dashboard
pmb model           # change the embedding model later
```

The dashboard runs on `http://127.0.0.1:8765` by default. If you also run the
shared daemon on that port, start the dashboard on another port, for example
`pmb dashboard --port 18888`.

## Next steps

- [Deleting memories](deleting-memories.md): remove or archive memory from the
  dashboard or the command line.
- [Command reference](../reference/COMMANDS.md): every command, grouped by task.
- [Usage notes](usage.md): deeper notes on day to day use.
