# PMB documentation

PMB (Personal Memory Brain) is local-first persistent memory for AI coding
agents. It runs fully on your machine, with no API keys, and works with Claude
Code, Codex, Cursor, and other agents over MCP.

New here? Start with the [Guide](#guide), then read
[How it works](concepts/how-it-works.md) when you want the details.

## Guide

Task-oriented pages to get things done.

- [Getting started](guide/getting-started.md): install with pip or npm, run
  setup, and understand the shared warm daemon.
- [Usage](guide/usage.md): day to day notes, per agent.
- [The dashboard](guide/dashboard.md): browse your memory, and delete from the UI.
- [Deleting memories](guide/deleting-memories.md): archive or permanently delete,
  from the dashboard or the command line.
- [Team and remote use](guide/TEAM.md): share one memory across machines.
- [Ollama setup](guide/SETUP_OLLAMA.md): use a local LLM for the background tier.

## Concepts

How PMB is built and why.

- [Architecture](concepts/architecture.md): the components and where they live in
  the code.
- [How it works](concepts/how-it-works.md): a step by step walk through the read
  path, the write path, the daemon, and the hooks.
- [Design and technology](concepts/design-and-tech.md): the design patterns, the
  stack, and the decisions behind them.

## Reference

Look things up.

- [Commands](reference/COMMANDS.md): every command, grouped by task.
- [Configuration](reference/configuration.md): the settings you are likely to
  touch, and how config layering works.

## Contributing

- [Adding a language](contributing/adding-a-language.md): extend the language
  packs (a good first contribution).

## Common tasks

| I want to... | Do this |
|---|---|
| Set up one agent | `pmb setup` |
| Set up every agent I have | `pmb setup --all` |
| Install from npm in one step | `npx pmb-ai setup` |
| See if memory is warm | `pmb daemon status` |
| Change the embedding model | `pmb model` |
| Archive a memory (reversible) | `pmb delete <ulid>` |
| Delete a memory permanently | `pmb delete <ulid> --hard` |
| Bring an archived memory back | `pmb restore <ulid>` |
| Open the web dashboard | `pmb dashboard` |
| Reset stray processes | `pmb daemon kill-all` |

## What makes it different

- **Zero token reads.** Memory is injected by local hooks before the model
  thinks, so recall does not cost agent tokens.
- **One warm daemon.** A single background process holds the embedding model and
  the index, and every connected agent reuses it.
- **Any language.** The default model is multilingual, and recall is tuned to
  find cross language answers.
- **Your data stays yours.** Everything is local. The dashboard binds to
  `127.0.0.1` only, and the sync commands are the only ones that touch the
  network, and only when you run them.

## Project

- [Roadmap](ROADMAP.md): what is planned next.
- [Changelog](../CHANGELOG.md): what changed in each release.
