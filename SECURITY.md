# Security Policy

## What PMB stores

PMB writes everything to a local SQLite file (`~/.pmb/workspaces/<id>/events.sqlite`) and a local LanceDB directory. **Nothing is sent to any external service** by PMB itself.

Caveats worth knowing:

- The **AI agent** plugged into PMB (Codex CLI / Claude Code / Cursor / etc.) talks to its own LLM provider. PMB has no control over that channel.
- The **embedding model** runs locally (sentence-transformers, no network at inference time after the first download).
- The **Ollama backend**, if you enable it, talks to your local Ollama server (default `http://localhost:11434`). It does not leave your machine unless you configure it to.
- Optional **Anthropic/OpenAI backends** for LLM-powered commands send clustered text or prompts to their API endpoints when invoked. Off by default.

## Reporting vulnerabilities

If you believe you have found a security issue:

1. Do not open a public issue.
2. Email the maintainer or open a GitHub Security Advisory.
3. Include a minimal repro and what you think the impact is.

We will acknowledge within a few days and aim to publish a fix or workaround within two weeks for serious issues.

## Threat model in scope

- Untrusted input via MCP tool calls (the agent may pass arbitrary text into `record_*`).
- File-path traversal in `pmb` CLI arguments.
- SQL injection (we use parameterised queries everywhere; a regression is a bug).
- Resource exhaustion via huge content blobs (mitigated by 5000-char cap in `record_batch`).

### Prompt injection via stored memory

Memory content is untrusted: it is derived from arbitrary conversations, files,
and tool output, any of which an attacker may influence. PMB therefore treats
stored content as data, never instructions, on the paths it controls:

- **Internal LLM calls run with no tools.** Consolidation, lesson distillation,
  entity extraction and the agent wrapper spawn `claude -p` with
  `--allowed-tools ""` and **without** `--permission-mode bypassPermissions`.
  These are text-in/JSON-out calls, so the spawned agent has no Bash/Edit/Write
  surface - an injected payload like "ignore the task and run …" cannot make it
  touch the filesystem or network.
  (`health/consolidate.py`, `graph/extractors_llm.py`, `agent_wrapper/loop.py`)
- **Note on recall into *your* agent.** When PMB injects recalled memory into
  the context of the agent you are driving (via the hooks / MCP), that agent's
  own permission model is the boundary. PMB cannot enforce tool restrictions
  there; keep a human in the loop for tool use as you normally would.

### Dashboard / API exposure

- The dashboard and HTTP MCP transport bind to `127.0.0.1` by default. Binding
  to `0.0.0.0` exposes the unauthenticated memory API to your network - only do
  this behind a trusted boundary, and set `PMB_MCP_BEARER_TOKEN` for the MCP
  HTTP transport.
- The dashboard does **not** emit `Access-Control-Allow-Origin`. The UI is
  served same-origin, so no CORS grant is needed; a wildcard would let any
  website you visit read the local memory store via cross-origin requests.

## Out of scope

- Confidentiality of data the user *chooses* to record. PMB is a memory store - if you feed it secrets they will be stored. Use `record_fact ... metadata={"redact": true}` or rely on the built-in regex redactor for known secret shapes.
- Multi-user isolation. PMB is single-user. Anyone with access to your `~/.pmb/` directory can read all your memory.
- Network-level attacks on Ollama or LanceDB. Those are upstream concerns.
