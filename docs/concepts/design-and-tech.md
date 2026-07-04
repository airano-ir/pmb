# Design and technology

This page covers the design patterns PMB leans on, the technology stack, and the
key decisions behind them. For how the pieces fit at runtime, see
[Architecture](architecture.md).

## Design patterns

- **Zero-LLM read path.** Memory is supplied by deterministic lifecycle hooks
  or explicit MCP `prepare` calls, not by asking a model what to recall. Reading
  costs no agent tokens, so the agent actually uses memory instead of avoiding
  it.
- **One warm shared process.** The cost of memory is loading the embedding
  model. A single warm daemon holds it, and every connected agent reuses it, so
  N agents cost about one model in RAM rather than N.
- **Thin client with a fast lane and graceful fallback.** `pmb-hook` is a
  stdlib only client that talks to the warm daemon in milliseconds, and falls
  back to a cold in process path the moment the daemon is absent. Nothing breaks
  when the daemon is down, it just gets slower for one message.
- **Bridge adapter for stdio only hosts.** Codex cannot take an HTTP MCP entry,
  so `pmb mcp proxy` bridges stdio to the daemon over HTTP. The same warm
  process serves both JSON hosts and stdio hosts.
- **Layered configuration.** Settings resolve in order: per call overrides, then
  the workspace config, then the global config, then schema defaults. One schema
  defines every key, its type, and its default.
- **Engine composed from mixins.** The Engine is assembled from focused mixins
  (write, recall, health, ambient) rather than one large class, so each concern
  stays small and testable.
- **Hybrid retrieval with min-max fusion.** A lexical channel (BM25) and a
  semantic channel (vectors) are normalized and fused, then ranked by importance,
  recency, and access. No single channel decides alone.
- **Anchor margin classification.** Short intent decisions, such as whether a
  message is conversational or worth a recall, use distance to small sets of
  example anchors with the same embedder, rather than a separate model.
- **Advisory guard, never a block.** The PreToolUse rule guard surfaces a
  relevant rule before an action, but it never takes control away from the
  agent. It informs, it does not hijack.
- **Registry as a singleton guard.** Running servers and the daemon record
  themselves in a small JSON registry, so a second heavy server is not started
  by accident and hooks can find the warm one.
- **Soft delete by default.** Deleting archives reversibly; a permanent purge is
  an explicit opt in. Destructive actions are never the default.
- **Capability tiers.** MCP tool profiles (minimal, lean, default, full) and
  host capability detection (hooks, rollout, mcp-only) tune behavior to what each
  host and task actually needs.

## Technology stack

| Layer | What we use |
|---|---|
| Language | Python 3.11+ |
| CLI and terminal UI | Typer, Rich |
| MCP server | FastMCP (3.x) |
| Embeddings | sentence-transformers (default multilingual MiniLM; bge-m3 and others optional) |
| Vector store | LanceDB |
| Lexical search | rank-bm25 |
| Numerics | numpy |
| Config | PyYAML |
| Primary storage | SQLite (standard library) |
| Daemon HTTP | uvicorn with Starlette middleware (bearer auth) |
| Codex config | tomli-w to write TOML, standard library tomllib to read |
| Dashboard | Python standard library HTTP server, vanilla HTML, CSS, and JavaScript, with vis-network for the entity graph |
| npm launcher | a thin Node wrapper that installs and forwards to the Python package (no bundled Python) |
| Optional extras | anthropic (Anthropic API backend), cryptography (encrypted workspace export); OpenAI API uses stdlib HTTP |

The dashboard deliberately has no build step and no JavaScript framework. It is
one HTML file with plain CSS and JavaScript, so it renders anywhere and there is
no toolchain to maintain.

## Key decisions, and why

- **Local only, no cloud, no keys.** Privacy, zero running cost, and it works
  offline. The dashboard binds to localhost, and only the explicit sync commands
  ever touch the network.
- **Reads must be free.** If memory costs tokens on every turn, agents skip it.
  Deterministic hooks and MCP prepare calls keep reading effectively free.
- **Share one warm model.** Loading the model is the expensive step, so it is
  loaded once and shared, not paid per agent.
- **Multilingual by default.** The default embedder handles many languages, and
  recall is tuned to find cross language answers, so it works in any language out
  of the box.
- **Archiving is reversible, purging is explicit.** Losing memory by accident is
  worse than keeping a bit too much, so the default delete is a reversible
  archive.
- **Structure over hardcoded lists.** Where a rule can be derived, it is derived.
  For example, the command names a rule guards are extracted structurally rather
  than from a fixed list, and lesson relevance uses corpus inverse document
  frequency rather than a static stopword list.

## Testing and quality

PMB ships with a large pytest suite covering the engine, retrieval quality, the
CLI, the MCP layer, the hooks, and integration paths. Recall quality has its own
regression gates with real, deterministic embeddings so a change that would lower
recall is caught before it ships. Linting is enforced with Ruff.
