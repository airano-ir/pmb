# Stability and guarantees

This page is the **contract**. The README and other guides describe how PMB
works today; this page states what you can *rely on* not changing under you.
It applies from 1.0 onward and is versioned with the project (see
[Versioning](#versioning)).

## Privacy and network

PMB is local-first and **offline by default**. No account, no API keys, nothing
leaves your machine for normal use. Memory lives in plain files on your disk.

The only ways PMB touches the network are **opt-in** and never happen unless you
turn them on:

- **LLM consolidation / distillation** (`pmb consolidate`, lesson distillation)
  when you point it at a cloud model. Use a local Ollama model to stay fully
  offline.
- **Team / remote mode** (`pmb mcp serve --transport streamable-http`,
  `pmb connect --remote`) when you deliberately share one memory over HTTP.
- **Git sync** when you wire a remote for backup.

If you never enable these, PMB makes zero network calls.

## Your data on disk

Everything lives under `PMB_HOME` (default `~/.pmb`):

```
~/.pmb/workspaces/<workspace-id>/
    events.sqlite          # all events (the source of truth)
    <lancedb tables>       # vectors, next to the SQLite
    <graph store>          # entity/association graph
    config.yaml            # optional per-workspace config
```

**Back up** by copying the workspace folder while PMB is idle, or use
`pmb workspace export` to write a single (optionally encrypted) bundle and
`pmb workspace import` to restore it. SQLite + LanceDB are just files; `cp`
works.

## Deleting memories

- `pmb forget <ulid>` **archives** an event (sets `archived_at`). It stops
  appearing in recall but is **recoverable** with `pmb unforget`.
- For permanent removal there is an explicit **hard delete** (`delete_event(...,
  hard=True)` / purge). Hard delete is irreversible by design.
- **Background maintenance never hard-deletes.** The maintenance tick is
  archive-only and report-only; it will not silently destroy data.
- Keyed facts (`record_keyed_fact`) **supersede** rather than overwrite: the old
  value is archived, and the full history stays queryable.

## Upgrades

- Schema changes ship with **automatic forward migrations**, applied when a
  workspace is opened (`events.sqlite` and the graph store each carry a schema
  version). Upgrading PMB does not lose data.
- `pmb migrate-workspaces` handles cross-version workspace moves/merges.
- **Downgrading** to an older PMB after a migration is not supported. Back up
  the workspace folder before downgrading.

## The MCP tool surface

What an agent sees is profile-gated (set via `PMB_TOOL_PROFILE`):

| Profile | Tools | When |
| :-- | :-- | :-- |
| `minimal` | 13 | the bare essentials |
| `lean` | 26 | set by `pmb connect claude-code` (hooks already cover the read-status tools) |
| `default` | 30 | day-to-day usage |
| `full` | all | debugging / development |

The **default** set is the stable contract: a tool in it will not be silently
removed or have its required arguments changed in a 1.x release. Admin/niche
tools outside the default set may change; they are still callable from the CLI.

## What is stable vs experimental

**Stable** (covered by this contract): hybrid `recall`, the `record_*` family,
lessons + `mark_lesson_followed`, goals, keyed facts, the lifecycle hooks, the
warm daemon, the `default` MCP tool surface, the everyday CLI commands, the
on-disk schema, and the export/import format.

**Experimental** (NOT covered; off by default; may change, move, or be removed
without a major version bump):

- the **semantic lesson-surfacing tier** (`recall.lesson_semantic`) - measured
  not to beat the lexical tier with the default embedder, so it stays opt-in;
- **warm keyed-anchor extraction** - powerful but unproven on precision data, so
  for 1.0 it stays experimental and default-off until field metrics justify
  flipping it (see the ALD validation work);
- the **predictive cache**;
- **multimodal** tools (image / code embedding);
- the optional **cross-encoder reranker**.

These are labelled in their config keys and docs. Turning one on is your choice,
and its behaviour may change between minor releases.

## Versioning

PMB follows semantic versioning. Everything under "stable" above holds across
all `1.x` releases; a breaking change to this contract waits for the next major
version and ships with a migration note.
