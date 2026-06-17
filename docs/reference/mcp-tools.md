# MCP tools

When you wire an agent with `pmb connect`, PMB appears as a set of MCP tools that
the agent calls itself. These tools and the [Commands](COMMANDS.md) on the CLI
share one engine and one workspace. This page lists the tools an agent sees.

Most reading is automatic: lifecycle hooks (Claude Code) or the installed rules
call `prepare` and `recall` when memory is relevant, so you rarely invoke these
by hand.

## Tool profiles

PMB trims the tool list so the agent is not overwhelmed. Pick a profile with the
`PMB_TOOL_PROFILE` environment variable, or let `pmb connect` choose one.

| Profile | Tools | For |
|---|---|---|
| `minimal` | 13 | The bare essentials, for the fastest path. |
| `lean` | 26 | Claude Code with ambient hooks, where hooks already cover recall. |
| `default` | 30 | Day-to-day usage. |
| `full` | 65 | Everything, including maintenance and sleep-stage operations. |

## Read first

| Tool | Parameters | Purpose |
|---|---|---|
| `prepare` | `message` | One call at the start of a task: project context, surfaced lessons (each with a `surface_id`), recent activity, and open goals. |
| `session_brief` | `minutes` | A digest of what this session already decided, did, and learned, to re-orient after a long session. |

## Recall and explore

| Tool | Parameters | Purpose |
|---|---|---|
| `recall` | `query`, `top_k=5`, `project` | Search memory; returns hits plus any matching lessons. |
| `recall_smart` | `query`, `top_k`, `confidence_threshold` | Fast recall that escalates to a deeper search only when confidence is low. |
| `recall_deep` | `query`, `top_k` | The slow path, with LLM query decomposition for hard questions. |
| `overview` | `topic`, `max_events` | A structured "what do I know about X": facts, decisions, lessons, goals, and a timeline. |
| `project_overview` | `name` | Full project context from the entity graph. |
| `find_lessons` | `query`, `limit` | Procedural lessons (rules), each with a `surface_id` for follow-tracking. |
| `mark_lesson_followed` | `surface_id`, `followed`, `note` | Report whether a surfaced lesson changed behavior; this powers the self-improvement loop. |

## Record

| Tool | Parameters | Purpose |
|---|---|---|
| `record_batch` | `items` | Store several items in one call: `fact`, `fact_tree`, `keyed_fact`, `goal`, `activity`, `milestone`, `plan`, or `lesson`. Prefer one batch per turn. |
| `record_fact` | `fact`, `importance=0.7` | Store one atomic fact. |
| `record_fact_tree` | `main`, `subfacts`, `importance` | A main fact plus linked subfacts. |
| `record_keyed_fact` | `subject`, `attribute`, `value`, `importance=0.85` | Upsert a mutable attribute; the old value is archived. |
| `record_goal` | `title`, `status`, `due_at`, ... | A goal or intention with a status. |
| `update_goal` | `goal_ulid`, `status`, `progress`, `note` | Update a goal's status or progress. |
| `record_activity` | `summary`, `kind`, `importance` | A lightweight log of finished work or a decision. |
| `record_milestone` | `chain_name`, `title`, `state` | A checkpoint in a named state-chain. |
| `pin` | `ulid` | Lock a memory at maximum importance. |
| `forget` | `ulid` | Archive a memory (reversible). |

## Status, goals, and graph

| Tool | Parameters | Purpose |
|---|---|---|
| `stats`, `workspace_info` | — | Workspace counts and identity. |
| `recent_activity`, `what_just_happened`, `list_recent` | — | Instant working-memory views, with no vector search. |
| `list_goals` | `status`, `limit` | List goals, optionally filtered by status. |
| `graph_stats`, `graph_top_entities`, `graph_neighbors` | — | Inspect the entity and association graph. |

## Ingest

| Tool | Parameters | Purpose |
|---|---|---|
| `index_pdf` | `path` | Add a PDF as searchable memory. |
| `index_project` | `path` | Index a code project's structure (symbols, imports, languages). |
| `record_code`, `record_image` | ... | Store code or an image; images support cross-modal search. |

## Maintenance and sleep operations

The `full` profile adds background and LLM-assisted tools that run off the recall
hot path: `consolidate_recent`, `reflect_batch`, `extract_facts`,
`cluster_into_arcs`, the `dedupe_*` family, `compact_storage`, `apply_daily_decay`,
`run_self_test`, and the predictive-cache tools. Most have a CLI equivalent on
[Commands](COMMANDS.md), so you do not need to expose them to the agent for
normal use.
