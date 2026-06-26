# Configuration

PMB works well with no configuration. Every setting has a sensible, ablation
tuned default. This page lists the knobs you are most likely to touch. For the
full set, including the advanced "pro" knobs, run `pmb config list --pro`.

## How configuration works

Settings resolve in this order, first match wins:

1. a per call override (passed in code or by a tool),
2. the workspace config (`~/.pmb/workspaces/<name>/config.yaml`),
3. the global config (`~/.pmb/config.yaml`),
4. the built in schema default.

View and change settings:

```bash
pmb config list             # the common keys, with value, source, and default
pmb config list --pro       # everything, including advanced knobs
pmb config get <key>        # one value
pmb config set <key> <value>   # write to the global config
pmb config set <key> <val>  # or the dashboard Settings tab
```

A value set globally applies to every workspace, which is what the daemon and
each agent read.

## Memory model

| Key | Default | What it does |
|---|---|---|
| `embedding.model` | multilingual MiniLM | the embedder that turns text into vectors. Change it with `pmb model`, which also re-embeds your memory. |
| `embedding.backend` | sentence-transformers | the embedding backend. |

Switching the model by hand needs a reindex, so prefer `pmb model <light\|balanced\|best>`, which downloads, re-embeds, and restarts the daemon in one step.

## Recall quality

| Key | Default | What it does |
|---|---|---|
| `recall.top_k` | 5 | how many results recall returns by default. |
| `recall.bm25_weight` | 0.7 | balance between the lexical and the semantic channel. |
| `recall.recency_half_life_days` | 30 | how fast a memory's recency boost decays. |
| `recall.keyed_fact_boost` | 0.35 | extra weight for current keyed facts (for example your city). |
| `recall.ppr_enabled` | true | use the entity graph to spread relevance to related memory. |
| `recall.rerank` | false | run a slower cross encoder reranker on results. |
| `recall.crosslingual_bm25_damp` | 0.3 (pro) | down weight the lexical channel when the query words are out of vocabulary, so cross language queries are decided by meaning. |

## The read hook (auto recall)

| Key | Default | What it does |
|---|---|---|
| `auto_recall.enabled` | true | inject relevant memory before the model thinks. |
| `auto_recall.budget_chars` | 4000 | the size cap on the injected block. |
| `auto_recall.correction_capture` | true | when a message reads as pushback/frustration, record a DRAFT lesson on the first complaint and nudge the agent to refine it. |
| `auto_recall.repeat_guard` | true | surface an existing rule LOUD when a message strongly overlaps it ("you've hit this before"). |

The precision gates that keep the read path quiet (an absolute evidence floor, a
specificity check, a conversational filter) are pro knobs under `auto_recall.*`.
The defaults are tuned, so you rarely need to touch them.

The action-time repeat guard (`hooks.pretool_guard`, on by default) fires the
"do NOT repeat this" corpus - failures + captured corrections - at the moment a
tool call would walk into a past mistake, even with no user message in the loop.
Lessons and decisions also keep a decay floor (0.6) so a critical rule cannot
fade out of ranking; retire one with `archive`/`forget`.

## Shared daemon and connect

| Key | Default | What it does |
|---|---|---|
| `connect.default_daemon` | true | `pmb connect` and `pmb setup` point JSON hosts at the one shared daemon instead of a per client server. |
| `daemon.autostart` | true | when a cold hook runs and no daemon is live, start one in the background for the next message. |
| `daemon.idle_exit_min` | 120 | exit after this many idle minutes so a forgotten daemon does not hold memory forever. 0 means never. Connecting pins this to 0 so the connection stays warm. |
| `daemon.tool_profile` | (set by connect) | the MCP tool profile the daemon serves: minimal, lean, default, or full. |

## Ambient auto write

| Key | Default | What it does |
|---|---|---|
| `autowrite.enabled` | true | journal a turn the agent did not record itself. |
| `autowrite.min_importance` | 0.45 | the quality bar a turn must clear to be journaled. |
| `autowrite.synthesizer` | template | how the entry is written. `template` needs no model; set an LLM backend for richer summaries. |

## Agent logging (active mode)

| Key | Default | What it does |
|---|---|---|
| `agent.active_mode` | false | install proactive logging rules by default (the agent records its own decisions and lessons). Turned on by `pmb setup --active`. |
| `agent.log_decisions` / `log_completed` / `log_lessons` / `log_failures` / `log_goals` | true | which categories the agent logs in active mode. |
| `agent.apply_lessons` | true | the self improvement loop: recall lessons at the start of a task and follow them. |
| `agent.context_continuity` | true | re-orient after a long session with `session_brief`. |

## Background brain and maintenance

| Key | Default | What it does |
|---|---|---|
| `consolidate.auto_trigger` | false | run sleep stage consolidation automatically. |
| `consolidate.backend` | auto (pro) | the offline LLM tier: auto, claude, ollama, or anthropic. Background only, never on the read path. |
| `decay.factor_per_day` | 0.985 | the forgetting curve applied by `pmb decay`. |
| `dedup.enable` | true | deduplicate near identical memories. |
| `graph.extractor` | regex | how entities are pulled from text for the graph. |

## See everything

```bash
pmb config list --pro
```

There are around 150 advanced knobs. They are advanced on purpose: the defaults
are tuned, and a blocking eval gate protects recall quality, so changing them is
opt in and rarely needed.
