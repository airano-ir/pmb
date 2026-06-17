# Memory model

This page describes what PMB stores, the kinds of memory it keeps, and how
memories age. For where these live on disk and how they are indexed, see
[Core engine](core-engine.md).

## Everything is an event

Every memory is an append-only event with a ULID, a type, content, an importance
score, a timestamp, access counters, and a tier. Events are not edited in place:
an update appends a new event, and a removal either archives (soft delete) or
purges (hard delete). That keeps history honest and recovery easy.

## Types of memory

PMB separates memory by type so the agent can treat each kind with the right
weight. The common types:

| Type | What it is | Where it comes from |
|---|---|---|
| `fact` | An atomic, durable fact | `record_fact`, `record_batch` |
| `keyed_fact` | A mutable personal attribute, stored as subject / attribute / value | `record_keyed_fact` |
| `decision` | A project-shaping choice (an activity that lands in long-term memory) | `record_batch` activity `kind=decision` |
| `lesson` | A procedural rule that surfaces before relevant work | `pmb learn`, `record_batch` |
| `goal` | A future intention with a status | `record_goal` |
| `activity` | A lightweight log of what happened in a turn | `record_activity` |
| `milestone` | A checkpoint in a named state-chain | `record_milestone` |
| `qa` | A stored question and answer pair | `remember` |
| `git`, `file`, `code` | Repository and code context | `pmb sync`, `index_project` |
| `image` | An image with an optional cross-modal embedding | `record_image` |

## Facts that change: keyed facts

Some facts have exactly one current value: your city, your employer, a pet's
name. `record_keyed_fact` stores these as `(subject, attribute, value)` and
upserts. Writing a new value for the same subject and attribute archives the old
one, so recall returns only the current value while the history stays queryable.

```text
record_keyed_fact("user", "city", "Warsaw")
record_keyed_fact("user", "employer", "Anthropic")
```

Keyed facts default to a high importance, since personal attributes usually
matter.

## Importance, pinning, and forgetting

- **Importance (0.0 to 1.0).** Every event has a score that drives ranking and
  decay. Defaults vary by type, so a fact starts higher than a routine activity.
- **Pin.** Pinning locks a memory at maximum importance, and a pinned memory is
  never auto-archived.
- **Forgetting curve.** A daily decay lowers importance over time, and an old,
  low-importance memory is eventually archived. Pinned memories are exempt. Tune
  it with `decay.factor_per_day`, or run `pmb decay` by hand.

## Three tiers: how fast memory fades

PMB keeps every event in one of three tiers that differ only in how fast they
fade. Recalling a memory promotes it toward a slower-fading tier, so what you use
sticks and what you never touch decays.

| Tier | Holds | Fades over |
|---|---|---|
| Working | New activity and routine events | days |
| Episodic | Specific events you have recalled a few times | weeks |
| Semantic | Facts, decisions, and lessons | months |

A memory moves Working → Episodic → Semantic as it proves useful through recall,
so frequently useful memory survives while one-off noise fades.

## Soft delete versus hard delete

Archiving (a soft delete) hides a memory from recall but keeps the row, so you
can restore it. A hard delete purges the row, its search vector, and its graph
links. For the full picture, see [Deleting memories](../guide/deleting-memories.md).
