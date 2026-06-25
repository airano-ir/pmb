# Does PMB actually help — and how we measure it honestly

Most memory tools assert that they improve your agent and back it with one
flattering number. PMB takes the opposite stance: **measure it conservatively,
on _your_ data, and say loudly when the signal isn't trustworthy yet.** A memory
system you can't measure is one you can't trust — so when PMB says a lesson
"helps", it has earned the word.

There are two different questions hiding inside "does it help?", and they need
two different methods.

| Question | Method | What it proves |
|---|---|---|
| Does recall find the **right** memory? | Retrieval benchmarks | quality of retrieval |
| Does **using** memory change outcomes? | Earned Memory | value of memory |

---

## 1. Retrieval quality — benchmarks

Can PMB surface the relevant memory at all? This is a search problem with
established benchmarks, and the numbers are reproducible on your machine — no
trust required:

```bash
python scripts/benchmarks/benchmark_locomo.py --n-conversations 10
python scripts/benchmarks/mega_stress_test.py
```

- **LoCoMo recall@10 ≈ 94.5%** (long-conversation QA)
- **Multilingual top-10 ≈ 99.2%** across ~11 languages

**Honest limit:** a retrieval benchmark proves PMB _finds_ the memory. It does
**not** prove that having it changed what your agent did. That is the harder
question, and it needs the next method.

---

## 2. Earned Memory — does memory change outcomes?

PMB already records two things as you work, with no extra LLM call:

1. **Which lessons were active** during each agent action (`surface_ids`).
2. **The outcome of each turn**, classified model-free from the actions
   themselves — tests passed/failed, a red→green fix, a build, a deploy.

Joining the two answers "when this lesson was in front of the agent, did the
turn go better?" — the thing that otherwise needs a manual A/B. The join runs
entirely over existing tables (`pmb health lessons-impact` /
the `lesson_impact` MCP tool).

The catch is that a naïve join lies in three ways, so Earned Memory is built in
**three layers of increasing rigour** — and refuses to overclaim at each one.

### Layer 1 — associational lift (weakest)

`lift = success_rate(turns with the lesson) − success_rate(turns with no lesson)`.

Useful as a first look, but **confounded**: lessons surface on harder, more
specific turns, so a genuinely helpful lesson can show _negative_ lift simply
because its turns were intrinsically harder (confounding by indication). Lift is
a flag for review, never ground truth.

### Layer 2 — statistical honesty

A point estimate over a handful of turns is noise. So every lesson also carries:

- a **95% Wilson confidence interval** on its success rate (sane at the tiny `n`
  the outcome signal actually produces, where the textbook interval breaks);
- a conservative **`verdict`** — `useful`/`harmful` **only** when the CI clears
  the baseline _and_ `n ≥ min_n`; otherwise `unverified` (seen, not yet
  provable) or `insufficient` (too few turns to judge);
- a top-level **`signal_sufficiency`** (`insufficient` / `thin` / `ok`) and a
  **`trustworthy`** flag.

The point: an `n=1` fluke can **never** read as a real effect.

### Layer 3 — within-lesson causal read (strongest)

The cleanest available control without randomisation: compare the **same
lesson** when it was **followed** versus **ignored**. Both arms share the same
trigger population ("this lesson was relevant and surfaced"), so this holds the
surfacing trigger fixed — the confounder that wrecks Layer 1.

`causal_verdict` is `helps` / `hurts` only when both arms clear `min_n` and
their Wilson CIs separate; otherwise `inconclusive` / `insufficient`. Fields:
`n_followed`, `n_ignored`, `followed_success_rate`, `ignored_success_rate`,
`followed_lift`.

---

## What PMB will not do

**An untrustworthy metric never drives behaviour.** Earned Memory is
measurement-only: it does not feed ranking or decay until the outcome signal is
dense enough to trust. We would rather show you `insufficient` than let a
flattering-but-wrong number quietly re-weight your memory.

## The limitations, stated plainly

These are in the harness's own `caveat` field, not buried:

- **Confounding by indication** — Layer 1 lift is associational, not causal.
- **Sparse outcome signal** — only turns with a test/build/deploy/red→green
  count, so the measured turns are a biased, test-heavy subset.
- **Surfaced ≠ followed** — a lesson can be shown and ignored.
- **Residual confound (Layer 3)** — the agent's _choice_ to follow a lesson may
  itself track task type. Better than Layer 1, still observational.

---

## Run it on your own data

```bash
pmb health lessons-impact -w 90
```

Read it like this:

- **`signal: insufficient`** and **`causal read pending`** early on is the
  honest answer, **not a bug** — outcome turns are rare, so a young workspace
  simply hasn't earned a verdict yet.
- A lesson only earns **`useful`/`helps`** once the statistics back it.
- Worst-lift-first ordering surfaces dead-weight or harmful rules for review.

That is the whole philosophy in one screen: PMB tells you what it can prove,
flags what it can't, and never dresses up noise as a result.

---

See also: [How it works](how-it-works.md) ·
[Core engine](core-engine.md) ·
[`pmb health lessons-impact`](../reference/COMMANDS.md)
