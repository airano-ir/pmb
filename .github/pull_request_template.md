## What this changes

A short summary.

## Why

The problem it solves or the feature it adds.

## How verified

- [ ] Tests pass (`pytest tests/engine/test_graph.py tests/engine/test_persons.py tests/engine/test_goals_chains.py tests/engine/test_fact_tree.py tests/recall/test_recall_cache.py tests/engine/test_config.py tests/security/test_redact.py tests/recall/test_causation.py -q`)
- [ ] If recall accuracy could change: ran `python scripts/benchmark_locomo.py --n-conversations 3` and pasted the number
- [ ] If write-path latency could change: timed `engine.record_batch_async` with the smoke test in `scripts/bench_qa_scenarios.py`

## Notes for reviewer

Anything non-obvious about the implementation.
