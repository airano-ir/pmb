# Contributing to PMB

Thanks for your interest. PMB is intentionally small and opinionated; here is what helps and what doesn't.

## Development setup

```bash
git clone <repo-url> pmb
cd pmb
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .
pip install textual                  # for the TUI
pip install pytest                   # for tests
```

Verify:

```bash
make test        # 88 passed in ~80s — the same set CI runs
make test-smoke  # 9 passed in ~5s   — import-weight regression tests
```

## Running tests

**Use `make test`, not `pytest tests/` directly.** The full pytest target (`pytest tests/`) has a known deadlock: several modules trigger parallel `huggingface_hub` model downloads against the same on-disk cache, and the second downloader blocks waiting for a lock the first one hasn't released. Symptom: pytest hangs forever on collection.

| Target | What it runs | When to use |
|---|---|---|
| `make test` (= `make test-core`) | 8 deterministic files from `.github/workflows/ci.yml`, 88 tests | Default during development; matches CI |
| `make test-smoke` | `tests/meta/test_lightweight_imports.py`, 9 tests | After touching `pmb/__init__.py` or any module added to the lazy-attribute table |
| `make test-all-WARN` | The full `tests/` directory | Only if your HF cache is already populated; otherwise it will hang |
| `pytest tests/test_X.py` | One specific file | Reproducing a single failure |

If you add a new test that depends on a HuggingFace download, please mark it (`@pytest.mark.heavy`, manual `pytest -k`) and document why - we'd rather not grow the deadlock surface.

## Project layout

```
src/pmb/
  core/             - engine, events, search, workspace, recall_cache
  graph/            - entities, persons, store (SQLite-backed)
  reasoning/        - facts, reflect, causation, arcs, temporal, dedup, typo_fix
  mcp/              - FastMCP server, perf tracking, tools schema
  dashboard/        - local web UI (HTTP, no framework)
  cli/              - typer entry points (main, tui_*, ollama_cmd, connect)
  agent_wrapper/    - pmb-chat (optional standalone chat loop)
  health/           - consolidation, doctor checks
  eval/             - LoCoMo judge helpers
tests/              - pytest, grouped by subsystem (lang/recall/engine/hooks/…)
scripts/            - benchmarks, demos, profilers
```

## Where to add things

| You want to… | Touch this |
|---|---|
| add a recall ranking signal | `core/engine.py` (search the `recall(` method, ~12 stages) |
| change a tunable knob | `config.py` (single SCHEMA dict) |
| add an MCP tool | `mcp/server.py` (decorate with `@mcp.tool()`) |
| add a CLI command | `cli/main.py` |
| add a dashboard tab | `dashboard/static/index.html` + a handler in `dashboard/server.py` |
| add a TUI tab | `cli/tui_workspace.py` |

## Code style

- Keep modules focused. The recall pipeline is already large; do not add new top-level layers without a clear reason.
- Public methods on `Engine` should be readable from a docstring alone.
- All write-path additions must be sub-100 ms on warm cache. Profile before merging if you add embedding work.
- Pure-Python where possible. PyTorch is fine (already a dep); avoid adding new heavy dependencies.

## Tests

- Tests are grouped by subsystem under `tests/`: `lang/`, `recall/`, `engine/`, `hooks/`, `mcp/`, `ingest/`, `maintenance/`, `security/`, `cli/`, `integration/`, `eval/`, `meta/`. Put a new test in the folder matching what it exercises; frozen baselines live in `tests/fixtures/`.
- Tests use temp workspaces (`tempfile.mkdtemp`); don't write to `~/.pmb/` from a test.
- For features that touch recall scoring, add a test in `tests/engine/test_graph.py` style that asserts ordering, not exact scores.
- The full LoCoMo bench (`scripts/benchmark_locomo.py`) is the integration test for retrieval quality.

## Pull requests

- One concern per PR.
- Include a short description of what changed and **a benchmark line** if recall accuracy or latency could be affected.
- Updating the README's benchmark numbers is fine when you have new data - link to the run that produced them.

## Hardening passes (style for cross-cutting refactors)

If you are doing a sweeping technical-debt pass (lazy imports, exception handling, type annotations, etc.) please follow this protocol so it stays reviewable:

1. **Phase 0 - inventory first.** Before changing anything, post the current state to the PR description: file/line numbers, what each call does, why it exists. For exception-handler changes specifically, see `docs/HARDENING_NOTES.md` for the categorisation we want to preserve.
2. **One phase per commit.** Lazy imports, conftest, smoke tests, exception logging - each is its own commit so we can bisect a regression.
3. **Run `make test` after every commit.** Not `make test-all-WARN`.
4. **Behaviour-preserving by default.** If a change alters runtime behaviour (e.g. converting a silent fallback into an exception), call it out explicitly and add a test.
5. **Update CHANGELOG.md.** Hardening counts as a real change; document the measurable effect (`import pmb` time, test count, etc.).

## What we are not looking for

- Multi-user, multi-device, cloud sync. PMB is single-user single-machine on purpose.
- Frontend frameworks. The dashboard is plain HTML/JS; the TUI is textual. Keep it that way.
- New embedding backends. We have sentence-transformers (default) and fastembed (optional). Adding a third needs a strong case.

## Filing issues

Useful issues include: a minimal repro (or a workspace dump), the version of PMB (`pip show pmb`), and what command/agent triggered it. "It feels slow" without timing data is harder to act on - `pmb tui` → `[3] Stats` shows the real numbers.

## Licensing of contributions

PMB is licensed under **Apache License 2.0** (see [`LICENSE`](LICENSE)).

By submitting a pull request, you agree that your contribution is licensed under the same terms - this is the default behaviour spelled out in Apache 2.0 §5 ("Submission of Contributions"). No separate CLA, no sign-off chain. Just open the PR.

If your contribution includes code or assets you did not write, list their origin and license in the PR description so we can add them to [`NOTICE`](NOTICE).
