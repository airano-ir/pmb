#!/usr/bin/env bash
# Run the test suite.
#
#   bash scripts/test.sh                # the whole suite (CI-equivalent)
#   bash scripts/test.sh tests/recall   # one directory
#   bash scripts/test.sh -k dashboard   # one keyword
#   bash scripts/test.sh tests/integration/test_dashboard.py -q
#
# Any arguments are passed straight through to pytest. With no arguments it runs
# the full blocking set the way CI does: skips the load-flaky `quarantined`
# tests and the `perf` benchmarks, and ignores two suites that need optional
# dev deps not always installed (hypothesis, pytest-asyncio).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ "$#" -gt 0 ]; then
  exec python -m pytest -q "$@"
fi

exec python -m pytest -q -m "not perf and not quarantined" \
  --ignore=tests/eval/test_property_invariants.py \
  --ignore=tests/integration/test_mcp_e2e.py
