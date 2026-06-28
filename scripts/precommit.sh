#!/usr/bin/env bash
# Pre-commit gate: lint (fast) + CodeQL security scan (the same security-extended
# suite CI runs). Blocks the commit if either finds a real problem.
#
#   bash scripts/precommit.sh            # run the gate by hand
#   SKIP_CODEQL=1 bash scripts/precommit.sh   # lint only (skip the ~1-2 min scan)
#
# Wire it as a git hook with: bash scripts/install-dev-hooks.sh
# Bypass once with: git commit --no-verify
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo ">> ruff"
ruff check src tests scripts

if [ "${SKIP_CODEQL:-0}" = "1" ]; then
  echo ">> CodeQL skipped (SKIP_CODEQL=1)"
else
  echo ">> CodeQL security-extended (~1-2 min; SKIP_CODEQL=1 to skip)"
  bash scripts/codeql_local.sh
fi

echo ">> pre-commit checks passed"
