#!/usr/bin/env bash
# Install the dev git hooks into this clone. Currently: a pre-commit hook that
# runs ruff + CodeQL (security-extended) so problems are caught BEFORE they
# reach CI. Coexists with PMB's own post-commit hook (`pmb track install`).
#
#   bash scripts/install-dev-hooks.sh
#
# Bypass a single commit with `git commit --no-verify`; skip just the slow
# CodeQL step with `SKIP_CODEQL=1 git commit ...`. Uninstall by deleting the
# hook file printed below.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

HOOK_DIR="$(git rev-parse --git-path hooks)"
mkdir -p "$HOOK_DIR"
cp scripts/hooks/pre-commit "$HOOK_DIR/pre-commit"
chmod +x "$HOOK_DIR/pre-commit"

echo "Installed pre-commit hook -> $HOOK_DIR/pre-commit"
echo "  runs: ruff + CodeQL (security-extended)"
echo "  bypass once:    git commit --no-verify"
echo "  skip CodeQL:    SKIP_CODEQL=1 git commit ..."
