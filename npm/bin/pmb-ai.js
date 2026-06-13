#!/usr/bin/env node
// Thin launcher: `pmb-ai` (npm) forwards to the `pmb` CLI from the PyPI package
// `pmb-ai`. It does NOT bundle Python. Resolution order:
//   1. an existing `pmb` on PATH (pip / pipx install) - used directly
//   2. else `uvx --from pmb-ai pmb ...` - runs the published PyPI release isolated
//   3. else print how to get Python/uv and exit non-zero
// NOTE: the PyPI package exposes `pmb` (CLI) and `pmb-mcp` (the stdio MCP server),
// so we launch by script name via `--from`, never `uvx pmb-ai` (no such script).
"use strict";

const { spawnSync } = require("node:child_process");

const args = process.argv.slice(2);
const isWin = process.platform === "win32";

function available(cmd, probe) {
  const r = spawnSync(cmd, probe, { stdio: "ignore", shell: isWin });
  return !r.error && r.status === 0;
}

function exec(cmd, cmdArgs) {
  const r = spawnSync(cmd, cmdArgs, { stdio: "inherit", shell: isWin });
  if (r.error) {
    return false;
  }
  process.exit(r.status === null ? 1 : r.status);
}

// 1. A pip/pipx-installed `pmb` already on PATH - fastest, respects the user's install.
if (available("pmb", ["--help"])) {
  exec("pmb", args);
}

// 2. Otherwise run the published PyPI package in an isolated env via uv.
if (available("uvx", ["--version"])) {
  exec("uvx", ["--from", "pmb-ai", "pmb", ...args]);
}

// 3. Neither is available - tell the user how to get one.
console.error("pmb-ai needs either the `pmb` CLI on your PATH or `uv` installed.\n");
console.error("Fastest (pip):");
console.error("  pip install pmb-ai && pmb setup\n");
console.error("Or install uv, then `npx pmb-ai` works with no Python setup:");
console.error(
  isWin
    ? '  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"'
    : "  curl -LsSf https://astral.sh/uv/install.sh | sh"
);
process.exit(1);
