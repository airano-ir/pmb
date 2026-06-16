#!/usr/bin/env node
// `pmb-ai` (npm) launcher. Resolves a working `pmb` - installing the Python
// `pmb-ai` package on first use if it is missing - then forwards every arg.
//
// So `npx pmb-ai setup` is a FULL cycle: it bootstraps the Python package
// (uv / pipx / pip) and then runs `pmb setup`, rather than only downloading a
// shim or printing that everything is fine. A `pmb` already on PATH is used
// directly and nothing is installed.
"use strict";

const { spawnSync } = require("node:child_process");
const { ensurePmb, printNoToolchain, isWin, c } = require("./bootstrap.js");

const args = process.argv.slice(2);

const resolved = ensurePmb();
if (!resolved) {
  printNoToolchain();
  process.exit(1);
}

const { cmd, pre } = resolved;
const r = spawnSync(cmd, [...pre, ...args], { stdio: "inherit", shell: isWin });
if (r.error) {
  c.warn(`could not launch ${cmd}: ${r.error.message}`);
  process.exit(1);
}
process.exit(r.status === null ? 1 : r.status);
