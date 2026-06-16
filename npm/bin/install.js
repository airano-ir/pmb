#!/usr/bin/env node
// postinstall: make `npm i -g pmb-ai` end with a WORKING `pmb`, not just a JS
// shim or an "ok" message. Installs the Python pmb-ai package (uv / pipx / pip).
// Always non-fatal and skippable, so it can never break `npm install` itself.
"use strict";

const { ensurePmb, probe, printNoToolchain, c } = require("./bootstrap.js");

// Opt-out for CI / Docker layers that don't want a Python install side effect.
if (process.env.PMB_SKIP_POSTINSTALL) {
  c.dim("PMB_SKIP_POSTINSTALL set - skipping the Python bootstrap.");
  process.exit(0);
}

try {
  if (probe("pmb", ["--help"])) {
    c.ok("pmb is already installed.");
    c.dim("Next: run `pmb setup` (or `npx pmb-ai setup`) to wire your agent.");
    process.exit(0);
  }
  c.step("pmb-ai: bootstrapping the Python package so `pmb` actually works...");
  const r = ensurePmb();
  if (r) {
    c.ok("PMB is ready.");
    c.dim("Next: run `pmb setup` (or `npx pmb-ai setup`) to wire your agent.");
  } else {
    printNoToolchain();
    c.dim("(npm install still succeeded - PMB will finish installing on your "
      + "first `npx pmb-ai` command.)");
  }
} catch (e) {
  c.warn("bootstrap skipped: " + (e && e.message ? e.message : String(e)));
}

// Never fail the npm install over the Python bootstrap.
process.exit(0);
