// Shared bootstrap for the npm package. The npm package is a thin launcher for
// the Python `pmb-ai` package; this module is what makes `npx pmb-ai ...` and
// `npm i -g pmb-ai` actually END UP WITH A WORKING `pmb`, instead of only
// downloading the JS shim or printing "ok".
//
// ensurePmb() resolves a runnable `pmb`:
//   1. an existing `pmb` on PATH (pip / pipx / uv tool / system) is used as-is
//   2. else it INSTALLS the Python package persistently, preferring:
//        uv tool install pmb-ai   ->   pipx install pmb-ai   ->   pip install --user pmb-ai
//      then re-probes PATH; if the freshly installed `pmb` is not yet on this
//      process's PATH, it falls back to `uvx --from pmb-ai pmb` (works now; the
//      persistent install means every future shell has `pmb`).
//   3. else returns null (no Python toolchain found) so the caller can guide.
"use strict";

const { spawnSync } = require("node:child_process");

const isWin = process.platform === "win32";

function probe(cmd, args) {
  const r = spawnSync(cmd, args, { stdio: "ignore", shell: isWin });
  return !r.error && r.status === 0;
}

function run(cmd, args) {
  // Inherit stdio so the user SEES the real install happening (download bars,
  // resolver output) instead of a silent spinner or a fake "ok".
  const r = spawnSync(cmd, args, { stdio: "inherit", shell: isWin });
  return !r.error && (r.status === 0 || r.status === null);
}

// Minimal ANSI so progress reads as a real, stepped install. All to stderr so
// stdout stays clean for any forwarded command output.
const c = {
  step: (m) => process.stderr.write(`\n\x1b[36m→ ${m}\x1b[0m\n`),
  ok: (m) => process.stderr.write(`\x1b[32m✓ ${m}\x1b[0m\n`),
  warn: (m) => process.stderr.write(`\x1b[33m! ${m}\x1b[0m\n`),
  dim: (m) => process.stderr.write(`\x1b[2m${m}\x1b[0m\n`),
};

function pythonCmd() {
  for (const py of ["python3", "python", "py"]) {
    if (probe(py, ["-m", "pip", "--version"])) return py;
  }
  return null;
}

// Try each installer in turn. Returns the method that succeeded, or null.
function installPmb() {
  if (probe("uv", ["--version"])) {
    c.step("Installing pmb-ai with uv (uv tool install pmb-ai)...");
    if (run("uv", ["tool", "install", "--force", "pmb-ai"])) return "uv";
    c.warn("uv install did not complete; trying the next method.");
  }
  if (probe("pipx", ["--version"])) {
    c.step("Installing pmb-ai with pipx (pipx install pmb-ai)...");
    if (run("pipx", ["install", "pmb-ai"])) return "pipx";
    c.warn("pipx install did not complete; trying the next method.");
  }
  const py = pythonCmd();
  if (py) {
    c.step(`Installing pmb-ai with pip (${py} -m pip install --user pmb-ai)...`);
    if (run(py, ["-m", "pip", "install", "--user", "--upgrade", "pmb-ai"])) return "pip";
    c.warn("pip install did not complete.");
  }
  return null;
}

// Resolve how to invoke pmb: { cmd, pre } where the full call is
// `cmd [...pre] [...userArgs]`. Installs the Python package if needed.
// opts.autoInstall=false only probes (never installs).
function ensurePmb(opts) {
  const autoInstall = !opts || opts.autoInstall !== false;

  if (probe("pmb", ["--help"])) {
    return { cmd: "pmb", pre: [], method: "path" };
  }
  if (!autoInstall) return null;

  const method = installPmb();
  if (method) {
    // Best case: the installer put `pmb` on PATH for this process too.
    if (probe("pmb", ["--help"])) {
      c.ok(`pmb installed via ${method} and is on your PATH.`);
      return { cmd: "pmb", pre: [], method };
    }
    // uv tool / pip --user can install to a bin dir not yet on THIS process's
    // PATH. uvx runs the just-published package now; the persistent install
    // means new shells get a bare `pmb`.
    if (probe("uvx", ["--version"])) {
      c.ok(`pmb installed via ${method}.`);
      c.dim("Its bin dir is not on this shell's PATH yet - using uvx for now. "
        + "Open a new terminal (or run the installer's PATH hint) for a bare `pmb`.");
      return { cmd: "uvx", pre: ["--from", "pmb-ai", "pmb"], method };
    }
    c.ok(`pmb installed via ${method}.`);
    c.warn("Could not find `pmb` on PATH in this shell. Open a NEW terminal, "
      + "then run `pmb setup`.");
    return null;
  }

  // No installer succeeded but uv exists: at least run the published package
  // ephemerally so the command still works this once.
  if (probe("uvx", ["--version"])) {
    return { cmd: "uvx", pre: ["--from", "pmb-ai", "pmb"], method: "uvx" };
  }
  return null;
}

function printNoToolchain() {
  c.warn("PMB needs Python (3.11+) to run, and none was found.");
  c.dim("Fastest path - install uv, then re-run, and PMB installs itself:");
  c.dim(isWin
    ? '  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"'
    : "  curl -LsSf https://astral.sh/uv/install.sh | sh");
  c.dim("Or with pip:  pip install pmb-ai && pmb setup");
}

module.exports = { ensurePmb, probe, run, printNoToolchain, c, isWin };
