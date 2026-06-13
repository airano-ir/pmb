# pmb-ai (npm launcher)

A thin npm wrapper for **[pmb-ai](https://pypi.org/project/pmb-ai/)**, the
Python package. It does not bundle Python; it forwards to the `pmb` CLI from the
PyPI release.

## Usage

```bash
npx pmb-ai setup        # wire your agent (Claude Code / Cursor / Codex / ...)
npx pmb-ai stats        # see what's stored
```

Or install globally:

```bash
npm install -g pmb-ai
pmb-ai setup
```

## How it works

`pmb-ai` resolves a runner in this order:

1. an existing `pmb` on your PATH (`pip install pmb-ai` / `pipx`), used directly;
2. otherwise `uvx --from pmb-ai pmb ...`, which runs the published PyPI release in
   an isolated environment (install [uv](https://docs.astral.sh/uv/) once);
3. otherwise it prints how to install Python/uv and exits.

So you need Python available one way or another - this wrapper is just a
convenient `npx` entry point. The full project, docs, and the MCP server
(`pmb-mcp`) live with the Python package.

- Main project: https://github.com/oleksiijko/pmb
- PyPI: https://pypi.org/project/pmb-ai/
- License: Apache-2.0
