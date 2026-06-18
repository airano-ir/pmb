# PMB in Docker

Optional containerized mode. The default install path (`pip install pmb-ai`
or `make install`) is unchanged - this is purely an alternative that keeps
PMB off your host Python and out of your host `~/.pmb`.

## Layout

```
docker/
  Dockerfile     single image used by every compose profile
  data/          PMB_HOME inside the container (bind-mounted, git-ignored)
  README.md      this file
compose.yaml     (repo root) shell / dashboard / mcp profiles
```

One image is enough: PMB is local-first with embedded SQLite + LanceDB, so
there is no separate database service to split out.

## Data isolation

- `PMB_HOME=/data` → bind-mounted to `./docker/data`. Your host `~/.pmb` is
  never touched. Back up or inspect memory by looking in `./docker/data`.
- The HuggingFace embedding model (downloaded on first recall) is cached in
  the named volume `pmb_hf_cache`, so rebuilds don't re-download it.
- Drop a `config.yaml` into `./docker/data/` to set global PMB config; place
  per-workspace config under `./docker/data/workspaces/<id>/config.yaml`.

## CPU vs GPU

The image ships **CPU-only by default**, because for normal PMB usage a GPU
buys you almost nothing. Here's the reasoning, then the commands.

### Why CPU is the default

The only GPU-relevant work PMB does is computing **embeddings** - one forward
pass through a small sentence-transformer model, done once per stored memory
(at write time) and once per query (at recall time). Everything else in the
recall pipeline - BM25, SQLite lookups, the entity graph, reranking fusion -
is plain CPU work that a GPU can't speed up.

For interactive use that means:

- **A single-query embedding is tiny.** On CPU it's milliseconds. The real
  cost of the first recall is **loading the model into memory** (cold start),
  not the math - and a GPU doesn't make loading faster.
- **Warm recall is already fast on CPU** (single-digit to tens of ms), which is
  far below the latency of the LLM you're feeding. The agent never waits on PMB.

So a GPU sits idle during normal "remember / recall" traffic. It only earns its
keep on **bulk embedding** jobs:

- `pmb reindex` re-embedding a large existing corpus,
- importing a big history (`pmb import …`),
- running the LoCoMo / stress benchmarks over thousands of items.

There, embeddings are computed in large batches and GPU throughput genuinely
helps. If that's not your workload, stay on CPU.

### Differences at a glance

| | **CPU** (default, `pmb:local`) | **GPU** (opt-in, `pmb:gpu`) |
| :-- | :-- | :-- |
| torch wheel | `+cpu` (no CUDA) | default CUDA wheel + bundled NVIDIA libs |
| Image size | ~1.9 GB | ~5.9 GB |
| Host requirements | none - runs anywhere | NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| Interactive recall | fast (model load dominates) | no meaningful gain |
| Bulk re-embed / import / benchmarks | slower | faster (batched throughput) |
| When to use | normal agent memory, dev, dashboard | you routinely re-embed large corpora |

Note: a CUDA image on a host **without** a GPU is not just bigger - it can be
*slower* to start, because torch probes for CUDA and falls back. Don't build the
GPU image unless you actually have (and pass through) a GPU.

### Commands

```bash
# CPU (default)
make docker-build           # → image pmb:local
make docker-shell

# GPU (opt-in)
make docker-build-gpu       # → image pmb:gpu (CUDA torch)
make docker-shell-gpu
make docker-dashboard-gpu
```

Under the hood the GPU targets layer `docker/compose.gpu.yaml` over the base
compose (`docker compose -f compose.yaml -f docker/compose.gpu.yaml …`), which
flips the `TORCH_VARIANT` build arg to `cuda` and reserves the GPU. The two
images are tagged separately (`pmb:local` vs `pmb:gpu`) so they don't clobber
each other.

## Quick start

```bash
# build (UID/GID default to your user so ./docker/data stays writable)
make docker-build

# interactive dev sandbox: pip, python, pytest, ruff, pmb …
make docker-shell
#   inside: pmb note "hello from the container"
#           pmb recall "hello"
#           make test-core

# web dashboard → http://127.0.0.1:8765
make docker-dashboard

# run the core test suite in the container
make docker-test

# stop everything
make docker-down
```

Without the Makefile, the equivalents are:

```bash
docker compose build
docker compose run --rm shell
docker compose --profile dashboard up
docker compose run --rm -i mcp
docker compose --profile dev --profile dashboard --profile mcp down
```

## Wiring the MCP server to an agent

The MCP server talks over stdio, so the agent must launch it as a command.
Point your agent's MCP config at:

```json
{
  "mcpServers": {
    "pmb": {
      "command": "docker",
      "args": ["compose", "-f", "/abs/path/to/pmb/compose.yaml",
               "run", "--rm", "-i", "mcp"]
    }
  }
}
```

(Use an absolute path to `compose.yaml`; agents don't run from the repo root.)

## Warm daemon (share one model across agents)

The `mcp` profile above spawns a fresh server per `compose run`, so each call
pays the model cold-start. To keep ONE warm process that every agent and the
host CLI share, run the `daemon` profile instead - it serves MCP over HTTP on
`127.0.0.1:8765`:

```bash
docker compose --profile daemon up -d
pmb connect claude-code --remote http://127.0.0.1:8765/mcp
```

The container holds the Engine + embedding model; agents on the host stay thin
and just talk to that URL. It is bound to localhost; for LAN/Tailscale add a
real bind plus `--bearer-token <secret>` on `pmb mcp serve` and the matching
`--bearer-token` on `pmb connect`. The daemon shares port 8765 with the
`dashboard` profile, so run one of the two at a time.

## Notes

- The repo is bind-mounted at `/app`, and the package is installed editable,
  so source edits on the host take effect immediately in the container.
- If `./docker/data` ends up root-owned, rebuild with your IDs:
  `make docker-build UID=$(id -u) GID=$(id -g)`.
