# Team / multi-machine mode (optional)

> **You don't need this for local use.** The default `pmb connect <agent>`
> runs the MCP server as a child process of your agent over **stdio** — no
> network, no port, no token. This page is only for sharing one memory
> across several machines or people.

PMB's memory is a SQLite file plus a vector index. By default it lives next
to you and only your local agent talks to it. If you want **one shared
memory** — a team on the same workspace, or your own laptop + desktop —
you run the MCP server once over HTTP and point every agent at its URL.

## When you'd want it

| Situation | Why HTTP |
|---|---|
| A team of 5–15 on one workspace | One person records "don't touch X", everyone sees it |
| Your own multiple machines | Work laptop + home desktop on one memory |
| Memory on a server / NAS | One persistent process; agents connect from anywhere |

If none of these apply, stop here and use the local stdio default.

## Run the shared server

One persistent process, living next to the SQLite file (a homelab box, a
Tailscale node, a small VM):

```bash
pmb mcp serve --transport http --host 0.0.0.0 --port 8765 \
              --bearer-token "$(openssl rand -hex 32)"
```

Then on each developer's machine:

```bash
pmb connect claude --remote http://memo.local:8765/mcp \
                   --bearer-token <same secret>
```

Same MCP, same tools, one shared memory.

## Why the bearer token is required

Over **stdio** there's no auth because there's no network surface — the
channel is a pipe between parent and child process, and the OS isolates it.

Over **HTTP** the moment you open a port, anyone who can reach it can read
and write your memory. The bearer token is the only thing separating "my
agent" from "a random host on the network / mesh". So in HTTP mode it isn't
optional hardening — it's the boundary.

PMB does a constant-time comparison (`hmac.compare_digest`) so a leaked log
line can't side-channel a partial match. CORS preflights (`OPTIONS`) and the
health endpoint (`/healthz`, `/`) pass through unauthenticated; everything
else needs `Authorization: Bearer <token>`.

## Config via environment

`pmb mcp serve` is a thin wrapper over env vars, if you'd rather set them in
a systemd unit / Docker:

| Env | Default | Meaning |
|---|---|---|
| `PMB_MCP_TRANSPORT` | `stdio` | `streamable-http` for the shared server |
| `PMB_MCP_HOST` | `127.0.0.1` | `0.0.0.0` to accept LAN / mesh |
| `PMB_MCP_PORT` | `8765` | bind port (don't collide with `pmb dashboard`) |
| `PMB_MCP_PATH` | `/mcp` | mount path |
| `PMB_MCP_BEARER_TOKEN` | _(empty)_ | shared secret; empty = unauthenticated |

> Heads-up: `pmb dashboard` also defaults to port 8765. If you run both on
> one host, give one of them a different port.

## Tests

The auth contract is pinned in `tests/test_http_bearer_auth.py` (no token →
401, wrong token → 401, correct token → 200, preflight/health pass through,
empty token disables the gate).
