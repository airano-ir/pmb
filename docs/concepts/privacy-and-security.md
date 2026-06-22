# Privacy and security

PMB is local-first by design. Your memory is a SQLite file and a vector index on
your own machine, and nothing leaves it unless you run a sync or export command
on purpose.

## What stays local

- **Storage.** The global config, the daemon token, and every workspace (SQLite,
  LanceDB, and the side indexes) live under `~/.pmb/` on your machine.
- **The read path.** Recall runs locally with no LLM call and no network request.
- **The dashboard.** It binds to `127.0.0.1` only, so it is never exposed to the
  network.
- **No telemetry, no call-home.** There is no PMB server to phone, by design.

The only commands that touch the network are the explicit sync and export
commands, and only when you run them.

## Secret redaction

Before an event is written, PMB scans its text for credentials and replaces each
one with a `[REDACTED:<kind>]` marker, so secrets are not stored in memory. It
recognizes common shapes, including:

- **API keys:** Anthropic, OpenAI, Google, Stripe, and AWS access keys.
- **Tokens:** GitHub, Slack, JWTs, and bearer or `Authorization` headers.
- **PEM private key blocks.**
- **`KEY=value` lines** for `API_KEY`, `SECRET`, `PASSWORD`, `TOKEN`,
  `DATABASE_URL`, and similar.

Redaction is pattern-based, so a bare high-entropy string with no recognizable
prefix can still slip through. Treat it as a strong safety net, not a guarantee.

## Memory poisoning and procedural lessons

Local-first storage keeps PMB private, but it does not make every remembered
fact equally authoritative. PMB can surface lessons and failures as rules before
an agent acts, so a bad or low-trust memory can become a future instruction if it
is recorded without review.

A few practical boundaries help:

- Treat recalled lessons as **evidence with provenance**, not invisible policy.
  Check the source, confidence, freshness, and `surface_id` when a recalled
  lesson would affect a risky edit, command, or credential-adjacent action.
- Review durable rules periodically with `pmb lessons` and the broader memory
  view with `pmb audit`.
- Be more conservative with automatic writes than explicit `remember` / `lesson`
  entries. If an untrusted session, imported transcript, issue body, or noisy
  agent run may have polluted memory, archive those automatic writes with
  `pmb forget-auto` before relying on them.
- Keep imported third-party histories in a separate workspace until you trust the
  resulting facts and lessons.

The goal is not to stop remembering; it is to keep remembered context from
silently becoming command authority.

## Team mode: the bearer token is the boundary

Locally there is no network surface: clients use stdio or a daemon bound to
`127.0.0.1`. The moment you expose the MCP server over HTTP for a team, the
bearer token is the only thing separating your agent from any host that can
reach the port.

- **Required in HTTP mode.** Set `PMB_MCP_BEARER_TOKEN`, or pass
  `pmb mcp serve --bearer-token`.
- **Constant-time check.** PMB compares tokens with `hmac.compare_digest`, so a
  leaked log line cannot side-channel a partial match.
- **What passes unauthenticated.** Only the CORS preflight (`OPTIONS`) and the
  health endpoints (`/healthz`, `/`). Everything else needs
  `Authorization: Bearer <token>`.
- **Keep it private.** Bind beyond localhost only in team mode, and put it behind
  a private network such as Tailscale or an SSH tunnel.

See [Team and remote](../guide/TEAM.md) for the full setup.

## Encrypted, portable export

`pmb workspace export` packs a whole workspace into one encrypted bundle that is
safe to store even on a public remote.

- **Key derivation:** scrypt from a passphrase, or a raw 32-byte key file.
- **Cipher:** authenticated encryption (AES plus HMAC), so tampering is detected
  on import rather than silently accepted.
- **Install:** needs `pip install 'pmb-ai[crypto]'`.

```bash
pmb workspace export memory.enc        # prompts for a passphrase
pmb workspace import memory.enc work   # restores into a workspace named "work"
```

## Deletion you can trust

Archiving is the reversible default. A hard delete purges the event row, its
search vector, and its graph links, so nothing points at a memory that no longer
exists. For the full picture, see [Deleting memories](../guide/deleting-memories.md).

## Non-goals

- No hosted PMB service, no telemetry, and no call-home.
- No silent network calls on the read or write path. Optional LLM passes such as
  `consolidate`, `reflect`, and `distill` are explicit and opt-in.
