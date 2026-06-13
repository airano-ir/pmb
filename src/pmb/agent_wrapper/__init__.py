"""
pmb.agent_wrapper - scaffold for a client-side wrapper around the Claude API
that controls its own context compaction and uses PMB for long-term memory.

This is the architecture the original "Cognitive Memory" pitch's mechanic #1
(in-session smart compression) actually requires. Claude Code's auto-compact
runs inside the agent and cannot be hooked from an external MCP server, so to
get *selective* compression you have to own the chat loop yourself.

## Status

Scaffold only. Bones, not muscle. See `agent_wrapper/PLAN.md` for the
concrete N-week buildout. Today this package contains:

- `loop.py`      - minimal chat loop that talks to Anthropic API and writes
                   each turn into PMB. **Works** as a basic agent.
- `budget.py`    - token budget accounting (approximate; uses Anthropic's
                   `count_tokens` when available). **Works.**
- `policy.py`    - interface for compression policies. One naive policy
                   (`DropOldestNarrative`) implemented. **Selective
                   compression is the hard part and is NOT done yet.**
- `__main__.py`  - `python -m pmb.agent_wrapper` entrypoint. **Works
                   for a one-shot CLI chat session.**

What this is NOT yet:

- A Claude Code replacement (no tool use, no file editing, no MCP client side).
- Anywhere near production-tested.
- Sufficient evidence the pitch's mechanic #1 actually helps a user.

Use this as: a place to start the wrapper project, with the memory plumbing
already wired up, so the only research question left is "what compression
policy actually works".
"""
