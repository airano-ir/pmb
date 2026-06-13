"""Thin, stdlib-only hook client (S2).

Every agent lifecycle event (UserPromptSubmit, SessionStart, PostToolUse, Stop)
spawns a fresh process. Routing those through the full `pmb` CLI loads typer +
numpy + rank_bm25 + rich (~0.8-1.0 s) before any hook logic runs - even when a
warm daemon will answer in ~10 ms. This package is the fast lane: it imports
ONLY stdlib plus the deliberately-light `pmb.mcp.registry` / `pmb.mcp.daemon`
(no engine, no fastmcp), talks to the daemon over localhost HTTP, and falls
back to the full CLI only when the daemon is absent.
"""
