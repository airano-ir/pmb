"""
PMB Web Dashboard.

A zero-dependency local web UI for inspecting PMB memory:
  - workspace stats
  - event timeline
  - entity graph
  - arcs
  - recall debugger (paste query, see results + per-layer scores)
  - manual pin / archive / feedback

Built on stdlib http.server - no Flask, no FastAPI, no Node. Single binary
loadable via `pmb dashboard`.
"""
