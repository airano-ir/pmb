"""
PMB v2 reasoning layer.

This is the *thinking* part of the system, distinct from store-and-search.
It runs in the background ("idle / sleep") and turns raw events into
understanding: reflections (why this matters), causation (what caused what),
arcs (narrative threads).

Modules:
  reflect.py   - LLM-driven 'why does this matter?' reflection
  causation.py - typed event-to-event edges (caused, influenced, references)
  arcs.py      - narrative arc clustering and summarization
  router.py    - at read time, classify the question and dispatch

Design principles:
  1. Reflection happens OFFLINE (sleep/idle), never blocks recall.
  2. Reflection items are stored as regular events (event_type='reflection')
     so existing recall pipeline finds them. No new index needed.
  3. Causation graph uses a new edge_type column on existing graph_edges +
     a separate event_edges table for direct event↔event links.
  4. Arcs cluster events into narrative threads; LLM writes summaries.
  5. Read time stays fast: query router decides whether to pay the multi-hop
     traversal cost or just use the cheap hybrid path.

The whole layer degrades gracefully - if the LLM client isn't available or
reasoning has never run, the system behaves exactly like PMB v1.
"""
