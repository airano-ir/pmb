"""
pmb.graph — entity-association graph layered on top of events.

We store events in SQLite as flat rows. The graph adds *entities* (concepts,
files, technologies, decisions) and *edges* (co-mentions). At recall time
the graph augments the existing BM25+vector hybrid by expanding a query
into its entities and pulling in events that share entities, even when
no direct token or embedding match exists.

Two tables (added via migration v2):
  - graph_entities (id, workspace_id, kind, name, n_mentions, last_seen)
  - graph_edges    (workspace_id, entity_a_id, entity_b_id, weight, last_seen)
  - graph_event_entities (event_ulid, entity_id)  — many-to-many

Entity kinds:
  - file       — paths like 'src/auth.py', 'db.py'
  - tech       — known technology names (Postgres, Redis, Docker, ...)
  - concept    — lowercase nouns extracted from content (best-effort regex)
  - decision   — events tagged as decision (fact type with keywords)

Extraction is rule-based and cheap (no LLM at write time). For deeper
extraction, `pmb consolidate` may add LLM-derived entities later.
"""

from pmb.graph.entities import (
    KNOWN_TECHS,
    EntityExtractor,
    extract_file_paths,
    extract_techs,
)
from pmb.graph.store import Edge, Entity, GraphStore

__all__ = [
    "EntityExtractor", "KNOWN_TECHS", "extract_file_paths", "extract_techs",
    "GraphStore", "Entity", "Edge",
]
