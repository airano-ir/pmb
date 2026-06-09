"""
Stdlib HTTP server exposing PMB internals to a local web UI.

Endpoints:
  GET  /                     → dashboard HTML
  GET  /static/<file>        → static assets
  GET  /api/stats            → workspace + memory stats
  GET  /api/events?limit=50  → recent events (paginated)
  GET  /api/entities?limit=30 → top entities by mentions
  GET  /api/entity_events?entity_id=N → events linked to one entity (for Map node-click)
  GET  /api/arcs             → narrative arcs
  GET  /api/event/<ulid>     → one event detail (+ reflections, facts, edges)
  POST /api/recall           → run recall (body: {"query": "...", "top_k": 10})
  POST /api/pin/<ulid>       → pin event
  POST /api/unpin/<ulid>     → unpin
  POST /api/archive/<ulid>   → archive
  POST /api/feedback         → log feedback (body: {ulid, verdict})

Binds to 127.0.0.1 only (no remote access by default).
"""

from __future__ import annotations

import json
import logging
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs


log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def make_handler(engine):
    """Factory that closes over the engine instance."""

    class _Handler(BaseHTTPRequestHandler):
        # ------------------------------------------------------------------
        # Output helpers
        # ------------------------------------------------------------------

        def _send_json(self, payload: dict | list, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            # No Access-Control-Allow-Origin: the dashboard UI is served
            # same-origin from this very server, so it needs no CORS grant.
            # A wildcard ("*") would let ANY website the user visits read the
            # local memory store via cross-origin fetch to 127.0.0.1:8765
            # (and POST to the archive/merge endpoints). See SECURITY.md.
            self.end_headers()
            self.wfile.write(body)

        def _send_static(self, path: Path) -> None:
            if not path.exists() or not path.is_file():
                self.send_error(404)
                return
            ctype, _ = mimetypes.guess_type(str(path))
            ctype = ctype or "application/octet-stream"
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):
            # Quiet — don't spam stdout per request
            log.debug(fmt, *args)

        # ------------------------------------------------------------------
        # Routing
        # ------------------------------------------------------------------

        def do_GET(self):
            parsed = urlparse(self.path)
            route = parsed.path
            qs = parse_qs(parsed.query or "")

            try:
                if route == "/" or route == "/index.html":
                    self._send_static(STATIC_DIR / "index.html")
                    return
                if route.startswith("/static/"):
                    rel = route[len("/static/"):]
                    self._send_static(STATIC_DIR / rel)
                    return
                if route == "/api/stats":
                    self._send_json(self._handle_stats())
                    return
                if route == "/api/events":
                    limit = int((qs.get("limit") or ["50"])[0])
                    self._send_json(self._handle_events(limit))
                    return
                if route == "/api/entities":
                    limit = int((qs.get("limit") or ["30"])[0])
                    kind = (qs.get("kind") or [None])[0]
                    self._send_json(self._handle_entities(limit, kind))
                    return
                if route == "/api/arcs":
                    self._send_json(self._handle_arcs())
                    return
                if route == "/api/graph":
                    limit = int((qs.get("limit") or ["80"])[0])
                    self._send_json(self._handle_graph(limit))
                    return
                if route.startswith("/api/event/"):
                    ulid = route[len("/api/event/"):]
                    self._send_json(self._handle_event_detail(ulid))
                    return
                if route == "/api/dedup_candidates":
                    limit = int((qs.get("limit") or ["100"])[0])
                    self._send_json(self._handle_dedup_candidates(limit))
                    return
                if route == "/api/perf":
                    hours = float((qs.get("hours") or ["24"])[0])
                    self._send_json(self._handle_perf(hours))
                    return
                if route == "/api/entity_events":
                    try:
                        eid = int((qs.get("entity_id") or ["0"])[0])
                    except ValueError:
                        eid = 0
                    elim = int((qs.get("limit") or ["20"])[0])
                    self._send_json(self._handle_entity_events(eid, elim))
                    return
                if route == "/api/lessons":
                    days = float((qs.get("days") or ["7"])[0])
                    self._send_json(self._handle_lesson_stats(days))
                    return
                if route == "/api/adherence":
                    days = float((qs.get("days") or ["7"])[0])
                    self._send_json(self._handle_adherence(days))
                    return
                self.send_error(404)
            except Exception as e:
                log.exception("GET %s failed", route)
                self._send_json({"error": str(e)}, status=500)

        def do_POST(self):
            parsed = urlparse(self.path)
            route = parsed.path
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length > 0 else b""
            try:
                payload = json.loads(body.decode("utf-8") or "{}")
            except Exception:
                payload = {}
            try:
                if route == "/api/recall":
                    self._send_json(self._handle_recall(payload))
                    return
                if route.startswith("/api/pin/"):
                    self._send_json(self._handle_pin(route[len("/api/pin/"):], pin=True))
                    return
                if route.startswith("/api/unpin/"):
                    self._send_json(self._handle_pin(route[len("/api/unpin/"):], pin=False))
                    return
                if route.startswith("/api/archive/"):
                    self._send_json(self._handle_archive(route[len("/api/archive/"):]))
                    return
                if route == "/api/feedback":
                    self._send_json(self._handle_feedback(payload))
                    return
                if route == "/api/dedup_merge":
                    self._send_json(self._handle_dedup_merge(payload))
                    return
                if route == "/api/dedup_keep_both":
                    self._send_json(self._handle_dedup_keep(payload))
                    return
                if route == "/api/dedup_sweep":
                    self._send_json(self._handle_dedup_sweep(payload))
                    return
                self.send_error(404)
            except Exception as e:
                log.exception("POST %s failed", route)
                self._send_json({"error": str(e)}, status=500)

        # ------------------------------------------------------------------
        # Handlers
        # ------------------------------------------------------------------

        def _handle_stats(self) -> dict:
            stats = engine.stats() if hasattr(engine, "stats") else {}
            graph_stats = engine.graph_stats() if hasattr(engine, "graph_stats") else {}
            return {"workspace": stats, "graph": graph_stats}

        def _handle_events(self, limit: int) -> list[dict]:
            evs = engine.events.list_active(engine.workspace.id, limit=limit)
            ulids = [e.ulid for e in evs]
            # Bulk-load top entities per event so the Timeline UI can group
            # by project without N+1 queries. We pick the 3 highest-mention
            # entities per event — that's plenty for the lane assignment
            # heuristic ("project = top entity") plus a tooltip.
            ent_map: dict[str, list[dict]] = {u: [] for u in ulids}
            if ulids:
                import sqlite3
                placeholders = ",".join("?" * len(ulids))
                with sqlite3.connect(engine.workspace.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        f"""
                        SELECT ee.event_ulid, e.id, e.kind, e.name, e.n_mentions
                        FROM graph_event_entities ee
                        JOIN graph_entities e ON e.id = ee.entity_id
                        WHERE ee.event_ulid IN ({placeholders})
                          AND e.workspace_id = ?
                        ORDER BY e.n_mentions DESC
                        """,
                        (*ulids, engine.workspace.id),
                    ).fetchall()
                for r in rows:
                    lst = ent_map.setdefault(r["event_ulid"], [])
                    if len(lst) < 3:
                        lst.append({
                            "id": r["id"],
                            "kind": r["kind"],
                            "name": r["name"],
                            "mentions": r["n_mentions"],
                        })
            return [
                {
                    "ulid": e.ulid,
                    "event_type": e.event_type,
                    "content": (e.content or "")[:500],
                    "timestamp": e.timestamp,
                    "importance": e.importance,
                    "tier": e.tier,
                    "access_count": e.access_count,
                    "metadata": e.metadata,
                    "top_entities": ent_map.get(e.ulid, []),
                }
                for e in evs
            ]

        def _handle_lesson_stats(self, days: float) -> dict:
            """Self-improvement loop dashboard data."""
            try:
                return engine.lesson_follow_stats(days=days)
            except Exception as e:
                log.exception("lesson stats failed")
                return {"error": str(e)}

        def _handle_adherence(self, days: float) -> dict:
            """Adherence stats + a day-by-day breakdown for the chart."""
            try:
                base = engine.adherence_stats(days=days)
            except Exception as e:
                log.exception("adherence stats failed")
                return {"error": str(e)}
            # Add a per-day series so the dashboard can draw the chart.
            try:
                import sqlite3, time as _t
                cutoff = _t.time() - days * 86400.0
                read_tools = (
                    "recall","prepare","project_overview","find_lessons",
                    "overview","recent_activity","what_just_happened",
                    "session_brief","list_goals","recall_smart",
                    "get_subfacts","list_recent",
                )
                write_tools = (
                    "record_batch","record_fact","record_fact_tree",
                    "record_keyed_fact","record_goal","record_activity",
                    "record_milestone","remember","index_pdf","index_project",
                )
                with sqlite3.connect(engine.workspace.db_path) as conn:
                    rows = conn.execute(
                        "SELECT DATE(timestamp,'unixepoch') AS d, tool_name, COUNT(*) "
                        "FROM mcp_calls WHERE workspace_id=? AND timestamp >= ? "
                        "GROUP BY d, tool_name ORDER BY d",
                        (engine.workspace.id, cutoff),
                    ).fetchall()
                by_day: dict[str, dict] = {}
                for d, tool, n in rows:
                    rec = by_day.setdefault(d, {"day": d, "read": 0, "write": 0,
                                               "prepare": 0, "by_tool": {}})
                    rec["by_tool"][tool] = n
                    if tool in read_tools: rec["read"] += n
                    if tool in write_tools: rec["write"] += n
                    if tool == "prepare": rec["prepare"] += n
                series = sorted(by_day.values(), key=lambda r: r["day"])
                base["series"] = series
            except Exception:
                base["series"] = []
            return base

        def _handle_entity_events(self, entity_id: int, limit: int) -> dict:
            """Return events linked to one entity, plus the entity itself.
            Used by the Map view's node-click panel."""
            import sqlite3
            ws = engine.workspace.id
            with sqlite3.connect(engine.workspace.db_path) as conn:
                conn.row_factory = sqlite3.Row
                ent = conn.execute(
                    "SELECT id, kind, name, n_mentions FROM graph_entities "
                    "WHERE id = ? AND workspace_id = ?",
                    (entity_id, ws),
                ).fetchone()
                if not ent:
                    return {"entity": None, "events": []}
                rows = conn.execute(
                    """
                    SELECT ev.ulid, ev.event_type, ev.content, ev.timestamp,
                           ev.importance, ev.metadata_json
                    FROM graph_event_entities ee
                    JOIN events ev ON ev.ulid = ee.event_ulid
                    WHERE ee.entity_id = ?
                      AND ev.workspace_id = ?
                      AND ev.archived_at IS NULL
                    ORDER BY ev.timestamp DESC
                    LIMIT ?
                    """,
                    (entity_id, ws, limit),
                ).fetchall()
                events = []
                for r in rows:
                    md = {}
                    if r["metadata_json"]:
                        try:
                            md = json.loads(r["metadata_json"])
                        except Exception:
                            md = {}
                    events.append({
                        "ulid": r["ulid"],
                        "event_type": r["event_type"],
                        "content": (r["content"] or "")[:300],
                        "timestamp": r["timestamp"],
                        "importance": r["importance"],
                        "metadata": md,
                    })
            return {
                "entity": dict(ent),
                "events": events,
            }

        def _handle_entities(self, limit: int, kind: Optional[str]) -> list[dict]:
            ents = engine.graph.top_entities(
                engine.workspace.id, kind=kind, limit=limit,
            )
            return [e.to_dict() for e in ents]

        def _handle_arcs(self) -> list[dict]:
            try:
                return engine.list_arcs(limit=100)
            except Exception:
                return []

        # Improvement JJ: MCP perf stats
        def _handle_perf(self, hours: float) -> dict:
            try:
                from pmb.mcp.perf import get_perf_stats
                return get_perf_stats(
                    db_path=engine.workspace.db_path,
                    workspace_id=engine.workspace.id,
                    hours=hours,
                )
            except Exception as e:
                log.exception("perf stats failed")
                return {"error": str(e)}

        # Improvement U: dedup endpoints
        def _handle_dedup_candidates(self, limit: int) -> list[dict]:
            try:
                return engine.dedupe_list_pending(limit=limit)
            except Exception as e:
                log.exception("dedup candidates failed")
                return [{"error": str(e)}]

        def _handle_dedup_merge(self, payload: dict) -> dict:
            """User clicked 'merge'. Archive `new_ulid`, point at `canonical`."""
            import sqlite3, time as _t, json as _j
            new_ulid = payload.get("new_ulid")
            canonical = payload.get("canonical_ulid")
            pending_id = payload.get("pending_id")
            if not new_ulid or not canonical:
                return {"error": "missing ulids"}
            try:
                from pmb.reasoning.dedup import _archive_with_pointer, mark_verdict
                _archive_with_pointer(engine.workspace.db_path, new_ulid, canonical)
                if pending_id:
                    mark_verdict(engine.workspace.db_path, int(pending_id), "merge")
                return {"ok": True, "merged": new_ulid, "into": canonical}
            except Exception as e:
                log.exception("dedup merge failed")
                return {"error": str(e)}

        def _handle_dedup_keep(self, payload: dict) -> dict:
            """User clicked 'keep both'. Just mark the pair resolved."""
            pending_id = payload.get("pending_id")
            if not pending_id:
                return {"error": "missing pending_id"}
            try:
                from pmb.reasoning.dedup import mark_verdict
                mark_verdict(engine.workspace.db_path, int(pending_id), "keep_both")
                return {"ok": True}
            except Exception as e:
                return {"error": str(e)}

        def _handle_dedup_sweep(self, payload: dict) -> dict:
            """Trigger a full workspace sweep (background-ish — blocks request)."""
            threshold = float(payload.get("threshold", 0.92))
            try:
                return engine.dedupe_sweep(threshold=threshold)
            except Exception as e:
                log.exception("dedup sweep failed")
                return {"error": str(e)}

        def _handle_graph(self, limit: int) -> dict:
            """Return nodes + edges for visualization.
            Top-N entities by mentions; only edges between included nodes.

            Honors `graph.viz_min_mentions` so one-off concepts (mentions=1) can
            be hidden from the dashboard without touching the DB — useful when
            the regex extractor produces a long tail of word-noise.
            """
            import sqlite3
            ws = engine.workspace.id
            try:
                min_mentions = int(engine.config.get("graph.viz_min_mentions") or 1)
            except Exception:
                min_mentions = 1
            with sqlite3.connect(engine.workspace.db_path) as conn:
                conn.row_factory = sqlite3.Row
                ent_rows = conn.execute(
                    "SELECT id, kind, name, n_mentions FROM graph_entities "
                    "WHERE workspace_id = ? AND n_mentions >= ? "
                    "ORDER BY n_mentions DESC LIMIT ?",
                    (ws, min_mentions, limit),
                ).fetchall()
                included = {r["id"] for r in ent_rows}
                if not included:
                    return {"nodes": [], "edges": []}
                placeholders = ",".join("?" * len(included))
                edge_rows = conn.execute(
                    f"SELECT entity_a, entity_b, weight FROM graph_edges "
                    f"WHERE workspace_id = ? AND entity_a IN ({placeholders}) "
                    f"AND entity_b IN ({placeholders}) "
                    f"ORDER BY weight DESC LIMIT 500",
                    (ws, *included, *included),
                ).fetchall()
            nodes = [
                {
                    "id": r["id"],
                    "kind": r["kind"],
                    "name": r["name"],
                    "mentions": r["n_mentions"],
                }
                for r in ent_rows
            ]
            edges = [
                {
                    "source": r["entity_a"],
                    "target": r["entity_b"],
                    "weight": r["weight"],
                    # All graph edges in PMB are entity co-occurrence within
                    # the same event. There is no verb / typed relation yet —
                    # we expose this explicitly so the UI can label correctly.
                    "kind": "co_occurrence",
                }
                for r in edge_rows
            ]
            return {"nodes": nodes, "edges": edges}

        def _handle_event_detail(self, ulid: str) -> dict:
            ev = engine.events.get_by_ulid(ulid)
            if ev is None:
                return {"error": "event not found"}
            # Linked entities
            import sqlite3
            with sqlite3.connect(engine.workspace.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT e.kind, e.name, e.n_mentions
                    FROM graph_event_entities ee
                    JOIN graph_entities e ON e.id = ee.entity_id
                    WHERE ee.event_ulid = ?
                    """,
                    (ulid,),
                ).fetchall()
                entities = [
                    {"kind": r["kind"], "name": r["name"], "mentions": r["n_mentions"]}
                    for r in rows
                ]
                # Reflections / fact_atoms pointing to this event
                derived = conn.execute(
                    """
                    SELECT ulid, event_type, content
                    FROM events
                    WHERE workspace_id = ?
                      AND json_extract(metadata_json, '$.source_ulid') = ?
                      AND archived_at IS NULL
                    """,
                    (engine.workspace.id, ulid),
                ).fetchall()
                derived_list = [
                    {"ulid": r["ulid"], "event_type": r["event_type"],
                     "content": (r["content"] or "")[:300]}
                    for r in derived
                ]
                # Edges in/out
                edges_out = conn.execute(
                    "SELECT target_ulid, edge_type, confidence FROM event_edges "
                    "WHERE source_ulid = ?",
                    (ulid,),
                ).fetchall()
                edges_in = conn.execute(
                    "SELECT source_ulid, edge_type, confidence FROM event_edges "
                    "WHERE target_ulid = ?",
                    (ulid,),
                ).fetchall()
            return {
                "ulid": ev.ulid,
                "event_type": ev.event_type,
                "content": ev.content,
                "timestamp": ev.timestamp,
                "importance": ev.importance,
                "tier": ev.tier,
                "access_count": ev.access_count,
                "metadata": ev.metadata,
                "entities": entities,
                "derived": derived_list,
                "edges_out": [
                    {"target": r["target_ulid"], "type": r["edge_type"],
                     "confidence": r["confidence"]}
                    for r in edges_out
                ],
                "edges_in": [
                    {"source": r["source_ulid"], "type": r["edge_type"],
                     "confidence": r["confidence"]}
                    for r in edges_in
                ],
            }

        def _handle_recall(self, payload: dict) -> dict:
            query = payload.get("query") or ""
            top_k = int(payload.get("top_k") or 10)
            rerank = bool(payload.get("rerank") or False)
            if not query.strip():
                return {"error": "empty query"}
            pack = engine.recall(query, top_k=top_k, rerank=rerank)
            return pack.to_dict()

        def _handle_pin(self, ulid: str, pin: bool) -> dict:
            try:
                if pin:
                    engine.pin(ulid)
                else:
                    engine.unpin(ulid)
                return {"ok": True, "ulid": ulid, "pin": pin}
            except Exception as e:
                return {"error": str(e)}

        def _handle_archive(self, ulid: str) -> dict:
            try:
                engine.events.archive(ulid)
                engine.recall_cache.bump_generation()
                return {"ok": True, "ulid": ulid}
            except Exception as e:
                return {"error": str(e)}

        def _handle_feedback(self, payload: dict) -> dict:
            try:
                ulid = payload.get("ulid")
                verdict = payload.get("verdict", "good")
                if not ulid:
                    return {"error": "ulid required"}
                from pmb.health.feedback import record_feedback
                record_feedback(engine, ulid=ulid, verdict=verdict)
                return {"ok": True}
            except Exception as e:
                return {"error": str(e)}

    return _Handler


def run_dashboard(engine, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Blocking — runs until KeyboardInterrupt."""
    handler = make_handler(engine)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"PMB dashboard at http://{host}:{port}")
    print(f"Workspace: {engine.workspace.name} ({engine.workspace.id})")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
