from __future__ import annotations

import json
from pathlib import Path

_PROJECT_ENTITY_BLACKLIST = {
    "auth", "design", "docs", "engine", "deploys", "deploy", "issue", "integrated",
    "fix", "built", "installed", "code", "done", "config", "sources", "tests",
    "test", "file", "script", "endpoint", "tool", "tools", "feature", "features",
    "agent", "agents", "session", "sessions", "model", "models", "user", "users",
    "project", "projects", "data", "memory", "memories", "event", "events",
    "log", "logs", "task", "tasks", "idea", "ideas", "update", "updates", "time",
    "work", "core", "source", "main", "module", "modules", "options", "option",
}
_NON_PROJECT_ENTITY_KINDS = {
    "concept", "file", "function", "method", "class", "import", "module",
    "package", "tech",
}
_PROJECT_ENTITY_KINDS = {
    "project", "repo", "repository", "product", "codebase",
    "app", "application", "service", "system",
}


class OverviewMixin:
    def active_arcs_for_project(self, project_name: str, limit: int = 2) -> list[dict]:
        """Return the top narrative arcs whose member events overlap with
        the project entity's events. Used by `prepare()` so the agent
        sees the bigger story (e.g. "Postgres adoption", "Auth refactor")
        when picking up project work.

        Output: list of {arc_id, title, summary, n_events, status,
        last_updated, event_ulids, overlap_count}.
        """
        import sqlite3
        ws = self.workspace.id
        nm = (project_name or "").strip().lower()
        if not nm:
            return []
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # Prefer exact structured project metadata for legacy graphs that
            # do not yet contain a project-kind entity.
            ev_rows = conn.execute(
                """
                SELECT ulid AS event_ulid FROM events
                WHERE workspace_id = ? AND archived_at IS NULL
                  AND LOWER(COALESCE(
                    json_extract(metadata_json, '$.project_name'),
                    json_extract(metadata_json, '$.project'),
                    ''
                  )) = ?
                """,
                (ws, nm),
            ).fetchall()
            ent = None if ev_rows else conn.execute(
                """
                SELECT id FROM graph_entities
                WHERE workspace_id = ? AND LOWER(name) LIKE ?
                ORDER BY
                  CASE
                    WHEN LOWER(name) = ?
                         AND kind IN ('project', 'repo', 'repository') THEN 0
                    WHEN LOWER(name) = ? THEN 1
                    WHEN kind IN ('project', 'repo', 'repository') THEN 2
                    ELSE 3
                  END,
                  n_mentions DESC
                LIMIT 1
                """,
                (ws, f"%{nm}%", nm, nm),
            ).fetchone()
            if not ent and not ev_rows:
                return []
            # Project events
            if ent:
                ev_rows = conn.execute(
                    "SELECT event_ulid FROM graph_event_entities WHERE entity_id = ?",
                    (ent["id"],),
                ).fetchall()
            ev_ulids = {r["event_ulid"] for r in ev_rows}
            if not ev_ulids:
                return []
            # Active arcs ranked by overlap with project events.
            arc_rows = conn.execute(
                """
                SELECT a.id, a.title, a.summary, a.status, a.n_events,
                       a.last_updated, a.first_event_ulid, a.last_event_ulid
                FROM arcs a
                WHERE a.workspace_id = ? AND a.status = 'active'
                ORDER BY a.last_updated DESC
                LIMIT 50
                """,
                (ws,),
            ).fetchall()
            # S5: one batched fetch of all candidate arcs' member ulids instead
            # of a query PER arc (was up to 50 round-trips). Group in Python.
            arc_ids = [ar["id"] for ar in arc_rows]
            members: dict = {}
            if arc_ids:
                ph = ",".join("?" * len(arc_ids))
                for mr in conn.execute(
                    f"SELECT arc_id, event_ulid FROM arc_events "
                    f"WHERE arc_id IN ({ph})",
                    arc_ids,
                ).fetchall():
                    members.setdefault(mr["arc_id"], []).append(mr["event_ulid"])
            scored = []
            for ar in arc_rows:
                ulids = members.get(ar["id"], [])
                overlap = len([u for u in ulids if u in ev_ulids])
                if overlap == 0:
                    continue
                scored.append({
                    "arc_id": ar["id"],
                    "title": ar["title"],
                    "summary": (ar["summary"] or "")[:300],
                    "status": ar["status"],
                    "n_events": ar["n_events"],
                    "last_updated": ar["last_updated"],
                    "event_ulids": ulids,
                    "overlap_count": overlap,
                })
        scored.sort(key=lambda a: -a["overlap_count"])
        return scored[:limit]

    def _project_entity_candidates(self, min_mentions: int = 3) -> list[dict]:
        """Return plausible project-name candidates, cached by write generation.

        Legacy workspaces sometimes classified product names such as LoadGuard
        as ``person`` or ``concept``. Keep those candidates, but exclude entity
        kinds that are structurally code/technology rather than projects.
        """
        import sqlite3

        gen = getattr(self.recall_cache, "_generation", 0)
        cache = getattr(self, "_project_candidate_cache", None)
        ckey = (gen, int(min_mentions))
        if cache is not None and cache[0] == ckey:
            return cache[1]

        ws = self.workspace.id
        current = self._workspace_project_aliases()
        # The min=1 path is used only for project-scoping a small lesson set.
        # Keep enough legacy person-kind entities to catch low-frequency product
        # names such as HackerNoon without admitting concept/tech noise.
        candidate_limit = 1000 if min_mentions <= 1 else 100
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, kind, name, n_mentions FROM graph_entities
                WHERE workspace_id = ? AND n_mentions >= ?
                ORDER BY n_mentions DESC LIMIT ?
                """,
                (ws, min_mentions, candidate_limit),
            ).fetchall()
        out: list[dict] = []
        for r in rows:
            nm = (r["name"] or "").lower()
            kind = (r["kind"] or "").lower()
            # Short names are noisy in the general graph, but the current
            # repository's explicit name is authoritative (notably "PMB").
            if (len(nm) < 4 and nm not in current) or len(nm) > 30:
                continue
            if nm in _PROJECT_ENTITY_BLACKLIST:
                continue
            if "." in nm or "/" in nm or "\\" in nm:
                continue
            if kind in _NON_PROJECT_ENTITY_KINDS and nm not in current:
                continue
            out.append({
                "id": r["id"],
                "name": r["name"],
                "kind": r["kind"],
                "n_mentions": r["n_mentions"],
            })
        # Avoid retaining sqlite.Row objects/connections in the cache.
        self._project_candidate_cache = (ckey, out)
        return out

    def _workspace_project_aliases(self) -> set[str]:
        """Names that identify the repository the current Engine was opened in."""
        out: set[str] = set()
        try:
            root_name = (self.workspace.root.name or "").strip().lower()
            if root_name:
                out.add(root_name)
        except Exception:
            pass
        try:
            raw = str(self.workspace.name or "").strip()
            if raw:
                out.add(raw.lower())
                out.add(Path(raw).name.lower())
        except Exception:
            pass
        return {x for x in out if x}

    def projects_in_text(self, text: str, min_mentions: int = 3) -> list[dict]:
        """All plausible project mentions, ranked for task routing.

        The current repository wins when explicitly named. Otherwise textual
        order wins: in "compare Alpha with Beta", Alpha is the primary subject.
        Mention-count is only a tie-breaker, never the main routing signal.
        """
        if not text:
            return []
        import re as _re

        text_lc = text.lower()
        current = self._workspace_project_aliases()
        matches: list[dict] = []
        for cand in self._project_entity_candidates(min_mentions=min_mentions):
            nm = (cand.get("name") or "").lower()
            m = _re.search(rf"\b{_re.escape(nm)}\b", text_lc)
            if not m:
                continue
            item = dict(cand)
            item["position"] = m.start()
            item["current_workspace"] = nm in current
            matched = text[m.start():m.end()]
            letters = [c for c in matched if c.isalpha()]
            item["projectish_spelling"] = bool(
                letters
                and (
                    all(c.isupper() for c in letters)
                    or any(c.isupper() for c in matched[1:])
                )
            )
            matches.append(item)
        matches.sort(
            key=lambda x: (
                not x["current_workspace"],
                x["position"],
                0 if (x.get("kind") or "").lower() in _PROJECT_ENTITY_KINDS else 1,
                -int(x.get("n_mentions") or 0),
            )
        )
        return matches

    def detect_project_in_text(self, text: str, min_mentions: int = 3) -> dict | None:
        """Return the primary project explicitly mentioned in ``text``.

        Unlike the old implementation, this does not let the most-mentioned
        project in the database hijack a multi-project sentence. The current
        repository wins when named; otherwise the first project mention wins.
        """
        matches = self.projects_in_text(text, min_mentions=min_mentions)
        if not matches:
            return None
        out = dict(matches[0])
        out.pop("position", None)
        out.pop("current_workspace", None)
        out.pop("projectish_spelling", None)
        return out

    def project_overview(self, name: str, max_per_section: int = 8) -> dict:
        """Graph-driven overview of one project / entity. Faster + more
        complete than topic_overview because we go directly via
        graph_event_entities instead of running the full recall pipeline.

        Use this when the agent (re)starts work on a known project - ONE
        call returns the full context: top facts, lessons, decisions, open
        goals, recent completions, related sub-entities (tech stack /
        people / files), and project span.

        `name` is matched case-insensitively against entity names; we pick
        the highest-mention entity that contains the query as a substring.

        S5: this is ~100 ms (500-row JOIN + bucket + a rescue LIKE scan) and
        `prepare()` fires it on nearly every message, so the result is memoized
        by (name, max_per_section, write-generation). The recall cache already
        bumps that generation on every write, so a 0.3 ms staleness probe
        replaces the full recompute and any write invalidates it correctly.
        """
        nm = (name or "").strip().lower()
        if not nm:
            return {"empty": True, "error": "empty name"}
        gen = getattr(self.recall_cache, "_generation", 0)
        ckey = (nm, max_per_section)
        cache = getattr(self, "_overview_cache", None)
        if cache is None:
            from collections import OrderedDict
            cache = self._overview_cache = OrderedDict()
        hit = cache.get(ckey)
        if hit is not None and hit[0] == gen:
            cache.move_to_end(ckey)
            return hit[1]
        result = self._project_overview_uncached(nm, name, max_per_section)
        cache[ckey] = (gen, result)
        while len(cache) > 16:
            cache.popitem(last=False)
        return result

    def project_structure(self, name: str, max_files: int = 400) -> dict:
        """Assemble a project's STRUCTURE from memory - no filesystem scan.

        Reads the file-index rows written by `index_project` (per-file symbols,
        imports, language), the one-line purposes written by `pmb track
        modules`, and the latest `pmb track changes` intents, then returns a
        compact map: languages, files grouped by top-level directory (each with
        its purpose + symbol count), key modules, and recent change intents.

        This is the "what does this project look like" view - complementary to
        `project_overview`, which answers "what do I know / decided about it".
        Returns {empty: True, ...} when the project was never indexed.
        """
        import json
        import sqlite3
        from pathlib import Path

        nm = (name or "").strip().lower()
        if not nm:
            return {"empty": True, "error": "empty name"}
        ws = self.workspace.id

        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # Resolve the project_path: whichever indexed project matches `nm`
            # by name OR path substring and has the most file rows.
            cand = conn.execute(
                """
                SELECT json_extract(metadata_json,'$.project_path') AS pp,
                       COUNT(*) AS n
                FROM events
                WHERE workspace_id=? AND archived_at IS NULL
                  AND json_extract(metadata_json,'$.source')='project'
                  AND json_extract(metadata_json,'$.index_artifact')=1
                  AND (
                    LOWER(COALESCE(json_extract(metadata_json,'$.project_name'),''))=?
                    OR LOWER(COALESCE(json_extract(metadata_json,'$.project_path'),'')) LIKE ?
                  )
                GROUP BY pp ORDER BY n DESC LIMIT 1
                """,
                (ws, nm, f"%{nm}%"),
            ).fetchone()
            if not cand or not cand["pp"]:
                return {
                    "empty": True, "name": name,
                    "hint": f"No indexed structure for '{name}'. "
                            f"Run `pmb index project` first.",
                }
            project_path = cand["pp"]

            file_rows = conn.execute(
                """
                SELECT metadata_json FROM events
                WHERE workspace_id=? AND archived_at IS NULL
                  AND json_extract(metadata_json,'$.source')='project'
                  AND json_extract(metadata_json,'$.index_artifact')=1
                  AND json_extract(metadata_json,'$.project_path')=?
                ORDER BY rowid DESC
                """,
                (ws, project_path),
            ).fetchall()
            sum_rows = conn.execute(
                """
                SELECT json_extract(metadata_json,'$.file_path') AS fp, content
                FROM events
                WHERE workspace_id=? AND archived_at IS NULL
                  AND json_extract(metadata_json,'$.source')='module-summary'
                  AND json_extract(metadata_json,'$.project_path')=?
                ORDER BY rowid DESC
                """,
                (ws, project_path),
            ).fetchall()
            chg_rows = conn.execute(
                """
                SELECT json_extract(metadata_json,'$.commit_short') AS c, content
                FROM events
                WHERE workspace_id=? AND archived_at IS NULL
                  AND json_extract(metadata_json,'$.source')='git-change'
                  AND json_extract(metadata_json,'$.project_path')=?
                ORDER BY rowid DESC LIMIT 5
                """,
                (ws, project_path),
            ).fetchall()

        # file_path -> one-line purpose (content is "Module <fp>: <summary>")
        purpose: dict[str, str] = {}
        for r in sum_rows:
            fp = r["fp"]
            if fp and fp not in purpose:
                c = r["content"] or ""
                purpose[fp] = c.split(":", 1)[1].strip() if ":" in c else c.strip()

        seen: set[str] = set()
        files: list[dict] = []
        langs: dict[str, int] = {}
        for r in file_rows:
            try:
                md = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
            except Exception:
                md = {}
            fp = md.get("file_path")
            if not fp or fp in seen:
                continue
            seen.add(fp)
            lang = md.get("language", "other")
            langs[lang] = langs.get(lang, 0) + 1
            files.append({
                "path": fp,
                "language": lang,
                "loc": md.get("loc"),
                "n_symbols": len(md.get("symbols") or []),
                "imports": (md.get("imports") or [])[:8],
                "purpose": purpose.get(fp, ""),
            })
            if len(files) >= max_files:
                break

        tree: dict[str, list] = {}
        for f in files:
            norm = f["path"].replace("\\", "/")
            top = norm.split("/", 1)[0] if "/" in norm else "."
            tree.setdefault(top, []).append({
                "file": f["path"], "purpose": f["purpose"],
                "symbols": f["n_symbols"], "loc": f["loc"],
            })

        # Key modules: prefer files that have a purpose, then by symbol count.
        key = sorted(files, key=lambda f: (bool(f["purpose"]), f["n_symbols"]),
                     reverse=True)[:12]
        key_modules = [
            {"file": f["path"], "purpose": f["purpose"], "symbols": f["n_symbols"]}
            for f in key
        ]

        recent_changes = []
        for r in chg_rows:
            c = r["content"] or ""
            recent_changes.append({
                "commit": r["c"],
                "summary": c.splitlines()[0] if c else "",
            })

        return {
            "project": Path(project_path).name,
            "path": project_path,
            "n_files": len(files),
            "languages": dict(sorted(langs.items(), key=lambda x: -x[1])),
            "n_with_purpose": sum(1 for f in files if f["purpose"]),
            "tree": tree,
            "key_modules": key_modules,
            "recent_changes": recent_changes,
        }

    def _project_overview_uncached(self, nm: str, name: str,
                                   max_per_section: int = 8) -> dict:
        import sqlite3
        ws = self.workspace.id
        if not nm:
            return {"empty": True, "error": "empty name"}
        with sqlite3.connect(self.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # Structured project metadata is authoritative. Legacy graphs may
            # not have a project:pmb node yet, while a noisy concept such as
            # tmp_pmb_home has many mentions. Prefer the exact metadata match
            # and synthesize a project entity when needed; regraph will create
            # the real node later, but reads are correct immediately.
            project_meta = conn.execute(
                """
                SELECT
                  COALESCE(
                    json_extract(metadata_json, '$.project_name'),
                    json_extract(metadata_json, '$.project')
                  ) AS name,
                  COUNT(*) AS n_mentions
                FROM events
                WHERE workspace_id = ?
                  AND archived_at IS NULL
                  AND LOWER(COALESCE(
                    json_extract(metadata_json, '$.project_name'),
                    json_extract(metadata_json, '$.project'),
                    ''
                  )) = ?
                GROUP BY name
                ORDER BY n_mentions DESC
                LIMIT 1
                """,
                (ws, nm),
            ).fetchone()
            if project_meta:
                ent = conn.execute(
                    """
                    SELECT id, kind, name, n_mentions FROM graph_entities
                    WHERE workspace_id = ? AND LOWER(name) = ?
                      AND kind IN ('project', 'repo', 'repository')
                    ORDER BY n_mentions DESC LIMIT 1
                    """,
                    (ws, nm),
                ).fetchone()
                if not ent:
                    ent = {
                        "id": None,
                        "kind": "project",
                        "name": project_meta["name"],
                        "n_mentions": project_meta["n_mentions"],
                    }
            else:
                # No structured project match: use the dominant graph entity.
                ent = conn.execute(
                    """
                    SELECT id, kind, name, n_mentions FROM graph_entities
                    WHERE workspace_id = ? AND LOWER(name) LIKE ?
                    ORDER BY
                      CASE
                        WHEN LOWER(name) = ?
                             AND kind IN ('project', 'repo', 'repository') THEN 0
                        WHEN LOWER(name) = ? THEN 1
                        WHEN kind IN ('project', 'repo', 'repository') THEN 2
                        ELSE 3
                      END,
                      n_mentions DESC
                    LIMIT 1
                    """,
                    (ws, f"%{nm}%", nm, nm),
                ).fetchone()
            if not ent:
                return {"empty": True, "name_query": name,
                        "hint": "no entity matched; try a shorter name or use overview() for hybrid search"}
            eid = ent["id"]
            # All linked events (cheap SQL JOIN, no recall).
            if eid is None:
                ev_rows = conn.execute(
                    """
                    SELECT ulid, event_type, content, timestamp, importance,
                           metadata_json
                    FROM events
                    WHERE workspace_id = ?
                      AND archived_at IS NULL
                      AND LOWER(COALESCE(
                        json_extract(metadata_json, '$.project_name'),
                        json_extract(metadata_json, '$.project'),
                        ''
                      )) = ?
                    ORDER BY timestamp DESC
                    LIMIT 250
                    """,
                    (ws, nm),
                ).fetchall()
            else:
                ev_rows = conn.execute(
                    """
                    SELECT ev.ulid, ev.event_type, ev.content, ev.timestamp,
                           ev.importance, ev.metadata_json
                    FROM graph_event_entities ee
                    JOIN events ev ON ev.ulid = ee.event_ulid
                    WHERE ee.entity_id = ?
                      AND ev.workspace_id = ?
                      AND ev.archived_at IS NULL
                    ORDER BY ev.timestamp DESC
                    LIMIT 250
                    """,
                    (eid, ws),
                ).fetchall()
            # Top related entities - neighbours in the co-occurrence graph,
            # by edge weight. Gives the "tech stack" feel without LLM.
            nbr_rows = [] if eid is None else conn.execute(
                    """
                    SELECT e.id, e.kind, e.name, e.n_mentions, ed.weight
                    FROM graph_edges ed
                    JOIN graph_entities e
                      ON (e.id = CASE WHEN ed.entity_a = ? THEN ed.entity_b ELSE ed.entity_a END)
                    WHERE ed.workspace_id = ?
                      AND (ed.entity_a = ? OR ed.entity_b = ?)
                    ORDER BY ed.weight DESC LIMIT 30
                    """,
                    (eid, ws, eid, eid),
                ).fetchall()

        # Bucket events by event_type / metadata.kind.
        facts, lessons, decisions, completed, goals_open, goals_done, activity, other = [], [], [], [], [], [], [], []
        timestamps = []
        for r in ev_rows:
            try:
                md = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
            except Exception:
                md = {}
            kind = md.get("kind") or md.get("activity_kind") or r["event_type"]
            item = {
                "ulid": r["ulid"],
                "content": (r["content"] or "")[:240],
                "timestamp": r["timestamp"],
                "importance": r["importance"],
                "kind": kind,
            }
            timestamps.append(r["timestamp"])
            # Per-file project-index rows are machine-generated lookup
            # artifacts, not project facts. Keep them searchable through
            # recall/graph, but do not let hundreds of file summaries crowd
            # the human-facing project overview.
            if md.get("source") == "project" and md.get("file_path"):
                continue
            if r["event_type"] == "lesson" or kind == "lesson":
                lessons.append(item)
            elif kind == "decision":
                decisions.append(item)
            elif kind == "completed":
                completed.append(item)
            elif r["event_type"] == "goal":
                status = md.get("goal_status") or md.get("status") or "in_progress"
                bucket = goals_open if status in ("pending", "in_progress") else goals_done
                bucket.append({**item, "status": status})
            elif r["event_type"] == "fact":
                facts.append(item)
            elif r["event_type"] == "activity":
                activity.append(item)
            else:
                other.append(item)

        # Hybrid: ALSO pull lessons / decisions that mention the project
        # name in content but weren't linked by the graph extractor. This
        # rescues real project rules that the regex / LLM extractor missed.
        try:
            with sqlite3.connect(self.workspace.db_path) as conn:
                conn.row_factory = sqlite3.Row
                seen_ulids = {x["ulid"] for x in (lessons + decisions)}
                extra = conn.execute(
                    """
                    SELECT ulid, event_type, content, timestamp, importance,
                           metadata_json
                    FROM events
                    WHERE workspace_id = ?
                      AND archived_at IS NULL
                      AND LOWER(content) LIKE ?
                      AND (event_type = 'lesson'
                           OR metadata_json LIKE '%"kind":"lesson"%'
                           OR metadata_json LIKE '%"kind":"decision"%'
                           OR metadata_json LIKE '%"kind": "lesson"%'
                           OR metadata_json LIKE '%"kind": "decision"%')
                    ORDER BY timestamp DESC LIMIT 50
                    """,
                    (ws, f"%{nm}%"),
                ).fetchall()
                for r in extra:
                    if r["ulid"] in seen_ulids:
                        continue
                    try:
                        md = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
                    except Exception:
                        md = {}
                    kind = md.get("kind") or r["event_type"]
                    item = {
                        "ulid": r["ulid"],
                        "content": (r["content"] or "")[:240],
                        "timestamp": r["timestamp"],
                        "importance": r["importance"],
                        "kind": kind,
                    }
                    if r["event_type"] == "lesson" or kind == "lesson":
                        lessons.append(item)
                    elif kind == "decision":
                        decisions.append(item)
        except Exception:
            pass

        # Sort facts by importance to surface the most-pinned first.
        facts.sort(key=lambda i: -(i.get("importance") or 0))

        # Span.
        import time as _t
        span = None
        if timestamps:
            span = {
                "from": _t.strftime("%Y-%m-%d", _t.gmtime(min(timestamps))),
                "to": _t.strftime("%Y-%m-%d", _t.gmtime(max(timestamps))),
                "n_events": len(timestamps),
            }

        # Related entities - keep variety across kinds, prune the project
        # entity itself.
        related_noise = {"file", "files", "symbol", "symbols", "import", "imports"}
        related = [
            {"name": r["name"], "kind": r["kind"],
             "mentions": r["n_mentions"], "weight": r["weight"]}
            for r in nbr_rows
            if (r["name"] or "").strip().lower() not in related_noise
        ]

        return {
            "entity": {
                "id": ent["id"],
                "name": ent["name"],
                "kind": ent["kind"],
                "n_mentions": ent["n_mentions"],
            },
            "span": span,
            "key_facts": facts[:max_per_section],
            "lessons": lessons[:max_per_section],
            "decisions": decisions[:max_per_section],
            "open_goals": goals_open[:max_per_section],
            "completed_goals": goals_done[:max_per_section],
            "recent_completed": completed[:max_per_section],
            "recent_activity": activity[:max_per_section],
            "other": other[:5],
            "related_entities": related[:15],
            "n_total": len(ev_rows),
            "empty": len(ev_rows) == 0,
        }

    def topic_overview(self, topic: str, max_events: int = 40) -> dict:
        """Structured "what do I know about <topic>?" overview - no LLM.

        Recalls memories related to the topic and aggregates them into key
        facts & decisions, lessons, failures, open goals, a compact timeline,
        and related topics (entity-graph neighbours). Pure read + aggregation:
        it reuses recall() but writes nothing and changes no ranking. Exposed
        on the CLI (`pmb overview`) and over MCP so an agent can get up to
        speed on a project/feature in one call.
        """
        import time as _t
        topic = (topic or "").strip()
        if not topic:
            return {"topic": topic, "n_memories": 0, "empty": True}

        pack = self.recall(query=topic, top_k=max_events)
        results = list(pack.results)

        def _item(r) -> dict:
            meta = r.metadata or {}
            return {
                "date": getattr(r, "resolved_date", None),
                "content": (r.content or "")[:240],
                "score": round(float(r.score), 3),
                "kind": meta.get("kind") or r.event_type,
            }

        facts, lessons, failures, goals, other = [], [], [], [], []
        for r in results:
            meta = r.metadata or {}
            k = meta.get("kind")
            if k == "lesson":
                lessons.append(_item(r))
            elif k == "failure":
                failures.append(_item(r))
            elif r.event_type == "goal":
                goals.append(_item(r))
            elif r.event_type == "fact" or k == "decision":
                facts.append(_item(r))
            else:
                other.append(_item(r))

        timeline = sorted(
            ({"date": getattr(r, "resolved_date", None) or "",
              "content": (r.content or "")[:120]} for r in results),
            key=lambda x: x["date"],
        )[:12]

        ts = [r.timestamp for r in results if r.timestamp]
        span = None
        if ts:
            span = {"from": _t.strftime("%Y-%m-%d", _t.gmtime(min(ts))),
                    "to": _t.strftime("%Y-%m-%d", _t.gmtime(max(ts)))}

        related: list[str] = []
        try:
            for tok in topic.split()[:3]:
                gn = self.graph_neighbors(tok, top_k=5)
                if gn.get("entity"):
                    for nbr in gn.get("neighbors", []):
                        nm = (nbr.get("entity") or {}).get("name")
                        if nm and nm.lower() not in topic.lower() and nm not in related:
                            related.append(nm)
        except Exception:
            pass

        return {
            "topic": topic,
            "n_memories": len(results),
            "span": span,
            "facts": facts[:10],
            "lessons": lessons[:10],
            "failures": failures[:10],
            "goals": goals[:10],
            "other": other[:5],
            "timeline": timeline,
            "related_topics": related[:8],
            "empty": len(results) == 0,
        }

    def session_brief(self, session_id: str | None = None,
                      minutes: float | None = None, limit: int = 100) -> dict:
        """Compact digest of the CURRENT (or given) work session - what was
        decided / done / learned so far.

        Built so an agent can re-orient after its OWN context window compacts
        in a long session: PMB is the durable session memory, so instead of
        re-asking the user what it already did, the agent calls this and picks
        the thread back up. Pure read; off the recall ranking path.

        Scopes to events tagged with the active session; if none, falls back to
        the last `minutes` (config `session.brief_minutes`).
        """
        import time as _t
        sess = None
        if session_id is None:
            try:
                sess = self.session_tracker.current(auto_create=False)
            except Exception:
                sess = None
            session_id = getattr(sess, "id", None) if sess else None
        if minutes is None:
            try:
                minutes = float(self.config.get("session.brief_minutes"))
            except Exception:
                minutes = 180.0

        now = _t.time()
        # Scope to the session as a UNION: every event recorded since the
        # session began PLUS anything explicitly tagged with this session id.
        # Why both: only `record_activity` auto-binds events to the session;
        # facts / goals / lessons recorded during the session carry no
        # session id, so a tag-only filter silently drops them. Falling back to
        # the session's start time captures that work too. Without an active
        # session, use the last `minutes` window.
        sess_started = getattr(sess, "started_at", None) if sess else None
        cutoff = now - minutes * 60.0
        if session_id and isinstance(sess_started, (int, float)):
            cutoff = min(cutoff, sess_started - 1.0)
        # S5: push the (timestamp >= cutoff OR session-tagged) scope into SQL
        # via idx_workspace_time instead of loading the whole table.
        scoped = self.events.list_since(
            self.workspace.id, cutoff, session_id=session_id, limit=5000)
        scoped.sort(key=lambda e: e.timestamp)

        def _kind(meta) -> str | None:
            # `record_activity` stores the kind under `activity_kind`; lessons /
            # failures use `kind`. Read both so decisions/done classify.
            return meta.get("kind") or meta.get("activity_kind")

        def _it(e) -> dict:
            meta = e.metadata or {}
            return {
                # ulid is required by `_log_lesson_surfaces` to mint a
                # `surface_id` the agent can pass to `mark_lesson_followed`.
                # Without it, session-restore lessons stayed invisible to the
                # adherence loop (the filter `if L.get("ulid")` dropped them all).
                "ulid": e.ulid,
                "when": _t.strftime("%m-%d %H:%M", _t.gmtime(e.timestamp)),
                "content": (e.content or "")[:200],
                "kind": _kind(meta) or e.event_type,
            }

        decisions, done, lessons, failures, goals, other = [], [], [], [], [], []
        for e in scoped:
            meta = e.metadata or {}
            k = _kind(meta)
            if k == "lesson":
                lessons.append(_it(e))
            elif k == "failure":
                failures.append(_it(e))
            elif e.event_type == "goal":
                goals.append(_it(e))
            elif k == "decision":
                decisions.append(_it(e))
            elif k == "completed":
                done.append(_it(e))
            else:
                other.append(_it(e))

        duration_min = None
        started = getattr(sess, "started_at", None) if sess else None
        if isinstance(started, (int, float)):
            duration_min = round((now - started) / 60.0)

        return {
            "session_id": session_id,
            "session_name": getattr(sess, "name", None) if sess else None,
            "duration_min": duration_min,
            "scope": "session" if (session_id and scoped) else f"last {minutes:g} min",
            "n_events": len(scoped),
            "decisions": decisions[:15],
            "done": done[:15],
            "lessons": lessons[:15],
            "failures": failures[:15],
            "goals": goals[:15],
            "other": other[:10],
            "empty": len(scoped) == 0,
        }
