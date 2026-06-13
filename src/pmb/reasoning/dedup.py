"""
Multi-layer dedup - Improvement U.

Three lines of defense + an async background queue for borderline cases:

  L1  exact-text match
      Cheapest possible - normalize whitespace + casefold; if identical
      string exists in last N events of same event_type in this workspace,
      return that ULID.

  L2  semantic match (conservative)
      Embed candidate, find nearest event of same type with active status.
      Threshold COSINE_HIGH (default 0.92) ⇒ definite duplicate, return
      existing ULID, bump the original's access/importance.

  L2.5  async LLM verify
      Cosine in [COSINE_MID .. COSINE_HIGH) ⇒ borderline. Write both, but
      enqueue the pair into `dedup_pending`. Background worker (Ollama or
      Anthropic) asks "are these the same fact?" and merges yes-cases.

  L3  user review (dashboard, see /api/dedup_candidates)
      Anything below COSINE_MID is left alone; user can still browse the
      "Duplicates" tab in the dashboard if they want.

Trade-off rationale (vs mem0's LLM-on-every-write):

  • mem0: 2-5s per write, costs ~$0.001, requires cloud LLM.
  • us:    ~50ms per write, $0, local. Borderline cases get LLM async.

False-merge protection:
  - High cosine threshold (0.92, not 0.85). Precision >> recall.
  - Same event_type only - never merges a 'goal' with a 'fact'.
  - Merged entries are ARCHIVED with metadata.merged_into=<canonical>, never
    deleted. `pmb dedupe --undo` can restore.

Storage:
  - `dedup_pending` table (created lazily) holds borderline pairs awaiting
    LLM verdict. Worker drains via `dedup_run_pending()`.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# Tunable thresholds. Lower = catch more dups, higher = fewer false merges.
# These are intentionally CONSERVATIVE - we lean precision over recall.
COSINE_HIGH: float = 0.92    # ≥ this  → definite duplicate, auto-merge
COSINE_MID:  float = 0.80    # ∈[MID..HIGH) → borderline, async LLM verify
# Below COSINE_MID we never even consider it a candidate - too different.

# How far back to look for candidates (in days). Older items are too stale
# to be likely dups; bounding the search keeps L2 fast on large workspaces.
LOOKBACK_DAYS: float = 90.0


@dataclass
class DedupHit:
    """Result of a dedup check: caller should skip the write and bump this."""
    canonical_ulid: str
    similarity: float        # 1.0 if L1 exact, else cosine
    method: str              # "exact" | "semantic"
    canonical_text: str


@dataclass
class DedupBorderline:
    """Pair to enqueue for async LLM verify."""
    new_ulid: str            # the new event we just wrote
    candidate_ulid: str      # the existing event it might be duplicating
    similarity: float


# ----------------------------------------------------------------------
# Text normalization for L1 exact match
# ----------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_PUNCT_NORMALIZE_RE = re.compile(r"[--]")  # em/en dash → hyphen


def normalize_for_exact_match(text: str) -> str:
    """Aggressive normalization for L1 dedup hashing.

    Casefold, collapse whitespace, strip surrounding punctuation, normalize
    dashes. Two strings that differ ONLY in these surface details are
    considered identical.
    """
    if not text:
        return ""
    t = _PUNCT_NORMALIZE_RE.sub("-", text)
    t = _WS_RE.sub(" ", t).strip().casefold()
    return t


# ----------------------------------------------------------------------
# Schema for borderline-pair queue
# ----------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS dedup_pending (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    new_ulid     TEXT NOT NULL,
    candidate_ulid TEXT NOT NULL,
    similarity   REAL NOT NULL,
    created_at   REAL NOT NULL,
    verdict      TEXT,
    resolved_at  REAL,
    UNIQUE(workspace_id, new_ulid, candidate_ulid)
);
"""

_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_dedup_pending_unresolved "
    "ON dedup_pending(workspace_id, resolved_at) WHERE resolved_at IS NULL",
)


def _ensure_schema(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(_DDL)
        for ddl in _INDEX_DDL:
            conn.execute(ddl)


# ----------------------------------------------------------------------
# L1: exact-text dedup
# ----------------------------------------------------------------------

def find_exact_duplicate(
    db_path: Path,
    workspace_id: str,
    event_type: str,
    content: str,
    lookback_days: float = LOOKBACK_DAYS,
) -> DedupHit | None:
    """L1: scan recent same-type active events for an identical normalized
    content string. Returns the canonical event's ULID if found.

    O(N) over recent events - bounded by lookback window. ~5ms for 200
    events in practice.
    """
    norm_target = normalize_for_exact_match(content)
    if not norm_target:
        return None
    cutoff = time.time() - lookback_days * 86400.0
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT ulid, content FROM events
            WHERE workspace_id = ?
              AND event_type = ?
              AND archived_at IS NULL
              AND timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT 500
            """,
            (workspace_id, event_type, cutoff),
        ).fetchall()
    for r in rows:
        if normalize_for_exact_match(r["content"] or "") == norm_target:
            return DedupHit(
                canonical_ulid=r["ulid"],
                similarity=1.0,
                method="exact",
                canonical_text=r["content"] or "",
            )
    return None


# ----------------------------------------------------------------------
# L2: semantic dedup (cosine) - needs an embedder
# ----------------------------------------------------------------------

def find_semantic_duplicate(
    db_path: Path,
    workspace_id: str,
    event_type: str,
    candidates: list[tuple[str, float]],    # [(ulid, cosine_sim), ...] from nearest-neighbor index
    threshold_high: float = COSINE_HIGH,
    threshold_mid: float = COSINE_MID,
) -> tuple[DedupHit | None, tuple[str, float] | None]:
    """L2: given pre-computed cosine similarities, decide if any is a dup.

    `candidates` should come from PMB's vector index (LanceDB) - engine
    builds it via `engine._dedup_nearest_candidates(content, event_type)`.
    Dedup module stays agnostic of where vectors live.

    Returns (hit, borderline):
      hit:        DedupHit if top candidate ≥ threshold_high → SKIP write
      borderline: (candidate_ulid, sim) if in [mid, high) → WRITE then enqueue
    """
    if not candidates:
        return None, None
    # Filter by event_type via SQLite (only candidates with the SAME type
    # qualify - never merge a goal with a fact)
    ulids = [c[0] for c in candidates]
    sim_by_ulid = {c[0]: c[1] for c in candidates}
    placeholders = ",".join("?" * len(ulids))
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT ulid, content, event_type FROM events "
            f"WHERE workspace_id = ? AND archived_at IS NULL "
            f"AND ulid IN ({placeholders})",
            (workspace_id, *ulids),
        ).fetchall()

    best: tuple[str, float, str] | None = None
    for r in rows:
        if r["event_type"] != event_type:
            continue
        sim = sim_by_ulid.get(r["ulid"], 0.0)
        if best is None or sim > best[1]:
            best = (r["ulid"], sim, r["content"] or "")

    if best is None:
        return None, None
    ulid, sim, text = best
    if sim >= threshold_high:
        return DedupHit(
            canonical_ulid=ulid, similarity=sim,
            method="semantic", canonical_text=text,
        ), None
    if sim >= threshold_mid:
        return None, (ulid, sim)
    return None, None


# ----------------------------------------------------------------------
# Borderline queue (L2.5)
# ----------------------------------------------------------------------

def enqueue_borderline(
    db_path: Path,
    workspace_id: str,
    new_ulid: str,
    candidate_ulid: str,
    similarity: float,
) -> None:
    """Enqueue a borderline pair for async LLM verification."""
    if new_ulid == candidate_ulid:
        return
    _ensure_schema(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO dedup_pending
              (workspace_id, new_ulid, candidate_ulid, similarity, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (workspace_id, new_ulid, candidate_ulid, similarity, time.time()),
        )


def list_pending(db_path: Path, workspace_id: str, limit: int = 100) -> list[dict]:
    """List unresolved borderline pairs (for dashboard or worker)."""
    _ensure_schema(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT p.*,
                   e1.content AS new_content,
                   e2.content AS candidate_content,
                   e1.event_type AS event_type
            FROM dedup_pending p
            JOIN events e1 ON e1.ulid = p.new_ulid
            JOIN events e2 ON e2.ulid = p.candidate_ulid
            WHERE p.workspace_id = ? AND p.resolved_at IS NULL
            ORDER BY p.similarity DESC, p.created_at DESC
            LIMIT ?
            """,
            (workspace_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def mark_verdict(
    db_path: Path,
    pending_id: int,
    verdict: str,                # "merge" | "keep_both"
) -> None:
    _ensure_schema(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE dedup_pending SET verdict = ?, resolved_at = ? WHERE id = ?",
            (verdict, time.time(), pending_id),
        )


# ----------------------------------------------------------------------
# One-shot cleanup over an existing workspace
# ----------------------------------------------------------------------

def sweep_workspace(
    db_path: Path,
    workspace_id: str,
    embeddings_provider,             # callable() -> list[(ulid, event_type, importance, access_count, timestamp, np.ndarray)]
    threshold: float = COSINE_HIGH,
    event_types: Iterable[str] | None = None,
    archive_fn=None,
) -> dict:
    """One-shot dedup pass over all active events.

    `embeddings_provider` must return a list of tuples - engine pulls these
    from LanceDB+SQLite (where vectors actually live). Dedup module stays
    agnostic of storage.

    For each event_type, finds clusters of items with pairwise cosine ≥
    threshold. Keeps the one with highest (importance, access_count, recency);
    archives the rest with metadata.merged_into = winner_ulid.

    Returns stats: {n_clusters, n_merged, by_type}.
    """
    items = embeddings_provider() or []
    if not items:
        return {"n_clusters": 0, "n_merged": 0, "by_type": {}}

    # Group by event_type - never merge across types.
    by_type_groups: dict[str, list] = {}
    for it in items:
        ulid, etype, imp, acc, ts, vec = it
        if event_types is not None and etype not in set(event_types):
            continue
        by_type_groups.setdefault(etype, []).append(it)

    by_type: dict[str, int] = {}
    n_clusters = 0
    n_merged = 0

    for etype, group in by_type_groups.items():
        if len(group) < 2:
            continue
        mat = np.stack([g[5] for g in group])
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        mat_n = mat / norms
        sim = mat_n @ mat_n.T
        np.fill_diagonal(sim, 0.0)

        n = len(group)
        assigned = [-1] * n
        cluster_id = 0
        for i in range(n):
            if assigned[i] != -1:
                continue
            cluster = [i]
            for j in range(i + 1, n):
                if assigned[j] != -1:
                    continue
                if sim[i, j] >= threshold:
                    cluster.append(j)
                    assigned[j] = cluster_id
            if len(cluster) > 1:
                for k in cluster:
                    assigned[k] = cluster_id
                cluster_id += 1
                n_clusters += 1

        if cluster_id == 0:
            continue

        clusters: dict[int, list[int]] = {}
        for idx, cid in enumerate(assigned):
            if cid == -1:
                continue
            clusters.setdefault(cid, []).append(idx)

        for cid, idxs in clusters.items():
            def score(idx, group=group):
                _, _, imp, acc, ts, _ = group[idx]
                age_days = (time.time() - ts) / 86400.0
                recency = max(0.0, 1.0 - age_days / 30.0)
                return (imp or 0.0) * 5.0 + np.log1p(acc or 0) + recency
            winner = max(idxs, key=score)
            winner_ulid = group[winner][0]
            for idx in idxs:
                if idx == winner:
                    continue
                loser_ulid = group[idx][0]
                if archive_fn:
                    archive_fn(loser_ulid, winner_ulid)
                else:
                    _archive_with_pointer(db_path, loser_ulid, winner_ulid)
                n_merged += 1
                by_type[etype] = by_type.get(etype, 0) + 1

    return {
        "n_clusters": n_clusters,
        "n_merged": n_merged,
        "by_type": by_type,
    }


def _archive_with_pointer(db_path: Path, loser_ulid: str, winner_ulid: str) -> None:
    """Soft-archive `loser_ulid`, store merged_into pointer in metadata.
    Reversible - `pmb dedupe --undo` can restore.
    """
    now = time.time()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        r = conn.execute(
            "SELECT metadata_json FROM events WHERE ulid = ?",
            (loser_ulid,),
        ).fetchone()
        try:
            meta = json.loads(r["metadata_json"] or "{}") if r else {}
        except Exception:
            meta = {}
        meta["merged_into"] = winner_ulid
        meta["merged_at"] = now
        conn.execute(
            "UPDATE events SET archived_at = ?, metadata_json = ? WHERE ulid = ?",
            (now, json.dumps(meta, ensure_ascii=False), loser_ulid),
        )


def undo_merges(db_path: Path, workspace_id: str) -> int:
    """Restore events archived by dedup (metadata.merged_into is set).
    Returns count of restored events.
    """
    n = 0
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT ulid, metadata_json FROM events
            WHERE workspace_id = ?
              AND archived_at IS NOT NULL
              AND json_extract(metadata_json, '$.merged_into') IS NOT NULL
            """,
            (workspace_id,),
        ).fetchall()
        for r in rows:
            try:
                meta = json.loads(r["metadata_json"] or "{}")
            except Exception:
                continue
            meta.pop("merged_into", None)
            meta.pop("merged_at", None)
            conn.execute(
                "UPDATE events SET archived_at = NULL, metadata_json = ? WHERE ulid = ?",
                (json.dumps(meta, ensure_ascii=False), r["ulid"]),
            )
            n += 1
    return n


# ----------------------------------------------------------------------
# L2.5 async worker - drain dedup_pending using a local/cloud LLM
# ----------------------------------------------------------------------

_LLM_PROMPT = """You are a memory dedup judge for a personal memory system.

Decide whether these TWO facts describe the SAME piece of information,
just phrased differently (e.g. translation, paraphrase, minor edit), or
whether they are TWO DIFFERENT facts that should both be kept.

Fact A:
{a}

Fact B:
{b}

Answer with exactly ONE word, no explanation:
  MERGE     - same fact, keep one (the canonical, fact B)
  KEEP_BOTH - different facts, keep both
"""


def _call_ollama(prompt: str, model: str = "llama3.2") -> str | None:
    """Call a local Ollama instance. Returns None on any failure (caller
    treats as 'cannot decide → keep_both')."""
    try:
        import urllib.error
        import urllib.request
        body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=body, headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return (data.get("response") or "").strip()
    except Exception as e:
        log.debug("Ollama dedup call failed: %s", e)
        return None


def _call_anthropic(prompt: str, model: str = "claude-haiku-4-5") -> str | None:
    """Call Anthropic via the local `claude` CLI if available, else SDK."""
    try:
        from anthropic import Anthropic
        client = Anthropic()
        msg = client.messages.create(
            model=model, max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip() if msg.content else None
    except Exception as e:
        log.debug("Anthropic dedup call failed: %s", e)
        return None


def run_pending(
    db_path: Path,
    workspace_id: str,
    backend: str = "auto",          # "auto" | "ollama" | "anthropic" | "none"
    limit: int = 50,
    archive_fn=None,
) -> dict:
    """Drain the dedup_pending queue. For each pair, ask an LLM whether
    they're the same fact; if yes, archive the new ulid (keep candidate
    as canonical since it's older).

    Returns: {n_processed, n_merged, n_kept, n_skipped}
    """
    _ensure_schema(db_path)
    pairs = list_pending(db_path, workspace_id, limit=limit)
    if not pairs:
        return {"n_processed": 0, "n_merged": 0, "n_kept": 0, "n_skipped": 0}

    # Choose backend
    def call_llm(prompt: str) -> str | None:
        if backend in ("auto", "ollama"):
            r = _call_ollama(prompt)
            if r is not None:
                return r
            if backend == "ollama":
                return None
        if backend in ("auto", "anthropic"):
            return _call_anthropic(prompt)
        return None

    n_merged = n_kept = n_skipped = 0
    for p in pairs:
        prompt = _LLM_PROMPT.format(a=p["new_content"], b=p["candidate_content"])
        verdict_raw = call_llm(prompt)
        if not verdict_raw:
            # No LLM available - keep both, mark resolved with 'unknown'
            mark_verdict(db_path, p["id"], "unknown")
            n_skipped += 1
            continue
        verdict = verdict_raw.upper().split()[0].strip(".,;:!?")
        if verdict == "MERGE":
            # Archive the newer (new_ulid), point at candidate (canonical)
            if archive_fn:
                archive_fn(p["new_ulid"], p["candidate_ulid"])
            else:
                _archive_with_pointer(db_path, p["new_ulid"], p["candidate_ulid"])
            mark_verdict(db_path, p["id"], "merge")
            n_merged += 1
        else:
            mark_verdict(db_path, p["id"], "keep_both")
            n_kept += 1

    return {
        "n_processed": len(pairs),
        "n_merged": n_merged,
        "n_kept": n_kept,
        "n_skipped": n_skipped,
    }
