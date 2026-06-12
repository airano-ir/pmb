"""F3 — close the X2 channel-weights learning loop (NEVER auto-applied).

X2 shipped `propose_channel_weights(samples)` but nothing fed it samples. This
collects weak-labelled `(channel_scores, useful)` samples — channels of a
surfaced top result, labelled useful when the agent later acts on that result
(feedback boost / lesson-followed) — into a small table, and the maintenance
tick runs `propose_channel_weights` over them and writes a SUGGESTION that
`pmb doctor` prints. The user inspects and applies it via
`pmb config set recall.channel_weights …`; recall never adopts it on its own
(the X2 contract: default identity, human-in-the-loop).

Gated by `recall.weight_learning` (default off) so the sample insert is opt-in.
"""
from __future__ import annotations

import json
import sqlite3
import time


def _table_ready(conn) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS channel_weight_samples ("
        " id INTEGER PRIMARY KEY,"
        " ts REAL NOT NULL,"
        " workspace_id TEXT,"
        " ulid TEXT,"
        " channels_json TEXT NOT NULL,"
        " useful INTEGER NOT NULL DEFAULT 0)")


def record_channel_sample(engine, channels: dict, ulid: str | None = None,
                          useful: bool = False) -> None:
    """Append one weak-label sample. Best-effort, never raises."""
    try:
        ws = getattr(engine.workspace, "id", None)
        with sqlite3.connect(str(engine.workspace.db_path)) as conn:
            _table_ready(conn)
            conn.execute(
                "INSERT INTO channel_weight_samples (ts, workspace_id, ulid, "
                "channels_json, useful) VALUES (?,?,?,?,?)",
                (time.time(), ws, ulid,
                 json.dumps({k: round(float(v), 4) for k, v in channels.items()}),
                 1 if useful else 0))
    except Exception:
        pass


def note_recall_useful(engine, ulid: str) -> None:
    """Flip the most recent sample for `ulid` to useful=1 (the agent acted on
    that surfaced result). Best-effort."""
    if not ulid:
        return
    try:
        ws = getattr(engine.workspace, "id", None)
        with sqlite3.connect(str(engine.workspace.db_path)) as conn:
            _table_ready(conn)
            conn.execute(
                "UPDATE channel_weight_samples SET useful=1 WHERE id = ("
                "  SELECT id FROM channel_weight_samples WHERE ulid=? "
                "  AND workspace_id IS ? ORDER BY ts DESC LIMIT 1)",
                (ulid, ws))
    except Exception:
        pass


def load_samples(engine, limit: int = 2000) -> list[tuple[dict, bool]]:
    out: list[tuple[dict, bool]] = []
    try:
        ws = getattr(engine.workspace, "id", None)
        with sqlite3.connect(str(engine.workspace.db_path)) as conn:
            _table_ready(conn)
            rows = conn.execute(
                "SELECT channels_json, useful FROM channel_weight_samples "
                "WHERE workspace_id IS ? ORDER BY ts DESC LIMIT ?",
                (ws, limit)).fetchall()
    except Exception:
        return []
    for channels_json, useful in rows:
        try:
            out.append((json.loads(channels_json), bool(useful)))
        except Exception:
            continue
    return out


def propose_from_samples(engine, min_samples: int = 20) -> dict:
    """Tick step: run propose_channel_weights over collected samples and cache
    the suggestion. Returns a summary; the cache is what `pmb doctor` reads."""
    samples = load_samples(engine)
    n_useful = sum(1 for _c, ok in samples if ok)
    if len(samples) < min_samples or n_useful == 0:
        return {"n_samples": len(samples), "n_useful": n_useful,
                "skipped": "insufficient samples"}
    from pmb.reasoning.scoring import propose_channel_weights
    weights = propose_channel_weights(samples)
    payload = {"weights": weights, "n_samples": len(samples),
               "n_useful": n_useful, "ts": time.time()}
    try:
        (engine.workspace.storage_dir / "channel_weights_suggestion.json").write_text(
            json.dumps(payload), encoding="utf-8")
    except Exception:
        pass
    return payload


def load_suggestion(engine) -> dict | None:
    try:
        p = engine.workspace.storage_dir / "channel_weights_suggestion.json"
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
