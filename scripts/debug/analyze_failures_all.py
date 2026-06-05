"""
Run the failure analyzer across ALL 10 LoCoMo conversations.

Goal: confirm the conv-26 pattern (many multi-hop failures are
*unsatisfiable* — empty gold evidence) holds across the dataset.

For each conv:
  - n cat-3 questions
  - n with empty gold evidence (unsatisfiable)
  - n hit / n missed among satisfiable

Aggregate: true multi-hop recall vs naive benchmark recall.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _bench_data import data_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def analyze_conv(conv, eng) -> dict:
    cat3 = [q for q in conv["qa"] if q.get("category") == 3]
    n_total = len(cat3)
    n_unsatisfiable = 0
    n_hit_satisfiable = 0
    n_miss_satisfiable = 0
    misses_solved_at_20 = 0
    misses_solved_at_50 = 0

    # Build dia_to_ulid map
    import sqlite3
    dia_to_ulid: dict = {}
    with sqlite3.connect(eng.workspace.db_path) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(
            "SELECT ulid, metadata_json FROM events WHERE workspace_id = ?",
            (eng.workspace.id,),
        ).fetchall():
            try:
                meta = json.loads(row["metadata_json"] or "{}")
                for d in (meta.get("dia_ids") or []):
                    dia_to_ulid[d] = row["ulid"]
            except Exception:
                continue

    for q in cat3:
        question = q.get("question", "")
        gold_evidence = q.get("evidence") or []
        if not gold_evidence:
            n_unsatisfiable += 1
            continue
        # Run recall at top_k=10
        pack = eng.recall(question, top_k=10)
        hit_at_10 = False
        for r in pack.results:
            meta = r.metadata or {}
            ids = meta.get("dia_ids") or ([meta.get("dia_id")] if meta.get("dia_id") else [])
            if any(d in gold_evidence for d in ids):
                hit_at_10 = True
                break
        if hit_at_10:
            n_hit_satisfiable += 1
        else:
            n_miss_satisfiable += 1
            # Check rank at top_k=50
            big_pack = eng.recall(question, top_k=50)
            gold_session_ulids = {dia_to_ulid.get(d) for d in gold_evidence if dia_to_ulid.get(d)}
            found_at = None
            for rank, r in enumerate(big_pack.results):
                if r.ulid in gold_session_ulids:
                    found_at = rank + 1
                    break
            if found_at is not None:
                if found_at <= 20:
                    misses_solved_at_20 += 1
                if found_at <= 50:
                    misses_solved_at_50 += 1
    return {
        "n_total": n_total,
        "n_unsatisfiable": n_unsatisfiable,
        "n_hit_satisfiable": n_hit_satisfiable,
        "n_miss_satisfiable": n_miss_satisfiable,
        "misses_solved_at_20": misses_solved_at_20,
        "misses_solved_at_50": misses_solved_at_50,
    }


def main():
    DATASET = data_path("locomo10.json")
    with open(DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Analyzing {len(dataset)} conversations on multi-hop (cat 3)...\n")

    agg = Counter()
    per_conv = []
    for ci, conv in enumerate(dataset):
        sample_id = conv.get("sample_id", f"conv-{ci}")

        tmp_home = Path(tempfile.mkdtemp(prefix=f"pmb-fa-{ci}-"))
        tmp_ws = Path(tempfile.mkdtemp(prefix=f"pmb-fa-ws-{ci}-"))
        os.environ["PMB_HOME"] = str(tmp_home)
        from pmb.core.engine import Engine
        eng = Engine(
            cwd=tmp_ws, pmb_home=tmp_home,
            config_overrides={
                "recall.cache_size": 0,
                "recall.spreading_activation": False,
                "recall.graph_boost": 0.15,
            },
        )

        # Session-chunked ingest
        for sname in sorted(
            [k for k in conv["conversation"]
             if k.startswith("session_") and not k.endswith("_date_time")],
            key=lambda k: int(k.split("_")[1]),
        ):
            turns = conv["conversation"].get(sname) or []
            if not turns:
                continue
            content_parts = [
                f"{t.get('speaker', '?')}: {t.get('text', '')}"
                for t in turns if t.get("text", "").strip()
            ]
            dia_ids = [t.get("dia_id") for t in turns if t.get("dia_id")]
            content = "\n".join(content_parts)[:8000]
            eng.record_event(
                event_type="qa", content=content, importance=0.5,
                metadata={"dia_ids": dia_ids, "session": sname, "locomo": True},
            )

        r = analyze_conv(conv, eng)
        per_conv.append({"sample_id": sample_id, **r})
        for k, v in r.items():
            agg[k] += v

        naive_recall = (r["n_hit_satisfiable"] / r["n_total"]) if r["n_total"] else 0
        true_recall_at_satisfiable = (
            r["n_hit_satisfiable"] / max(1, r["n_total"] - r["n_unsatisfiable"])
        )
        print(f"{sample_id}: n={r['n_total']:3d} "
              f"unsatisfiable={r['n_unsatisfiable']:2d} "
              f"hit={r['n_hit_satisfiable']:3d} miss={r['n_miss_satisfiable']:2d} "
              f"  naive={naive_recall:.1%}  true(sat)={true_recall_at_satisfiable:.1%}  "
              f"miss-solved@20={r['misses_solved_at_20']} @50={r['misses_solved_at_50']}")

        # Cleanup
        import gc, shutil
        del eng
        gc.collect()
        for p in (tmp_home, tmp_ws):
            for _ in range(3):
                try:
                    shutil.rmtree(p, ignore_errors=False)
                    break
                except (OSError, PermissionError):
                    time.sleep(0.3)
                    gc.collect()

    # Aggregate
    print("\n" + "=" * 78)
    print("AGGREGATE")
    print("=" * 78)
    total_n = agg["n_total"]
    n_unsat = agg["n_unsatisfiable"]
    n_sat = total_n - n_unsat
    n_hit = agg["n_hit_satisfiable"]
    n_miss = agg["n_miss_satisfiable"]
    print(f"Total cat-3 questions:           {total_n}")
    print(f"  Unsatisfiable (empty gold):    {n_unsat} ({n_unsat/total_n:.1%})")
    print(f"  Satisfiable:                   {n_sat}")
    print(f"    Hit @ top_k=10:              {n_hit}")
    print(f"    Miss @ top_k=10:             {n_miss}")
    print(f"      Of which solvable @ 20:    {agg['misses_solved_at_20']}")
    print(f"      Of which solvable @ 50:    {agg['misses_solved_at_50']}")
    print()
    print(f"Naive cat-3 recall:              {n_hit / total_n:.1%}")
    print(f"True recall on satisfiable:      {n_hit / max(1, n_sat):.1%}  ← honest metric")
    print(f"Recall @ top_k=20 (sat):         {(n_hit + agg['misses_solved_at_20']) / max(1, n_sat):.1%}")

    # Save
    out = {
        "per_conversation": per_conv,
        "aggregate": dict(agg),
    }
    out_path = data_path("pmb_failures_all.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nDetails: {out_path}")


if __name__ == "__main__":
    main()
