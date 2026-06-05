"""
Failure analysis for LoCoMo benchmark.

For each multi-hop (cat 3) question PMB gets WRONG:
  - print the query
  - print the gold answer + evidence dia_ids
  - print the actual gold event(s) content
  - print PMB's top-10 retrieved events
  - flag obvious patterns

Then we can SEE what's failing and target it surgically instead of
throwing more techniques at the wall.

Usage:
  python scripts/analyze_failures.py
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


def main():
    DATASET = data_path("locomo10.json")
    with open(DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    conv = dataset[0]  # conv-26

    tmp_home = Path(tempfile.mkdtemp(prefix="pmb-analyze-"))
    tmp_ws = Path(tempfile.mkdtemp(prefix="pmb-analyze-ws-"))
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

    # Ingest session-chunked (matches our best benchmark setup)
    print("Ingesting...")
    n = 0
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
        n += 1
    print(f"  {n} session-chunked events\n")

    # Build a dia_id -> session-event map so we can show the actual gold content
    dia_to_ulid: dict = {}
    import sqlite3
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

    # Iterate cat 3 questions
    cat3 = [q for q in conv["qa"] if q.get("category") == 3]
    print(f"Cat 3 multi-hop questions: {len(cat3)}\n")

    failures = []
    successes = 0
    for q in cat3:
        question = q.get("question", "")
        gold_evidence = set(q.get("evidence") or [])
        gold_answer = str(q.get("answer", ""))

        pack = eng.recall(question, top_k=10)
        hit = False
        for r in pack.results:
            meta = r.metadata or {}
            dia = meta.get("dia_id")
            dia_ids = meta.get("dia_ids") or []
            if (dia and dia in gold_evidence) or any(d in gold_evidence for d in dia_ids):
                hit = True
                break

        if hit:
            successes += 1
        else:
            failures.append({
                "question": question,
                "gold_answer": gold_answer,
                "gold_evidence": list(gold_evidence),
                "retrieved": [(r.ulid, r.content[:200]) for r in pack.results],
                "gold_session_ulids": list({dia_to_ulid.get(d) for d in gold_evidence if dia_to_ulid.get(d)}),
            })

    print(f"Successes: {successes}/{len(cat3)}")
    print(f"Failures: {len(failures)}/{len(cat3)}\n")

    print("=" * 80)
    print("FAILURE SAMPLES (first 8)")
    print("=" * 80)
    for i, f in enumerate(failures[:8]):
        print(f"\n--- Failure {i+1} ---")
        print(f"  Q: {f['question']}")
        print(f"  Gold answer: {f['gold_answer']}")
        print(f"  Gold dia_ids: {f['gold_evidence']}")
        # Show the gold session content
        for u in f["gold_session_ulids"]:
            ev = eng.events.get_by_ulid(u)
            if ev:
                print(f"  Gold session ({u}):")
                print(f"    {(ev.content or '')[:400]}")
        print(f"  Top-10 retrieved:")
        for j, (u, text) in enumerate(f["retrieved"][:5]):
            in_gold = u in f["gold_session_ulids"]
            mark = " ★" if in_gold else "  "
            print(f"    {j+1}.{mark}{text[:150]}")

    # Pattern analysis
    print("\n" + "=" * 80)
    print("PATTERN ANALYSIS")
    print("=" * 80)
    keywords = Counter()
    for f in failures:
        for w in f["question"].lower().split():
            if len(w) > 3:
                keywords[w] += 1
    print("\nTop words in failed questions:")
    for w, c in keywords.most_common(15):
        print(f"  {w}: {c}")

    # Length of gold sessions
    gold_lens = []
    for f in failures:
        for u in f["gold_session_ulids"]:
            ev = eng.events.get_by_ulid(u)
            if ev:
                gold_lens.append(len(ev.content or ""))
    if gold_lens:
        import statistics
        print(f"\nGold-session content length stats:")
        print(f"  min: {min(gold_lens)}, max: {max(gold_lens)}")
        print(f"  mean: {statistics.mean(gold_lens):.0f}")
        print(f"  median: {statistics.median(gold_lens):.0f}")

    # How many failures had gold-session in top-50?
    print("\nWhere did gold appear in extended retrieval?")
    rank_distribution = Counter()
    for f in failures:
        # Re-run with top_k=50
        big_pack = eng.recall(f["question"], top_k=50)
        gold_ulids = set(f["gold_session_ulids"])
        found_at = None
        for rank, r in enumerate(big_pack.results):
            if r.ulid in gold_ulids:
                found_at = rank + 1
                break
        if found_at is None:
            rank_distribution["never"] += 1
        elif found_at <= 20:
            rank_distribution["11-20"] += 1
        elif found_at <= 30:
            rank_distribution["21-30"] += 1
        elif found_at <= 50:
            rank_distribution["31-50"] += 1
    for k, v in sorted(rank_distribution.items()):
        print(f"  {k}: {v}/{len(failures)}")

    # Save detailed
    out = {
        "n_total": len(cat3), "n_successes": successes,
        "n_failures": len(failures),
        "failures": failures,
        "top_failure_keywords": keywords.most_common(20),
    }
    out_path = data_path("pmb_failures.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nDetails: {out_path}")

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


if __name__ == "__main__":
    main()
