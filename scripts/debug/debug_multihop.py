"""Pick a few cat-3 (multi-hop) questions from LoCoMo conv-26 and show
what PMB retrieves vs the gold evidence. Goal: understand the failure
pattern."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _bench_data import data_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pmb.core.engine import Engine


def main():
    dataset_path = data_path("locomo10.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    conv = dataset[0]  # conv-26

    tmp_home = Path(tempfile.mkdtemp(prefix="pmb-debug-"))
    tmp_ws = Path(tempfile.mkdtemp(prefix="pmb-debug-ws-"))
    os.environ["PMB_HOME"] = str(tmp_home)
    eng = Engine(
        cwd=tmp_ws, pmb_home=tmp_home,
        config_overrides={
            "recall.cache_size": 0,
            "recall.spreading_activation": False,
            "recall.graph_boost": 0.15,
        },
    )

    # Ingest turn-chunked
    sessions = sorted(
        [k for k in conv["conversation"] if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )
    n_ingested = 0
    for sname in sessions:
        for turn in conv["conversation"].get(sname) or []:
            text = turn.get("text", "")
            if not text.strip():
                continue
            content = f"{turn.get('speaker', '?')}: {text}"
            eng.record_event(
                event_type="qa", content=content, importance=0.5,
                metadata={"dia_id": turn.get("dia_id"), "session": sname},
            )
            n_ingested += 1
    print(f"Ingested {n_ingested} events.\n")

    # Find cat 3 questions
    cat3 = [q for q in conv["qa"] if q.get("category") == 3]
    print(f"Cat 3 multi-hop questions in conv-26: {len(cat3)}\n")

    # Show 3 examples
    for i, q in enumerate(cat3[:3]):
        question = q.get("question", "")
        gold_evidence = set(q.get("evidence") or [])
        gold_answer = q.get("answer", "")

        print("=" * 70)
        print(f"Q{i+1}: {question}")
        print(f"  gold_answer: {gold_answer}")
        print(f"  gold_evidence (dia_ids): {gold_evidence}")
        print()

        # Show the gold evidence content
        print("  Gold evidence content:")
        import sqlite3
        with sqlite3.connect(eng.workspace.db_path) as conn:
            conn.row_factory = sqlite3.Row
            for dia_id in gold_evidence:
                row = conn.execute(
                    "SELECT content FROM events WHERE workspace_id = ? AND "
                    "json_extract(metadata, '$.dia_id') = ?",
                    (eng.workspace.id, dia_id),
                ).fetchone()
                if row:
                    print(f"    [{dia_id}] {row['content'][:150]}")

        # What does query entity extraction find?
        ext = eng.entity_extractor.extract(question)
        print(f"\n  Query entities extracted:")
        print(f"    techs={ext.techs}  files={ext.files}  concepts={ext.concepts}")

        # Run recall
        pack = eng.recall(question, top_k=10)
        print(f"\n  PMB top-10 retrieved (in rank order):")
        hit_gold = False
        for j, r in enumerate(pack.results):
            dia = (r.metadata or {}).get("dia_id")
            is_hit = dia in gold_evidence
            marker = "*GOLD*" if is_hit else "      "
            if is_hit:
                hit_gold = True
            print(f"    {j+1}. {marker} [{dia}] {r.content[:100]}")
        print(f"\n  Evidence hit: {hit_gold}")
        print()

    # Cleanup
    import gc, shutil, time
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
