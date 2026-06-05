"""Debug LLM-judge: show reader+gold+judge for first 5 LoCoMo cat-1 questions."""
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


def main():
    DATASET = data_path("locomo10.json")
    with open(DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    conv = dataset[0]

    tmp_home = Path(tempfile.mkdtemp(prefix="pmb-jdbg-"))
    tmp_ws = Path(tempfile.mkdtemp(prefix="pmb-jdbg-ws-"))
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
            metadata={"dia_ids": dia_ids, "session": sname},
        )

    from pmb.eval.locomo_judge import LocomoJudge
    from pmb.health.consolidate import resolve_llm_client
    llm = resolve_llm_client(backend="claude")
    judge = LocomoJudge(reader_llm=llm, judge_llm=llm)

    cat1 = [q for q in conv["qa"] if q.get("category") == 1][:6]
    for i, q in enumerate(cat1):
        question = q["question"]
        gold = str(q.get("answer", ""))
        print(f"\n{'='*70}")
        print(f"Q{i+1}: {question}")
        print(f"GOLD: {gold}")

        pack = eng.recall(question, top_k=20)
        contents = []
        for r in pack.results:
            meta = r.metadata or {}
            dt = meta.get("session_dt") or ""
            if dt:
                contents.append(f"[Session date: {dt}]\n{r.content}")
            else:
                contents.append(r.content)
        print(f"  (retrieved {len(contents)} chunks, top hit preview: "
              f"{contents[0][:100] if contents else 'none'}...)")

        result = judge.run_question(
            question=question, gold=gold,
            retrieved_contents=contents, category=1,
        )
        print(f"PREDICTED: {result.prediction[:300]}")
        print(f"VERDICT  : correct={result.correct}, reason={result.reasoning}")

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
                time.sleep(0.3); gc.collect()


if __name__ == "__main__":
    main()
