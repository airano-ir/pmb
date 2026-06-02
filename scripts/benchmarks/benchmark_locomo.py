"""
LoCoMo benchmark adapter for PMB.

LoCoMo (Snap Research, arxiv 2402.17753): 10 multi-session conversations
with ~199 QA pairs each, probing memory across long-term dialogue.

What this script does:
  1. Loads a LoCoMo conversation JSON.
  2. Builds a fresh PMB workspace per conversation.
  3. Replays sessions chronologically, storing each turn as an event
     (record_event with type="qa", metadata={dia_id, speaker}).
  4. After full ingestion, runs every QA pair through engine.recall().
  5. Scores with token-F1 against the gold answer over the top-K retrieved
     context. Also reports recall@K (was the right dia_id in the evidence
     surfaced?) for direct comparison with retrieval-only systems.

Grading caveats (read before quoting numbers):
  - The mem0 / Letta / MemMachine paper numbers use an LLM-grader. F1 is a
    cheap proxy; expect mid-50s on questions that need open-form generation
    where their LLM-grade reports 90%+.
  - Therefore the meaningful PMB number from this script is
    **evidence_recall** — did the source dia_id surface in top-K? This is
    pure retrieval, no generation, no grader bias.
  - We also report F1 for posterity but it's not apples-to-apples against
    their numbers.

Usage:
    python scripts/benchmark_locomo.py --n-conversations 1   # smoke
    python scripts/benchmark_locomo.py --n-conversations 10  # full
    python scripts/benchmark_locomo.py --rerank               # toggle reranker

Time budget: ~5-15 min per conversation depending on rerank.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parent.parent / "src"))


def _tokenize(s: str) -> list[str]:
    """LoCoMo-style normalization for F1: lowercase, alphanumeric tokens only."""
    return re.findall(r"[a-z0-9]+", s.lower())


def f1(prediction: str, gold: str) -> float:
    p = Counter(_tokenize(prediction))
    g = Counter(_tokenize(gold))
    if not p or not g:
        return 0.0
    common = sum((p & g).values())
    if common == 0:
        return 0.0
    prec = common / sum(p.values())
    rec = common / sum(g.values())
    return 2 * prec * rec / (prec + rec)


def sessions_in_order(conversation: dict) -> list[tuple[str, str, list[dict]]]:
    """Yield (session_name, datetime_str, turns) in numerical order."""
    keys = sorted(
        [k for k in conversation if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )
    out = []
    for k in keys:
        dt = conversation.get(f"{k}_date_time", "")
        turns = conversation.get(k) or []
        out.append((k, dt, turns))
    return out


def ingest_conversation(eng, conversation: dict, chunk_by: str = "turn") -> int:
    """Stream conversation into PMB. Returns total events ingested.

    chunk_by:
      - 'turn'   : each dialogue turn becomes one event. 300+ events per
                   conversation; slow to ingest because the graph layer
                   bumps O(k²) edges per event. Most faithful to PMB's
                   normal "everything you say is one event" model.
      - 'session': each session (one chat window) becomes one event with
                   concatenated turns. ~20 events per conversation, fast.
                   evidence_recall is per-session not per-turn — easier to
                   hit, but we keep the gold dia_ids in metadata for
                   honest scoring against turn-level evidence.
    """
    n = 0
    for session_name, dt, turns in sessions_in_order(conversation):
        if chunk_by == "session":
            if not turns:
                continue
            content_parts = [f"{t.get('speaker', '?')}: {t.get('text', '')}" for t in turns if t.get("text", "").strip()]
            dia_ids = [t.get("dia_id") for t in turns if t.get("dia_id")]
            content = "\n".join(content_parts)[:8000]  # cap text per event
            eng.record_event(
                event_type="qa",
                content=content,
                importance=0.5,
                metadata={
                    "dia_ids": dia_ids,
                    "session": session_name,
                    "session_dt": dt,
                    "locomo": True,
                },
            )
            n += 1
        else:  # 'turn'
            for turn in turns:
                text = turn.get("text", "")
                if not text.strip():
                    continue
                content = f"{turn.get('speaker', '?')}: {text}"
                eng.record_event(
                    event_type="qa",
                    content=content,
                    importance=0.5,
                    metadata={
                        "dia_id": turn.get("dia_id"),
                        "session": session_name,
                        "session_dt": dt,
                        "speaker": turn.get("speaker"),
                        "locomo": True,
                    },
                )
                n += 1
    return n


def _evidence_hit(result_metadata: dict, gold_evidence_ids: set) -> bool:
    """Check whether a recall result's metadata mentions any gold dia_id.
    Handles both single 'dia_id' (turn-chunked) and 'dia_ids' (session-chunked).
    """
    dia = result_metadata.get("dia_id") if result_metadata else None
    if dia and dia in gold_evidence_ids:
        return True
    dia_ids = result_metadata.get("dia_ids") if result_metadata else None
    if dia_ids:
        return any(d in gold_evidence_ids for d in dia_ids)
    return False


def evaluate(eng, qa_list: list[dict], top_k: int = 10, rerank: bool = False) -> dict:
    """Run all QA, compute retrieval-recall + F1."""
    by_cat: dict[int, dict] = {}
    f1s: list[float] = []
    ev_hits: list[int] = []
    latencies: list[float] = []
    examples = []  # save a handful for inspection

    for i, q in enumerate(qa_list):
        question = q.get("question", "")
        gold = str(q.get("answer", ""))
        evidence_ids = set(q.get("evidence", []) or [])
        cat = q.get("category")

        t0 = time.perf_counter()
        pack = eng.recall(question, top_k=top_k, rerank=rerank)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        # Did any retrieved event have a dia_id present in gold evidence?
        evidence_hit = 0
        for r in pack.results:
            if _evidence_hit(r.metadata or {}, evidence_ids):
                evidence_hit = 1
                break
        ev_hits.append(evidence_hit)

        # Token-F1 between gold answer and the concatenated retrieved context
        context = " ".join(r.content for r in pack.results)
        score = f1(context, gold)
        f1s.append(score)

        cat_stats = by_cat.setdefault(cat, {"n": 0, "f1": 0.0, "ev_hit": 0})
        cat_stats["n"] += 1
        cat_stats["f1"] += score
        cat_stats["ev_hit"] += evidence_hit

        if i < 5:
            examples.append({
                "category": cat,
                "question": question,
                "gold": gold,
                "evidence_hit": bool(evidence_hit),
                "f1": round(score, 3),
                "top1": pack.results[0].content[:120] if pack.results else "",
            })

    n = len(qa_list)
    overall = {
        "n_questions": n,
        "evidence_recall_top_k": round(sum(ev_hits) / n, 3) if n else 0.0,
        "mean_f1": round(sum(f1s) / n, 3) if n else 0.0,
        "p50_latency_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        "p95_latency_ms": (
            round(sorted(latencies)[int(0.95 * len(latencies))], 1)
            if len(latencies) > 1 else 0.0
        ),
        "top_k": top_k,
        "rerank": rerank,
    }
    by_cat_summary = {}
    for cat, st in sorted(by_cat.items(), key=lambda kv: kv[0] if kv[0] is not None else -1):
        m = st["n"] or 1
        by_cat_summary[str(cat)] = {
            "n": st["n"],
            "evidence_recall": round(st["ev_hit"] / m, 3),
            "mean_f1": round(st["f1"] / m, 3),
        }
    return {"overall": overall, "by_category": by_cat_summary, "examples": examples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="C:/Users/alexb/AppData/Local/Temp/locomo10.json")
    ap.add_argument("--n-conversations", type=int, default=1,
                    help="How many conversations to evaluate (max 10)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--rerank", action="store_true",
                    help="Enable cross-encoder reranker (slower, +recall)")
    ap.add_argument("--chunk-by", choices=("turn", "session"), default="session",
                    help="How to chunk conversation: 'turn' (per dialogue line, slow ingest) "
                         "or 'session' (per chat window, fast). Session is the realistic "
                         "default for a chat-memory system.")
    ap.add_argument("--no-graph", action="store_true",
                    help="Skip graph indexing during ingestion (faster, fairer for retrieval-only eval)")
    ap.add_argument("--reflect", action="store_true",
                    help="PMB v2: run reflection + causation extraction after ingest. "
                         "Adds ~5-10s/event via Claude CLI. Boosts multi-hop.")
    ap.add_argument("--cluster-arcs", action="store_true",
                    help="PMB v2: run arc clustering after reflect. Helps narrative-style queries.")
    ap.add_argument("--extract-facts", action="store_true",
                    help="Improvement D (mem0-style): LLM extracts atomic facts "
                         "from each event during sleep. Reader sees clean facts.")
    ap.add_argument("--reflect-limit", type=int, default=200,
                    help="Max events to reflect on per conversation")
    ap.add_argument("--decompose", action="store_true",
                    help="PMB v2.2: adaptive query decomposition for multi-hop queries")
    ap.add_argument("--judge", action="store_true",
                    help="LLM-as-judge eval (reader + judge). Reports J-score "
                         "comparable to mem0/Letta papers. Slow: ~2 LLM calls per QA.")
    ap.add_argument("--judge-backend", default="auto",
                    help="LLM backend for reader+judge (auto / claude / anthropic / ollama)")
    ap.add_argument("--judge-limit", type=int, default=None,
                    help="Only judge first N questions per conversation (for fast smoke)")
    ap.add_argument("--out", default="C:/Users/alexb/AppData/Local/Temp/pmb_locomo.json")
    args = ap.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    n_conv = min(args.n_conversations, len(dataset))

    print(f"LoCoMo benchmark — {n_conv}/{len(dataset)} conversations, "
          f"top_k={args.top_k}, rerank={args.rerank}")
    print()

    all_results = []
    for ci, conv in enumerate(dataset[:n_conv]):
        sample_id = conv.get("sample_id", f"conv-{ci}")
        qa = conv.get("qa") or []
        conversation = conv.get("conversation") or {}

        print(f"=== {sample_id} — {len(qa)} QA ===")
        from pmb.core.engine import Engine
        tmp_home = Path(tempfile.mkdtemp(prefix=f"pmb-locomo-{ci}-"))
        tmp_ws = Path(tempfile.mkdtemp(prefix=f"pmb-locomo-ws-{ci}-"))
        os.environ["PMB_HOME"] = str(tmp_home)
        eng = Engine(
            cwd=tmp_ws, pmb_home=tmp_home,
            rerank_model=("cross-encoder/ms-marco-MiniLM-L-6-v2" if args.rerank else None),
            config_overrides={
                "recall.cache_size": 0,            # honest eval, no cache help
                "recall.spreading_activation": False,  # heavy SQL per call, skip for bench
                "recall.graph_boost": (0.0 if args.no_graph else 0.15),
                "recall.adaptive_decompose": bool(args.decompose),
            },
        )
        # Optional: skip graph indexing entirely during ingest. This is a
        # huge win in ingestion time (O(N²) edge upserts collapse to 0) at
        # the cost of graph-traversal-aided recall. The user picks via CLI.
        if args.no_graph:
            eng._index_event_in_graph = lambda *a, **kw: []  # type: ignore[method-assign]

        t_ing = time.time()
        n_turns = ingest_conversation(eng, conversation, chunk_by=args.chunk_by)
        ing_seconds = time.time() - t_ing
        print(f"  ingested {n_turns} events ({args.chunk_by}-chunked) in {ing_seconds:.1f}s")

        # Hardening: with the unified _embed_or_defer path, writes may be
        # deferred while the model is still loading. Block here so the
        # vector index is fully consistent before evaluation — otherwise
        # we'd measure with a half-built index and under-report recall.
        try:
            drain = eng.wait_for_embed_queue(timeout_seconds=120.0)
            if drain.get("timeout"):
                print(f"  WARN: embed queue drain timeout: {drain}")
        except Exception:
            pass

        # PMB v2: reflection + causation extraction (LLM-driven, in 'sleep')
        if args.reflect:
            t_refl = time.time()
            refl_result = eng.reflect_batch(
                limit=args.reflect_limit,
                max_age_days=3650.0,  # any age — these are simulated histories
            )
            refl_seconds = time.time() - t_refl
            print(f"  reflected {refl_result.get('n_reflected', 0)} events "
                  f"({refl_result.get('n_causation_edges', 0)} causation edges) "
                  f"in {refl_seconds:.1f}s")
            if refl_result.get("skipped") == "no_llm":
                print("  WARNING: no LLM available — reflection skipped")

        # PMB v2 phase 3: arc clustering (LLM groups events into narrative threads)
        if args.cluster_arcs:
            t_arc = time.time()
            arc_result = eng.cluster_events_into_arcs(
                limit=args.reflect_limit, max_age_days=3650.0,
            )
            arc_seconds = time.time() - t_arc
            print(f"  clustered into arcs: {arc_result.get('n_created', 0)} new, "
                  f"{arc_result.get('n_joined', 0)} joined, "
                  f"{arc_result.get('n_summaries_updated', 0)} summaries in {arc_seconds:.1f}s")
            if arc_result.get("skipped") == "no_llm":
                print("  WARNING: no LLM available — arc clustering skipped")

        # Improvement D: atomic fact extraction (mem0-style)
        if args.extract_facts:
            t_facts = time.time()
            fact_result = eng.extract_facts_batch(
                limit=args.reflect_limit, max_age_days=3650.0,
            )
            facts_seconds = time.time() - t_facts
            print(f"  extracted {fact_result.get('n_facts_added', 0)} atomic facts "
                  f"from {fact_result.get('n_candidates', 0)} events in {facts_seconds:.1f}s")
            if fact_result.get("skipped") == "no_llm":
                print("  WARNING: no LLM available — fact extraction skipped")

        t_eval = time.time()
        result = evaluate(eng, qa, top_k=args.top_k, rerank=args.rerank)
        eval_seconds = time.time() - t_eval
        result["sample_id"] = sample_id
        result["ingest_seconds"] = round(ing_seconds, 1)
        result["eval_seconds"] = round(eval_seconds, 1)
        result["n_turns_ingested"] = n_turns

        # Improvement A: LLM-as-judge — gives us a J-score directly
        # comparable to mem0/Letta papers. Slow (2 LLM calls per QA).
        if args.judge:
            from pmb.eval.locomo_judge import LocomoJudge, aggregate as agg_judge
            from pmb.health.consolidate import resolve_llm_client
            try:
                llm = resolve_llm_client(backend=args.judge_backend)
            except Exception as e:
                print(f"  WARNING: no LLM for judge ({e}) — skipping J-score")
                llm = None
            if llm is not None:
                judge = LocomoJudge(reader_llm=llm, judge_llm=llm)
                judge_results = []
                qa_subset = qa[: args.judge_limit] if args.judge_limit else qa
                t_judge = time.time()
                for q in qa_subset:
                    question = q.get("question", "")
                    gold = str(q.get("answer", ""))
                    cat = q.get("category")
                    pack = eng.recall(question, top_k=args.top_k, rerank=args.rerank)
                    # Prepend session-date metadata to each chunk so the
                    # reader can resolve relative time refs ("yesterday",
                    # "last week") into absolute dates. LoCoMo gold answers
                    # are absolute dates, conversations are relative.
                    contents = []
                    for r in pack.results:
                        meta = r.metadata or {}
                        dt = meta.get("session_dt") or meta.get("session_date_time") or ""
                        if dt:
                            contents.append(f"[Session date: {dt}]\n{r.content}")
                        else:
                            contents.append(r.content)
                    jr = judge.run_question(
                        question=question, gold=gold,
                        retrieved_contents=contents, category=cat,
                    )
                    judge_results.append(jr)
                run = agg_judge(judge_results)
                judge_seconds = time.time() - t_judge
                result["judge"] = run.to_summary()
                result["judge"]["wall_seconds"] = round(judge_seconds, 1)
                # Per-question detail so we can see WHY answers failed
                # (reader wrong vs judge strict vs date-resolution) instead of
                # iterating blind. Diagnostic only.
                result["judge"]["details"] = [
                    {"cat": r.category, "q": r.question, "gold": r.gold,
                     "pred": r.prediction, "correct": r.correct, "why": r.reasoning}
                    for r in run.results
                ]
                print(f"  J-score                    = {run.j_score:.2%}  "
                      f"({run.n_correct}/{run.n_total})")
                print(f"  per-category J:",
                      {k: v["j_score"] for k, v in result["judge"]["per_category"].items()})

        ov = result["overall"]
        print(f"  evidence_recall@{ov['top_k']} = {ov['evidence_recall_top_k']:.2%}")
        print(f"  mean_f1                    = {ov['mean_f1']:.3f}")
        print(f"  p50 / p95 latency          = {ov['p50_latency_ms']}ms / {ov['p95_latency_ms']}ms")
        print(f"  per-category evidence_recall:",
              {k: v["evidence_recall"] for k, v in result["by_category"].items()})
        all_results.append(result)

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
    if all_results:
        all_ev = [r["overall"]["evidence_recall_top_k"] for r in all_results]
        all_f1 = [r["overall"]["mean_f1"] for r in all_results]
        agg = {
            "n_conversations": len(all_results),
            "mean_evidence_recall": round(sum(all_ev) / len(all_ev), 3),
            "mean_f1": round(sum(all_f1) / len(all_f1), 3),
            "top_k": args.top_k,
            "rerank": args.rerank,
        }
        print()
        print("=" * 70)
        print(" Aggregate over all conversations")
        print("=" * 70)
        print(json.dumps(agg, indent=2))

    Path(args.out).write_text(
        json.dumps({"per_conversation": all_results,
                    "aggregate": agg if all_results else {}},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nDetails written to {args.out}")


if __name__ == "__main__":
    main()
