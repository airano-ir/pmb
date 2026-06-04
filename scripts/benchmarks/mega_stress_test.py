"""
MEGA STRESS TEST — exercises everything PMB ships post-hardening.

Phases:
  1. Setup + multi-domain / multi-language ingest (~500 events)
  2. Feature smoke (atomic extraction RU/UK, keyed-upsert, pattern split,
     auto vocab bridges, fact trees, multilingual check, warmup)
  3. Stress recall — 30 base queries × 100 paraphrases = 3000 queries,
     mix of EN/RU/UK, A/B with new defaults ON vs OFF
  4. Adversarial — declensions, mixed-language, multi-hop, "don't answer"
  5. Final headline numbers

Goal: one place to point at and say "yes, all of this works".

Runtime budget: ~10-20 minutes on a warm machine.
Output: stdout + JSON snapshot at /tmp/pmb_mega.json.

Usage:
    python scripts/benchmarks/mega_stress_test.py
    python scripts/benchmarks/mega_stress_test.py --phases 1,2,3  # subset
    python scripts/benchmarks/mega_stress_test.py --n-paraphrases 50
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _bench_data import data_path

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parent.parent.parent / "src"))


# ============================================================
# Data
# ============================================================


CODING_FACTS = [
    "We picked Postgres over MySQL because of JSONB support.",
    "We dropped MySQL after replication kept breaking under load.",
    "Production runs on Google Cloud, region europe-west1.",
    "API service is hosted on Cloud Run with auto-scaling.",
    "Primary database is Postgres 15 on Cloud SQL with HA replicas.",
    "Secrets are stored in Secret Manager, injected at deploy time.",
    "Static assets served via Cloud CDN with 1-hour TTL.",
    "Logging through Cloud Logging with 30-day retention.",
    "Authentication uses httpOnly cookies, not localStorage.",
    "We use mypy for type checking, only on new code (CI-enforced).",
    "Refresh token rotation is implemented in the auth service.",
    "Migration 0042 added COALESCE for NULL JSONB columns.",
    "We use ruff for linting (replaces flake8 + isort + black).",
    "GitHub Actions runs the test suite on every PR.",
    "Deployment is blue-green via Terraform, 5-minute drain.",
    "We chose Stripe over Adyen for payment processing.",
    "Rate limiting is token-bucket per API key.",
    "Bug found: order-service crashes on NULL metadata.",
    "Decision: enforce NOT NULL on JSONB columns going forward.",
    "Code review for PR #1234 added webhook support.",
    "Released v0.3.2 to staging; smoke tests passed.",
    "Wrote integration tests for OAuth flow with token rotation.",
    "Fixed login bug: race condition between session refresh and cookie set.",
    "Implemented rate limiting middleware using token bucket per API key.",
    "Alice leads the backend team.",
    "Bob leads infra.",
    "Carol owns frontend.",
    "Dana is the senior engineer at Acme, owns the API gateway.",
    "We use GitHub Actions for CI.",
    "We use Docker for local development.",
]


PERSONAL_RU = [
    "Меня зовут Алексей.",
    "Я живу в Киеве.",
    "Мой день рождения 7 июня.",
    "У меня кошка Мурка, ей 4 года, аллергия на курицу.",
    "Мне нравятся спокойные видеоигры.",
    "Предпочитаю чёрный чай без сахара.",
    "Мой любимый редактор — Helix, переключился с Neovim три месяца назад.",
    "Я работаю инженером в компании Acme.",
    "Мой друг Борис работает врачом.",
    "Моя сестра Анна живёт в Берлине.",
    "Я хожу в спортзал три раза в неделю.",
    "У меня анафилактическая аллергия на арахис.",
    "Я не ем мясо, я вегетарианец уже пять лет.",
    "Мой любимый фильм — Inception.",
    "Я учу испанский, занимаюсь по 30 минут в день.",
]


PERSONAL_UK = [
    "Мене звати Олексій.",
    "Я живу у Києві.",
    "Мій день народження 7 червня.",
    "Друг користувача Олексій працює програмістом.",
    "Моя дружина Олена дизайнер.",
    "Я люблю каву без цукру.",
    "Я переїхав до Львова два роки тому.",
    "Моя донька Софія ходить до школи.",
    "У мене собака на ім'я Рекс, лабрадор.",
    "Я грав на гітарі з дитинства.",
]


MIXED_FACTS = [
    "We chose Postgres because в нашем продукте важна JSONB поддержка.",
    "Користувач любить Python and TypeScript для backend.",
    "Production деплоится through GitHub Actions.",
    "Мій улюблений IDE is VSCode with Python extension.",
    "Discussed migration plan: с MongoDB to Postgres next quarter.",
]


# 30 base queries with expected substring + language tag
EXPECTED = [
    # ----- Coding domain -----
    ("Why did we choose Postgres?", ["Postgres", "JSONB"], "en"),
    ("Where does our production run?", ["Google Cloud", "europe-west1"], "en"),
    ("What database do we use?", ["Postgres 15", "Cloud SQL"], "en"),
    ("How is authentication done?", ["httpOnly", "cookies"], "en"),
    ("Who leads the backend?", ["Alice"], "en"),
    ("How long are logs retained?", ["30-day", "Cloud Logging"], "en"),
    ("Where are secrets stored?", ["Secret Manager"], "en"),
    ("How is rate limiting done?", ["token bucket", "API key"], "en"),
    ("What CI do we use?", ["GitHub Actions"], "en"),
    ("What's the rollback strategy?", ["blue-green", "Terraform"], "en"),
    # ----- Personal RU -----
    ("Как меня зовут?", ["Алексей"], "ru"),
    ("Где я живу?", ["Киев"], "ru"),
    ("Когда у меня день рождения?", ["7 июня"], "ru"),
    ("Как зовут мою кошку?", ["Мурка"], "ru"),
    ("Какие у меня аллергии?", ["арахис", "курицу"], "ru"),
    ("Где работает Борис?", ["врач"], "ru"),
    ("Я ем мясо?", ["вегетарианец", "не ем"], "ru"),
    ("Какой у меня любимый редактор?", ["Helix", "Neovim"], "ru"),
    ("Сколько раз в неделю я в спортзале?", ["три раза"], "ru"),
    ("Какой язык я учу?", ["испанский"], "ru"),
    # ----- Personal UK -----
    ("Як мене звати?", ["Олексій"], "uk"),
    ("Де я живу?", ["Києв", "Львов"], "uk"),
    ("Коли у мене день народження?", ["7 червня"], "uk"),
    ("Хто моя дружина?", ["Олена", "дизайнер"], "uk"),
    ("Як звати мою доньку?", ["Софія"], "uk"),
    ("Що працює Олексій?", ["програміст"], "uk"),
    # ----- Cross-lingual -----
    ("когда у меня день рождения?", ["7 июня", "червня"], "ru→uk"),
    ("Where does Алексей live?", ["Киев"], "en→ru"),
    # ----- Compound -----
    ("Where do we deploy and how is auth done?",
     ["Cloud Run", "httpOnly"], "en-compound"),
    ("Кто я и где я живу?", ["Алексей", "Киев"], "ru-compound"),
]


# ============================================================
# Paraphrase generator (for stress phase)
# ============================================================


def paraphrase(base: str, n: int = 100, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    out = [base]
    # Case + whitespace variants
    out.extend([
        base.lower(), base.upper(),
        " ".join(base.split()),  # whitespace normalised
        base.replace("?", "."),
        base.rstrip("?"),
    ])
    # Prefix/suffix decorations
    prefixes_en = ["", "Quick: ", "Hey, ", "Just curious — ", "btw, "]
    prefixes_ru = ["", "Скажи, ", "Слушай, ", "Кстати, ", "А вот "]
    prefixes_uk = ["", "Скажи, ", "Гей, ", "А ще ", "До речі, "]
    suffixes = ["", " please", " thanks", " — quickly", "?"]
    has_cyrillic = bool(re.search(r"[а-яёА-ЯЁіїєґІЇЄҐ]", base))
    prefixes = (
        prefixes_ru if has_cyrillic and "червня" not in base.lower() and "Олекс" not in base
        else prefixes_uk if "Олекс" in base or "червня" in base.lower()
        else prefixes_en
    )
    # Iteration cap (not output cap) — prevents infinite loop when unique
    # combinations are exhausted before reaching n. With ~5 prefixes ×
    # ~7 suffixes = ~35 combos, for n=50 we'd otherwise loop forever.
    iterations = 0
    max_iter = max(20 * n, 500)
    while len(out) < n and iterations < max_iter:
        iterations += 1
        p = rng.choice(prefixes)
        s = rng.choice(suffixes)
        q = (p + base + s).strip()
        if q not in out:
            out.append(q)
    return out[:n]


# ============================================================
# Helpers
# ============================================================


def matches(content: str, expected_terms: list[str]) -> bool:
    c = content.lower()
    return any(t.lower() in c for t in expected_terms)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(pct * len(s))))
    return s[idx]


def fresh_engine(overrides: dict | None = None):
    from pmb.core.engine import Engine
    tmp_home = Path(tempfile.mkdtemp())
    tmp_ws = Path(tempfile.mkdtemp())
    os.environ["PMB_HOME"] = str(tmp_home)
    eng = Engine(
        cwd=tmp_ws, pmb_home=tmp_home, rerank_model=None,
        config_overrides=overrides or {},
    )
    _ = eng.search.model
    return eng


def ingest_all(eng):
    items = []
    for f in CODING_FACTS:    items.append({"type": "fact", "content": f})
    for f in PERSONAL_RU:     items.append({"type": "fact", "content": f})
    for f in PERSONAL_UK:     items.append({"type": "fact", "content": f})
    for f in MIXED_FACTS:     items.append({"type": "fact", "content": f})
    # Chunk into batches of 20
    for i in range(0, len(items), 20):
        eng.record_batch(items[i:i + 20])
    eng.wait_for_embed_queue(timeout_seconds=60.0)


def evaluate(eng, queries: list[tuple[str, list[str], str]]) -> dict:
    top1 = top3 = top10 = 0
    per_lang: dict[str, dict] = defaultdict(lambda: {"n": 0, "top1": 0,
                                                       "top3": 0, "top10": 0})
    lats: list[float] = []
    misses: list[tuple[str, str]] = []
    total = len(queries)
    progress_every = max(1, total // 20)
    t_eval_start = time.time()
    for i, (q, exp, lang) in enumerate(queries):
        if i and i % progress_every == 0:
            elapsed = time.time() - t_eval_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(f"    progress {i}/{total}  ({100*i/total:.0f}%)  "
                  f"rate={rate:.1f} q/s  ETA {eta:.0f}s",
                  flush=True)
        t0 = time.perf_counter()
        pack = eng.recall(q, top_k=10)
        lats.append((time.perf_counter() - t0) * 1000)
        per_lang[lang]["n"] += 1
        if not pack.results:
            if len(misses) < 30:
                misses.append((q, "(no results)"))
            continue
        if matches(pack.results[0].content, exp):
            top1 += 1
            per_lang[lang]["top1"] += 1
        elif len(misses) < 30:
            misses.append((q, pack.results[0].content[:60]))
        if any(matches(r.content, exp) for r in pack.results[:3]):
            top3 += 1
            per_lang[lang]["top3"] += 1
        if any(matches(r.content, exp) for r in pack.results):
            top10 += 1
            per_lang[lang]["top10"] += 1
    n = len(queries)
    return {
        "n": n,
        "top1": top1, "top1_pct": round(100 * top1 / n, 2) if n else 0,
        "top3": top3, "top3_pct": round(100 * top3 / n, 2) if n else 0,
        "top10": top10, "top10_pct": round(100 * top10 / n, 2) if n else 0,
        "p50_ms": round(percentile(lats, 0.5), 1),
        "p95_ms": round(percentile(lats, 0.95), 1),
        "p99_ms": round(percentile(lats, 0.99), 1),
        "per_lang": {
            k: {"n": v["n"],
                "top1_pct": round(100 * v["top1"] / v["n"], 2) if v["n"] else 0,
                "top3_pct": round(100 * v["top3"] / v["n"], 2) if v["n"] else 0}
            for k, v in per_lang.items()
        },
        "misses_sample": misses[:10],
    }


def section(title):
    bar = "=" * 75
    print(f"\n{bar}\n  {title}\n{bar}")


# ============================================================
# Phases
# ============================================================


def phase1_ingest(out: dict, args) -> tuple:
    section("PHASE 1 — Setup + multi-domain / multi-language ingest")
    eng = fresh_engine({
        "recall.pamvr_enabled": True,
        "recall.pattern_split": True,
        "recall.auto_vocab_bridges": True,
        "write.atomic_fact_extract": True,
        "recall.cache_size": 0,
    })
    t = eng.warmup(with_first_query=False)
    print(f"  warmup: {t['total_ms']:.0f}ms")
    t0 = time.time()
    ingest_all(eng)
    n_total = (len(CODING_FACTS) + len(PERSONAL_RU) +
               len(PERSONAL_UK) + len(MIXED_FACTS))
    ing_s = time.time() - t0
    print(f"  ingested {n_total} base facts in {ing_s:.1f}s")
    # Plus atomic atoms extracted
    import sqlite3
    with sqlite3.connect(str(eng.workspace.db_path)) as conn:
        n_active = conn.execute(
            "SELECT COUNT(*) FROM events WHERE archived_at IS NULL"
        ).fetchone()[0]
    print(f"  total active events (incl. atoms): {n_active}")
    out["phase1"] = {
        "base_facts": n_total,
        "total_active": n_active,
        "atoms_added": n_active - n_total,
        "ingest_seconds": round(ing_s, 1),
        "warmup_ms": t["total_ms"],
    }
    return eng, out


def phase2_features(eng, out: dict):
    section("PHASE 2 — Feature smoke (atomic, keyed-upsert, vocab, etc.)")
    p2: dict = {}

    # (a) RU/UK atomic extraction
    from pmb.reasoning.fact_extract import extract_atomic_facts
    ru_facts = extract_atomic_facts(
        "Меня зовут Алексей. Я живу в Киеве. Мой день рождения 7 июня. "
        "Я люблю спокойные игры.", min_len=20, min_sentences=1,
    )
    uk_facts = extract_atomic_facts(
        "Мене звати Олексій. Я живу у Києві. Мій день народження 7 червня.",
        min_len=20, min_sentences=1,
    )
    print(f"  (a) RU atoms: {len(ru_facts)}, UK atoms: {len(uk_facts)}")
    p2["ru_atoms_count"] = len(ru_facts)
    p2["uk_atoms_count"] = len(uk_facts)

    # (b) Keyed-upsert supersession
    r1 = eng.record_keyed_fact("user", "residence_test", "Киев")
    r2 = eng.record_keyed_fact("user", "residence_test", "Варшава")
    r3 = eng.record_keyed_fact("user", "residence_test", "Берлин")
    hist = eng.get_keyed_fact_history("user", "residence_test")
    current = [h for h in hist if h["is_current"]]
    archived = [h for h in hist if not h["is_current"]]
    print(f"  (b) keyed-upsert: history={len(hist)} "
          f"(current={len(current)}, archived={len(archived)})")
    p2["keyed_upsert_history_len"] = len(hist)
    p2["keyed_upsert_current_value"] = current[0]["value"] if current else None

    # (c) Auto VOCAB_BRIDGES — workspace lexicon
    eng._refresh_vocab_bridges(force=True)
    n_bridges = sum(len(v) for v in eng._vocab_bridges.values())
    print(f"  (c) auto VOCAB_BRIDGES: {len(eng._vocab_bridges)} keys, "
          f"{n_bridges} bridge terms")
    p2["vocab_bridge_keys"] = len(eng._vocab_bridges)
    p2["vocab_bridge_terms"] = n_bridges

    # (d) Pattern split firing on compound queries
    pack = eng.recall("where do we deploy and how is auth done", top_k=5)
    fired = getattr(eng, "_pattern_split_last_fired", None)
    returned = getattr(eng, "_pattern_split_last_returned", None)
    print(f"  (d) pattern split: fired={fired} returned={returned}")
    p2["pattern_split_fired"] = bool(fired)
    p2["pattern_split_returned"] = bool(returned)

    # (e) Multilingual model check
    from pmb.health.multilingual_check import evaluate as mlc_evaluate
    ml_result = mlc_evaluate(eng.workspace.db_path,
                              "paraphrase-multilingual-MiniLM-L12-v2")
    print(f"  (e) multilingual check: {ml_result['severity']} "
          f"(non-Latin={ml_result['non_latin_ratio']*100:.0f}%)")
    p2["non_latin_ratio"] = ml_result["non_latin_ratio"]
    p2["multilingual_severity"] = ml_result["severity"]

    # (f) Fact tree (hierarchical)
    res = eng.record_batch([{
        "type": "fact_tree",
        "main": "Production for orders service is hosted on Google Cloud.",
        "subfacts": [
            "Application code runs on Cloud Run, europe-west1.",
            "Primary DB is Postgres 15.",
            "Secrets are in Secret Manager.",
        ],
        "pin": True,
    }])
    eng.wait_for_embed_queue(timeout_seconds=30)
    pack = eng.recall("где хранятся секреты orders сервиса", top_k=3)
    subfact_hit = any("Secret Manager" in r.content for r in pack.results[:3])
    print(f"  (f) fact_tree: subfact retrievable individually = {subfact_hit}")
    p2["fact_tree_subfact_recall"] = subfact_hit

    # (g) Durable embed queue stats
    if hasattr(eng, "_durable_embed_queue") and eng._durable_embed_queue:
        pending = eng._durable_embed_queue.pending_count()
        dead = eng._durable_embed_queue.dead_letter_count()
        print(f"  (g) durable queue: pending={pending}, dead-letter={dead}")
        p2["durable_pending"] = pending
        p2["durable_dead_letter"] = dead

    out["phase2"] = p2
    return out


def phase3_stress(eng, out: dict, args):
    section(f"PHASE 3 — Stress recall ({len(EXPECTED)} bases × "
            f"{args.n_paraphrases} paraphrases)")
    queries: list[tuple[str, list[str], str]] = []
    for i, (q, exp, lang) in enumerate(EXPECTED):
        for p in paraphrase(q, n=args.n_paraphrases, seed=i):
            queries.append((p, exp, lang))
    print(f"  generated {len(queries)} total queries", flush=True)
    t0 = time.time()
    stats = evaluate(eng, queries)
    elapsed = time.time() - t0
    print(f"  eval elapsed: {elapsed:.0f}s")
    print(f"\n  HEADLINE: top-1={stats['top1_pct']:.1f}% "
          f"top-3={stats['top3_pct']:.1f}% top-10={stats['top10_pct']:.1f}%")
    print(f"  Latency: p50={stats['p50_ms']}ms p95={stats['p95_ms']}ms "
          f"p99={stats['p99_ms']}ms")
    print(f"\n  Per-language top-1:")
    for lang, v in sorted(stats["per_lang"].items()):
        print(f"    {lang:<12} n={v['n']:>4}  top1={v['top1_pct']:>5.1f}%  "
              f"top3={v['top3_pct']:>5.1f}%")
    out["phase3"] = stats
    return out


def phase4_adversarial(eng, out: dict):
    section("PHASE 4 — Adversarial / edge cases")
    p4 = {}

    # (a) Name declensions
    cases_decl = [
        ("Что ты знаешь про Алексея?", "Алексей"),
        ("Расскажи об Алексее", "Алексей"),
        ("Кто такой Алексей?", "Алексей"),
        ("Що знаєш про Олексія?", "Олексій"),
    ]
    hits = 0
    for q, name in cases_decl:
        pack = eng.recall(q, top_k=3)
        if pack.results:
            contents = " ".join(r.content for r in pack.results[:3]).lower()
            if name.lower() in contents:
                hits += 1
    print(f"  (a) declensions: {hits}/{len(cases_decl)} surfaced correct name")
    p4["declensions_pct"] = round(100 * hits / len(cases_decl), 1)

    # (b) Multiple persons disambiguation
    pack_b = eng.recall("Кем работает Борис?", top_k=3)
    pack_a = eng.recall("Where does Алексей live?", top_k=3)
    boris_ok = (pack_b.results and "Борис" in pack_b.results[0].content
                and "врач" in pack_b.results[0].content.lower())
    alexey_ok = (pack_a.results and "Алексей" in pack_a.results[0].content
                 and ("Киев" in pack_a.results[0].content or
                      "Україн" in pack_a.results[0].content))
    print(f"  (b) disambiguation: Борис={boris_ok}, Алексей={alexey_ok}")
    p4["disambig_boris"] = boris_ok
    p4["disambig_alexey"] = alexey_ok

    # (c) Off-topic should NOT promote personal facts
    off_topic_queries = [
        "сколько планет в солнечной системе",
        "what is the capital of Brazil",
        "as melhores praias do Algarve",  # Portuguese — completely off-domain
    ]
    promotions = 0
    for q in off_topic_queries:
        pack = eng.recall(q, top_k=1)
        if pack.results and pack.results[0].score > 0.7:
            # If a personal fact slipped in with high score, count as promotion
            top_c = pack.results[0].content.lower()
            if any(name in top_c for name in
                   ["алексей", "олекс", "борис", "мурка"]):
                promotions += 1
    print(f"  (c) off-topic personal-fact promotions: "
          f"{promotions}/{len(off_topic_queries)} (lower=better)")
    p4["off_topic_promotions"] = promotions

    # (d) Multi-hop compound queries
    compound = [
        ("Where do we deploy and what database do we use?",
         ["Cloud", "Postgres"]),
        ("Кто я и где я живу?", ["Алексей", "Киев"]),
    ]
    compound_hits = 0
    for q, terms in compound:
        pack = eng.recall(q, top_k=5)
        contents = " ".join(r.content for r in pack.results[:5]).lower()
        if all(t.lower() in contents for t in terms):
            compound_hits += 1
    print(f"  (d) compound queries (both sub-answers): "
          f"{compound_hits}/{len(compound)}")
    p4["compound_complete"] = compound_hits
    p4["compound_total"] = len(compound)

    out["phase4"] = p4
    return out


def phase5_ablation(out: dict, args):
    """A/B: full hardening defaults ON vs everything OFF."""
    section("PHASE 5 — A/B ablation: hardening ON vs OFF")

    # Build a fixed subset for fair compare (no need for full 3000)
    queries: list[tuple[str, list[str], str]] = []
    n_para = min(args.n_paraphrases, 20)  # smaller for A/B speed
    for i, (q, exp, lang) in enumerate(EXPECTED):
        for p in paraphrase(q, n=n_para, seed=i):
            queries.append((p, exp, lang))
    print(f"  comparing on {len(queries)} queries")

    print(f"\n  Run A: ALL OFF (raw baseline)")
    eng_off = fresh_engine({
        "recall.pamvr_enabled": False,
        "recall.pattern_split": False,
        "recall.auto_vocab_bridges": False,
        "write.atomic_fact_extract": False,
        "recall.cache_size": 0,
    })
    eng_off.warmup(with_first_query=False)
    ingest_all(eng_off)
    stats_off = evaluate(eng_off, queries)
    print(f"    top-1={stats_off['top1_pct']:.1f}%  "
          f"top-3={stats_off['top3_pct']:.1f}%  "
          f"top-10={stats_off['top10_pct']:.1f}%")
    try: eng_off.close()
    except Exception: pass

    print(f"\n  Run B: ALL ON (current hardened defaults)")
    eng_on = fresh_engine({
        "recall.pamvr_enabled": True,
        "recall.pattern_split": True,
        "recall.auto_vocab_bridges": True,
        "write.atomic_fact_extract": True,
        "recall.cache_size": 0,
    })
    eng_on.warmup(with_first_query=False)
    ingest_all(eng_on)
    stats_on = evaluate(eng_on, queries)
    print(f"    top-1={stats_on['top1_pct']:.1f}%  "
          f"top-3={stats_on['top3_pct']:.1f}%  "
          f"top-10={stats_on['top10_pct']:.1f}%")
    try: eng_on.close()
    except Exception: pass

    print(f"\n  Δ (ON - OFF):")
    for k in ("top1_pct", "top3_pct", "top10_pct"):
        delta = stats_on[k] - stats_off[k]
        sign = "+" if delta >= 0 else ""
        print(f"    {k:<12} {sign}{delta:>5.2f}pp")
    print(f"\n  Latency (p50/p95):")
    print(f"    OFF: {stats_off['p50_ms']}/{stats_off['p95_ms']}ms")
    print(f"    ON:  {stats_on['p50_ms']}/{stats_on['p95_ms']}ms")
    out["phase5"] = {"off": stats_off, "on": stats_on}
    return out


# ============================================================
# Main
# ============================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="1,2,3,4,5",
                    help="Comma-separated phase numbers to run (1-5)")
    ap.add_argument("--n-paraphrases", type=int, default=100,
                    help="Paraphrases per base query (default 100)")
    ap.add_argument("--out",
                    default=data_path("pmb_mega.json"))
    args = ap.parse_args()
    phases = set(args.phases.split(","))
    out: dict = {"n_paraphrases": args.n_paraphrases}

    eng = None
    if "1" in phases:
        eng, out = phase1_ingest(out, args)
    elif {"2", "3", "4"} & phases:
        # Need engine even if skipping ingest
        eng, out = phase1_ingest(out, args)

    if "2" in phases:
        out = phase2_features(eng, out)
    if "3" in phases:
        out = phase3_stress(eng, out, args)
    if "4" in phases:
        out = phase4_adversarial(eng, out)
    if eng:
        try: eng.close()
        except Exception: pass
    if "5" in phases:
        out = phase5_ablation(out, args)

    section("FINAL SUMMARY")
    s = out
    if "phase1" in s:
        print(f"  Setup:   {s['phase1']['total_active']} events, "
              f"warmup {s['phase1']['warmup_ms']:.0f}ms")
    if "phase3" in s:
        print(f"  Stress:  top-1={s['phase3']['top1_pct']:.1f}%  "
              f"top-3={s['phase3']['top3_pct']:.1f}%  "
              f"top-10={s['phase3']['top10_pct']:.1f}%")
        print(f"  Latency: p50={s['phase3']['p50_ms']}ms  "
              f"p95={s['phase3']['p95_ms']}ms  "
              f"p99={s['phase3']['p99_ms']}ms")
    if "phase4" in s:
        print(f"  Adversarial:")
        print(f"    declensions: {s['phase4']['declensions_pct']}%")
        print(f"    compound:    "
              f"{s['phase4']['compound_complete']}/{s['phase4']['compound_total']}")
        print(f"    off-topic promotions (lower=better): "
              f"{s['phase4']['off_topic_promotions']}")
    if "phase5" in s:
        d1 = s["phase5"]["on"]["top1_pct"] - s["phase5"]["off"]["top1_pct"]
        d3 = s["phase5"]["on"]["top3_pct"] - s["phase5"]["off"]["top3_pct"]
        print(f"  Hardening lift: top-1 {d1:+.1f}pp  top-3 {d3:+.1f}pp")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
