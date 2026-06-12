#!/usr/bin/env python
"""Anchor calibration (Phase A2) — pick each anchor set's margin threshold `tau`
from data, NOT by hand.

The Semantic Anchor Engine (src/pmb/lang/anchors.py) classifies a text by MARGIN
(its best cosine to a set's English positives, minus the strongest competing
signal). A1 shipped with placeholder taus (0.10 everywhere). That global guess
is wrong in both directions: too high for sets whose negatives sit far away
(throwing away real multilingual hits — the Russian/German "what's left to do"
that landed at margin 0.07), too low for sets whose negatives crowd in.

This harness fixes that empirically. For each set S:

  * POSITIVES  = a multilingual labelled corpus of texts that SHOULD fire S
                 (English + Russian/Ukrainian/German/Spanish/French/Polish/
                  Italian/Portuguese/Japanese/Chinese — written by hand below).
  * NEGATIVES  = every distractor PLUS every other set's positives (one-vs-rest:
                 a German goals-question is a NEGATIVE for self_intent).
  * tau(S)     = the LOWEST threshold whose false-positive rate on NEGATIVES is
                 <= MAX_FPR (default 1%). Lowest = highest recall under the FPR
                 budget. We take it just above the (1 - MAX_FPR) quantile of the
                 negative margins, floored at TAU_FLOOR so a tiny/clean negative
                 set can't drive tau to ~0.

The result is written to src/pmb/lang/anchor_calibration.json, keyed by the SAME
(embedder model id, anchor-definition hash) signature the .npz cache uses — so
editing an anchor's exemplars or upgrading the embedder invalidates the snapshot
(AnchorIndex ignores a mismatched key; the freeze test then fails loudly).

Run:  python scripts/calibrate_anchors.py            # writes the snapshot
      python scripts/calibrate_anchors.py --dry-run  # print table only
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

# ── labelled multilingual corpus ──────────────────────────────────────────────
# Positives per set. These are NOT anchors (the anchors stay English-only); they
# are the held-out, many-language texts we MEASURE recall against. The embedder
# never saw any of this — that is the whole point.

LABELLED: dict[str, list[str]] = {
    "intent.goals_query": [
        "what are my open goals", "what should I work on next",
        "what's left to do", "what tasks are still pending",
        "какие у меня открытые цели", "что мне делать дальше",
        "что осталось сделать", "какие задачи ещё не завершены",
        "які в мене відкриті цілі", "що залишилось зробити",
        "was sind meine offenen Ziele", "woran soll ich als Nächstes arbeiten",
        "was muss noch erledigt werden",
        "cuáles son mis objetivos abiertos", "qué queda por hacer",
        "quels sont mes objectifs en cours", "qu'est-ce qu'il reste à faire",
        "jakie są moje otwarte cele", "co zostało do zrobienia",
        "quali sono i miei obiettivi aperti", "o que ainda falta fazer",
        "未完了のタスクは何ですか", "次に何をすべきですか",
        "我还有哪些未完成的目标",
    ],
    "intent.past_query": [
        "what did I do yesterday", "why did we choose this database",
        "what did we decide last week", "who is Alice",
        "что я делал вчера", "почему мы выбрали эту базу данных",
        "что мы решили на прошлой неделе", "кто такой Алекс",
        "що я робив учора", "чому ми обрали цю базу даних",
        "was habe ich gestern gemacht", "warum haben wir diese Datenbank gewählt",
        "qué hice ayer", "por qué elegimos esta base de datos",
        "qu'ai-je fait hier", "pourquoi avons-nous choisi cette base de données",
        "co robiłem wczoraj", "dlaczego wybraliśmy tę bazę danych",
        "昨日は何をしましたか", "なぜこのデータベースを選んだのですか",
        "我昨天做了什么",
    ],
    "intent.recent_query": [
        "what did we just discuss", "what was I working on a moment ago",
        "where did I leave off",
        "что мы только что обсуждали", "над чем я только что работал",
        "на чём я остановился",
        "що ми щойно обговорювали", "на чому я зупинився",
        "worüber haben wir gerade gesprochen", "woran habe ich gerade gearbeitet",
        "qué acabamos de discutir", "en qué me quedé",
        "de quoi venons-nous de parler", "où en étais-je",
        "o czym przed chwilą rozmawialiśmy", "na czym skończyłem",
        "今何を話していましたか", "さっきまで何をしていましたか",
    ],
    "intent.lessons_query": [
        "what are the project conventions", "do we have a rule about commits",
        "how should I do this in this project",
        "какие в проекте соглашения", "есть ли у нас правило про коммиты",
        "как это принято делать в этом проекте",
        "які в проєкті домовленості", "як це прийнято робити тут",
        "was sind die Projektkonventionen", "haben wir eine Regel für Commits",
        "cuáles son las convenciones del proyecto",
        "tenemos una regla sobre los commits",
        "quelles sont les conventions du projet",
        "avons-nous une règle pour les commits",
        "jakie są konwencje w projekcie", "czy mamy zasadę dotyczącą commitów",
        "このプロジェクトの規約は何ですか", "コミットに関するルールはありますか",
    ],
    "intent.work_request": [
        "fix the auth module", "refactor this function",
        "add a test for the parser", "deploy the service",
        "почини модуль авторизации", "отрефактори эту функцию",
        "добавь тест для парсера", "задеплой сервис",
        "виправ модуль авторизації", "додай тест для парсера",
        "behebe das Auth-Modul", "refaktoriere diese Funktion",
        "deploye den Dienst",
        "arregla el módulo de autenticación", "refactoriza esta función",
        "corrige le module d'authentification", "refactorise cette fonction",
        "napraw moduł logowania", "zrefaktoryzuj tę funkcję",
        "認証モジュールを修正して", "この関数をリファクタリングして",
        "修复认证模块", "重构这个函数",
    ],
    "intent.self_intent": [
        "who am I", "where do I live", "what's my name",
        "what do I prefer", "what is my job",
        "кто я", "где я живу", "как меня зовут", "кем я работаю",
        "хто я", "де я живу",
        "wer bin ich", "wo wohne ich", "wie heiße ich",
        "quién soy", "dónde vivo", "cómo me llamo",
        "qui suis-je", "où est-ce que j'habite",
        "kim jestem", "gdzie mieszkam",
        "私は誰ですか", "私はどこに住んでいますか",
    ],
    "intent.trivial_ack": [
        "ok", "thanks", "got it", "sounds good", "great", "cool", "perfect",
        "ок", "спасибо", "понял", "отлично", "круто",
        "дякую", "зрозумів",
        "danke", "alles klar", "super",
        "vale", "gracias", "perfecto",
        "d'accord", "merci", "parfait",
        "dzięki", "jasne",
        "はい", "ありがとう", "了解",
        "好的", "谢谢",
    ],
    "query.lesson_intent": [
        "how should I do this here", "what's the convention for commits",
        "what should I avoid here", "what's the best way to test this",
        "как правильно это делать в проекте", "какие у нас соглашения по коммитам",
        "чего тут стоит избегать",
        "wie sollte ich das hier machen", "was ist die Konvention für Commits",
        "cómo debería hacer esto aquí", "cuál es la mejor manera de probar esto",
        "comment devrais-je faire cela ici", "quelle est la convention pour les commits",
        "jak powinienem to tutaj zrobić", "jakie są konwencje commitów",
        "come dovrei farlo qui", "ここではどうやるべきですか",
    ],
    "statement.future_intent": [
        "next we'll add the export feature", "the plan is to migrate to Postgres",
        "I will write the tests tomorrow", "going forward we should cache results",
        "the next step is to wire up the API",
        "дальше мы добавим экспорт", "план — мигрировать на Postgres",
        "завтра я напишу тесты",
        "als Nächstes fügen wir den Export hinzu", "der Plan ist, zu Postgres zu migrieren",
        "luego añadiremos la exportación", "el plan es migrar a Postgres",
        "ensuite nous ajouterons l'export", "le plan est de migrer vers Postgres",
        "następnie dodamy eksport", "planujemy migrację do Postgresa",
        "poi aggiungeremo l'esportazione", "次にエクスポート機能を追加します",
    ],
    "statement.about_self": [
        "I live in Tampa", "my name is Alex", "I work at Stripe",
        "I'm allergic to peanuts", "my job is software engineer",
        "я живу в Киеве", "меня зовут Алексей", "я работаю в Stripe",
        "у меня аллергия на арахис",
        "ich wohne in Berlin", "ich heiße Alex", "ich arbeite bei Stripe",
        "vivo en Madrid", "me llamo Alex", "trabajo en Stripe",
        "j'habite à Paris", "je m'appelle Alex", "je travaille chez Stripe",
        "mieszkam w Warszawie", "nazywam się Alex",
        "vivo a Roma", "mi chiamo Alex",
        "私は東京に住んでいます",
    ],
}

# Distractors — real-shaped texts that should fire NOTHING: statements, external
# facts, chit-chat that is not an ack, code. They are negatives for EVERY set, so
# they set the precision floor. Kept clearly out-of-class (a borderline same-act
# text would inflate every tau and is the embedder's job to separate, not ours).
DISTRACTORS: list[str] = [
    "the meeting is at 3pm tomorrow", "Python uses indentation for blocks",
    "what is the capital of France", "the weather is nice today",
    "I had coffee this morning", "the API returned a 500 error",
    "she went to the store", "the cat sat on the mat",
    "interest rates rose last quarter", "the train was late again",
    "def foo(): return 42", "SELECT * FROM users WHERE id = 1",
    "the package weighs two kilograms", "he plays the guitar on weekends",
    "встреча завтра в три часа", "сегодня хорошая погода",
    "какая столица Франции", "я выпил кофе утром",
    "кошка сидит на коврике", "поезд снова опоздал",
    "das Wetter ist heute schön", "was ist die Hauptstadt von Frankreich",
    "die Katze schläft auf dem Sofa",
    "hace buen tiempo hoy", "cuál es la capital de Francia",
    "el gato duerme en el sofá",
    "il fait beau aujourd'hui", "quelle est la capitale de la France",
    "dziś jest ładna pogoda", "jaka jest stolica Francji",
    "今日はいい天気です", "フランスの首都はどこですか",
    "今天天气很好", "法国的首都是哪里",
    "the quarterly report is due on Friday", "water boils at 100 degrees",
    "my flight lands at noon", "the documentary was three hours long",
]

MAX_FPR = 0.01
TAU_FLOOR = 0.04


def pick_tau(neg_margins: list[float], max_fpr: float = MAX_FPR,
             lo: float = TAU_FLOOR) -> float:
    """Lowest tau whose FPR on `neg_margins` is <= max_fpr (highest recall under
    the budget): just above the (1 - max_fpr) quantile, floored at `lo`."""
    if not neg_margins:
        return lo
    neg = sorted(neg_margins)
    allowed = math.floor(max_fpr * len(neg))          # tolerated false positives
    idx = max(0, min(len(neg) - 1 - allowed, len(neg) - 1))
    return round(max(neg[idx] + 1e-3, lo), 4)


def _build_index():
    """Construct an AnchorIndex on the real embedder (temp home so we never
    touch the user's workspace)."""
    from pmb.core.engine import Engine
    from pmb.lang.anchors import ALL_ANCHORS, AnchorIndex

    home = Path(tempfile.mkdtemp(prefix="pmb-calib-home-"))
    eng = Engine(cwd=Path.cwd(), pmb_home=home)
    model_id = getattr(eng.search, "model_name", "default")
    cache = Path(tempfile.mkdtemp(prefix="pmb-calib-cache-"))
    idx = AnchorIndex(eng.search.embed_batch, model_id, ALL_ANCHORS, cache)
    return idx, model_id


def main(argv: list[str]) -> int:
    dry = "--dry-run" in argv
    print("[calibrate] loading embedder + building anchor index ...", flush=True)
    idx, model_id = _build_index()

    names = list(LABELLED.keys())
    # Pre-score every corpus text once: text -> {set_name -> margin}.
    corpus = sorted({t for texts in LABELLED.values() for t in texts}
                    | set(DISTRACTORS))
    print(f"[calibrate] scoring {len(corpus)} texts across {len(names)} sets ...",
          flush=True)
    margins: dict[str, dict[str, float]] = {}
    pos_cos: dict[str, dict[str, float]] = {}
    for t in corpus:
        sm = idx.score_map(t)
        margins[t] = {n: sm[n].margin for n in names}
        pos_cos[t] = {n: sm[n].pos for n in names}

    label_of: dict[str, str] = {}
    for n, texts in LABELLED.items():
        for t in texts:
            label_of[t] = n   # last wins; corpus phrasings are unique per set

    # Group-aware negatives: a set competes only WITHIN its task group (the same
    # rule _scores uses for rivals). A text positive for a DIFFERENT-group set is
    # neither a positive nor a counted negative — so a guidance query ("what's
    # the convention") legitimately fires BOTH intent.lessons_query and
    # query.lesson_intent without each inflating the other's tau. Distractors are
    # negatives for everyone.
    group_of = {s.name: s.group for s in idx._sets}

    def _is_negative(text: str, name: str) -> bool:
        lbl = label_of.get(text)
        if lbl == name:
            return False                       # a positive of this set
        if lbl is None:
            return True                        # a distractor → negative for all
        return group_of.get(lbl) == group_of.get(name)   # same-task rival only

    sets_out: dict[str, dict] = {}
    print()
    print(f"  {'set':<22} {'tau':>6} {'floor':>6} {'recall':>7} "
          f"{'fpr':>6} {'npos':>5} {'nneg':>5}")
    print("  " + "-" * 62)
    floors = {s.name: s.floor for s in idx._sets}
    for n in names:
        pos_texts = LABELLED[n]
        neg_texts = [t for t in corpus if _is_negative(t, n)]
        neg_margins = [margins[t][n] for t in neg_texts]
        tau = pick_tau(neg_margins)
        floor = floors[n]
        # recall: positives clearing BOTH the calibrated tau and the floor.
        hits = sum(1 for t in pos_texts
                   if margins[t][n] >= tau and pos_cos[t][n] >= floor)
        recall = hits / max(1, len(pos_texts))
        fp = sum(1 for m in neg_margins if m >= tau)
        fpr = fp / max(1, len(neg_margins))
        sets_out[n] = {
            "tau": tau, "floor": round(floor, 4),
            "recall": round(recall, 4), "fpr": round(fpr, 4),
            "n_pos": len(pos_texts), "n_neg": len(neg_texts),
        }
        print(f"  {n:<22} {tau:>6.3f} {floor:>6.3f} {recall:>7.2f} "
              f"{fpr:>6.3f} {len(pos_texts):>5} {len(neg_texts):>5}")

    macro_recall = sum(s["recall"] for s in sets_out.values()) / len(sets_out)
    macro_fpr = sum(s["fpr"] for s in sets_out.values()) / len(sets_out)
    print("  " + "-" * 62)
    print(f"  macro recall={macro_recall:.3f}  macro fpr={macro_fpr:.3f}  "
          f"(target fpr <= {MAX_FPR})")

    snapshot = {
        "schema": 1,
        "key": idx._key,
        "model_id": model_id,
        "max_fpr": MAX_FPR,
        "tau_floor": TAU_FLOOR,
        "generated_by": "scripts/calibrate_anchors.py",
        "corpus": {"n_positives": sum(len(v) for v in LABELLED.values()),
                   "n_distractors": len(DISTRACTORS),
                   "languages": ["en", "ru", "uk", "de", "es", "fr", "pl",
                                 "it", "pt", "ja", "zh"]},
        "macro": {"recall": round(macro_recall, 4), "fpr": round(macro_fpr, 4)},
        "sets": sets_out,
    }
    out = Path(__file__).resolve().parents[1] / "src" / "pmb" / "lang" / \
        "anchor_calibration.json"
    if dry:
        print(f"\n[calibrate] --dry-run: NOT writing {out}")
        return 0
    out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"\n[calibrate] wrote {out}")
    print(f"[calibrate] key={idx._key}  model={model_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
