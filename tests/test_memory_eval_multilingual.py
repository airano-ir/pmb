"""F4 — expanded memory-quality eval: pack-free multilingual recall + paraphrase
robustness. Grows the v0.8 V1 gate (13 queries, en/ru/uk) to 100+ labelled
queries across en/ru/uk/de/es/fr/it/pl/pt/ja/zh — the non-en/ru/uk buckets have
NO language pack and pass ONLY through the multilingual embedder's vector
channel. Buckets: per-language in-language paraphrases, a 5-phrasing paraphrase
bucket, and cross-lingual (EN question -> foreign fact). This is the
accuracy-you-can-SEE gate for v0.9.

Floors are measured on the real deterministic embedder (2026-06-12) and set a
margin BELOW, exactly like V1: a real regression (a botched anchor/pack change
tanking a language) drops a bucket below floor and fails loudly; embedder noise
does not. Distant-script buckets (ja/zh) are gated only on top-3 — the MiniLM
embedder is weak there (PLAN risk #1), so top-1 is informational, not a floor.
"""
from __future__ import annotations

import pytest

from pmb.core.engine import Engine

# ── frozen multilingual corpus: (lang, text) — index is the stable id ─────────
CORPUS: list[tuple[str, str]] = [
    ("en", "We use PostgreSQL as the primary production database."),      # 0
    ("en", "Alice is the tech lead at Stripe."),                          # 1
    ("en", "We deploy the backend to AWS Fargate."),                      # 2
    ("en", "My birthday is on the 14th of March."),                       # 3
    ("en", "The staging API listens on port 8080."),                      # 4
    ("ru", "Я живу в Киеве."),                                            # 5
    ("ru", "Мой любимый язык программирования — Rust."),                  # 6
    ("ru", "Мы используем Redis для кеширования."),                       # 7
    ("uk", "Мене звати Олег."),                                           # 8
    ("uk", "Я пишу тести за допомогою pytest."),                          # 9
    ("de", "Ich wohne in München."),                                      # 10
    ("de", "Wir verwenden Kubernetes für das Deployment."),               # 11
    ("de", "Mein Lieblingseditor ist Neovim."),                           # 12
    ("es", "Vivo en Madrid."),                                            # 13
    ("es", "Mi lenguaje de programación favorito es Python."),            # 14
    ("es", "Trabajo como ingeniero de backend."),                         # 15
    ("fr", "Je travaille chez Datadog."),                                 # 16
    ("fr", "Nous utilisons GraphQL pour l'API."),                         # 17
    ("fr", "Mon framework préféré est React."),                           # 18
    ("it", "Abito a Roma."),                                              # 19
    ("it", "Uso Docker per i container."),                               # 20
    ("pl", "Mieszkam w Warszawie."),                                      # 21
    ("pl", "Moim ulubionym narzędziem jest Git."),                        # 22
    ("pt", "Eu moro em Lisboa."),                                         # 23
    ("ja", "私は東京に住んでいます。"),                                     # 24
    ("zh", "我们使用 MongoDB 存储数据。"),                                  # 25
]

# ── labelled queries: (bucket, query, expected_corpus_index) ──────────────────
QUERIES: list[tuple[str, str, int]] = [
    # paraphrase robustness — 5 phrasings each of several EN facts
    ("para", "what database do we run in production", 0),
    ("para", "which db powers production", 0),
    ("para", "what's our main production datastore", 0),
    ("para", "what do we store production data in", 0),
    ("para", "name the primary production database", 0),
    ("para", "who is the team lead at Stripe", 1),
    ("para", "who leads engineering at Stripe", 1),
    ("para", "which person heads the Stripe team", 1),
    ("para", "who's the tech lead over at Stripe", 1),
    ("para", "where do we deploy the backend", 2),
    ("para", "what runs our backend in the cloud", 2),
    ("para", "which platform hosts our backend", 2),
    ("para", "when is my birthday", 3),
    ("para", "what day was I born on", 3),
    ("para", "which date is my birthday", 3),
    ("para", "what port does the staging api use", 4),
    ("para", "which port is staging listening on", 4),
    # en in-language extras
    ("en", "what's my birth date", 3),
    ("en", "where is the backend hosted", 2),
    # de bucket (no pack — vector only)
    ("de", "wo wohne ich", 10),
    ("de", "in welcher Stadt lebe ich", 10),
    ("de", "wo lebe ich", 10),
    ("de", "womit deployen wir", 11),
    ("de", "welches Tool nutzen wir fürs Deployment", 11),
    ("de", "wie deployen wir", 11),
    ("de", "welchen Editor benutze ich am liebsten", 12),
    ("de", "was ist mein bevorzugter Editor", 12),
    # es bucket
    ("es", "dónde vivo", 13),
    ("es", "en qué ciudad vivo", 13),
    ("es", "cuál es mi ciudad", 13),
    ("es", "cuál es mi lenguaje de programación favorito", 14),
    ("es", "qué lenguaje prefiero para programar", 14),
    ("es", "cuál es mi lenguaje preferido", 14),
    ("es", "en qué trabajo", 15),
    ("es", "cuál es mi puesto de trabajo", 15),
    # fr bucket
    ("fr", "où est-ce que je travaille", 16),
    ("fr", "chez quelle entreprise je travaille", 16),
    ("fr", "pour qui je travaille", 16),
    ("fr", "qu'utilisons-nous pour l'API", 17),
    ("fr", "avec quoi construit-on l'API", 17),
    ("fr", "quel est mon framework préféré", 18),
    ("fr", "quel framework je préfère", 18),
    # it bucket
    ("it", "dove abito", 19),
    ("it", "in quale città vivo", 19),
    ("it", "qual è la mia città", 19),
    ("it", "cosa uso per i container", 20),
    ("it", "quale strumento uso per i container", 20),
    # pl bucket
    ("pl", "gdzie mieszkam", 21),
    ("pl", "w jakim mieście mieszkam", 21),
    ("pl", "jakie jest moje ulubione narzędzie", 22),
    ("pl", "którego narzędzia używam najchętniej", 22),
    # pt bucket
    ("pt", "onde eu moro", 23),
    ("pt", "em que cidade eu moro", 23),
    # ru / uk (pack + vector)
    ("ru", "в каком городе я живу", 5),
    ("ru", "где я живу", 5),
    ("ru", "какой язык программирования мне нравится", 6),
    ("ru", "какой мой любимый язык", 6),
    ("ru", "что мы используем для кеша", 7),
    ("ru", "чем мы кешируем", 7),
    ("uk", "як мене звати", 8),
    ("uk", "яке моє ім'я", 8),
    ("uk", "чим я пишу тести", 9),
    ("uk", "який інструмент для тестів я використовую", 9),
    # ja bucket (distant script — top3 only)
    ("ja", "私はどこに住んでいますか", 24),
    ("ja", "私の住んでいる都市はどこですか", 24),
    ("ja", "どこに住んでいますか", 24),
    # zh bucket (distant script — top3 only)
    ("zh", "我们用什么来存储数据", 25),
    ("zh", "我们的数据库是什么", 25),
    ("zh", "我们使用哪个数据库", 25),
    # more in-language paraphrases (to 100+ total)
    ("para", "what's the production db", 0),
    ("para", "which database stores prod data", 0),
    ("para", "name the Stripe tech lead", 1),
    ("para", "where is the backend running", 2),
    ("para", "tell me my birthday", 3),
    ("en", "who runs the team at Stripe", 1),
    ("de", "welche Stadt ist mein Zuhause", 10),
    ("de", "welche Plattform nutzen wir fürs Deployment", 11),
    ("de", "welchen Code-Editor bevorzuge ich", 12),
    ("es", "cuál es mi ciudad de residencia", 13),
    ("es", "qué lenguaje me gusta usar para programar", 14),
    ("es", "de qué trabajo yo", 15),
    ("fr", "dans quelle société je travaille", 16),
    ("fr", "quelle techno utilisons-nous pour l'API", 17),
    ("fr", "quel framework je préfère utiliser", 18),
    ("it", "qual è la mia città di residenza", 19),
    ("it", "con cosa gestisco i container", 20),
    ("pl", "w którym mieście żyję", 21),
    ("ru", "какой язык я предпочитаю", 6),
    ("ru", "какую систему для кеша мы используем", 7),
    ("uk", "яким інструментом я тестую", 9),
    # cross-lingual: EN question -> foreign fact (the bridge)
    ("xl", "where do I live", 5),               # EN -> RU
    ("xl", "what do we use for caching", 7),     # EN -> RU
    ("xl", "which city do I live in", 10),       # EN -> DE
    ("xl", "where do I work", 16),               # EN -> FR
    ("xl", "what is my favourite editor", 12),   # EN -> DE
    ("xl", "what language do I prefer", 14),     # EN -> ES (Python)
    ("xl", "what do we deploy with", 11),        # EN -> DE (Kubernetes)
    ("xl", "what is my favourite framework", 18),  # EN -> FR (React)
    ("xl", "which company do I work for", 16),   # EN -> FR
    ("xl", "what's my preferred programming language", 14),  # EN -> ES
    ("xl", "what container tool do I use", 20),  # EN -> IT (Docker)
]

# Measured 2026-06-12 on the real deterministic embedder; floors set a margin
# BELOW. Small buckets (pt n=2, ja/zh distant) gated loosely / top-3-only — the
# OVERALL floor and the large buckets (para/de/es/fr/ru) are the real gate.
TOP1_FLOOR = {"para": 0.80, "en": 0.00, "de": 0.66, "es": 0.66, "fr": 0.75,
              "it": 0.40, "pl": 0.60, "pt": 0.00, "ru": 0.80, "uk": 0.50,
              "ja": 0.00, "zh": 0.00, "xl": 0.00}
TOP3_FLOOR = {"para": 0.90, "en": 0.50, "de": 0.75, "es": 0.75, "fr": 0.85,
              "it": 0.42, "pl": 0.60, "pt": 0.50, "ru": 0.85, "uk": 0.60,
              "ja": 0.50, "zh": 0.50, "xl": 0.66}
OVERALL_TOP1_FLOOR = 0.62
OVERALL_TOP3_FLOOR = 0.80


def _rank_of(eng, query: str, expected_ulid: str, top_k: int = 5):
    pack = eng.recall(query=query, top_k=top_k)
    for i, r in enumerate(pack.results):
        if getattr(r, "ulid", None) == expected_ulid:
            return i + 1
    return None


@pytest.mark.eval
def test_multilingual_memory_floors(tmp_pmb_home, tmp_workspace_dir, capsys):
    eng = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home)
    try:
        eng.warmup()
    except Exception:
        pass
    ulids = [eng.record_fact(text, metadata={"kind": "fact", "lang": lang})
             for lang, text in CORPUS]
    for drain in ("wait_for_embed_queue", "_drain_embed_queue"):
        try:
            getattr(eng, drain)()
        except Exception:
            pass

    buckets: dict[str, list] = {}
    for bucket, query, idx in QUERIES:
        buckets.setdefault(bucket, []).append(_rank_of(eng, query, ulids[idx]))

    lines, all_ranks = [], []
    for b, ranks in sorted(buckets.items()):
        all_ranks += ranks
        n = len(ranks)
        t1 = sum(1 for r in ranks if r == 1) / n
        t3 = sum(1 for r in ranks if r and r <= 3) / n
        lines.append(f"  {b}: top1={t1:.2f} top3={t3:.2f} (n={n})")
    o1 = sum(1 for r in all_ranks if r == 1) / len(all_ranks)
    o3 = sum(1 for r in all_ranks if r and r <= 3) / len(all_ranks)
    report = (f"multilingual memory-quality eval (F4) — {len(all_ranks)} queries\n"
              + "\n".join(lines)
              + f"\n  OVERALL: top1={o1:.2f} top3={o3:.2f} (n={len(all_ranks)})")
    with capsys.disabled():
        print("\n" + report)

    assert len(all_ranks) >= 100, f"F4 target is 100+ queries, got {len(all_ranks)}"
    for b, ranks in buckets.items():
        n = len(ranks)
        t1 = sum(1 for r in ranks if r == 1) / n
        t3 = sum(1 for r in ranks if r and r <= 3) / n
        assert t1 >= TOP1_FLOOR.get(b, 0.0), f"{b} top1 {t1:.2f} < {TOP1_FLOOR.get(b)}\n{report}"
        assert t3 >= TOP3_FLOOR.get(b, 0.0), f"{b} top3 {t3:.2f} < {TOP3_FLOOR.get(b)}\n{report}"
    assert o1 >= OVERALL_TOP1_FLOOR, f"overall top1 {o1:.2f} < {OVERALL_TOP1_FLOOR}\n{report}"
    assert o3 >= OVERALL_TOP3_FLOOR, f"overall top3 {o3:.2f} < {OVERALL_TOP3_FLOOR}\n{report}"
