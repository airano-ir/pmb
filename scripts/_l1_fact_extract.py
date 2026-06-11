"""L1: generate the fact_extract RU/UK pattern packs from the live _PATTERNS
(exact: re=pattern.pattern) and capture an extract_atomic_facts behavior
baseline, BEFORE relocating the patterns out of the .py module."""
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import pmb.reasoning.fact_extract as FE  # noqa: E402

PACKS = pathlib.Path(__file__).resolve().parent.parent / "src" / "pmb" / "lang" / "packs"
BASE = pathlib.Path(__file__).resolve().parent.parent / "tests" / "_fact_extract_baseline.json"


def _flags(p):
    return "i" if (p.flags & re.IGNORECASE) else ""


ru, uk = [], []
en_count = 0
for pat, kind, tmpl in FE._PATTERNS:
    if kind.startswith("ru_"):
        ru.append({"re": pat.pattern, "kind": kind, "template": tmpl, "flags": _flags(pat)})
    elif kind.startswith("uk_"):
        uk.append({"re": pat.pattern, "kind": kind, "template": tmpl, "flags": _flags(pat)})
    else:
        en_count += 1

import yaml  # noqa: E402


def _append(path, key, items):
    txt = path.read_text(encoding="utf-8")
    block = yaml.safe_dump({key: items}, allow_unicode=True, sort_keys=False,
                           default_flow_style=False, width=10000)
    path.write_text(txt.rstrip() + "\n" + block, encoding="utf-8")


_append(PACKS / "ru.yaml", "fact_extract_patterns", ru)
_append(PACKS / "uk.yaml", "fact_extract_patterns", uk)
print(f"EN inline patterns: {en_count}; RU pack: {len(ru)}; UK pack: {len(uk)}")

# behavior baseline
PROBES = [
    "Today I met Alice. She lives in Berlin and is the tech lead at Stripe.",
    "We use Postgres for storage. We chose Redis over Memcached.",
    "Bob is 30 years old and is married. He moved from Texas.",
    "Меня зовут Алексей. Я живу в Киеве и работаю инженером.",
    "Его зовут Иван. Иван переехал в Берлин. Мой друг Олег.",
    "Мой день рождения 14 марта. Я люблю спокойные игры.",
    "У меня кот по имени Барсик.",
    "Мене звати Олег. Я живу в Києві. Я працюю дизайнером.",
    "Його звуть Тарас. Тарас переїхав до Львова. Мій день народження 5 травня.",
    "random text with no extractable facts here at all",
]
snap = {}
for p in PROBES:
    try:
        facts = FE.extract_atomic_facts(p)
        snap[p] = sorted((f.content, f.kind) for f in facts)
    except Exception as e:
        snap[p] = f"ERR:{e}"
BASE.write_text(json.dumps(snap, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
print("baseline ->", BASE, "| probes:", len(PROBES))
