"""Language packs — extend PMB's lexical fast-paths to new languages as DATA.

PMB's stopwords / function-words / verb-synonyms / attribute-aliases ship in
code as defaults covering EN + RU + UK (the "floor"). A language pack is a YAML
file that EXTENDS those lists for another language. Packs are **opt-in and
file-based**, exactly like ``reference.yaml``:

  * Built-in templates live in ``pmb/lang/packs/*.yaml`` (de, es, …) — NOT
    active by themselves.
  * A pack becomes ACTIVE when its file is present in ``$PMB_HOME/lang/``.
    ``pmb lang enable de`` copies the template there; you can also drop your
    own ``$PMB_HOME/lang/<code>.yaml``.

Why opt-in and not script-auto: German and English share the Latin script, so
auto-activating ``de`` on any Latin corpus would pollute an English workspace's
stopwords. Activation is therefore explicit (``pmb lang enable``); ``pmb lang
detect`` SUGGESTS packs from the corpus but never silently changes behaviour.

Because packs are EXTEND-ONLY and the EN/RU/UK floor stays in code, a workspace
with no ``$PMB_HOME/lang/`` files behaves byte-identically to before.

Pack schema (all keys optional):

    code: de
    name: German
    stopwords: [der, die, das, und, ist, ...]
    not_proper: [wann, warum, wo, wer, ...]       # sentence-initial non-nouns
    first_person: [ich, mein, meine, mir, mich]
    verb_synonyms:                                # canonical -> [stems]
      live: [wohne, wohnt, lebe, lebt]
      work: [arbeite, arbeitet]
    attribute_aliases:                            # canonical -> [labels]
      city: [stadt, wohnort]
      employer: [arbeitgeber, firma]
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_BUILTIN_DIR = Path(__file__).parent / "packs"


def _pmb_home() -> Path:
    return Path(os.environ.get("PMB_HOME") or (Path.home() / ".pmb"))


def user_dir() -> Path:
    return _pmb_home() / "lang"


def builtin_templates() -> dict[str, Path]:
    """{code: path} of shipped pack templates (not active until enabled)."""
    out: dict[str, Path] = {}
    try:
        for p in sorted(_BUILTIN_DIR.glob("*.yaml")):
            out[p.stem] = p
    except Exception:
        pass
    return out


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Ignoring malformed lang pack %s: %s",
                                             path, e)
        return {}


@lru_cache(maxsize=1)
def active_packs() -> dict[str, dict]:
    """{code: pack_data} for every pack the user has ACTIVATED (a file present
    in ``$PMB_HOME/lang/``). Cached; call clear_cache() after enabling one."""
    out: dict[str, dict] = {}
    d = user_dir()
    if not d.exists():
        return out
    try:
        for p in sorted(d.glob("*.yaml")):
            data = _load_yaml(p)
            if data:
                out[p.stem] = data
    except Exception:
        pass
    return out


def clear_cache() -> None:
    active_packs.cache_clear()


def active_codes() -> list[str]:
    return sorted(active_packs().keys())


def _iter_category(category: str):
    for data in active_packs().values():
        val = data.get(category)
        if val is not None:
            yield val


def merged_set(category: str, defaults):
    """defaults ∪ the `category` list from every active pack. Returns the same
    container kind (set / frozenset) as `defaults`. Extend-only."""
    merged = set(defaults)
    for val in _iter_category(category):
        if isinstance(val, (list, set, tuple)):
            merged.update(str(x).strip().lower() for x in val if str(x).strip())
    return frozenset(merged) if isinstance(defaults, frozenset) else merged


def merged_groups(category: str, defaults: dict) -> dict:
    """Merge a canonical->set/list mapping (verb_synonyms / attribute_aliases)
    from active packs INTO `defaults` (extend-only; canonical keys union)."""
    out = {k: set(v) for k, v in defaults.items()}
    for val in _iter_category(category):
        if not isinstance(val, dict):
            continue
        for canon, members in val.items():
            if isinstance(members, (list, set, tuple)):
                out.setdefault(str(canon), set()).update(
                    str(m).strip().lower() for m in members if str(m).strip())
    return out
