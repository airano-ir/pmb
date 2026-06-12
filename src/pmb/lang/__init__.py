"""Language packs — extend PMB's lexical fast-paths to new languages as DATA.

PMB's stopwords / function-words / verb-synonyms / attribute-aliases ship in
code as a small **English** bootstrap floor. RU/UK (and every other language)
are carried by the anchor tier + ALD, not by a pack (see the G3 note below). A
language pack is an optional YAML file that EXTENDS the lexical lists for a
language's COLD path. Packs are **opt-in and file-based**, exactly like
``reference.yaml``:

  * Built-in templates live in ``pmb/lang/packs/*.yaml`` (de, es, …) — NOT
    active by themselves.
  * A pack becomes ACTIVE when its file is present in ``$PMB_HOME/lang/``.
    ``pmb lang enable de`` copies the template there; you can also drop your
    own ``$PMB_HOME/lang/<code>.yaml``.

Why opt-in and not script-auto: German and English share the Latin script, so
auto-activating ``de`` on any Latin corpus would pollute an English workspace's
stopwords. Activation is therefore explicit (``pmb lang enable``); ``pmb lang
detect`` SUGGESTS packs from the corpus but never silently changes behaviour.

Because packs are EXTEND-ONLY and the English floor stays in code, a workspace
with no ``$PMB_HOME/lang/`` files behaves byte-identically to the shipped core.

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

# G3 (v0.9): NO built-in default-active packs. The RU/UK lexical floor that used
# to live in ru.yaml/uk.yaml is now carried by the anchor tier (WARM: English
# exemplars + cross-lingual embedder) and the ALD-distilled
# $PMB_HOME/lang/auto.yaml (COLD: grows from the user's own traffic), so the two
# packs were DELETED. The loader still merges any pack a user drops into
# $PMB_HOME/lang/ (override escape hatch) and the opt-in de/es TEMPLATES under
# packs/ — but nothing is active by default. RU/UK intents/keyed-extraction are
# WARM-anchor classified now (the cold path self-heals via ALD); the regex/pack
# tests below were migrated to assert that path.
_DEFAULT_ACTIVE: tuple[str, ...] = ()


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
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Ignoring malformed lang pack %s: %s",
                                             path, e)
        return {}


@lru_cache(maxsize=1)
def active_packs() -> dict[str, dict]:
    """{code: pack_data} for every ACTIVE pack: the built-in always-active floor
    packs (``_DEFAULT_ACTIVE`` = ru, uk) plus anything the user has enabled (a
    file present in ``$PMB_HOME/lang/``). A user file with the same code EXTENDS
    the built-in one (extend-only union, like reference.yaml). Cached; call
    clear_cache() after enabling one."""
    out: dict[str, dict] = {}
    # Built-in floor packs (ru, uk) — always active, shipped in packs/. The
    # G2 packs-off ratchet sets PMB_DISABLE_DEFAULT_PACKS=1 to prove the anchor
    # tier carries recall with NO built-in packs active (deletion soak, pre-G3).
    _defaults = () if os.environ.get("PMB_DISABLE_DEFAULT_PACKS") else _DEFAULT_ACTIVE
    for code in _defaults:
        p = _BUILTIN_DIR / f"{code}.yaml"
        if p.exists():
            data = _load_yaml(p)
            if data:
                out[code] = data
    # User-enabled packs (de/es templates, overrides, or extensions).
    d = user_dir()
    if d.exists():
        try:
            for p in sorted(d.glob("*.yaml")):
                data = _load_yaml(p)
                if not data:
                    continue
                if p.stem in out:
                    _merge_into(out[p.stem], data)  # extend the built-in
                else:
                    out[p.stem] = data
        except Exception:
            pass
    return out


def _merge_into(base: dict, extra: dict) -> None:
    """Extend-only merge of `extra` into `base` (user pack over built-in)."""
    for k, v in extra.items():
        if isinstance(v, list) and isinstance(base.get(k), list):
            base[k] = base[k] + v
        elif isinstance(v, dict) and isinstance(base.get(k), dict):
            for ck, cv in v.items():
                if isinstance(cv, list) and isinstance(base[k].get(ck), list):
                    base[k][ck] = base[k][ck] + cv
                else:
                    base[k][ck] = cv
        else:
            base[k] = v


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


_RE_FLAG = {"i": 2, "m": 8, "s": 16, "x": 64, "u": 32, "a": 256, "l": 4}


def compile_patterns(category: str, default_flags=None) -> list:
    """Compile the regex pattern fragments an active pack contributes for
    `category` (L1 — relocating RU/UK regexes out of .py). Each pack item is
    either a raw pattern string (compiled with `default_flags`) or a dict
    ``{re: "...", flags: "im"}``. Returns a list of compiled patterns; the
    caller PREPENDS its English inline patterns. Order within a category is
    irrelevant for the iterate-all matchers these feed."""
    import re as _re
    if default_flags is None:
        default_flags = _re.IGNORECASE
    out = []
    for val in _iter_category(category):
        if not isinstance(val, list):
            continue
        for item in val:
            try:
                if isinstance(item, str):
                    out.append(_re.compile(item, default_flags))
                elif isinstance(item, dict) and item.get("re"):
                    fl = 0
                    for ch in str(item.get("flags") or ""):
                        fl |= _RE_FLAG.get(ch.lower(), 0)
                    out.append(_re.compile(item["re"], fl or default_flags))
            except Exception:
                # a malformed pattern in a pack must never crash recall
                continue
    return out


def merged_list(category: str) -> list:
    """Concatenate the list items every active pack contributes for `category`
    (order = pack order). Used for relocated regex FRAGMENTS (verb stems,
    subject cues) that the caller joins into an alternation, and for
    prefix+attr dicts (current-state patterns)."""
    out: list = []
    for val in _iter_category(category):
        if isinstance(val, list):
            out.extend(val)
    return out


def compile_patterns_labeled(category: str, default_flags=None) -> list:
    """Like compile_patterns but each pack item carries a `label`; returns a
    list of ``(compiled, label)`` tuples (for matchers keyed by label, e.g.
    query-split markers). Item: ``{re: "...", label: "x", flags: "i"}``."""
    import re as _re
    if default_flags is None:
        default_flags = _re.IGNORECASE
    out = []
    for val in _iter_category(category):
        if not isinstance(val, list):
            continue
        for item in val:
            if not isinstance(item, dict) or not item.get("re"):
                continue
            try:
                fl = 0
                for ch in str(item.get("flags") or ""):
                    fl |= _RE_FLAG.get(ch.lower(), 0)
                out.append((_re.compile(item["re"], fl or default_flags),
                            str(item.get("label") or "")))
            except Exception:
                continue
    return out


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
