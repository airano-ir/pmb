"""Optional per-deployment reference-data overlay.

Stable word-lists (attribute aliases, known techs, stopwords, function-words,
kind priorities) live in code as DEFAULTS. If `PMB_HOME/reference.yaml` exists,
its entries EXTEND those defaults (or, for `kind_priority`, OVERRIDE them).
A missing or malformed file → defaults only (identical behaviour), with a
logged warning on malformed input.

This keeps PMB's deterministic core as DATA the user can grow without editing
Python - a step toward the "small core + config + offline-LLM" architecture.

reference.yaml schema (all keys optional):

    alias_groups:            # extend keyed-fact attribute aliases
      city: [hometown, where_i_live]
      employer: [gig]
    known_techs:             # extend the entity tech map (canonical: [variants])
      kubernetes: [k8s, kube]
    stopwords: [um, uh]      # extend recall stopwords (flat list)
    not_proper: [bonjour]    # extend PAMVR non-proper-noun function words
    kind_priority:           # OVERRIDE / add event-kind ranking priorities
      decision: 95
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

DEFAULT_PMB_HOME = Path.home() / ".pmb"


def _reference_path() -> Path:
    home = Path(os.environ.get("PMB_HOME", DEFAULT_PMB_HOME))
    return home / "reference.yaml"


@lru_cache(maxsize=1)
def _load_raw() -> dict:
    path = _reference_path()
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:  # malformed yaml / unreadable → defaults + warn
        import logging
        logging.getLogger(__name__).warning(
            "Ignoring malformed %s: %s", path, e
        )
        return {}


def clear_cache() -> None:
    """Drop the cached file (tests / after editing reference.yaml at runtime)."""
    _load_raw.cache_clear()


def extend_alias_groups(defaults: dict) -> dict:
    """Merge reference.yaml `alias_groups` INTO code defaults (extend-only)."""
    out = {k: set(v) for k, v in defaults.items()}
    raw = _load_raw().get("alias_groups")
    if isinstance(raw, dict):
        for canon, aliases in raw.items():
            if isinstance(aliases, (list, set, tuple)):
                out.setdefault(str(canon), set()).update(
                    str(a).strip().lower() for a in aliases
                )
    return out


def extend_tech_map(defaults: dict) -> dict:
    """Merge reference.yaml `known_techs` (canonical: [variants]) into the code
    default dict-of-lists, extend-only (dedup, preserves order)."""
    out = {k: list(v) for k, v in defaults.items()}
    raw = _load_raw().get("known_techs")
    if isinstance(raw, dict):
        for tech, variants in raw.items():
            if not isinstance(variants, (list, set, tuple)):
                continue
            cur = out.setdefault(str(tech), [])
            for v in variants:
                if v not in cur:
                    cur.append(v)
    return out


def extend_set(key: str, defaults):
    """Extend-only merge of reference.yaml[key] (a list) into a default set.
    Returns the SAME container type as `defaults` (set or frozenset)."""
    merged = set(defaults)
    raw = _load_raw().get(key)
    if isinstance(raw, (list, set, tuple)):
        merged.update(str(x) for x in raw)
    return frozenset(merged) if isinstance(defaults, frozenset) else merged


def override_dict(key: str, defaults: dict) -> dict:
    """reference.yaml[key] entries OVERRIDE / add to the defaults dict."""
    out = dict(defaults)
    raw = _load_raw().get(key)
    if isinstance(raw, dict):
        out.update(raw)
    return out
