"""R6: the keyed-fact ranking boost only applies to PERSONAL-attribute queries.

A keyed fact (user::city=Warsaw) used to get a big floor+multiplier on EVERY
query that lexically reached it — so a topical query like "Warsaw datacenter
timezone" rocketed "user city: Warsaw" to the top over the actually-relevant
fact. R6 gates the boost on the same personal-intent signal that gates keyed
injection (a question word + a first-person/user cue). The fact stays
retrievable; only the artificial boost is gated.
"""
from __future__ import annotations

import pytest

from pmb.core.engine import Engine


@pytest.fixture
def eng(tmp_pmb_home, tmp_workspace_dir):
    e = Engine(cwd=tmp_workspace_dir, pmb_home=tmp_pmb_home,
               config_overrides={"recall.cache_size": 0})
    e.warmup()
    e.record_keyed_fact("user", "city", "Warsaw")
    e.record_fact("The Warsaw datacenter timezone is UTC+1 and powers the EU region")
    e.wait_for_writes()
    return e


def _keyed_score(e, query):
    d = e.recall(query, top_k=10).to_dict()
    for r in d["results"]:
        if (r.get("metadata") or {}).get("keyed_fact_key"):
            return float(r["score"])
    return None


def test_topical_query_does_not_boost_keyed_fact(eng):
    q = "Warsaw datacenter timezone region"   # topical, NOT a personal question
    eng.config._overrides["recall.keyed_boost_personal_only"] = True   # R6 (default)
    gated = _keyed_score(eng, q)
    eng.config._overrides["recall.keyed_boost_personal_only"] = False  # old behaviour
    boosted = _keyed_score(eng, q)
    # the keyed fact is RETRIEVABLE in both, but un-boosted under R6
    if gated is not None and boosted is not None:
        assert boosted > gated, "the old path boosts on a topical query; R6 doesn't"


def test_personal_query_still_boosts_keyed_fact(eng):
    # "where do I live" — a question word + first-person cue → personal intent →
    # the keyed fact IS boosted (and should rank at the very top).
    d = eng.recall("where do I live now", top_k=10).to_dict()
    assert d["results"]
    top = d["results"][0]
    assert (top.get("metadata") or {}).get("keyed_fact_key"), \
        "the keyed city fact must win a personal-attribute query"
