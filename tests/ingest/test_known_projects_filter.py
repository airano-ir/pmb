"""R5: `_known_projects` must not treat every graph concept as a project.

The hook's project detector built its "known projects" set from
graph_top_entities(kind=None), so junk concept entities ('tests', 'fails',
'cloud') and mis-classified tool names ('person' kind) faked a PROJECT_PREP for
messages like "fix the tests". Now a graph entity must have a project-ish kind,
enough mentions, and a non-stopword name. The workspace name still passes
unconditionally.
"""
from __future__ import annotations

from pmb.hooks.auto_recall import _known_projects


class _Cfg:
    def get(self, key):
        return 3 if key == "auto_recall.project_min_mentions" else None


class _WS:
    name = "MyProj"


class _FakeEngine:
    workspace = _WS()
    config = _Cfg()

    def __init__(self, entities):
        self._entities = entities

    def graph_top_entities(self, kind=None, limit=200):
        return self._entities


def test_filters_junk_concepts_and_persons():
    eng = _FakeEngine([
        {"name": "tests", "kind": "concept", "n_mentions": 9},     # junk concept
        {"name": "fails", "kind": "concept", "n_mentions": 5},     # junk concept
        {"name": "pytest-benchmark", "kind": "person", "n_mentions": 4},  # misclass
        {"name": "the", "kind": "project", "n_mentions": 20},      # stopword name
        {"name": "LoadGuard", "kind": "project", "n_mentions": 7},  # real project
        {"name": "Narch", "kind": "repo", "n_mentions": 3},        # real repo
        {"name": "OneOff", "kind": "project", "n_mentions": 1},    # too few mentions
    ])
    known = _known_projects(eng)
    assert "MyProj" in known            # workspace name always passes
    assert "LoadGuard" in known
    assert "Narch" in known
    assert "tests" not in known         # concept kind → excluded
    assert "fails" not in known
    assert "pytest-benchmark" not in known  # person kind → excluded
    assert "the" not in known           # stopword name → excluded
    assert "OneOff" not in known        # n_mentions 1 < 3 → excluded


def test_unknown_kind_with_enough_mentions_is_kept_lenient():
    # an entity with no kind info but real recurrence is still allowed through
    # (we only EXCLUDE on a known non-project kind, never require one).
    eng = _FakeEngine([{"name": "Mystery", "n_mentions": 8}])
    assert "Mystery" in _known_projects(eng)
