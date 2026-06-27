"""Project-scoped recall must find decisions that mention the project in their
content even when they carry no project metadata tag.

Real-scenario bug: an agent records a decision via record_batch as
type='activity'; record_batch drops custom metadata on activity items, so the
decision has no project tag. A later recall(project=X) used to filter it out -
breaking cross-session memory - even though the decision text names the project.
"""
from __future__ import annotations

from pmb.core.engine.recall import _result_in_project


class _R:
    def __init__(self, content, metadata):
        self.content = content
        self.metadata = metadata


def test_matches_via_metadata_tag():
    assert _result_in_project(_R("anything", {"project": "PMB"}), "pmb")
    assert _result_in_project(_R("x", {"project_name": "PMB"}), "pmb")
    assert _result_in_project(_R("x", {"project_path": "C:/code/pmb"}), "pmb")


def test_matches_via_content_when_metadata_dropped():
    # the agent-recorded-decision case: no project tag, but content names it
    r = _R(
        "PMB demo decision: structured JSON logging at INFO level",
        {"actor": "agent", "activity_kind": "decision"},
    )
    assert _result_in_project(r, "pmb")


def test_unrelated_event_not_in_project():
    assert not _result_in_project(_R("a note about cooking pasta", {}), "pmb")
