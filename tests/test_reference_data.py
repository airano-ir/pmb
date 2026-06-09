"""T10: reference.yaml overlay loader — extend-only merge with code defaults.

No file → identical behaviour. File present → extends (alias_groups,
known_techs, stopwords, not_proper) or overrides (kind_priority). Malformed →
warning + defaults, never a crash.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pmb.reference_data as rd


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("PMB_HOME", str(tmp_path))
    rd.clear_cache()
    yield
    rd.clear_cache()


def test_no_file_is_defaults_only(tmp_path):
    assert rd.extend_set("stopwords", {"a", "b"}) == {"a", "b"}
    assert rd.extend_alias_groups({"city": {"town"}}) == {"city": {"town"}}
    assert rd.extend_tech_map({"pg": ["postgres"]}) == {"pg": ["postgres"]}
    assert rd.override_dict("kind_priority", {"tech": 0}) == {"tech": 0}


def test_file_extends_and_overrides(tmp_path):
    (tmp_path / "reference.yaml").write_text(
        "alias_groups:\n"
        "  city: [hometown]\n"
        "known_techs:\n"
        "  kubernetes: [kube]\n"
        "stopwords: [um, uh]\n"
        "not_proper: [bonjour]\n"
        "kind_priority:\n"
        "  decision: 95\n",
        encoding="utf-8",
    )
    rd.clear_cache()

    ag = rd.extend_alias_groups({"city": {"town"}})
    assert ag["city"] == {"town", "hometown"}  # extended, default kept

    tm = rd.extend_tech_map({"kubernetes": ["kubernetes", "k8s"]})
    assert tm["kubernetes"] == ["kubernetes", "k8s", "kube"]  # appended, deduped

    sw = rd.extend_set("stopwords", frozenset({"the"}))
    assert {"the", "um", "uh"} <= set(sw)
    assert isinstance(sw, frozenset)  # container type preserved

    np = rd.extend_set("not_proper", {"когда"})
    assert "bonjour" in np and "когда" in np

    kp = rd.override_dict("kind_priority", {"tech": 0})
    assert kp["tech"] == 0 and kp["decision"] == 95  # default kept, new added


def test_malformed_file_falls_back_to_defaults(tmp_path):
    (tmp_path / "reference.yaml").write_text(
        "alias_groups: [unclosed, list\n::::not valid yaml", encoding="utf-8")
    rd.clear_cache()
    # must NOT raise; returns defaults
    assert rd.extend_set("stopwords", {"a"}) == {"a"}
    assert rd.extend_alias_groups({"city": {"town"}}) == {"city": {"town"}}


def test_alias_overlay_visible_through_attributes_loader(tmp_path):
    """The attributes module exposes the merge via extend_alias_groups; verify
    a yaml alias canonicalizes correctly through the public helper."""
    (tmp_path / "reference.yaml").write_text(
        "alias_groups:\n  city: [where_i_crash]\n", encoding="utf-8")
    rd.clear_cache()
    merged = rd.extend_alias_groups({"city": {"town"}})
    assert "where_i_crash" in merged["city"]
