"""C2/C3 + L1: language packs.

EN lexical data stays in code; the RU/UK floor now lives in the built-in
ALWAYS-ACTIVE packs ``pmb/lang/packs/{ru,uk}.yaml`` (L1 — keeps the .py modules
Cyrillic-free while EN+RU+UK matching is byte-identical; the parity is gated by
tests/test_lang_pack_parity.py). Other built-in templates (de, es, fr) stay
OPT-IN: active only when copied into ``$PMB_HOME/lang/``. A pack EXTENDS the
merged sets/groups — never removes — so enabling de adds German on top of the
EN+RU+UK floor.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from pmb import lang


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PMB_HOME", str(tmp_path))
    lang.clear_cache()
    yield tmp_path
    lang.clear_cache()


def _write_pack(home: Path, code: str, body: str) -> None:
    d = home / "lang"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{code}.yaml").write_text(textwrap.dedent(body), encoding="utf-8")
    lang.clear_cache()


# ── built-in floor: ru + uk active by default, extend-only ──────────────────

def test_no_builtin_packs_active_by_default(tmp_home):
    # G3: NO pack is active by default — ru/uk were deleted and RU/UK moved to
    # the warm anchor tier. The loader floor still returns the caller's defaults
    # unchanged, extend-only, with container kind preserved.
    assert lang.active_codes() == []
    base_set = {"the", "a", "is"}
    assert base_set <= lang.merged_set("stopwords", base_set)
    merged = lang.merged_groups("verb_synonyms", {"live": {"live"}})
    assert merged["live"] == {"live"}        # EN floor only, no RU/UK verbs
    fs = frozenset({"x"})
    assert isinstance(lang.merged_set("stopwords", fs), frozenset)


# ── extension: an enabled pack adds, never removes ──────────────────────────

def test_pack_extends_stopwords(tmp_home):
    _write_pack(tmp_home, "de", "stopwords: [der, die, das]")
    out = lang.merged_set("stopwords", {"the"})
    assert out == {"the", "der", "die", "das"}      # floor kept + extended


def test_pack_extends_groups(tmp_home):
    _write_pack(tmp_home, "de", """
        verb_synonyms:
          live: [wohne, wohnt]
          work: [arbeite]
    """)
    out = lang.merged_groups("verb_synonyms", {"live": {"live"}})
    # de extends on top of the EN inline + the built-in ru/uk floor
    assert {"live", "wohne", "wohnt"} <= out["live"]
    assert {"arbeite"} <= out["work"]               # new canonical added


def test_active_codes_lists_enabled(tmp_home):
    _write_pack(tmp_home, "de", "stopwords: [der]")
    _write_pack(tmp_home, "es", "stopwords: [el]")
    # G3: only user-enabled packs are active now (no built-in default packs)
    assert lang.active_codes() == ["de", "es"]


def test_malformed_pack_is_ignored(tmp_home):
    d = tmp_home / "lang"
    d.mkdir(parents=True, exist_ok=True)
    (d / "broken.yaml").write_text("this: [is: not: valid: yaml", encoding="utf-8")
    lang.clear_cache()
    # malformed file is skipped, no crash, no spurious active code
    assert "broken" not in lang.active_codes()


# ── built-in templates ──────────────────────────────────────────────────────

def test_c4_user_cue_is_pack_aware(tmp_home):
    """C4: the offline keyed-suggestion prefilter recognises the user in a
    packed language. German 'ich' passes only once de is enabled; third-party
    'Alice wohnt' is still rejected; EN works with no pack."""
    from pmb.reasoning.attributes import has_user_subject_cue
    assert has_user_subject_cue("I moved to Berlin")          # EN built-in
    assert not has_user_subject_cue("Ich wohne in Berlin")    # de not enabled yet
    _write_pack(tmp_home, "de", "first_person: [ich, mein]")
    assert has_user_subject_cue("Ich wohne in Berlin")        # de enabled → passes
    assert not has_user_subject_cue("Alice wohnt in Berlin")  # third party rejected


def test_builtin_templates_present_and_load():
    bt = lang.builtin_templates()
    assert "de" in bt and "es" in bt and "fr" in bt
    de = lang._load_yaml(bt["de"])
    assert "der" in de["stopwords"]
    assert "wohnt" in de["verb_synonyms"]["live"]
    assert "stadt" in de["attribute_aliases"]["city"]
    fr = lang._load_yaml(bt["fr"])
    assert fr["name"] == "French"
    assert "où" in fr["stopwords"]
    assert "je" in fr["first_person"]
    assert "habite" in fr["verb_synonyms"]["live"]
    assert "ville" in fr["attribute_aliases"]["city"]


# ── integration: a real module picks up an enabled pack (subprocess so the
#    import-time merge runs with the pack already present) ───────────────────

# G3: a user-enabled de pack reaches the modules; the RU floor is gone (RU now
# rides the warm anchor tier), so the probe asserts only the EN inline floor + de.
_PROBE = """
import pmb.reasoning.pamvr as pamvr
from pmb.reasoning.attributes import canonicalize_attribute
assert "the" in pamvr._STOP and "der" in pamvr._STOP, "stopword floor+de"
assert "live" in pamvr.VERB_SYNS["live"] and "wohnt" in pamvr.VERB_SYNS["live"]
assert "warum" in pamvr._NOT_PROPER
assert pamvr._FIRST_PERSON_RE.search("ich wohne in Berlin")
assert canonicalize_attribute("stadt") == "city"
print("INTEGRATION_OK")
"""

_PROBE_FR = """
import pmb.reasoning.pamvr as pamvr
from pmb.reasoning.attributes import canonicalize_attribute, has_user_subject_cue
assert "the" in pamvr._STOP and "où" in pamvr._STOP, "stopword floor+fr"
assert "live" in pamvr.VERB_SYNS["live"] and "habite" in pamvr.VERB_SYNS["live"]
assert "pourquoi" in pamvr._NOT_PROPER
assert pamvr._FIRST_PERSON_RE.search("je travaille à Paris")
assert has_user_subject_cue("Je travaille chez Datadog")
assert has_user_subject_cue("J'ai une maison")
assert has_user_subject_cue("J’ai une maison")
assert not has_user_subject_cue("Alice travaille chez Datadog")
assert canonicalize_attribute("ville") == "city"
assert canonicalize_attribute("société") == "employer"
print("FR_INTEGRATION_OK")
"""


def test_enabled_pack_reaches_modules(tmp_home):
    src = lang.builtin_templates()["de"]
    (tmp_home / "lang").mkdir(parents=True, exist_ok=True)
    (tmp_home / "lang" / "de.yaml").write_text(
        src.read_text(encoding="utf-8"), encoding="utf-8")
    env = dict(os.environ)
    env["PMB_HOME"] = str(tmp_home)
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "src")
    r = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True,
                       text=True, env=env, timeout=120)
    assert "INTEGRATION_OK" in r.stdout, (r.stdout, r.stderr)


def test_enabled_french_template_reaches_modules(tmp_home):
    src = lang.builtin_templates()["fr"]
    (tmp_home / "lang").mkdir(parents=True, exist_ok=True)
    (tmp_home / "lang" / "fr.yaml").write_text(
        src.read_text(encoding="utf-8"), encoding="utf-8")
    env = dict(os.environ)
    env["PMB_HOME"] = str(tmp_home)
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "src")
    r = subprocess.run([sys.executable, "-c", _PROBE_FR], capture_output=True,
                       text=True, env=env, timeout=120)
    assert "FR_INTEGRATION_OK" in r.stdout, (r.stdout, r.stderr)


def test_no_pack_floor_only_in_module(tmp_home):
    """Mirror of the above with NO pack: German must be ABSENT (byte-identical
    floor)."""
    probe = (
        "import pmb.reasoning.pamvr as p\n"
        "assert 'the' in p._STOP and 'der' not in p._STOP\n"
        "assert 'wohnt' not in p.VERB_SYNS['live']\n"
        "print('FLOOR_OK')\n"
    )
    env = dict(os.environ)
    env["PMB_HOME"] = str(tmp_home)  # no lang/ dir
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "src")
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                       text=True, env=env, timeout=120)
    assert "FLOOR_OK" in r.stdout, (r.stdout, r.stderr)
