"""X4 — every declared SLO must have a real enforcing test (no dangling gates),
and the gate must actually be wired into the suite."""
from __future__ import annotations

from pathlib import Path

from pmb.health.slo import SLOS

_ROOT = Path(__file__).resolve().parent.parent


def test_every_slo_points_at_a_present_test():
    assert SLOS, "the SLO registry must not be empty"
    for slo in SLOS:
        assert "::" in slo.enforced_by, f"{slo.name}: malformed enforced_by"
        rel, test_name = slo.enforced_by.split("::", 1)
        path = _ROOT / rel
        assert path.exists(), f"{slo.name}: enforcing file missing: {rel}"
        src = path.read_text(encoding="utf-8")
        assert f"def {test_name}" in src, (
            f"{slo.name}: enforcing test {test_name} not found in {rel}")


def test_slos_are_uniquely_named():
    names = [s.name for s in SLOS]
    assert len(names) == len(set(names)), "duplicate SLO names"
