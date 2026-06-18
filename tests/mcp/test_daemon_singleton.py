"""The daemon is a per-PMB_HOME singleton via an atomic OS lock - NOT a
check-then-spawn race (which let two concurrent starts both pass
find_live_daemon() and produce duplicate daemons on 8765/8766).

Two claims for the same home -> exactly one wins; releasing the holder frees the
slot, so a crashed daemon never wedges it.
"""
from __future__ import annotations


def test_singleton_lock_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("PMB_HOME", str(tmp_path / "home"))
    from pmb.mcp.daemon import _acquire_singleton_lock

    fh1, st1, home1 = _acquire_singleton_lock()
    assert st1 == "acquired" and fh1 is not None
    assert str(tmp_path) in home1

    # A second claim while the first is held must NOT also acquire.
    fh2, st2, _ = _acquire_singleton_lock()
    assert st2 == "held"
    assert fh2 is None

    # Releasing the holder frees the slot (crash-safe semantics).
    fh1.close()
    fh3, st3, _ = _acquire_singleton_lock()
    assert st3 == "acquired"
    fh3.close()
