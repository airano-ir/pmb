"""Tests for the Read-Guard decision logic (redundant-re-read blocking)."""
from __future__ import annotations

from pmb.hooks.read_guard import deny_payload, evaluate


def test_first_read_allowed():
    led: dict = {}
    r = evaluate(led, "s", "a.py", "sha1", 40)
    assert r["decision"] == "allow"


def test_unchanged_reread_denied():
    led: dict = {}
    evaluate(led, "s", "a.py", "sha1", 40)          # first read
    r = evaluate(led, "s", "a.py", "sha1", 40)      # same content, soon after
    assert r["decision"] == "deny"
    assert "a.py" in r["reason"]


def test_changed_file_allowed():
    led: dict = {}
    evaluate(led, "s", "a.py", "sha1", 40)
    r = evaluate(led, "s", "a.py", "sha2", 40)      # content changed
    assert r["decision"] == "allow"


def test_outside_recency_window_allowed():
    led: dict = {}
    evaluate(led, "s", "a.py", "sha1", 1)           # seq 1
    evaluate(led, "s", "b.py", "x", 1)              # seq 2
    r = evaluate(led, "s", "a.py", "sha1", 1)       # seq 3: 3-1=2 > window 1
    assert r["decision"] == "allow"


def test_sessions_are_isolated():
    led: dict = {}
    evaluate(led, "s1", "a.py", "sha1", 40)
    r = evaluate(led, "s2", "a.py", "sha1", 40)     # different session, first time
    assert r["decision"] == "allow"


def test_no_sha_never_denies():
    led: dict = {}
    evaluate(led, "s", "a.py", None, 40)
    r = evaluate(led, "s", "a.py", None, 40)
    assert r["decision"] == "allow"


def test_deny_payload_shape():
    p = deny_payload("nope")["hookSpecificOutput"]
    assert p["hookEventName"] == "PreToolUse"
    assert p["permissionDecision"] == "deny"
    assert p["permissionDecisionReason"] == "nope"
