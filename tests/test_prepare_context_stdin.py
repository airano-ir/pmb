"""Regression: the hook reads stdin as UTF-8, not the platform locale.

Claude Code pipes the user message as UTF-8. On Windows, sys.stdin.read()
defaults to cp1251 and mangles Cyrillic into mojibake, which makes the
intent regexes miss — auto-recall silently produces nothing for any
non-ASCII message. _read_stdin_utf8 must decode UTF-8 explicitly.
"""

from __future__ import annotations

import io
import json

from pmb.cli.main import _read_stdin_utf8, _extract_hook_prompt


class _FakeStdin:
    """Mimics sys.stdin with a .buffer that yields raw bytes."""
    def __init__(self, raw: bytes):
        self.buffer = io.BytesIO(raw)

    def read(self):  # text fallback path
        return self.buffer.getvalue().decode("utf-8", errors="replace")


def test_reads_utf8_cyrillic(monkeypatch):
    msg = "почему мы выбрали playwright для e2e"
    monkeypatch.setattr("sys.stdin", _FakeStdin(msg.encode("utf-8")))
    assert _read_stdin_utf8() == msg


def test_reads_utf8_mixed_scripts(monkeypatch):
    msg = "виправ баг у LoadGuard — чому ми обрали osrm?"
    monkeypatch.setattr("sys.stdin", _FakeStdin(msg.encode("utf-8")))
    assert _read_stdin_utf8() == msg


def test_bad_bytes_do_not_crash(monkeypatch):
    # Invalid UTF-8 must not raise — errors='replace'.
    monkeypatch.setattr("sys.stdin", _FakeStdin(b"\xff\xfe bad \x80 bytes"))
    out = _read_stdin_utf8()
    assert isinstance(out, str)
    assert "bad" in out


def test_no_buffer_falls_back_to_read(monkeypatch):
    # pytest-style StringIO stdin (no .buffer) must still work.
    monkeypatch.setattr("sys.stdin", io.StringIO("plain text message"))
    assert _read_stdin_utf8() == "plain text message"


# ─── _extract_hook_prompt: the host pipes JSON, not raw text ──────────────
# Regression for the live-e2e finding: the real UserPromptSubmit hook sends a
# JSON payload, and json.dumps escapes Cyrillic to \uXXXX — so treating the
# blob as the message made intent classification blind to non-ASCII prompts.


def test_extract_prompt_from_json_payload():
    payload = json.dumps({"hook_event_name": "UserPromptSubmit",
                          "prompt": "fix the auth bug", "session_id": "s1"})
    assert _extract_hook_prompt(payload) == "fix the auth bug"


def test_extract_prompt_from_json_with_escaped_cyrillic():
    # json.dumps default ensure_ascii=True → Cyrillic becomes \uXXXX. The
    # extractor must recover the real Unicode text, not the escaped blob.
    msg = "поправь баг в auth — какие там правила?"
    payload = json.dumps({"hook_event_name": "UserPromptSubmit", "prompt": msg})
    assert "\\u04" in payload          # sanity: the wire form IS escaped
    assert _extract_hook_prompt(payload) == msg


def test_extract_prompt_raw_text_passthrough():
    assert _extract_hook_prompt("just a plain message") == "just a plain message"


def test_extract_prompt_json_without_text_field_falls_back():
    # An object with no prompt-ish field → return the raw (don't blank it).
    raw = json.dumps({"hook_event_name": "Stop", "session_id": "s1"})
    assert _extract_hook_prompt(raw) == raw


def test_extract_prompt_alternate_keys():
    assert _extract_hook_prompt(json.dumps({"message": "hi there"})) == "hi there"
    assert _extract_hook_prompt(json.dumps({"user_prompt": "do x"})) == "do x"


def test_extract_prompt_empty():
    assert _extract_hook_prompt("") == ""
