"""Tests for secret redaction."""
from __future__ import annotations

from pmb.security.redact import redact, redact_metadata


def test_openai_key_redacted():
    text = "use key sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789 to call api"
    out, stats = redact(text)
    assert "sk-proj-AbCdEf" not in out
    assert "[REDACTED:openai-key]" in out
    assert stats.total == 1


def test_anthropic_key_redacted():
    text = "ANTHROPIC_API_KEY=sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789xyz"
    out, stats = redact(text)
    assert "sk-ant-api03-AbCdEf" not in out
    assert stats.total >= 1


def test_aws_access_key_redacted():
    out, stats = redact("AKIAIOSFODNN7EXAMPLE found in config")
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "[REDACTED:aws-access-key]" in out


def test_github_token_redacted():
    text = "token: ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789"
    out, stats = redact(text)
    assert "ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ" not in out
    assert stats.total == 1


def test_jwt_redacted():
    text = "Auth: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTYifQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    out, stats = redact(text)
    assert "eyJhbGciOiJIUzI1NiJ9" not in out


def test_bearer_token_redacted():
    out, stats = redact("Authorization: Bearer abcdef1234567890ghijklmnop")
    assert "abcdef1234567890" not in out


def test_env_style_password_redacted():
    text = "DB_PASSWORD=hunter2supersecret\nDATABASE_URL=postgres://user:pass@host/db"
    out, stats = redact(text)
    assert "hunter2supersecret" not in out
    assert "user:pass@host" not in out or "[REDACTED" in out


def test_pem_private_key_redacted():
    text = (
        "Here is the key:\n"
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEAwJK0...veryLongBase64Body...AQAB\n"
        "moreLines==\n"
        "-----END RSA PRIVATE KEY-----\n"
        "after"
    )
    out, stats = redact(text)
    assert "MIIEowIBAAKCAQEAwJK0" not in out
    assert "[REDACTED:pem-private-key]" in out
    assert "after" in out


def test_clean_text_passes_through():
    text = "Postgres 17 on port 5432, docker-compose"
    out, stats = redact(text)
    assert out == text
    assert stats.total == 0


def test_redact_metadata_strings():
    md = {
        "query": "what is sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789 for?",
        "branch": "main",
        "count": 42,
    }
    out, stats = redact_metadata(md)
    assert "sk-proj-AbCdEf" not in out["query"]
    assert out["branch"] == "main"
    assert out["count"] == 42
    assert stats.total >= 1


def test_redact_metadata_list_values():
    md = {"files": ["src/db.py", "AKIAIOSFODNN7EXAMPLE.txt"]}
    out, stats = redact_metadata(md)
    assert out["files"][0] == "src/db.py"
    assert "AKIAIOSFODNN7EXAMPLE" not in out["files"][1]


def test_empty_input():
    out, stats = redact("")
    assert out == ""
    assert stats.total == 0
    out2, _ = redact_metadata(None)
    assert out2 == {}
