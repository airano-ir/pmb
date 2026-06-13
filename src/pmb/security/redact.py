"""
Secret redaction before write.

Goal: stop obvious credentials from landing in events.sqlite verbatim.
Not a substitute for keeping secrets out of agent transcripts entirely -
just a final guard at the persistence boundary.

Strategy:
- Match known-shape tokens (API keys, JWTs, PEM blocks) with conservative regex.
- For .env-style `KEY=value` lines, only redact when KEY name looks sensitive.
- Replace with `[REDACTED:<kind>]` so context is preserved for the human reader.
- Return stats so callers can surface "n secrets redacted" warnings.

Limitations (acknowledged, not patched over):
- Pure high-entropy strings without a known prefix are NOT touched. Catching
  those reliably needs entropy analysis and produces false positives.
- Multi-line PEM blocks redact the body but keep the header so the kind is
  visible.
- Metadata values that aren't strings are left as-is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Order matters: more specific patterns first so they consume before the
# generic Bearer / KEY=value patterns catch them.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Anthropic explicit (more specific - must run before generic sk-)
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    # OpenAI style API keys (excludes sk-ant- which is Anthropic)
    ("openai-key", re.compile(r"\bsk-(?!ant-)(?:proj-)?[A-Za-z0-9_\-]{20,}\b")),
    # AWS Access Key ID
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    # AWS Secret Access Key (heuristic - 40 char base64 after explicit context only)
    ("aws-secret", re.compile(
        r"(?i)\baws_secret_access_key\s*[=:]\s*[\"']?([A-Za-z0-9/+=]{40})[\"']?"
    )),
    # GitHub tokens
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    # Slack
    ("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    # Google API key
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    # Stripe
    ("stripe-key", re.compile(r"\b(?:sk|pk|rk)_(?:test|live)_[A-Za-z0-9]{20,}\b")),
    # JWT (header.payload.signature, base64url)
    ("jwt", re.compile(
        r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"
    )),
    # Bearer / Authorization header (case-insensitive)
    ("bearer-token", re.compile(r"(?i)\b(?:bearer|authorization:\s*bearer)\s+[A-Za-z0-9._\-]{16,}")),
    # Private key blocks (PEM) - redact the body, keep the header line
    ("pem-private-key", re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"
        r"[\s\S]+?"
        r"-----END (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----",
    )),
]


# Sensitive KEY names in KEY=value / KEY: value lines
_SENSITIVE_KEYS = re.compile(
    r"(?im)^\s*("
    r"(?:[A-Z][A-Z0-9_]*_)?"
    r"(?:API[_-]?KEY|SECRET|PASSWORD|PASSWD|PWD|TOKEN|ACCESS[_-]?KEY|PRIVATE[_-]?KEY|"
    r"CLIENT[_-]?SECRET|AUTH[_-]?TOKEN|REFRESH[_-]?TOKEN|SESSION[_-]?KEY|"
    r"DATABASE[_-]?URL|DB[_-]?PASSWORD|CONN(?:ECTION)?[_-]?STRING)"
    r")\s*[:=]\s*[\"']?([^\s\"'\r\n]{4,})[\"']?"
)


@dataclass
class RedactionStats:
    """How many of each kind we redacted in a single redact() call."""
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, kind: str, n: int = 1) -> None:
        if n <= 0:
            return
        self.counts[kind] = self.counts.get(kind, 0) + n

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def merge(self, other: RedactionStats) -> None:
        for k, v in other.counts.items():
            self.add(k, v)

    def to_dict(self) -> dict:
        return {"total": self.total, "by_kind": dict(self.counts)}


def redact(text: str, stats: RedactionStats | None = None) -> tuple[str, RedactionStats]:
    """
    Redact obvious secrets from `text`. Returns (redacted_text, stats).

    Always returns a fresh stats object unless one is passed in (and mutated).
    """
    if stats is None:
        stats = RedactionStats()
    if not text or not isinstance(text, str):
        return text, stats

    result = text

    # Specific shape-matched secrets
    for kind, pattern in _PATTERNS:
        def _sub(m: re.Match, _kind: str = kind) -> str:
            stats.add(_kind)
            if _kind == "pem-private-key":
                # Preserve outer markers so reader can see what kind of key it was
                return "-----BEGIN PRIVATE KEY-----\n[REDACTED:pem-private-key]\n-----END PRIVATE KEY-----"
            if _kind == "aws-secret":
                # We only matched the value group; keep key= prefix
                return m.group(0).replace(m.group(1), "[REDACTED:aws-secret]")
            return f"[REDACTED:{_kind}]"
        result = pattern.sub(_sub, result)

    # KEY=value style sensitive lines
    def _sub_kv(m: re.Match) -> str:
        key = m.group(1)
        value = m.group(2)
        # Don't re-redact already-redacted values (avoids double-counting)
        if value.startswith("[REDACTED"):
            return m.group(0)
        stats.add(f"env:{key.lower()}")
        return f"{key}=[REDACTED:env-secret]"
    result = _SENSITIVE_KEYS.sub(_sub_kv, result)

    return result, stats


def redact_metadata(metadata: dict | None, stats: RedactionStats | None = None) -> tuple[dict, RedactionStats]:
    """
    Redact string values inside a metadata dict (shallow).

    Non-string values are passed through unchanged. Keys are not modified.
    """
    if stats is None:
        stats = RedactionStats()
    if not metadata:
        return metadata or {}, stats

    out: dict[str, Any] = {}
    for k, v in metadata.items():
        if isinstance(v, str):
            new_v, _ = redact(v, stats)
            out[k] = new_v
        elif isinstance(v, list):
            out[k] = [
                redact(item, stats)[0] if isinstance(item, str) else item
                for item in v
            ]
        else:
            out[k] = v
    return out, stats
