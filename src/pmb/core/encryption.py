"""
Encrypted workspace bundles - export/import a whole PMB workspace as a single
authenticated-encrypted file.

Why
---
A workspace is plaintext SQLite + LanceDB on disk. That's fine on your own
laptop, but the moment you want to BACK IT UP or SYNC IT (Dropbox, a USB
stick, a *public* git repo), the privacy model breaks: anyone with the file
can read every fact you ever stored.

This module makes the workspace portable AND private:

    pmb workspace export memory.enc      # → one encrypted file
    pmb workspace import memory.enc work # → restored into workspace 'work'

The combo with git sync is the real unlock: you can `pmb workspace export`
into a repo and push it to a PUBLIC GitHub repo with zero exposure - the
remote only ever sees ciphertext.

Crypto
------
  • Key derivation: scrypt(passphrase, random 16-byte salt, N=2**15) → 32B,
    or a raw 32-byte `--key-file`.
  • Cipher: Fernet (AES-128-CBC + HMAC-SHA256, authenticated). Tampering is
    detected on decrypt (InvalidToken), not silently accepted.
  • Bundle = magic header + version + mode + salt + Fernet token over a
    gzip-tar of the workspace files.

Dependency
----------
Requires the `cryptography` package (optional extra: `pip install pmb-ai[crypto]`).
Import errors surface a clear install hint rather than a stack trace.
"""

from __future__ import annotations

import io
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path

_MAGIC = b"PMBENC"        # 6 bytes
_VERSION = 1              # 1 byte
_MODE_PASSPHRASE = 1      # scrypt KDF, salt present
_MODE_KEYFILE = 2        # raw 32-byte key, salt zeroed
_SALT_LEN = 16
_SCRYPT_N = 2 ** 15
_SCRYPT_R = 8
_SCRYPT_P = 1

# Files/dirs that constitute a portable workspace (mirror git_sync).
_BUNDLE_MEMBERS = [
    "events.sqlite", "meta.yaml", "bm25_index.pkl",
    "vocab_bridges.json", "vectors.lance",
]


class EncryptionUnavailable(RuntimeError):
    pass


def _require_crypto():
    try:
        import base64  # noqa: F401

        from cryptography.fernet import Fernet  # noqa: F401
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise EncryptionUnavailable(
            "Workspace encryption needs the `cryptography` package. "
            "Install it with: pip install 'pmb-ai[crypto]'  (or: pip install cryptography)"
        ) from e


def _derive_key_passphrase(passphrase: str, salt: bytes) -> bytes:
    import base64

    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    kdf = Scrypt(salt=salt, length=32, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    raw = kdf.derive(passphrase.encode("utf-8"))
    return base64.urlsafe_b64encode(raw)


def _key_from_keyfile(key_bytes: bytes) -> bytes:
    import base64
    if len(key_bytes) < 32:
        raise ValueError("key file must contain at least 32 bytes of key material")
    return base64.urlsafe_b64encode(key_bytes[:32])


@dataclass
class CryptoResult:
    ok: bool
    action: str
    detail: str = ""
    extra: dict | None = None

    def as_dict(self) -> dict:
        d = {"ok": self.ok, "action": self.action, "detail": self.detail}
        if self.extra:
            d.update(self.extra)
        return d


def _tar_workspace(storage_dir: Path) -> bytes:
    """Gzip-tar the portable members of a workspace into memory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for member in _BUNDLE_MEMBERS:
            p = storage_dir / member
            if p.exists():
                tar.add(p, arcname=member)
    return buf.getvalue()


def _untar_workspace(data: bytes, dest_dir: Path) -> list[str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar.getmembers():
            # path-traversal guard: members must stay within dest
            target = (dest_dir / m.name).resolve()
            if not str(target).startswith(str(dest_dir.resolve())):
                raise ValueError(f"unsafe path in bundle: {m.name!r}")
            names.append(m.name)
        # Python 3.12+ supports the 'data' filter (rejects unsafe members);
        # fall back gracefully on older runtimes.
        try:
            tar.extractall(dest_dir, filter="data")
        except TypeError:
            tar.extractall(dest_dir)  # noqa: S202 - path-traversal guarded above
    return names


def export_workspace(
    storage_dir: Path,
    out_path: Path,
    *,
    passphrase: str | None = None,
    key_file: Path | None = None,
) -> CryptoResult:
    """Encrypt a workspace into a single bundle file."""
    _require_crypto()
    from cryptography.fernet import Fernet

    if not passphrase and not key_file:
        return CryptoResult(False, "export", "provide either a passphrase or --key-file")

    payload = _tar_workspace(Path(storage_dir))
    if not payload:
        return CryptoResult(False, "export", "workspace has no data to export")

    if key_file:
        mode = _MODE_KEYFILE
        salt = b"\x00" * _SALT_LEN
        key = _key_from_keyfile(Path(key_file).read_bytes())
    else:
        mode = _MODE_PASSPHRASE
        salt = os.urandom(_SALT_LEN)
        key = _derive_key_passphrase(passphrase, salt)

    token = Fernet(key).encrypt(payload)
    header = _MAGIC + bytes([_VERSION, mode]) + salt
    Path(out_path).write_bytes(header + token)
    return CryptoResult(
        True, "export",
        f"encrypted {len(payload):,} bytes → {out_path}",
        {"bundle_bytes": Path(out_path).stat().st_size,
         "mode": "keyfile" if key_file else "passphrase"},
    )


def import_workspace(
    bundle_path: Path,
    dest_dir: Path,
    *,
    passphrase: str | None = None,
    key_file: Path | None = None,
) -> CryptoResult:
    """Decrypt a bundle into a workspace storage directory."""
    _require_crypto()
    from cryptography.fernet import Fernet, InvalidToken

    raw = Path(bundle_path).read_bytes()
    if len(raw) < len(_MAGIC) + 2 + _SALT_LEN or raw[:len(_MAGIC)] != _MAGIC:
        return CryptoResult(False, "import", "not a PMB encrypted bundle (bad magic)")
    version = raw[len(_MAGIC)]
    mode = raw[len(_MAGIC) + 1]
    if version != _VERSION:
        return CryptoResult(False, "import", f"unsupported bundle version {version}")
    salt = raw[len(_MAGIC) + 2: len(_MAGIC) + 2 + _SALT_LEN]
    token = raw[len(_MAGIC) + 2 + _SALT_LEN:]

    if mode == _MODE_KEYFILE:
        if not key_file:
            return CryptoResult(False, "import", "bundle needs a --key-file to decrypt")
        key = _key_from_keyfile(Path(key_file).read_bytes())
    elif mode == _MODE_PASSPHRASE:
        if not passphrase:
            return CryptoResult(False, "import", "bundle needs a passphrase to decrypt")
        key = _derive_key_passphrase(passphrase, salt)
    else:
        return CryptoResult(False, "import", f"unknown bundle mode {mode}")

    try:
        payload = Fernet(key).decrypt(token)
    except InvalidToken:
        return CryptoResult(False, "import",
                            "decryption failed - wrong passphrase/key or tampered bundle")

    dest = Path(dest_dir)
    if dest.exists() and any(dest.iterdir()):
        return CryptoResult(False, "import",
                            f"destination already exists and is not empty: {dest}")
    names = _untar_workspace(payload, dest)
    return CryptoResult(True, "import",
                        f"restored {len(names)} member(s) → {dest}",
                        {"members": names})
