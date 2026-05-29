"""Tests for encrypted workspace bundles (export/import).

Skips cleanly if `cryptography` isn't installed.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

cryptography = pytest.importorskip("cryptography")

from pmb.core.encryption import export_workspace, import_workspace


def _seed(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    (d / "events.sqlite").write_bytes(b"SQLite format 3\x00secret-events")
    (d / "meta.yaml").write_text("id: t\nname: Secret WS\n", encoding="utf-8")
    (d / "vocab_bridges.json").write_text('{"a": ["b"]}', encoding="utf-8")
    lance = d / "vectors.lance"
    lance.mkdir(exist_ok=True)
    (lance / "v.bin").write_bytes(b"vecs")


@pytest.fixture
def tree():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        ws = root / "ws"
        _seed(ws)
        yield root, ws


def test_export_import_passphrase_roundtrip(tree):
    root, ws = tree
    bundle = root / "out.enc"
    r = export_workspace(ws, bundle, passphrase="hunter2")
    assert r.ok, r.detail
    assert bundle.exists()

    dest = root / "restored"
    r2 = import_workspace(bundle, dest, passphrase="hunter2")
    assert r2.ok, r2.detail
    assert (dest / "events.sqlite").read_bytes() == b"SQLite format 3\x00secret-events"
    assert "Secret WS" in (dest / "meta.yaml").read_text(encoding="utf-8")
    assert (dest / "vectors.lance" / "v.bin").exists()


def test_wrong_passphrase_fails(tree):
    root, ws = tree
    bundle = root / "out.enc"
    export_workspace(ws, bundle, passphrase="correct")
    r = import_workspace(bundle, root / "restored", passphrase="wrong")
    assert not r.ok
    assert "decryption failed" in r.detail


def test_ciphertext_does_not_leak_plaintext(tree):
    root, ws = tree
    bundle = root / "out.enc"
    export_workspace(ws, bundle, passphrase="pw")
    blob = bundle.read_bytes()
    # the secret strings must NOT appear in the ciphertext
    assert b"secret-events" not in blob
    assert b"Secret WS" not in blob
    assert blob[:6] == b"PMBENC"  # magic header present


def test_keyfile_roundtrip(tree):
    root, ws = tree
    keyf = root / "key.bin"
    keyf.write_bytes(b"0" * 32)
    bundle = root / "out.enc"
    r = export_workspace(ws, bundle, key_file=keyf)
    assert r.ok, r.detail
    r2 = import_workspace(bundle, root / "restored", key_file=keyf)
    assert r2.ok, r2.detail
    assert (root / "restored" / "events.sqlite").exists()


def test_keyfile_wrong_key_fails(tree):
    root, ws = tree
    keyf = root / "key.bin"
    keyf.write_bytes(b"0" * 32)
    bundle = root / "out.enc"
    export_workspace(ws, bundle, key_file=keyf)
    badkey = root / "bad.bin"
    badkey.write_bytes(b"1" * 32)
    r = import_workspace(bundle, root / "restored", key_file=badkey)
    assert not r.ok


def test_import_refuses_nonempty_dest(tree):
    root, ws = tree
    bundle = root / "out.enc"
    export_workspace(ws, bundle, passphrase="pw")
    dest = root / "occupied"
    dest.mkdir()
    (dest / "x").write_text("hi", encoding="utf-8")
    r = import_workspace(bundle, dest, passphrase="pw")
    assert not r.ok
    assert "not empty" in r.detail


def test_export_requires_a_secret(tree):
    root, ws = tree
    r = export_workspace(ws, root / "out.enc")
    assert not r.ok
    assert "passphrase" in r.detail or "key-file" in r.detail


def test_bad_magic_rejected(tree):
    root, ws = tree
    fake = root / "fake.enc"
    fake.write_bytes(b"NOTPMB" + b"\x00" * 50)
    r = import_workspace(fake, root / "restored", passphrase="pw")
    assert not r.ok
    assert "magic" in r.detail
