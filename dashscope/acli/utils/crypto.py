# -*- coding: utf-8 -*-
"""Lightweight local-only value obfuscation helpers.

These are NOT encryption for security-critical storage: they just keep
plain secrets from being immediately visible in local config files.
"""

from __future__ import annotations

import base64
import hashlib
import platform
import uuid

_ENC_PREFIX = "ENC:"


def _machine_key() -> bytes:
    node = uuid.getnode()
    mac = ":".join(f"{(node >> i) & 0xff:02x}" for i in range(0, 48, 8))
    seed = f"{platform.node()}:{mac}:acli-secret-salt"
    return hashlib.sha256(seed.encode()).digest()


def _xor_cipher(data: bytes, key: bytes) -> bytes:
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


def encrypt_value(plaintext: str) -> str:
    """Obfuscate a plaintext value with a machine-local XOR cipher."""
    if not plaintext:
        return ""
    key = _machine_key()
    encrypted = _xor_cipher(plaintext.encode(), key)
    return _ENC_PREFIX + base64.b64encode(encrypted).decode()


def decrypt_value(stored: str) -> str:
    """Reverse :func:`encrypt_value`."""
    if not stored:
        return ""
    if not stored.startswith(_ENC_PREFIX):
        return stored
    key = _machine_key()
    encrypted = base64.b64decode(stored[len(_ENC_PREFIX) :])
    return _xor_cipher(encrypted, key).decode()
