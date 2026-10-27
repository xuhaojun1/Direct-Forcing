"""Streaming file checksums shared by model assets and training references."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
