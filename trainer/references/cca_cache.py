"""Exact lazy disk cache of frozen CCA target tokens, separate from MMD.

Content keys include actual latents and conditioning. Namespace contracts
identify weights, layer, preprocessing and decoder. Atomic writes allow reuse
across ranks/runs. Preserve dtype and all tokens; no quantization or pooling.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import torch


class CCATargetTokenCache:
    def __init__(self, root, contract):
        manifest = json.dumps({'version': 1, **contract}, sort_keys=True, indent=2)
        self.namespace = hashlib.sha256(manifest.encode()).hexdigest()
        self.path = Path(root) / self.namespace
        self.path.mkdir(parents=True, exist_ok=True)
        self._atomic_write(self.path / 'contract.json', lambda f: f.write(manifest.encode()))
        self.reset_stats()

    @staticmethod
    def _atomic_write(path, write):
        fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as handle:
                write(handle)
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def reset_stats(self):
        self.stats = dict(hits=0, misses=0, key_seconds=0., read_seconds=0.,
                          compute_seconds=0., write_seconds=0., bytes_written=0)

    @staticmethod
    def _key(tensors):
        digest = hashlib.sha256()
        for tensor in tensors:
            if tensor is None:
                digest.update(b'none')
                continue
            value = tensor.detach().contiguous().cpu()
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def get(self, target, embedding, compute):
        start = time.perf_counter()
        key = self._key((target, embedding))
        self.stats['key_seconds'] += time.perf_counter() - start
        path = self.path / (key + '.pt')
        start = time.perf_counter()
        if path.is_file():
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved['namespace'] != self.namespace or saved['key'] != key:
                raise ValueError(f'CCA target cache identity mismatch: {path}')
            tokens = saved['tokens'].to(target.device)
            self.stats['hits'] += 1
            self.stats['read_seconds'] += time.perf_counter() - start
            return tokens
        tokens = compute().detach()
        if tokens.ndim != 3 or not torch.isfinite(tokens).all():
            raise ValueError('CCA cache requires finite full [B,N,C] target tokens')
        cpu = tokens.cpu().contiguous()
        self.stats['compute_seconds'] += time.perf_counter() - start
        start = time.perf_counter()
        self._atomic_write(path, lambda f: torch.save(
            dict(namespace=self.namespace, key=key, tokens=cpu), f))
        self.stats['write_seconds'] += time.perf_counter() - start
        self.stats['bytes_written'] += cpu.numel() * cpu.element_size()
        self.stats['misses'] += 1
        return tokens
