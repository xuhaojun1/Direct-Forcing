"""Atomic tensor-cache writer shared by offline builders."""
import os
from pathlib import Path
import torch

def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
