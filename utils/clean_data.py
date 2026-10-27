"""Identity and access for the canonical causal-forcing clean-latent LMDB."""
from pathlib import Path
import json


def clean_data_identity(directory):
    identity = Path(directory) / 'identity.json'
    if identity.is_file():
        # Content identity survives copying between machines and filesystems.
        return dict(gt_latents_sha256=json.loads(identity.read_text())['content_sha256'])
    # Compatibility for pre-existing, locally validated research references.
    stat = (Path(directory)/'data.mdb').stat()
    return dict(gt_lmdb_bytes=stat.st_size, gt_lmdb_mtime_ns=stat.st_mtime_ns)


def open_clean_latents(directory):
    from utils.clean_lmdb import WanCleanLatentDataset
    return WanCleanLatentDataset(directory)
