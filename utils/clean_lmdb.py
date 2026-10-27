"""Read Causal-Forcing's normalized Wan clean latents without decoding videos."""

import os
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class WanCleanLatentDataset(Dataset):
    """Official rows are FP16 [steps,T,C,H,W]; the last step is clean.

    These are already normalized Wan latents. Only transpose T/C for PFM;
    applying VAE mean/std a second time would change the training target.
    The source is opened read-only without touching its lock.mdb.
    """

    def __init__(self, directory):
        self.directory = str(Path(directory).expanduser().resolve())
        if not (Path(self.directory) / "data.mdb").is_file():
            raise FileNotFoundError(f"Wan GT needs a clean LMDB: {self.directory}/data.mdb")
        self._env = None
        self._pid = None
        env = self._open()
        try:
            with env.begin() as txn:
                raw = txn.get(b"latents_shape")
                if raw is None:
                    raise ValueError("LMDB has no latents_shape; a prompt-only RDM bundle is insufficient")
                shape = tuple(map(int, raw.decode().split()))
                if len(shape) not in (5, 6) or any(size <= 0 for size in shape):
                    raise ValueError(f"Expected [N,(steps),T,C,H,W], got {shape}")
                if shape[-4:] != (21, 16, 60, 104):
                    raise ValueError(f"Wan GT expects 21-frame 480x832 latents, got {shape}")
                prompt_shape = txn.get(b"prompts_shape")
                if prompt_shape is None or prompt_shape.decode().split() != [str(shape[0])]:
                    raise ValueError("Prompt and latent row counts must match")
                self.count, self.row_shape = shape[0], shape[1:]
                self.row_bytes = int(np.prod(self.row_shape)) * np.dtype(np.float16).itemsize
        finally:
            env.close()

    def _open(self):
        return lmdb.open(self.directory, readonly=True, lock=False,
                         readahead=False, meminit=False)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        if self._env is None or self._pid != os.getpid():
            self.close()
            self._env, self._pid = self._open(), os.getpid()
        with self._env.begin() as txn:
            raw = txn.get(f"latents_{index}_data".encode())
            prompt = txn.get(f"prompts_{index}_data".encode())
        if raw is None or len(raw) != self.row_bytes or prompt is None:
            raise ValueError(f"Missing or malformed Wan GT row {index} in {self.directory}")
        array = np.frombuffer(raw, dtype=np.float16).reshape(self.row_shape)
        clean = array[-1] if array.ndim == 5 else array
        if not np.isfinite(clean).all():
            raise ValueError(f"Non-finite Wan GT latent at row {index}")
        latent = torch.from_numpy(clean.transpose(1, 0, 2, 3).copy())
        return {"wan_gt_latents": latent, "caption": prompt.decode("utf-8"),
                "key": str(index), "url": self.directory, "bucket_id": 0}

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
        self._pid = None

    def __getstate__(self):
        # Spawn workers must open their own LMDB environment.
        return {**self.__dict__, "_env": None, "_pid": None}


class RepeatingDistributedSampler(DistributedSampler):
    """Shuffle and shard each epoch, then continue past the first 6,505 rows."""

    def set_start_offset(self, samples_per_rank):
        """Resume the shuffle stream without reading skipped latent tensors."""
        if not isinstance(samples_per_rank, int) or samples_per_rank < 0:
            raise ValueError("sampler start offset must be a nonnegative integer")
        self.start_offset = samples_per_rank

    def __iter__(self):
        from itertools import islice
        epoch_offset, skip = divmod(getattr(self, "start_offset", 0), self.num_samples)
        epoch = self.epoch + epoch_offset
        while True:
            self.set_epoch(epoch)
            yield from islice(super().__iter__(), skip, None)
            skip = 0
            epoch += 1


def collate_wan_clean_latents(rows):
    # Keep log metadata as Python lists, matching the video dataloaders.
    # Default collation would turn bucket_id into a non-JSON-serializable tensor.
    return {"wan_gt_latents": torch.stack([row["wan_gt_latents"] for row in rows]),
            **{key: [row[key] for row in rows] for key in ("caption", "key", "url", "bucket_id")}}


def build_wan_clean_lmdb_dataloader(directory, *, rank, world_size, seed=42, num_workers=4):
    dataset = WanCleanLatentDataset(directory)
    if len(dataset) < world_size:
        raise ValueError("Wan GT needs at least one sample per distributed rank")
    # Drop only the epoch remainder so a batch contains no duplicated rows.
    # Shuffling each epoch rotates which rows are omitted (2 for 7 GPUs, 1 for 8).
    sampler = RepeatingDistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                         seed=seed, shuffle=True, drop_last=True)
    return DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=num_workers,
                      collate_fn=collate_wan_clean_latents,
                      pin_memory=True, persistent_workers=num_workers > 0)
