"""Frozen RDM reference loading and Nyström loss assembly."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch
from utils.legacy import canonical_metadata


def file_sha256(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RDMReference:
    Z: torch.Tensor
    Z2: torch.Tensor
    alpha: torch.Tensor
    sigma: float
    k_rr: float
    metadata: Mapping[str, Any]
    sha256: str
    beta: float | None = None

    @property
    def M(self) -> int:
        return int(self.Z.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.Z.shape[1])

    def to(self, device: torch.device | str) -> "RDMReference":
        """A temporary device view; never cache a GPU copy on a CPU reference.

        Autograd retains Z only until the feature-level MMD backward finishes.
        The encoder VJP and generator replay therefore need no GPU reference bank.
        """
        device = torch.device(device)
        if all(t.device == device for t in (self.Z, self.Z2, self.alpha)):
            return self
        return replace(self, Z=self.Z.to(device), Z2=self.Z2.to(device),
                       alpha=self.alpha.to(device))


def load_reference(
    path: str | Path,
    *,
    device: torch.device | str,
    expected_encoder_id: str,
    expected_checkpoint_sha256: str,
    expected_height: int,
    expected_width: int,
    expected_rows: int,
    expected_reference_revision: str,
    expected_encoder_contract: Mapping[str, Any],
    expected_landmarks: int = 4096,
    expected_feature_dim: int = 1024,
    expected_input_mode: str = "native",
    expected_pool: str = "cls",
    expected_rows_per_video: int = 81,
    cpu_offload: bool = False,
) -> RDMReference:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"RDM reference bundle not found: {path}")
    # All ranks can share the read-only file-backed FP32 pages. Do not pin or
    # clone the 20 GiB token matrix separately in each worker's CPU memory.
    bundle = torch.load(path, map_location="cpu", weights_only=False, mmap=cpu_offload)
    metadata = canonical_metadata(dict(bundle.get("metadata", {})))
    expected = {
        "encoder_id": str(expected_encoder_id),
        "encoder_checkpoint_sha256": str(expected_checkpoint_sha256),
        "input_mode": str(expected_input_mode),
        "input_height": int(expected_height),
        "input_width": int(expected_width),
        "pool": str(expected_pool),
        "feature_dim": int(expected_feature_dim),
        "feature_normalize": False,
        "num_rows": int(expected_rows),
        "rows_per_video": int(expected_rows_per_video),
        "rdm_reference_revision": str(expected_reference_revision),
        "nystrom_landmarks": int(expected_landmarks),
    }
    expected.update(dict(expected_encoder_contract))
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"RDM reference metadata {key}={metadata.get(key)!r}, expected {value!r}"
            )
    Z = bundle["Z"].float()
    alpha = bundle["alpha"].float()
    if tuple(Z.shape) != (int(expected_landmarks), int(expected_feature_dim)):
        raise ValueError(f"RDM Z shape {tuple(Z.shape)} is invalid")
    if tuple(alpha.shape) != (int(expected_landmarks),):
        raise ValueError(f"RDM alpha shape {tuple(alpha.shape)} is invalid")
    sigma = float(bundle["sigma"])
    k_rr = float(bundle["k_rr"])
    beta_value = bundle.get("beta")
    beta = float(beta_value) if beta_value is not None else None
    if not sigma > 0 or not torch.isfinite(torch.tensor([sigma, k_rr])).all():
        raise ValueError(f"invalid RDM sigma/k_rr: {sigma}, {k_rr}")
    if beta is not None and (
        not beta > 0 or not bool(torch.isfinite(torch.tensor(beta)))
    ):
        raise ValueError(f"invalid RDM joint beta: {beta}")
    device_Z = Z.to(device)
    # Preserve the original device-side FP32 reduction exactly, including its
    # rounding. Only the storage location changes, not the kernel arithmetic.
    Z2 = (device_Z * device_Z).sum(1)
    if cpu_offload:
        Z2 = Z2.cpu()
        del device_Z
    else:
        Z = device_Z
        alpha = alpha.to(device)
    return RDMReference(
        Z=Z,
        Z2=Z2,
        alpha=alpha,
        sigma=sigma,
        k_rr=k_rr,
        beta=beta,
        metadata=metadata,
        sha256=file_sha256(path),
    )
