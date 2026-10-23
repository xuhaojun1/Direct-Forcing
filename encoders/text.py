"""Frozen text tables and joint video-text feature utilities."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch

from references.bank import file_sha256


RDM_JOINT_TEXT_MODEL = "ViT-SO400M-16-SigLIP2-256"
RDM_JOINT_TEXT_PRETRAINED = "webli"
RDM_JOINT_TEXT_DIM = 1152
RDM_JOINT_BANDWIDTH_SCALE = 1.0
RDM_JOINT_CONSTRUCTION = "concat_single_rbf"
RDM_JOINT_POOL = "fixed_tokens_flatten_sqrt_k_v1_concat_siglip2_text"


def prompt_rows_sha256(prompts: Iterable[str]) -> str:
    """Hash an ordered prompt table without delimiter ambiguity."""

    digest = hashlib.sha256()
    for index, prompt in enumerate(prompts):
        encoded = str(prompt).encode("utf-8")
        digest.update(int(index).to_bytes(8, byteorder="little", signed=False))
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def load_frozen_text_table(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    expected_dim: int = RDM_JOINT_TEXT_DIM,
    device: torch.device | str = "cpu",
    norm_atol: float = 5.0e-4,
) -> torch.Tensor:
    """Load the official frozen, L2-normalized ``tau(c)`` table strictly."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"joint text table not found: {path}")
    actual_sha256 = file_sha256(path)
    if actual_sha256 != str(expected_sha256):
        raise RuntimeError(
            f"joint text table SHA-256 {actual_sha256} != configured {expected_sha256}"
        )
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.float32:
        raise ValueError(f"joint text table must be float32, got {array.dtype}")
    if tuple(array.shape) != (int(expected_rows), int(expected_dim)):
        raise ValueError(
            f"joint text table shape {tuple(array.shape)} != "
            f"{(int(expected_rows), int(expected_dim))}"
        )
    table = torch.from_numpy(np.ascontiguousarray(array)).float()
    if not bool(torch.isfinite(table).all()):
        raise ValueError("joint text table contains NaN or Inf")
    norm_error = float((torch.linalg.vector_norm(table, dim=1) - 1.0).abs().max())
    if norm_error > float(norm_atol):
        raise ValueError(
            f"joint text table is not L2-normalized: max norm error {norm_error:.6g}"
        )
    return table.to(device=device)


def rdm_joint_contract(
    visual_contract: Mapping[str, object],
    *,
    text_table_sha256: str,
    prompt_rows_sha256_value: str,
    bandwidth_scale: float = RDM_JOINT_BANDWIDTH_SCALE,
    visual_feature_dim: int = 1024,
    text_feature_dim: int = RDM_JOINT_TEXT_DIM,
    text_model: str = RDM_JOINT_TEXT_MODEL,
    text_pretrained: str = RDM_JOINT_TEXT_PRETRAINED,
    text_kernel_eta: float = 1.,
) -> dict[str, object]:
    """Return the immutable metadata contract for a joint reference bundle."""

    if not math.isfinite(float(text_kernel_eta)) or float(text_kernel_eta) <= 0:
        raise ValueError('Text kernel eta must be finite and positive')
    suffix = ('_concat_siglip2_text' if text_model == RDM_JOINT_TEXT_MODEL
              else '_concat_qwen3vl_text' if text_model == 'Qwen/Qwen3-VL-Embedding-2B'
              else None)
    if suffix is None:
        raise ValueError(f'Unsupported joint text model: {text_model}')
    contract = dict(visual_contract)
    contract.update(
        {
            "pool": str(visual_contract['pool']) + suffix,
            "pooled_rows_per_video": 1,
            "rows_per_video": 1,
            "feature_dim": int(visual_feature_dim) + int(text_feature_dim),
            "joint_enable": True,
            "joint_construction": RDM_JOINT_CONSTRUCTION,
            "joint_visual_feature_dim": int(visual_feature_dim),
            "joint_text_feature_dim": int(text_feature_dim),
            "joint_text_encoder_id": str(text_model),
            "joint_text_encoder_pretrained": str(text_pretrained),
            "joint_text_feature_normalize": True,
            "joint_text_table_sha256": str(text_table_sha256),
            "joint_prompt_rows_sha256": str(prompt_rows_sha256_value),
            "joint_pairing": "lmdb_row_index",
            "joint_bandwidth_scale": float(bandwidth_scale),
        }
    )
    # Preserve existing eta=1 reference contracts byte-for-byte. Weighted
    # references carry an explicit field and have a separate reference revision.
    if float(text_kernel_eta) != 1.:
        contract['joint_text_kernel_eta'] = float(text_kernel_eta)
    return contract


def couple_video_text(
    visual_features: torch.Tensor,
    text_rows: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Build ``[visual_features | beta * text_rows]``."""

    text_block = (float(beta) * text_rows).to(visual_features.dtype)
    return torch.cat([visual_features, text_block], dim=1)


def feature_reference_contract(config, encoder):
    """Keep visual-only references distinct from SigLIP2 joint references."""
    if not config.get('rdm_joint_text_enabled', True):
        return dict(encoder.contract(), joint_enable=False, rows_per_video=1,
                    joint_construction='visual_only',
                    joint_prompt_rows_sha256=str(config.rdm_joint_prompt_rows_sha256),
                    rdm_reference_revision=str(config.rdm_reference_revision))
    return rdm_joint_contract(encoder.contract(),
        text_table_sha256=str(config.rdm_joint_text_table_sha256),
        prompt_rows_sha256_value=str(config.rdm_joint_prompt_rows_sha256),
        bandwidth_scale=float(config.rdm_joint_bandwidth_scale),
        visual_feature_dim=int(config.rdm_feature_dim),
        text_feature_dim=int(config.rdm_joint_text_feature_dim),
        text_model=str(config.rdm_joint_text_encoder_id),
        text_pretrained=str(config.rdm_joint_text_encoder_pretrained),
        text_kernel_eta=float(config.get('rdm_joint_text_kernel_eta', 1.)))
