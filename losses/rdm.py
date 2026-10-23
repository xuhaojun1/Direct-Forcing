"""Differentiable Nyström MMD statistics used by RDM."""
import time
import torch
import torch.utils.checkpoint
from references.bank import RDMReference


def self_normalize_rdm_loss(raw: torch.Tensor, eps: float = 1.0e-7) -> torch.Tensor:
    return raw / (raw.detach().abs() + float(eps))


def mmd_nystrom_with_terms(
    feat_g: torch.Tensor,
    reference: RDMReference,
    *,
    gen_chunk: int = 4096,
    timings: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the biased generated kernel and frozen Nyström attraction."""

    # For CPU residency this GPU copy lives only through this kernel's backward.
    # In no-grad evaluation it is released as soon as this function returns.
    moving = reference.Z.device != feat_g.device
    if timings is not None:
        timings['reference_h2d_seconds'] = 0.
    if moving and timings is not None and feat_g.device.type == 'cuda':
        # Exclude preceding feature/gather work from the transfer measurement.
        torch.cuda.synchronize(feat_g.device)
        started = time.perf_counter()
        reference = reference.to(feat_g.device)
        torch.cuda.synchronize(feat_g.device)
        timings['reference_h2d_seconds'] = time.perf_counter()-started
    else:
        reference = reference.to(feat_g.device)

    def self_kernel_mean(features: torch.Tensor, gamma: float) -> torch.Tensor:
        row_count = features.shape[0]

        def block(values, start, stop):
            rows = values[start:stop]
            row_norm = (rows * rows).sum(1)
            value_norm = (values * values).sum(1)
            distance = (
                row_norm[:, None] + value_norm[None, :] - 2.0 * (rows @ values.T)
            ).clamp_min(0)
            return torch.exp(-gamma * distance).sum()

        total = features.new_zeros(())
        for start in range(0, row_count, int(gen_chunk)):
            stop = min(start + int(gen_chunk), row_count)
            total = total + torch.utils.checkpoint.checkpoint(
                block, features, start, stop, use_reentrant=False
            )
        return total / (row_count * row_count)

    gamma = 1.0 / (2.0 * reference.sigma * reference.sigma)

    g = feat_g.float()
    k_gg = self_kernel_mean(g, gamma)
    distance = (
        g.pow(2).sum(1)[:, None] + reference.Z2[None, :] - 2.0 * (g @ reference.Z.T)
    ).clamp_min(0)
    k_gr = (torch.exp(-gamma * distance) @ reference.alpha).mean()
    k_rr = g.new_tensor(reference.k_rr)
    raw = k_gg - 2.0 * k_gr + k_rr
    return raw, {"k_gg": k_gg, "k_gr_nystrom": k_gr, "k_rr": k_rr}
