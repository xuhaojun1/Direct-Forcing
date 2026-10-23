"""Exact sample OT and MMD on ordered token vectors without huge CPU SVDs."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from losses.rdm import mmd_nystrom_with_terms
from encoders.text import couple_video_text


@torch.no_grad()
def compare_tokens(x, y, reference, text, device):
    x = torch.as_tensor(x, device=device).double()
    y = torch.as_tensor(y, device=device).double()
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError('Fixed token evaluation requires equal-sized video populations')
    n = len(x)
    cost = (x.square().sum(1)[:, None]+y.square().sum(1)[None, :]-2*x@y.T).clamp_min(0).cpu().numpy()
    i, j = linear_sum_assignment(cost)
    xc, yc = x-x.mean(0), y-y.mean(0)
    variance = float(xc.square().sum()/n)
    eigenvalues = torch.linalg.eigvalsh(xc@xc.T/n).clamp_min(0)
    probabilities = eigenvalues/eigenvalues.sum().clamp_min(1e-30)
    mean_shift = float((x.mean(0)-y.mean(0)).square().sum())
    gaussian = max(0., mean_shift+variance+float(yc.square().sum()/n)
                   -2*float(torch.linalg.svdvals(xc@yc.T/n).sum()))
    del xc, yc
    if reference.metadata.get('joint_enable', True):
        if text is None:
            raise ValueError('Text-coupled reference requires its text rows')
        joint = couple_video_text(x.float(), torch.as_tensor(text, device=device).float(), reference.beta)
    else:
        if reference.beta is not None or text is not None:
            raise ValueError('Visual-only evaluation must not contain text features')
        joint = x.float()
    raw, terms = mmd_nystrom_with_terms(joint, reference, gen_chunk=32)
    ot = float(cost[i, j].mean())
    return dict(empirical_w2_squared=ot,
        solver='scipy.linear_sum_assignment; uniform equal-size empirical measures',
        representation='ordered token matrix; squared Frobenius distance / K; no pooling',
        gaussian_moment_w2_squared=gaussian, squared_mean_shift=mean_shift,
        centered_empirical_w2_squared=max(0., ot-mean_shift),
        paired_squared_distance=float((x-y).square().sum(1).mean()),
        visual_ot_same_prompt_mass=float(np.mean(i == j)),
        moments=dict(variance_trace=variance,
            effective_rank=float(torch.exp(-(probabilities*probabilities.clamp_min(1e-30).log()).sum()))),
        official_joint_nystrom_mmd=dict(raw_biased_mmd2=float(raw), joint_text_enabled=reference.metadata.get('joint_enable', True),
            **{k: float(v) for k,v in terms.items()}))
