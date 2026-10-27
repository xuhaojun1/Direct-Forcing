"""Equal-weight feature objectives; feature spaces keep separate kernels."""

import torch



def combine_rdm_branches(values, gradients, weights=None, divisor=None):
    """Pack separate feature VJPs, applying each branch coefficient once."""
    if not values or len(values) != len(gradients):
        raise ValueError('Expected matching feature/gradient branches')
    rows = len(values[0])
    if any(len(branch) != rows for branch in values + gradients):
        raise ValueError('Feature branches must use the same rollout population')
    count = len(values)
    weights = [1.] * count if weights is None else list(weights)
    divisor = float(count if divisor is None else divisor)
    import math
    if len(weights) != count or not math.isfinite(divisor) or divisor <= 0 or any(
            not math.isfinite(float(w)) or float(w) < 0 for w in weights):
        raise ValueError('Invalid MMD encoder weights/divisor')
    packed_values = [torch.cat([v[i] for v in values], dim=-1) for i in range(rows)]
    packed_gradients = [torch.cat([g[i]*(float(w)/divisor) for g, w in zip(gradients, weights)], dim=-1) for i in range(rows)]
    return packed_values, packed_gradients
