"""Position-preserving video representation for a single-encoder MMD.

Flattening is an isometry of the selected token matrix, not pooling. Division
by sqrt(K) reports mean per-token squared distance; recalibrating sigma and
text beta by the same factor leaves the RBF kernel unchanged.
"""

import math
import torch
from torch import nn

TOKEN_POOL = 'fixed_tokens_flatten_sqrt_k_v1'


def fixed_token_indices(total, count, device=None):
    if not 1 <= count <= total:
        raise ValueError(f'Token count must be in [1,{total}], got {count}')
    # Same deterministic linspace-round rule as Elastic-Forcing. Identical for
    # reference/generated videos and every iteration; never shuffle positions.
    return torch.linspace(0, total-1, count, dtype=torch.float64,
                          device=device).round().long()


def selected_token_indices(base, count, device=None):
    segments = int(base.get('token_sampling_segments', 1))
    if segments == 1:
        return fixed_token_indices(int(base['token_count']), count, device)
    if count % segments:
        raise ValueError('Selected token count must divide the number of segments')
    total = int(base['raw_tokens_per_segment'])
    indices = fixed_token_indices(total, count//segments, device)
    return torch.cat([indices + segment*total for segment in range(segments)])


def token_vector(tokens, count, indices=None):
    if tokens.ndim != 3:
        raise ValueError('Expected [video, position, channel] tokens')
    if indices is None:
        indices = fixed_token_indices(tokens.shape[1], count, tokens.device)
    else:
        indices = indices.to(tokens.device)
    return tokens.index_select(1, indices).float().flatten(1)/math.sqrt(count)


def token_feature_contract(base, count):
    return {**base, 'pool': TOKEN_POOL, 'feature_dim':int(base['feature_dim'])*count,
            'mmd_representation':'position_preserving_tokens', 'selected_token_count':count,
            'token_channel_dim':int(base['feature_dim']),
            'token_sampling':('uniform_linspace_round_per_segment' if base.get('token_sampling_segments', 1) > 1
                              else 'uniform_linspace_round'),
            'token_indices':selected_token_indices(base,count).tolist(),
            'token_distance':'frobenius_squared_divided_by_selected_token_count',
            'pooled_normalization':'none_raw_backbone_tokens', 'token_implementation':'fixed_tokens_v1'}


class TokenVideoEncoder(nn.Module):
    """Share one frozen backbone; paired CCA still consumes its complete grid."""

    def __init__(self, backbone, count, config=None):
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.count = int(count)
        base = backbone.contract()
        self.source_tokens = int(base['token_count'])
        self.register_buffer('indices', selected_token_indices(base, self.count), persistent=False)
        self.feature_dim = int(base['feature_dim'])*self.count
        self.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(False)
        return self

    @property
    def checkpoint_sha256(self):
        return self.backbone.checkpoint_sha256

    def _forward_tokens(self, video, *conditioning):
        return self.backbone._forward_tokens(video, *conditioning)

    def forward(self, video, *conditioning):
        return token_vector(self._forward_tokens(video, *conditioning), self.count, self.indices)

    def forward_views(self, video, *conditioning):
        tokens = self._forward_tokens(video, *conditioning)
        base = self.backbone.contract()
        temporal_count = base.get('temporal_tokens', base.get('token_grid', [None])[0])
        temporal = tokens.float().reshape(tokens.shape[0], int(temporal_count),
                                  -1, tokens.shape[-1]).mean(2)
        if hasattr(self.backbone, 'fc_norm'):
            temporal = self.backbone.fc_norm(temporal)
        return {'mmd_features': token_vector(tokens, self.count, self.indices),
                'mean_spatial_per_tubelet': temporal}

    def contract(self):
        base = self.backbone.contract()
        return token_feature_contract(base,self.count)

    def reference_contract(self):
        from model.encoders.text import feature_reference_contract
        cfg = self.config
        contract = feature_reference_contract(cfg, self)
        contract['rdm_reference_revision'] = str(cfg.rdm_reference_revision)
        return contract
