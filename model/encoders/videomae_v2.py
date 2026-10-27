"""Frozen VideoMAEv2-Large, block 16 or 24, five chronological 16-frame clips.

One shared frozen prefix processes frames 1:17, 17:33, ..., 65:81. CCA keeps
all 7,840 tokens; TokenVideoEncoder selects 256 per segment for MMD. The
preprocessing and post-residual block output match Elastic's v2 adapter.
"""

import hashlib
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

MODEL_ID = 'videomae_v2_vitl_l15_5x16_224'
MODEL_LAYERS = {MODEL_ID: 15, 'videomae_v2_vitl_l23_5x16_224': 23}
SOURCE_REVISION = '9981a9c8f77118c421e5228e1b219468a4b0238d'
CHECKPOINT_SHA256 = 'c27064402bfec2c7495e2226a5e0d8f46374b4df4b03026af1e10a57e11b2996'
WINDOW_STARTS = (1, 17, 33, 49, 65)
PINNED_FILES = {
    'config.json': 'b30ba8c947c437ed0d25678370fdec9615f3dcb988d74b900e2ef8f4a01c6d5e',
    'modeling_config.py': 'e645f9b518e152523cfa097dd049cdd86131f5a0b7d96bb52edbce64b2c0335f',
    'modeling_videomaev2.py': 'f821aac219d959254c7189995afd4dd313d98c74a89d4efff05057771bae1079',
    'preprocessor_config.json': 'fa12f1db567e88119a097a96cfe288faccc3af2144476f39890fbd255bd24a68',
}


class VideoMAEv2LargeVideoEncoder(nn.Module):
    def __init__(self, model_id, checkpoint_path, *, activation_checkpointing=True,
                 device='cuda', **kwargs):
        super().__init__()
        if model_id not in MODEL_LAYERS:
            raise ValueError(f'Unsupported VideoMAE v2 model: {model_id}')
        self.layer_index = MODEL_LAYERS[model_id]
        expected = dict(checkpoint_key='safetensors', input_height=480, input_width=832,
            input_frames=81, padded_frames=81, temporal_pad='skip_first',
            feature_dim=1024)
        for key, value in kwargs.items():
            if key not in expected or value != expected[key]:
                raise ValueError(f'Unsupported VideoMAE v2 option: {key}={value!r}')
        directory = Path(checkpoint_path).resolve().parent
        for name, digest in PINNED_FILES.items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Pinned VideoMAE v2 {name} changed')
        from utils.hashing import file_sha256
        if file_sha256(checkpoint_path) != CHECKPOINT_SHA256:
            raise ValueError('VideoMAE v2 checkpoint checksum mismatch')
        from transformers import AutoModel
        loaded, info = AutoModel.from_pretrained(directory, local_files_only=True,
            trust_remote_code=True, use_safetensors=True, output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise RuntimeError(f'VideoMAE v2 checkpoint mismatch: {info}')
        cfg = loaded.config.model_config
        if tuple(cfg[k] for k in ('embed_dim', 'depth', 'num_frames', 'img_size',
                                  'patch_size', 'tubelet_size', 'cos_attn')) != (1024, 24, 16, 224, 16, 2, False):
            raise ValueError('Expected the pinned VideoMAEv2-Large architecture')
        self.model = loaded.model
        self.model.blocks = nn.ModuleList(list(self.model.blocks[:self.layer_index+1]))
        for name in ('norm', 'fc_norm', 'head_dropout', 'head'):
            if hasattr(self.model, name):
                setattr(self.model, name, None)
        self.activation_checkpointing = activation_checkpointing
        self.temporal_tokens, self.height_tokens, self.width_tokens = 40, 14, 14
        self.feature_dim = 1024
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 1, 3, 1, 1), persistent=False)
        self.to(device=device, dtype=torch.float32).eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(False)
        return self

    def preprocess(self, video):
        if video.ndim != 5 or tuple(video.shape[1:]) != (81, 3, 480, 832):
            raise ValueError(f'Expected RGB [B,81,3,480,832], got {tuple(video.shape)}')
        frames = F.interpolate(video[:, 1:].flatten(0, 1).float(), size=(224, 388),
            mode='bilinear', align_corners=False, antialias=True)
        frames = frames[..., 82:306].reshape(video.shape[0], 80, 3, 224, 224)
        return (frames-self.mean)/self.std

    def _encode_window(self, pixels):
        with torch.autocast('cuda', enabled=pixels.is_cuda, dtype=torch.bfloat16):
            values = self.model.patch_embed(pixels.permute(0, 2, 1, 3, 4).contiguous())
            position = self.model.pos_embed
            if position is not None:
                values = values + position.type_as(values).to(values.device).clone().detach()
            values = self.model.pos_drop(values)
            for block in self.model.blocks:
                values = block(values)
        return values.float()

    def _forward_tokens(self, video):
        pixels = self.preprocess(video)
        windows = []
        for start in range(0, 80, 16):
            clip = pixels[:, start:start+16]
            if self.activation_checkpointing and torch.is_grad_enabled() and clip.requires_grad:
                values = checkpoint(self._encode_window, clip, use_reentrant=False)
            else:
                values = self._encode_window(clip)
            windows.append(values)
        return torch.cat(windows, dim=1)

    def forward(self, video):
        return self._forward_tokens(video)

    def contract(self):
        layer = getattr(self, 'layer_index', 15)
        return dict(input_mode='native_video', input_height=480, input_width=832,
            input_frames=81, padded_frames=81, temporal_pad='skip_first',
            patch_size=16, tubelet_size=2, temporal_tokens=40, height_tokens=14,
            width_tokens=14, token_count=7840, token_output=f'block{layer}_post_residual',
            pool='global_mean_all_tokens', pooled_rows_per_video=1, feature_dim=1024,
            feature_normalize=False, is_causal=False, layer_index=layer,
            encoder_preprocess='bilinear_antialias_short224_center224_float_v1',
            mean=[.485, .456, .406], std=[.229, .224, .225],
            window_frames=16, window_starts=list(WINDOW_STARTS),
            token_sampling_segments=5, raw_tokens_per_segment=1568,
            pooled_normalization='none_raw_block_tokens', source_repo='OpenGVLab/VideoMAEv2-Large',
            source_revision=SOURCE_REVISION, checkpoint_key='safetensors',
            encoder_implementation=f'videomaev2_l{layer}_shared_prefix_v1')
