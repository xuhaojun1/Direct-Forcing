"""Frozen Wan 14B block-9 features and the matching reference identity."""

import hashlib

import torch
import torch.distributed as dist

ENCODER_ID = 'wan21_14b_block8_bid_clean_tau300_pooled'
MODEL_SHA256 = '441488bc362d9248d5828c66066398b2c0186cbf05c03c87679cac55a29a7c0c'


def dit_layer_index(config):
    layer = int(config.get('rdm_dit_layer_index', 8))
    if layer != 8:
        raise ValueError('The release uses DiT block 9 (zero-based index 8)')
    return layer


def dit_model_sha256(config):
    dit_layer_index(config)
    return MODEL_SHA256


def dit_reference_contract(config):
    from utils.clean_data import clean_data_identity
    tau = int(config.get('rdm_dit_timestep',300))
    layer = dit_layer_index(config)
    return {
        'encoder_id': f'wan21_14b_block{layer}_bid_clean_tau{tau}_pooled',
        'encoder_checkpoint_sha256': dit_model_sha256(config),
        'diffusers_version': '0.37.1', 'layer_index': layer, 'timestep': float(tau),
        'attention': 'bidirectional', 'input_mode': 'normalized_wan_latent',
        'input_height': 480, 'input_width': 832, 'latent_shape': [16,21,60,104],
        'noised_input': False, 'pool': 'raw_block_output_global_mean_video_text_joint',
        'feature_normalize': False, 'feature_dim': 6272, 'visual_feature_dim': 5120,
        'num_rows': int(config.get('rdm_expected_reference_rows',6505)), 'rows_per_video': 1,
        'prompt_rows_sha256': str(config.rdm_joint_prompt_rows_sha256),
        'text_table_sha256': str(config.rdm_joint_text_table_sha256),
        'dit_prompt_condition': 'positive_wan_t5_max512', 'text_feature_normalize': True,
        **clean_data_identity(config.clean_data_path),
        'rdm_reference_revision': 'pfm_dit_joint_reference_v1', 'reference_seed': 0,
    }


def dit_visual_contract(config):
    return {**dit_reference_contract(config), 'feature_dim':5120,
            'pool':'global_mean_all_tokens', 'token_grid':[21,30,52],
            'token_count':32760, 'token_output':f'raw_block{dit_layer_index(config)}_output',
            'conditioning_implementation':'shared_WanTextEncoder_bf16_max512; matched_to_retained_Diffusers_conditions'}


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class WanDiTFeatureEncoder(torch.nn.Module):
    """Direct normalized [B,C,T,H,W] latents; positive canonical Wan text conditions."""

    def __init__(self, config, device):
        super().__init__()
        import diffusers
        from model.dit import WanTransformer3DModel

        if diffusers.__version__ != '0.37.1':
            raise ValueError('The retained DiT reference requires diffusers 0.37.1')
        self.device = torch.device('cuda', device) if isinstance(device, int) else torch.device(device)
        self.timestep = int(config.get('rdm_dit_timestep', 300))
        self.layer_index = dit_layer_index(config)
        if self.timestep != 300:
            raise ValueError('The released DiT reference uses timestep conditioning 300')
        self.encoder_id = f'wan21_14b_block{self.layer_index}_bid_clean_tau{self.timestep}_pooled'
        if str(config.rdm_encoder_id) != self.encoder_id:
            raise ValueError('DiT encoder ID and feature timestep disagree')
        self.model = WanTransformer3DModel.from_pretrained(config.rdm_encoder_checkpoint_path,
            subfolder='transformer', torch_dtype=torch.bfloat16, local_files_only=True)
        if self.layer_index >= len(self.model.blocks):
            raise ValueError('Requested DiT feature block is missing from the checkpoint')
        self.model.blocks = self.model.blocks[:self.layer_index+1]
        self.model.to(dtype=torch.bfloat16)
        self.model.requires_grad_(False).eval()
        digest = [model_hash(self.model) if not dist.is_initialized() or dist.get_rank() == 0 else None]
        if dist.is_initialized():
            dist.broadcast_object_list(digest, src=0)
        self.checkpoint_sha256 = digest[0]
        expected_sha256 = dit_model_sha256(config)
        configured_sha256 = str(config.rdm_encoder_checkpoint_sha256)
        if self.checkpoint_sha256 != expected_sha256 or self.checkpoint_sha256 != configured_sha256:
            raise ValueError(
                f'Frozen Wan prefix differs from the retained DiT reference '
                f'(block_index={self.layer_index}, dtype=bfloat16, '
                f'actual={self.checkpoint_sha256}, expected={expected_sha256}, '
                f'configured={configured_sha256})')
        self.model.to(self.device)
        if config.rdm_encoder_activation_checkpointing:
            self.model.enable_gradient_checkpointing()
        self.temporal_tokens, self.height_tokens, self.width_tokens = 21, 30, 52
        self.feature_dim = 5120
        self._reference_contract = dit_reference_contract(config)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def contract(self):
        return {**self._reference_contract, 'feature_dim': 5120,
                'pool': 'global_mean_all_tokens', 'token_grid': [21, 30, 52],
                'token_count': 32760, 'token_output': f'raw_block{self.layer_index}_output',
                'conditioning_implementation': 'shared_WanTextEncoder_bf16_max512; matched_to_retained_Diffusers_conditions'}

    def reference_contract(self):
        return dict(self._reference_contract)

    def _forward_tokens(self, latent, embedding):
        if tuple(latent.shape) != (1, 16, 21, 60, 104):
            raise ValueError(f'DiT expects normalized [1,16,21,60,104], got {tuple(latent.shape)}')
        if tuple(embedding.shape) != (1, 512, 4096):
            raise ValueError(f'DiT requires canonical Wan text [1,512,4096], got {tuple(embedding.shape)}')
        embedding = embedding.detach().to(device=latent.device, dtype=torch.bfloat16)
        tau = torch.full((1,), float(self.timestep), device=latent.device, dtype=torch.float32)
        tokens, = dit_features(self.model, latent, embedding, tau, [self.layer_index])
        if tuple(tokens.shape) != (1, 32760, 5120):
            raise ValueError(f'Unexpected DiT token grid: {tuple(tokens.shape)}')
        return tokens

    def forward(self, latent, embedding):
        return self._forward_tokens(latent, embedding)


def wan_blocks(model):
    for name in ("blocks", "transformer_blocks"):
        blocks = getattr(model, name, None)
        if isinstance(blocks, torch.nn.ModuleList):
            return name, blocks
    raise ValueError("cannot locate Wan transformer blocks")


def dit_features(model, clean, prompt, timestep, layers):
    captured, handles = {}, []

    def capture(index):
        def hook(_module, _inputs, output):
            captured[index] = output[0] if isinstance(output, tuple) else output
        return hook

    try:
        _, blocks = wan_blocks(model)
        for index in layers:
            handles.append(blocks[index].register_forward_hook(capture(index)))
        model(hidden_states=clean.to(next(model.parameters()).dtype),
              timestep=timestep, encoder_hidden_states=prompt,
              return_dict=False)
    finally:
        for handle in handles:
            handle.remove()
    return [captured[index] for index in layers]
