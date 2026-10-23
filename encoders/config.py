"""Frozen feature specifications and independent causal-alignment selection."""
from pathlib import Path
from omegaconf import OmegaConf

DIT_LAYERS = {"dit": 8}

def configure_token_assets(config, name, project_root, token_count=None, dit_timestep=300):
    if name not in ("videomae_v2", "videomae_v2_l24", "dit"):
        raise ValueError(f"Unsupported release encoder: {name}")
    if token_count is not None or int(dit_timestep) != 300:
        raise ValueError("The release recipe fixes V16/V24 K1280, DiT K256 and tau300")
    preset = OmegaConf.load(Path(project_root)/"configs/encoders"/(name+".yaml"))
    config.merge_with(preset)
    models = config.asset_paths
    config.rdm_encoder_checkpoint_path = (models.dit if name == "dit"
        else str(Path(models.videomae)/"model.safetensors"))
    config.joint_rdm_encoder = name
    return config

CCA_FEATURE_ENCODERS = ('videomae_v2', 'videomae_v2_l24', 'dit')


def _cca_backend_settings(config):
    """Frozen encoder settings without loading an RDM reference for CCA."""
    from omegaconf import OmegaConf
    return {k: v for k, v in OmegaConf.to_container(config, resolve=True).items()
            if (k.startswith('rdm_') and not k.startswith('rdm_reference_bundle')
                and k != 'rdm_pairing_mode')
            or k == 'clean_data_path'}


def cca_feature_backends(config):
    """Independent CCA source, if it does not share the RDM encoder."""
    source = config.get('joint_cca_backend')
    return [] if source is None else [source]


def configure_cca_feature_backend(config, name, project_root, dit_timestep=300):
    """Select a frozen CCA encoder without changing/loading any MMD reference."""
    from omegaconf import OmegaConf
    if name not in CCA_FEATURE_ENCODERS or config.get('joint_cca_mode') != 'teacher_forcing':
        raise ValueError('Independent CCA features require CCA pairing and a supported encoder')
    dual = config.get('joint_secondary_backend') is not None
    if dual and config.joint_rdm_encoder != 'videomae_v2_dit':
        raise ValueError('Independent CCA selection supports one MMD encoder or VideoMAE+DiT')
    cca = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    # A latent-space MMD encoder must not leak its input mode into video CCA.
    cca.rdm_input_mode = 'native_video'
    configure_token_assets(cca, name, project_root, dit_timestep=dit_timestep)
    config.joint_cca_encoder = name
    shared = (name == config.joint_rdm_encoder and
              cca.rdm_encoder_id == config.rdm_encoder_id)
    config.joint_cca_backend = None if shared else _cca_backend_settings(cca)
    config.joint_cca_feature_contract = {
        'encoder_id': cca.rdm_encoder_id,
        'checkpoint_path': cca.rdm_encoder_checkpoint_path,
        'checkpoint_sha256': cca.rdm_encoder_checkpoint_sha256,
        'input_mode': cca.rdm_input_mode,
        'block_number': {'videomae_v2': 16, 'videomae_v2_l24': 24, 'dit': 9}[name],
        'output': 'raw_block_output',
        'dit_timestep': int(dit_timestep) if name in DIT_LAYERS else None,
        'loss': 'mean cosine over all paired tokens; no token subsampling',
        'shares_rdm_encoder': shared,
        'loads_cca_reference': False}
    if dual:
        config.joint_encoder_combination = 'independent_cca_and_mean_self_normalized_mmd_v1'
    return config


def feature_runtime_config(primary, feature_config):
    """Attach trainer-entry runtime options without changing the pinned recipe.

    Feature configurations are captured by the launcher before the trainer sets
    its output paths and logging switches. Each feature branch must inherit those
    later options, including when W&B is enabled on the main rank.
    """
    from omegaconf import OmegaConf

    result = OmegaConf.create(OmegaConf.to_container(feature_config, resolve=True))
    for key in ('logdir', 'wandb_save_dir', 'disable_wandb', 'no_save',
                'no_visualize', 'config_name'):
        result[key] = primary[key]
    if 'rdm_cleanup_policy' in primary:
        result.rdm_cleanup_policy = primary.rdm_cleanup_policy
    return result
