"""Portable public configuration, compiled into the validated training recipe."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value, root=ROOT):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else Path(root) / path


def load_spec(path, overrides=(), _stack=()):
    path = resolve_path(path).resolve()
    if path in _stack:
        raise ValueError(f"Cyclic config inheritance: {path}")
    config = OmegaConf.load(path)
    parent = config.pop('_base_', None)
    if parent:
        config = OmegaConf.merge(load_spec(path.parent / parent, _stack=(*_stack, path)), config)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    for key, value in config.paths.items():
        config.paths[key] = str(resolve_path(value).resolve())
    validate_spec(config)
    return config


def validate_spec(spec):
    t = spec.training
    if int(t.world_size) not in (1, 4, 8):
        raise ValueError('Supported single-node world sizes: 1, 4, 8')
    if int(t.rdm_batch) not in (32, 64, 128, 256, 512, 1024):
        raise ValueError('Unsupported global RDM batch')
    if not 0 < int(t.cca_batch) <= int(t.rdm_batch):
        raise ValueError('CCA batch must be positive and no larger than RDM batch')
    if int(t.cca_batch) % int(t.world_size) or int(t.rdm_batch) % int(t.world_size):
        raise ValueError('Global batches must be divisible by world size')
    if int(t.steps) < 1 or int(t.checkpoint_every) < 1 or int(t.preview_every) < 1:
        raise ValueError('Training and logging intervals must be positive')
    if int(t.reference_landmarks) != 4096:
        raise ValueError('This recipe uses min(4096, dataset size) reference landmarks')
    for key in ('fixed_ratio', 'fixed_weight'):
        if not math.isfinite(float(spec.controller[key])) or float(spec.controller[key]) < 0:
            raise ValueError(f'controller.{key} must be finite and nonnegative')
    if not 0 <= float(spec.controller.r_min) <= 1:
        raise ValueError('r_min must lie in [0, 1]')
    if spec.controller.rew and not spec.controller.gnm:
        raise ValueError('REW uses the GNM-calibrated gradient; enable GNM or disable REW')
    if spec.memory.reference_residency not in ('cpu', 'cuda'):
        raise ValueError('Reference residency must be cpu or cuda')
    if spec.logging.wandb_mode not in ('disabled', 'offline', 'online'):
        raise ValueError('W&B mode must be disabled, offline, or online')
    if tuple(spec.features.rdm) not in (('videomae_v2',), ('dit',), ('videomae_v2', 'dit')):
        raise ValueError('Release RDM choices: [videomae_v2], [dit], [videomae_v2, dit]')
    if spec.features.cca not in ('videomae_v2', 'videomae_v2_l24', 'dit'):
        raise ValueError('Release CCA choices: videomae_v2, videomae_v2_l24, dit')


def read_metadata(spec):
    path = Path(spec.paths.metadata)
    if not path.is_file():
        raise FileNotFoundError(f'Data metadata missing: {path}. Run scripts/prepare_data.py first.')
    metadata = json.loads(path.read_text())
    if int(metadata['num_rows']) < 160 + int(spec.training.rdm_batch):
        raise ValueError('Data must contain 160 fixed probes plus at least one complete RDM batch')
    return metadata


def validate_assets(spec, *, training=True):
    """Check required files before any rank allocates a model; runtime verifies hashes."""
    p = spec.paths
    wan = Path(p.wan)
    required = [wan / 'config.json', wan / 'models_t5_umt5-xxl-enc-bf16.pth',
                wan / 'Wan2.1_VAE.pth', wan / 'google/umt5-xxl/tokenizer_config.json']
    if not list(wan.glob('diffusion_pytorch_model*.safetensors')):
        raise FileNotFoundError(f'Wan generator weights missing under {wan}; run scripts/download_models.py')
    if training:
        required += [Path(p.taew), Path(p.clean_lmdb) / 'data.mdb',
                     Path(p.prompt_lmdb) / 'data.mdb', Path(p.text_table), Path(p.probes)]
        encoders = set(spec.features.rdm) | {spec.features.cca}
        if encoders & {'videomae_v2', 'videomae_v2_l24'}:
            required += [Path(p.videomae) / 'model.safetensors', Path(p.videomae) / 'config.json',
                         Path(p.videomae) / 'modeling_videomaev2.py', Path(p.videomae) / 'modeling_config.py']
        if 'dit' in encoders:
            directory = Path(p.dit) / 'transformer'
            required.append(directory / 'config.json')
            if not list(directory.glob('diffusion_pytorch_model*.safetensors')):
                raise FileNotFoundError(f'DiT transformer weights missing under {directory}')
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('Required assets missing: ' + ', '.join(missing))


def base_runtime(spec, metadata):
    c = OmegaConf.load(ROOT / 'configs/runtime/base.yaml')
    p, t = spec.paths, spec.training
    c.asset_paths = OmegaConf.to_container(p, resolve=True)
    c.data_path, c.clean_data_path = p.prompt_lmdb, p.clean_lmdb
    c.rdm_joint_text_table_path = p.text_table
    c.rdm_joint_text_table_sha256 = metadata['text_table_sha256']
    c.rdm_joint_prompt_rows_sha256 = metadata['prompt_rows_sha256']
    c.rdm_expected_reference_rows = int(metadata['num_rows'])
    c.rdm_reference_revision = 'pfm_fixed_tokens_joint_v1'
    c.rdm_taew_checkpoint_path = p.taew
    c.rdm_taew_checkpoint_sha256 = 'd26151e76cdc2c9424bef988de874b33d9a53f30ef3060cd556c429c469c797e'
    c.rdm_nystrom_landmarks = min(int(t.reference_landmarks), int(metadata['num_rows']))
    c.max_steps, c.seed = int(t.steps), int(t.seed)
    c.lr, c.lr_floor = float(t.learning_rate), float(t.learning_rate_floor)
    c.lr_cosine_end = int(t.cosine_end_step)
    c.log_iters, c.preview_every = int(t.checkpoint_every), int(t.preview_every)
    c.rdm_expected_world_size = c.rdm_expected_local_world_size = int(t.world_size)
    c.rdm_expected_global_rows = c.joint_global_batch = int(t.rdm_batch)
    c.rdm_grad_accum_steps = int(t.rdm_batch) // int(t.world_size)
    c.joint_cca_batch = int(t.cca_batch)
    c.joint_probe_manifest = p.probes
    c.rdm_reference_residency = spec.memory.reference_residency
    c.generator_activation_cpu_offload = bool(spec.memory.activation_offload)
    c.cca_target_cache = bool(spec.memory.cache_cca_targets)
    c.cca_target_cache_root = str(Path(p.target_cache) / 'cca')
    c.wandb_project, c.wandb_entity = spec.logging.project, spec.logging.entity
    c.wandb_mode = spec.logging.wandb_mode
    c.joint_ratio_start, c.joint_ratio_min = 1., float(spec.controller.r_min)
    if spec.controller.rew:
        c.joint_controller_mode = 'gnm_rew_v1'
    elif spec.controller.gnm:
        c.joint_controller_mode = 'gnm_v1'
        c.joint_fixed_gradient_ratio = float(spec.controller.fixed_ratio)
    else:
        c.joint_controller_mode = 'fixed_cca_v1'
        c.joint_cca_fixed_weight = float(spec.controller.fixed_weight)
    return c


def encoder_runtime(spec, name, metadata=None, load_reference=True):
    from model.encoders.config import configure_token_assets
    metadata = read_metadata(spec) if metadata is None else metadata
    c = base_runtime(spec, metadata)
    configure_token_assets(c, name, ROOT)
    c.feature_cache_root = str(Path(spec.paths.target_cache) / name)
    directory = Path(spec.paths.references) / name
    c.rdm_reference_bundle_path = str(directory / f'reference_M{c.rdm_nystrom_landmarks}.pt')
    c.rdm_reference_bundle_sha256 = ''
    if load_reference:
        receipt_path = directory / 'receipt.json'
        if not receipt_path.is_file():
            raise FileNotFoundError(f'Reference missing: {receipt_path}. Run scripts/prepare_references.sh.')
        receipt = json.loads(receipt_path.read_text())
        expected = dict(encoder_id=c.rdm_encoder_id,
                        encoder_checkpoint_sha256=c.rdm_encoder_checkpoint_sha256,
                        num_rows=c.rdm_expected_reference_rows,
                        nystrom_landmarks=c.rdm_nystrom_landmarks,
                        selected_token_count=c.rdm_token_count,
                        text_table_sha256=c.rdm_joint_text_table_sha256,
                        prompt_rows_sha256=c.rdm_joint_prompt_rows_sha256)
        mismatched = [k for k, v in expected.items() if receipt.get(k) != v]
        if mismatched:
            raise ValueError(f'Reference contract mismatch ({name}): {mismatched}')
        if not Path(c.rdm_reference_bundle_path).is_file():
            raise FileNotFoundError(c.rdm_reference_bundle_path)
        c.rdm_reference_bundle_sha256 = receipt['reference_bundle_sha256']
    return c


def compile_runtime(spec, load_references=True):
    from model.encoders.config import configure_cca_feature_backend
    metadata = read_metadata(spec)
    names = list(spec.features.rdm)
    if len(names) == 2:
        c = encoder_runtime(spec, 'dit', metadata, load_references)
        secondary = encoder_runtime(spec, 'videomae_v2', metadata, load_references)
        c.joint_secondary_backend = OmegaConf.to_container(secondary, resolve=True)
        c.joint_rdm_encoder = 'videomae_v2_dit'
        c.joint_encoder_combination = 'mean_cca_cosine_and_mean_self_normalized_mmd_v1'
    else:
        c = encoder_runtime(spec, names[0], metadata, load_references)
    configure_cca_feature_backend(c, spec.features.cca, ROOT)
    c.joint_rdm_representation = 'tokens'
    return c


def configure_environment(spec):
    os.environ['DIRECT_FORCING_WAN_ROOT'] = spec.paths.wan
    os.environ['WANDB_MODE'] = spec.logging.wandb_mode
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('CAUSAL_DISABLE_FLEX_ATTENTION', '1')


def inference_runtime(spec):
    """Inference requires only Wan and a Direct-Forcing checkpoint, never feature/data caches."""
    c = OmegaConf.load(ROOT / 'configs/runtime/base.yaml')
    c.num_frame_per_block = 3
    c.num_training_frames = 21
    c.denoising_step_list = [1000, 750, 500, 250]
    c.model_kwargs.local_attn_size = -1
    return c
