"""Translate research-era names only when reading serialized metadata.

The implementation uses CCA/RDM names. These aliases keep existing controller
states, resume contracts, and reference bundles readable without changing model
or optimizer tensors. MMD remains the name of the kernel statistic.
"""
from __future__ import annotations

import re


NAME_ALIASES = {
    'JointTrainer': 'DirectForcingTrainer',
    'RDMModel': 'CausalVideoModel',
    'RDMTrainingPipeline': 'SelfRolloutPipeline',
    'PairMMDTrainer': 'DirectForcingTrainer',
    'DualPairMMDTrainer': 'MultiEncoderTrainer',
    'PairedTrainer': 'TrainingRuntime',
    'Trainer': 'RDMTrainer',
    'PairFitWeights': 'GNMREWController',
    'PairGradientWeights': 'GNMController',
    'FixedPairWeights': 'FixedCCAController',
    'LimitedPairWeights': 'ScheduledCCAController',
    'CURRENT_WEIGHTING_KIND': 'GNM_REW_KIND',
    'GRADIENT_WEIGHTING_KIND': 'GNM_KIND',
    'FIXED_WEIGHTING_KIND': 'FIXED_CCA_KIND',
    'ViRDM': 'CausalVideoModel',
    'ViRDMTrainingPipeline': 'SelfRolloutPipeline',
    'ViRDMReference': 'RDMReference',
    'PairedVideoFeatureLoss': 'CCAFeatureLoss',
    'PairedDiTFeatureLoss': 'DiTCCALoss',
    '_PairedLatentLoss': '_CCALatentLoss',
    'MeanPairedFeatureLoss': 'MeanCCALoss',
    'DualPairedFeatureLoss': 'DualCCALoss',
    'TFTargetTokenCache': 'CCATargetTokenCache',
    'TF_COMBINATIONS': 'CCA_COMBINATIONS',
    'TF_FEATURE_ENCODERS': 'CCA_FEATURE_ENCODERS',
    'MULTI_TF_MMD_ENCODERS': 'MULTI_CCA_RDM_ENCODERS',
    'paired_feature_backends': 'cca_feature_backends',
    '_paired_encoder_objective': '_cca_encoder_objective',
    '_paired_backend_objective': '_cca_backend_objective',
    '_prepare_research_state': '_prepare_training_state',
    'validate_research_initialization': 'validate_initialization',
    'pack_mmd_branches': 'combine_rdm_branches',
    'mmd_feature_encoder': 'rdm_feature_encoder',
    'mmd_sources': 'rdm_sources',
    'mmd_name': 'rdm_name',
    'self_normalize_virdm_loss': 'self_normalize_rdm_loss',
    'joint_feature_encoder': 'joint_rdm_encoder',
    'joint_pair_population': 'joint_cca_batch',
    'joint_pair_mode': 'joint_cca_mode',
    'joint_pair_coefficient': 'joint_cca_fixed_weight',
    'joint_weighting': 'joint_controller_mode',
    'joint_gradient_ratio': 'joint_fixed_gradient_ratio',
    'joint_mmd_representation': 'joint_rdm_representation',
    'joint_mmd_encoder_weights': 'joint_rdm_encoder_weights',
    'joint_mmd_encoder_divisor': 'joint_rdm_encoder_divisor',
    'pair_mode': 'rdm_pairing_mode',
    'pair_clean_data': 'clean_data_path',
    'pair_cache_root': 'feature_cache_root',
    'pair_control_every': 'control_every',
    'pair_report_every': 'preview_every',
    'pair_control_count': 'control_probe_count',
    'pair_report_count': 'report_probe_count',
    'pair_calibration_count': 'calibration_probe_count',
    'pair_split_seed': 'probe_split_seed',
    'pair_eval_seed': 'eval_seed',
    'pair_smoke_test': 'smoke_test',
    'pair_loss': 'cca_loss',
    'pair_norm': 'cca_norm',
    'mmd_norm': 'rdm_norm',
    'current_pair_norm': 'current_cca_norm',
    'current_mmd_norm': 'current_rdm_norm',
    'current_pair': 'current_cca',
    'current_mmd': 'current_rdm',
    'pair_coefficient': 'cca_coefficient',
    'pair_weight': 'cca_weight',
    'pair_weight_used': 'cca_weight_used',
    'pair_grad_ratio': 'cca_grad_ratio',
    'pair_gradient_combination': 'cca_gradient_combination',
    'pair_cos': 'cca_cos',
    'pair_gradient_norm_current': 'cca_gradient_norm_current',
    'mmd_gradient_norm_current': 'rdm_gradient_norm_current',
    'pair_seconds': 'cca_seconds',
    'pair_objective': 'cca_objective',
    'pair_input': 'cca_input',
    'pair_decoder': 'cca_decoder',
    'pair_encoders': 'cca_encoders',
    'pair_encoder_weights': 'cca_encoder_weights',
    'pair_timestep_weighting': 'cca_timestep_weighting',
    'pair_token_selection': 'cca_token_selection',
    'pair_to_mmd_grad_ratio': 'cca_to_rdm_grad_ratio',
    'pair_to_sf_grad_ratio': 'cca_to_rdm_grad_ratio',
    'shares_mmd_encoder': 'shares_rdm_encoder',
    'shared_mmd_branch': 'shared_rdm_branch',
    'fit_gradient_current_v2': 'gnm_rew_v1',
    'gradient_only_current_v1': 'gnm_v1',
    'fixed_loss_v1': 'fixed_cca_v1',
    'limited_pair_v1': 'scheduled_cca_v1',
    'VIRDM_RUN_NAME': 'DIRECT_FORCING_RUN_NAME',
    'VIRDM_WAN_MODEL_ROOT': 'DIRECT_FORCING_WAN_ROOT',
    'VIRDM_BASE_SEED': 'DIRECT_FORCING_BASE_SEED',
    'TF': 'CCA',
    'virdm': 'rdm',
    'tf': 'cca',
}


def canonical_identifier(name):
    if name in NAME_ALIASES:
        return NAME_ALIASES[name]
    if name.startswith('pair_lr_'):
        return 'lr_' + name[len('pair_lr_'):]
    if name.startswith('pair_resume_'):
        return 'resume_' + name[len('pair_resume_'):]
    if name.startswith('pair_cos_'):
        return 'cca_cos_' + name[len('pair_cos_'):]
    if name.startswith('virdm_'):
        return 'rdm_' + name[len('virdm_'):]
    if name.startswith('_virdm_'):
        return '_rdm_' + name[len('_virdm_'):]
    if name.startswith('VIRDM_'):
        return 'RDM_' + name[len('VIRDM_'):]
    if name.startswith('tf_'):
        return 'cca_' + name[len('tf_'):]
    name = name.replace('_virdm_', '_rdm_').replace('_tf_', '_cca_')
    if name.endswith('_tf'):
        name = name[:-3] + '_cca'
    return name


def canonical_text(value):
    if value == 'tf':
        return 'teacher_forcing'
    if value.startswith(('/', './', '../', 'https://', 'http://', 's3://')):
        return value
    value = re.sub(r'\b[A-Za-z_]\w*\b',
                   lambda m: 'RDM' if m[0] == 'ViRDM' else canonical_identifier(m[0]), value)
    # These phrases occurred in serialized controller descriptions. Normalize
    # their terminology as well, while preserving numeric/objective checks.
    for old, new in (
        ('pair loss', 'CCA loss'), ('pair norm', 'CCA norm'),
        ('MMD norm', 'RDM norm'), ('pair coefficient', 'CCA coefficient'),
        ('paired global mean', 'CCA global mean'), ('paired mean', 'CCA mean'),
        ('fit/gradient-calibrated', 'REW/GNM-calibrated'),
        ('rew/gradient-calibrated', 'REW/GNM-calibrated'),
    ):
        value = value.replace(old, new)
    return value


def canonical_metadata(value):
    """Normalize plain contracts/controllers, never parameter state dictionaries.

    Conflicting old/new spellings are rejected instead of silently overwriting
    an objective or encoder identity during migration.
    """
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = (canonical_text(key).replace('pair/', 'cca/').replace('fit/', 'rew/')
                    if isinstance(key, str) else key)
            item = canonical_metadata(item)
            if name == 'trainer' and item == 'rdm':
                item = 'direct_forcing'
            if name in result and result[name] != item:
                raise ValueError(f'Conflicting legacy and current metadata for {name}')
            result[name] = item
        return result
    if isinstance(value, (list, tuple)):
        return [canonical_metadata(item) for item in value]
    if isinstance(value, str):
        return canonical_text(value)
    return value
