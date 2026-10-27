"""Small W&B view; full diagnostics remain in per-run JSONL files."""


TIMING_KEYS = (
    'timing/generator_update_seconds', 'timing/rollout_seconds',
    'timing/feature_and_kernel_seconds', 'timing/reference_h2d_seconds',
    'timing/mmd_kernel_backward_seconds', 'timing/feature_vjp_seconds',
    'timing/generator_backward_seconds', 'timing/cca_seconds',
    'timing/cca_target_key_seconds', 'timing/cca_target_read_seconds',
    'timing/cca_target_compute_seconds', 'timing/cca_target_write_seconds',
    'cache/cca_target_hit_rate',
)


def timing_metrics(logs):
    return {key: logs[key] for key in TIMING_KEYS if key in logs}


def training_metrics(logs):
    names = {'raw_mmd': 'train/rdm_mmd', 'cca/loss': 'train/cca_loss',
             'cca/weight': 'train/cca_weight',
             'cca/feature_gradient_ratio': 'train/cca_grad_ratio',
             'lr': 'optim/lr', 'generator_grad_norm': 'optim/grad_norm',
             'rollout_video': 'rollout/rollout_video'}
    return {**{target: logs[source] for source, target in names.items() if source in logs},
            **timing_metrics(logs)}


def evaluation_metrics(role, logs):
    if role == 'control':
        return {'control/ot': logs['control/ot_w2_squared']}
    if role == 'report':
        return {f'eval/{target}': logs[f'report/{source}'] for source, target in (
            ('ot_w2_squared', 'ot'), ('official_mmd', 'mmd'), ('variance_ratio', 'variance_ratio'))}
    raise ValueError(f'Unknown evaluation role: {role}')
