"""Keep content identities in checkpoint contracts without machine placement."""

PLACEMENT_KEYS = {
    'asset_paths', 'data_path', 'clean_data_path', 'joint_probe_manifest',
    'resume_from', 'resume_fork', 'logdir', 'wandb_mode',
    'wandb_entity', 'wandb_project', 'wandb_host', 'wandb_key',
}


def portable_contract(value):
    """Runtime validates assets by hash; their storage paths may change on resume."""
    if isinstance(value, dict):
        return {k: portable_contract(v) for k, v in value.items()
                if k not in PLACEMENT_KEYS and not k.endswith(('_path', '_root', '_dir'))}
    if isinstance(value, (list, tuple)):
        return [portable_contract(v) for v in value]
    return value
