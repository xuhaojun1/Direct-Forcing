"""Explicit checkpoint forks that preserve the objective and global batches."""
import hashlib
from utils.legacy import canonical_metadata

MEMORY_RUNTIME_KEYS = ('rdm_reference_residency', 'generator_activation_cpu_offload')
MEMORY_BACKEND_KEYS = ('joint_secondary_backend', 'joint_tertiary_backend',
                       'joint_cca_backend', 'joint_cca_aux_backend', 'joint_cca_tertiary_backend')


def memory_runtime_paths(*, contract=False):
    """Copy placement may change on a fork; feature values/objectives stay strict."""
    paths = set(MEMORY_RUNTIME_KEYS)
    prefix = 'joint.' if contract else ''
    paths.update(prefix + backend + '.' + key
                 for backend in MEMORY_BACKEND_KEYS for key in MEMORY_RUNTIME_KEYS)
    return paths


def contract_differences(saved, current, prefix=()):
    if isinstance(saved, dict) and isinstance(current, dict):
        result = {}
        for key in sorted(set(saved) | set(current)):
            path = prefix + (key,)
            if key not in saved or key not in current:
                result['.'.join(path)] = (saved.get(key), current.get(key))
            else:
                result.update(contract_differences(saved[key], current[key], path))
        return result
    return {} if saved == current else {'.'.join(prefix): (saved, current)}


def validate_resume_contract(saved, current, step, *, fork=False, disable_early_stop=False):
    """A fork may repartition a global batch or change the fixed-LR stopping point.

    All loss, encoder, reference, controller, data and LR parameters stay strict.
    CPU/GPU placement of references and saved activations may change on a fork.
    Removing a checkpoint OT stop policy requires its separate explicit switch.
    No checkpoint metadata or optimizer tensors are rewritten to pass this check.
    """
    saved, current = canonical_metadata(saved), canonical_metadata(current)
    step = int(step)
    if not 0 <= step <= int(saved['max_steps']) or step > int(current['max_steps']):
        raise ValueError('Checkpoint step is outside the source/target training range')
    changes = contract_differences(saved, current)
    old_world = int(saved['rdm_expected_world_size'])
    new_world = int(current['rdm_expected_world_size'])
    if changes and not fork:
        raise ValueError('Resume contract changed; an explicit resume fork is required: ' + ', '.join(changes))
    allowed = {
        'max_steps', 'rdm_expected_world_size',
        'joint.joint_cca_backend.rdm_expected_world_size',
        'joint.joint_cca_backend.rdm_expected_local_world_size',
        'joint.joint_cca_backend.rdm_grad_accum_steps',
    }
    allowed.update(memory_runtime_paths(contract=True))
    # Secondary MMD feature configs copy launch metadata, although they do not
    # own a separate optimizer or training loop. Keep feature/loss fields strict.
    allowed.update('joint.joint_secondary_backend.'+key for key in (
        'max_steps', 'resume_from', 'resume_fork',
        'resume_disable_early_stop', 'wandb_mode',
        'rdm_expected_world_size', 'rdm_expected_local_world_size', 'rdm_grad_accum_steps'))
    if disable_early_stop:
        if not fork or current.get('joint', {}).get('joint_early_stop') is not None:
            raise ValueError('Disabling checkpoint OT stopping requires an explicit fork and no target stop policy')
        allowed.add('joint.joint_early_stop')
    unexpected = set(changes) - allowed
    if unexpected:
        raise ValueError('Resume fork changed the objective/data/LR contract: ' + ', '.join(sorted(unexpected)))
    if 'max_steps' in changes and (
            saved.get('lr_schedule') != 'cosine_window_v1'
            or current.get('lr_schedule') != 'cosine_window_v1'):
        raise ValueError('Changing the stop step requires a fixed-horizon cosine LR schedule')
    for contract, world in ((saved, old_world), (current, new_world)):
        population = int(contract['rdm_expected_global_rows'])
        if world < 1 or population % world:
            raise ValueError('Global MMD batch must divide the process count')
        joint = contract.get('joint', {})
        if joint.get('joint_cca_batch') is not None and int(joint['joint_cca_batch']) % world:
            raise ValueError('Global CCA batch must divide the process count')
        for backend in (joint.get('joint_cca_backend'), joint.get('joint_secondary_backend')):
            if backend is None:
                continue
            expected = {'rdm_expected_world_size': world,
                        'rdm_expected_local_world_size': world,
                        'rdm_grad_accum_steps': population // world}
            if any(backend.get(key) != value for key, value in expected.items()):
                raise ValueError('Feature backend repartition metadata disagrees with global batch/world size')
    return {'mode': 'fork' if fork else 'exact', 'source_step': step,
            'source_world_size': old_world, 'target_world_size': new_world,
            'source_max_steps': int(saved['max_steps']), 'target_max_steps': int(current['max_steps']),
            'changes': changes, 'optimizer': 'full Adam state reshared by FSDP',
            'controller': 'restored unchanged',
            'checkpoint_ot_stopping': 'explicitly disabled' if disable_early_stop else 'unchanged',
            'rng_policy': 'reseed_per_rank_on_repartition' if old_world != new_world else 'restore_saved_per_rank'}


def repartition_seed(seed, step, world, rank):
    if world < 1 or not 0 <= rank < world:
        raise ValueError('Invalid target rank/world size')
    value = f'wan_resume_fork_v1:{int(seed)}:{int(step)}:{int(world)}:{int(rank)}'
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little') % (2**63 - 1)
