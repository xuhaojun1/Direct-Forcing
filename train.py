"""Train Direct-Forcing with a portable YAML configuration under torchrun."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import traceback

from omegaconf import OmegaConf
from utils.config import load_spec, compile_runtime, configure_environment, resolve_path, validate_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/chunkwise_4step.yaml')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE')
    parser.add_argument('--prepare-only', action='store_true', help='Validate inputs and configuration without allocating a model')
    parser.add_argument('--resume', type=Path, help='Checkpoint directory containing model.pt and training.pt')
    parser.add_argument('--output', type=Path, help='Explicit fresh run directory')
    parser.add_argument('--no-visualize', action='store_true')
    args = parser.parse_args()
    spec = load_spec(args.config, args.set)
    configure_environment(spec)
    config = compile_runtime(spec)
    validate_assets(spec)
    from trainer.joint import DirectForcingTrainer
    from trainer.multi_encoder import MultiEncoderTrainer
    cls = MultiEncoderTrainer if config.get('joint_secondary_backend') is not None else DirectForcingTrainer
    cls._validate_static_config(config)
    if args.resume:
        source = resolve_path(args.resume).resolve()
        for name in ('model.pt', 'training.pt'):
            if not (source / name).is_file():
                raise FileNotFoundError(source / name)
        config.resume_from, config.resume_fork = str(source), True
    if args.prepare_only:
        print(json.dumps(dict(status='validated', cca=spec.features.cca,
            rdm=list(spec.features.rdm), steps=config.max_steps,
            global_batches=dict(cca=config.joint_cca_batch, rdm=config.rdm_expected_global_rows),
            controller=config.joint_controller_mode, reference_residency=config.rdm_reference_residency), indent=2))
        return
    import torch
    import torch.distributed as dist
    import wandb
    world, rank = int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('RANK', 0))
    if world != spec.training.world_size:
        raise ValueError(f'Launch world size {world} != training.world_size {spec.training.world_size}')
    # The elastic run ID is shared by workers and avoids timestamp races.
    run_id = os.environ.get('TORCHELASTIC_RUN_ID', '')
    if not run_id or run_id == 'none':
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = resolve_path(args.output) if args.output else Path(spec.paths.output) / run_id
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        OmegaConf.save(spec, output / 'config.yaml')
        OmegaConf.save(config, output / 'launch_config.yaml')
    config.logdir = str(output)
    config.wandb_save_dir = str(output / 'wandb')
    config.config_name = spec.name
    config.no_save, config.no_visualize = False, args.no_visualize
    config.disable_wandb = spec.logging.wandb_mode == 'disabled'
    os.environ['DIRECT_FORCING_RUN_NAME'] = f'{spec.name}_{run_id}'
    failed = True
    try:
        trainer = cls(config)
        trainer.train()
        failed = False
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        wandb.finish(exit_code=1 if failed else 0)
        if not failed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
