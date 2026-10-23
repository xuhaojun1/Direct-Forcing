"""Four-step chunk-wise generation from a Direct-Forcing checkpoint."""
import argparse
import os
from pathlib import Path
import sys

from omegaconf import OmegaConf
from utils.config import load_spec, inference_runtime, configure_environment, resolve_path, validate_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/chunkwise_4step.yaml')
    parser.add_argument('--set', nargs='*', default=[])
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--prompts', default='assets/prompts/demo.txt', type=Path)
    parser.add_argument('--extended-prompts', type=Path)
    parser.add_argument('--output', default='outputs/samples', type=Path)
    parser.add_argument('--samples-per-prompt', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--vbench-names', action='store_true')
    parser.add_argument('--decode-offload', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    spec = load_spec(args.config, args.set)
    configure_environment(spec)
    validate_assets(spec, training=False)
    checkpoint, prompts = resolve_path(args.checkpoint), resolve_path(args.prompts)
    for path in (checkpoint, prompts):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.samples_per_prompt < 1:
        raise ValueError('samples-per-prompt must be positive')
    from utils.dataset import TextDataset
    extended = resolve_path(args.extended_prompts) if args.extended_prompts else None
    dataset = TextDataset(prompts, extended)
    config = inference_runtime(spec)
    if args.prepare_only:
        print(f'Validated 4-step chunk-wise inference: {len(dataset)} prompts, 81 RGB frames at 16 fps')
        return
    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get('RANK', 0))
    runtime_path = output / f'.inference_rank{rank}.yaml'
    OmegaConf.save(config, runtime_path)
    from pipeline.export_videos import main as export
    previous = sys.argv
    sys.argv = ['export_videos', '--config_path', str(runtime_path),
        '--checkpoint_path', str(checkpoint), '--prompt_path', str(prompts),
        '--output_folder', str(output), '--num_output_frames', '21',
        '--num_samples_per_prompt', str(args.samples_per_prompt), '--seed', str(args.seed),
        '--limit', str(args.limit), '--per_sample_seed',
        '--naming', 'raw_prompt_index' if args.vbench_names else 'prompt_index']
    if extended:
        sys.argv += ['--extended_prompt_path', str(extended)]
    if args.decode_offload:
        sys.argv += ['--offload_generator_before_decode']
    try:
        export()
    finally:
        sys.argv = previous


if __name__ == '__main__':
    main()
