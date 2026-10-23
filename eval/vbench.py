"""Generate and score the fixed expanded VBench protocol used in our experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from utils.config import ROOT, load_spec, resolve_path
from eval.protocol import validate_prompts, expected_filenames
from eval.score import DEFAULT_DIMENSIONS


def run(command, environment=None):
    import shlex
    print(shlex.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, cwd=ROOT, env=environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/chunkwise_4step.yaml')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--vbench-root', default='third_party/VBench', type=Path)
    parser.add_argument('--vbench-python', default=sys.executable)
    parser.add_argument('--cache', default='cache/vbench', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--generate-only', action='store_true')
    mode.add_argument('--score-only', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.gpus not in (1, 2, 3, 4, 8):
        raise ValueError('Use 1, 2, 3, 4, or 8 inference/scoring workers')
    spec = load_spec(args.config)
    checkpoint = resolve_path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    bench = resolve_path(args.vbench_root)
    full_info = bench / 'vbench/VBench_full_info.json'
    prompts, expanded = validate_prompts(ROOT / 'assets/prompts/all_dimension.txt',
        ROOT / 'assets/prompts/all_dimension_extended.txt', full_info)
    if args.prepare_only:
        print(json.dumps(dict(prompts=len(prompts), samples_per_prompt=5,
                             videos=len(prompts) * 5, frames=81, fps=16,
                             dimensions=DEFAULT_DIMENSIONS), indent=2))
        return
    output = resolve_path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with checkpoint.open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            digest.update(part)
    manifest = dict(checkpoint_sha256=digest.hexdigest(), frames=81, fps=16,
                    samples_per_prompt=5, prompt_count=len(prompts), seed=0,
                    prompt_sha256=hashlib.sha256('\n'.join(expanded).encode()).hexdigest())
    record = output / 'evaluation.json'
    if record.exists() and json.loads(record.read_text()) != manifest:
        raise ValueError('Output directory contains a different evaluation; choose a fresh directory')
    record.write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'original.txt').write_text('\n'.join(prompts) + '\n')
    (output / 'extended.txt').write_text('\n'.join(expanded) + '\n')
    if not args.score_only:
        run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
             f'--nproc_per_node={args.gpus}', 'inference.py', '--config', args.config,
             '--checkpoint', checkpoint, '--prompts', output / 'original.txt',
             '--extended-prompts', output / 'extended.txt', '--output', output / 'videos',
             '--samples-per-prompt', '5', '--seed', '0', '--vbench-names'])
    if args.generate_only:
        return
    missing = [name for name in expected_filenames(prompts)
               if not (output / 'videos' / name).is_file() or (output / 'videos' / name).stat().st_size == 0]
    if missing:
        raise ValueError(f'Incomplete generation: {len(missing)} missing videos; first: {missing[:3]}')
    env = dict(os.environ, PYTHONPATH=str(bench), VBENCH_CACHE_DIR=str(resolve_path(args.cache)),
               WANDB_MODE='disabled')
    # Keep the scorer's interpreter isolated from training-environment imports.
    env.pop('PYTHONHOME', None)
    run([args.vbench_python, '-m', 'torch.distributed.run', '--standalone',
         f'--nproc_per_node={args.gpus}', ROOT / 'eval/score.py', '--videos-path', output / 'videos',
         '--full-info', full_info, '--output-dir', output / 'scores', '--name', 'final',
         '--device', 'cuda', '--local-models'], env)
    run([args.vbench_python, ROOT / 'eval/summarize.py', '--input', output / 'scores',
         '--output', output / 'scores.json'], env)


if __name__ == '__main__':
    main()
