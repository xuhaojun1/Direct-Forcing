"""Import Causal-Forcing clean latents and cache SigLIP2 text features."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import load_spec, configure_environment, resolve_path


def source_lmdb_paths(directory):
    """Accept a merged LMDB or all official chunk-wise shards, in numeric order."""
    directory = resolve_path(directory).resolve()
    for merged in (directory, directory / 'clean_data'):
        if (merged / 'data.mdb').is_file():
            return [merged]
    shards = [directory / f'ODE6KCausal_chunkwise_{i}' for i in range(15)]
    missing = [shard.name for shard in shards if not (shard / 'data.mdb').is_file()]
    if missing:
        raise FileNotFoundError(
            f'Incomplete Causal-Forcing data under {directory}; missing chunk-wise shards: {missing}. '
            'Complete the download shown in README.md, or pass --source-lmdb with a merged LMDB.'
        )
    return shards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/chunkwise_4step.yaml')
    parser.add_argument('--set', nargs='*', default=[])
    parser.add_argument('--source-lmdb', type=Path, default=Path('data/Causal-Forcing'),
                        help='Downloaded Causal-Forcing directory or a merged latent LMDB')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--text-batch', type=int, default=64)
    args = parser.parse_args()
    spec = load_spec(args.config, args.set)
    configure_environment(spec)
    import lmdb
    import numpy as np
    import torch
    import open_clip
    from utils.clean_lmdb import WanCleanLatentDataset
    from model.encoders.text import prompt_rows_sha256
    from utils.hashing import file_sha256

    paths = spec.paths
    if Path(paths.metadata).exists():
        raise FileExistsError('Prepared metadata already exists; select a new data directory to rebuild.')
    datasets = [WanCleanLatentDataset(path) for path in source_lmdb_paths(args.source_lmdb)]
    n = sum(len(dataset) for dataset in datasets)
    minimum = 160 + int(spec.training.rdm_batch)
    if n < minimum:
        raise ValueError(f'Use at least {minimum} rows: 160 fixed probes and one complete RDM batch')
    for directory in (paths.clean_lmdb, paths.prompt_lmdb):
        Path(directory).mkdir(parents=True, exist_ok=True)
        if (Path(directory) / 'data.mdb').exists():
            raise FileExistsError(f'Refusing to replace an existing dataset: {directory}')
    clean = lmdb.open(paths.clean_lmdb, map_size=max(1 << 30, n * 21 * 16 * 60 * 104 * 2 * 2))
    prompt_db = lmdb.open(paths.prompt_lmdb, map_size=1 << 30)
    prompts, digest = [], hashlib.sha256()
    try:
        rows = (dataset[index] for dataset in datasets for index in range(len(dataset)))
        for i, row in enumerate(rows):
            prompt = row['caption']
            latent = row['wan_gt_latents'].transpose(0, 1).numpy().astype(np.float16)
            if latent.shape != (21, 16, 60, 104) or not np.isfinite(latent).all():
                raise ValueError(f'Invalid clean latent at row {i}')
            raw, caption = latent.tobytes(), prompt.encode('utf-8')
            digest.update(i.to_bytes(8, 'little')); digest.update(len(caption).to_bytes(8, 'little'))
            digest.update(caption); digest.update(raw)
            with clean.begin(write=True) as txn:
                txn.put(f'latents_{i}_data'.encode(), raw)
                txn.put(f'prompts_{i}_data'.encode(), caption)
            with prompt_db.begin(write=True) as txn:
                txn.put(f'prompts_{i}_data'.encode(), caption)
            prompts.append(prompt)
            if i % 25 == 0:
                print(f'Latents {i + 1}/{n}', flush=True)
        with clean.begin(write=True) as txn:
            txn.put(b'latents_shape', f'{n} 21 16 60 104'.encode())
            txn.put(b'prompts_shape', str(n).encode())
        with prompt_db.begin(write=True) as txn:
            txn.put(b'prompts_shape', str(n).encode())
    finally:
        clean.close(); prompt_db.close()
        for dataset in datasets:
            dataset.close()
    (Path(paths.clean_lmdb) / 'identity.json').write_text(json.dumps({'content_sha256': digest.hexdigest()}, indent=2) + '\n')
    text_model, _, _ = open_clip.create_model_and_transforms('ViT-SO400M-16-SigLIP2-256', pretrained='webli', device=args.device)
    text_model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer('ViT-SO400M-16-SigLIP2-256')
    text_rows = []
    with torch.no_grad():
        for start in range(0, n, args.text_batch):
            tokens = tokenizer(prompts[start:start + args.text_batch]).to(args.device)
            features = text_model.encode_text(tokens, normalize=True).float()
            text_rows.append(features.cpu().numpy())
    table = np.concatenate(text_rows).astype(np.float32)
    if table.shape != (n, 1152):
        raise ValueError(f'Expected SigLIP2 [N,1152], got {table.shape}')
    Path(paths.text_table).parent.mkdir(parents=True, exist_ok=True)
    np.save(paths.text_table, table)
    metadata = dict(num_rows=n, prompt_rows_sha256=prompt_rows_sha256(prompts),
                    text_table_sha256=file_sha256(paths.text_table), content_sha256=digest.hexdigest(),
                    frames=81, fps=16, height=480, width=832,
                    latent_shape=[21, 16, 60, 104], text_encoder='ViT-SO400M-16-SigLIP2-256')
    selected = random.Random(20261002).sample(range(n), 160)
    probes = dict(control_ids=selected[:32], report_ids=selected[32:96], calibration_ids=selected[96:],
                  contract=dict(text_sha256=metadata['text_table_sha256']),
                  rows=[dict(index=i, prompt=prompts[i]) for i in selected])
    Path(paths.probes).parent.mkdir(parents=True, exist_ok=True)
    Path(paths.probes).write_text(json.dumps(probes, indent=2) + '\n')
    Path(paths.metadata).write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Prepared {n} rows: {paths.metadata}', flush=True)


if __name__ == '__main__':
    main()
