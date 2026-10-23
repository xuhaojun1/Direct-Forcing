"""Import Causal-Forcing latents or encode captioned videos, then cache SigLIP2 text."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import load_spec, configure_environment, resolve_path


def read_manifest(path):
    path = Path(path)
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
            raise ValueError('Every manifest row needs a nonempty prompt')
        video = Path(row['video']).expanduser()
        row['video'] = str((path.parent / video).resolve() if not video.is_absolute() else video)
        if not Path(row['video']).is_file():
            raise FileNotFoundError(row['video'])
        rows.append(row)
    return rows


def decode_video(row):
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    cap = cv2.VideoCapture(row['video'])
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or count < 1:
        raise ValueError(f'Cannot read video metadata: {row["video"]}')
    start = float(row.get('start_seconds', 0.))
    if start < 0 or count / fps < start + 5.0 - 1 / fps:
        raise ValueError('Videos must cover the requested five-second interval')
    indices = np.minimum(np.rint((start + np.arange(81) / 16) * fps).astype(int), count - 1)
    frames = []
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Could not decode frame {index}: {row["video"]}')
            frame = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
            h, w = frame.shape[-2:]
            scale = max(480 / h, 832 / w)
            frame = F.interpolate(frame[None], size=(round(h * scale), round(w * scale)),
                                  mode='bilinear', align_corners=False, antialias=True)[0]
            y, x = (frame.shape[-2] - 480) // 2, (frame.shape[-1] - 832) // 2
            frames.append(frame[:, y:y + 480, x:x + 832])
    finally:
        cap.release()
    return torch.stack(frames, dim=1)[None] * 2 - 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/chunkwise_4step.yaml')
    parser.add_argument('--set', nargs='*', default=[])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest', type=Path, help='JSONL with video and prompt; video paths are relative to the manifest')
    source.add_argument('--source-lmdb', type=Path, help='Existing normalized Causal-Forcing latent LMDB')
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
    from encoders.text import prompt_rows_sha256
    from references.bank import file_sha256

    paths = spec.paths
    if Path(paths.metadata).exists():
        raise FileExistsError('Prepared metadata already exists; select a new data directory to rebuild.')
    dataset = WanCleanLatentDataset(resolve_path(args.source_lmdb)) if args.source_lmdb else None
    manifest = None if dataset else read_manifest(resolve_path(args.manifest))
    n = len(dataset) if dataset else len(manifest)
    minimum = 160 + int(spec.training.rdm_batch)
    if n < minimum:
        raise ValueError(f'Use at least {minimum} rows: 160 fixed probes and one complete RDM batch')
    for directory in (paths.clean_lmdb, paths.prompt_lmdb):
        Path(directory).mkdir(parents=True, exist_ok=True)
        if (Path(directory) / 'data.mdb').exists():
            raise FileExistsError(f'Refusing to replace an existing dataset: {directory}')
    clean = lmdb.open(paths.clean_lmdb, map_size=max(1 << 30, n * 21 * 16 * 60 * 104 * 2 * 2))
    prompt_db = lmdb.open(paths.prompt_lmdb, map_size=1 << 30)
    vae = None
    if dataset is None:
        from model.wan_adapter import WanVAEWrapper
        vae = WanVAEWrapper().to(args.device, dtype=torch.bfloat16).eval().requires_grad_(False)
    prompts, digest = [], hashlib.sha256()
    try:
        for i in range(n):
            if dataset:
                row = dataset[i]
                prompt = row['caption']
                latent = row['wan_gt_latents'].transpose(0, 1).numpy().astype(np.float16)
            else:
                prompt = manifest[i]['prompt']
                with torch.no_grad():
                    latent = vae.encode_to_latent(decode_video(manifest[i]).to(args.device, torch.bfloat16))[0].cpu().numpy().astype(np.float16)
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
        if dataset:
            dataset.close()
        del vae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
