"""Distributed frozen-feature extraction and reproducible Nyström construction."""
import argparse
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import load_spec, read_metadata, encoder_runtime, configure_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/chunkwise_4step.yaml')
    parser.add_argument('--encoder', required=True, choices=['videomae_v2', 'videomae_v2_l24', 'dit'])
    parser.add_argument('--set', nargs='*', default=[])
    parser.add_argument('--extract-only', action='store_true')
    args = parser.parse_args()
    spec = load_spec(args.config, args.set)
    configure_environment(spec)
    os.environ['WANDB_MODE'] = 'disabled'
    import numpy as np
    import torch
    import torch.distributed as dist
    from utils.clean_lmdb import WanCleanLatentDataset
    from utils.atomic import atomic_save
    from utils.dataset import PromptLMDBDataset
    from model.wan_adapter import WanVAEWrapper, WanTextEncoder
    from encoders.factory import build_video_encoder
    from references.bank import file_sha256, load_reference
    from encoders.text import load_frozen_text_table, prompt_rows_sha256, feature_reference_contract
    from references.builder import median_bandwidth
    from references.tokens import pool_cost_matrix, build_token_reference
    from utils.clean_data import clean_data_identity

    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    device = torch.device('cuda', local)
    if world > 1:
        dist.init_process_group('gloo', timeout=timedelta(hours=12))
    metadata = read_metadata(spec)
    cfg = encoder_runtime(spec, args.encoder, metadata, load_reference=False)
    output = Path(spec.paths.references) / args.encoder
    output.mkdir(parents=True, exist_ok=True)
    n, d, landmarks = int(cfg.rdm_expected_reference_rows), int(cfg.rdm_feature_dim), int(cfg.rdm_nystrom_landmarks)
    if (output / 'receipt.json').is_file():
        checked = encoder_runtime(spec, args.encoder, metadata, load_reference=True)
        if file_sha256(checked.rdm_reference_bundle_path) != checked.rdm_reference_bundle_sha256:
            raise ValueError('Completed reference checksum mismatch')
        if rank == 0:
            print(f'Reference already complete: {output}', flush=True)
        if world > 1:
            dist.destroy_process_group()
        return
    encoder = build_video_encoder(cfg, device, activation_checkpointing=False)
    contract = feature_reference_contract(cfg, encoder)
    contract['rdm_reference_revision'] = str(cfg.rdm_reference_revision)
    prompts_db = PromptLMDBDataset(cfg.data_path, readahead=False)
    prompts = [str(prompts_db[i]['prompts']) for i in range(len(prompts_db))]
    prompts_db.env.close()
    prompt_sha = prompt_rows_sha256(prompts)
    if len(prompts) != n or prompt_sha != cfg.rdm_joint_prompt_rows_sha256:
        raise ValueError('Prompt order differs from the prepared text table')
    text = load_frozen_text_table(cfg.rdm_joint_text_table_path,
        expected_sha256=cfg.rdm_joint_text_table_sha256, expected_rows=n)
    dataset = WanCleanLatentDataset(cfg.clean_data_path)
    if len(dataset) != n:
        raise ValueError('Clean latents and prompts have different lengths')
    latent_encoder = args.encoder == 'dit'
    decode_contract = dict(encoder=encoder.contract(), checkpoint=cfg.rdm_encoder_checkpoint_sha256,
        reference_decoder='none_normalized_clean_latent' if latent_encoder else 'original_Wan2.1_VAE_BF16',
        prompt_sha256=prompt_sha, **clean_data_identity(cfg.clean_data_path))
    checkpoint_name = 'models_t5_umt5-xxl-enc-bf16.pth' if latent_encoder else 'Wan2.1_VAE.pth'
    decode_contract['conditioning_or_vae_sha256'] = file_sha256(Path(spec.paths.wan) / checkpoint_name)
    cache_hash = hashlib.sha256(json.dumps(decode_contract, sort_keys=True).encode()).hexdigest()
    rows_dir = output / 'rows' / cache_hash[:20]
    rows_dir.mkdir(parents=True, exist_ok=True)
    pending = [i for i in range(rank, n, world) if not (rows_dir / f'{i:06d}.pt').is_file()]
    auxiliary = None
    if pending:
        auxiliary = (WanTextEncoder(dtype=torch.bfloat16, assign_load=True).to(device)
                     if latent_encoder else WanVAEWrapper().to(device=device, dtype=torch.bfloat16))
        auxiliary.eval().requires_grad_(False)
        with torch.no_grad():
            for done, index in enumerate(pending, 1):
                row = dataset[index]
                if row['caption'] != prompts[index]:
                    raise ValueError(f'Caption mismatch at row {index}')
                latent = row['wan_gt_latents'].unsqueeze(0).transpose(1, 2).to(device, torch.bfloat16)
                if latent_encoder:
                    embedding = auxiliary(text_prompts=[row['caption']])['prompt_embeds']
                    value = encoder(latent.transpose(1, 2), embedding).float().cpu()
                else:
                    rgb = (auxiliary.decode_to_pixel(latent, use_cache=False).float() * .5 + .5).clamp(0, 1)
                    value = encoder(rgb).float().cpu()
                    del rgb
                if value.shape != (1, d) or not torch.isfinite(value).all():
                    raise ValueError(f'Invalid feature at row {index}')
                atomic_save(dict(index=index, feature=value, contract_sha256=cache_hash), rows_dir / f'{index:06d}.pt')
                del latent, value
                if done == 1 or done % 25 == 0:
                    print(f'Rank {rank}: {done}/{len(pending)} features', flush=True)
    dataset.close()
    del auxiliary, dataset, encoder
    gc.collect(); torch.cuda.empty_cache()
    if world > 1:
        dist.barrier(); dist.destroy_process_group()
    if rank != 0 or args.extract_only:
        return
    pool = np.lib.format.open_memmap(output / 'joint_features_fp32.npy', mode='w+', dtype='float32', shape=(n, d + 1152))
    for i in range(n):
        row = torch.load(rows_dir / f'{i:06d}.pt', map_location='cpu', weights_only=True)
        if row['index'] != i or row['contract_sha256'] != cache_hash or row['feature'].shape != (1, d):
            raise ValueError(f'Incompatible cached row {i}')
        pool[i, :d] = row['feature'].numpy()[0]
    pool.flush()
    distances = pool_cost_matrix(pool[:, :d], device=device)
    upper = torch.triu_indices(n, n, 1)
    sigma = max(float(distances[upper[0], upper[1]].median().sqrt()), 1e-5)
    del upper
    torch.manual_seed(0)
    text_sigma = median_bandwidth(text, max_subsample=2000)
    beta = sigma / text_sigma
    pool[:, d:] = (beta * text).numpy(); pool.flush()
    distances += torch.cdist(text, text).square() * beta ** 2
    k_rr = float(torch.exp(-distances.double() / (2 * sigma * sigma)).mean())
    del distances
    def progress(**values):
        print(json.dumps(values), flush=True)
    reference_metadata = dict(contract)
    reference_metadata.update(encoder_id=cfg.rdm_encoder_id,
        encoder_checkpoint_sha256=cfg.rdm_encoder_checkpoint_sha256,
        num_rows=n, num_videos=n, nystrom_landmarks=landmarks,
        rdm_reference_revision=cfg.rdm_reference_revision, reference_decode_contract=decode_contract,
        joint_beta=beta, joint_sigma_img=sigma, joint_s_txt=text_sigma, sigma_scale=1.,
        fit_n=n, kmeans_iters=0 if landmarks == n else 20, krr_n=n, seed=0, jitter=1e-6,
        builder='streaming_fp32_kmeans_fp64_solve_v1')
    bundle = build_token_reference(pool, sigma, n_landmarks=landmarks, device=device, k_rr=k_rr, progress=progress)
    bundle.update(beta=beta, s_txt=text_sigma, sigma_scale=1., d_img=d, d_txt=1152, metadata=reference_metadata)
    reference_path = Path(cfg.rdm_reference_bundle_path)
    atomic_save(bundle, reference_path)
    loaded = load_reference(reference_path, device='cpu', cpu_offload=True,
        expected_encoder_id=cfg.rdm_encoder_id, expected_checkpoint_sha256=cfg.rdm_encoder_checkpoint_sha256,
        expected_height=480, expected_width=832, expected_rows=n,
        expected_reference_revision=cfg.rdm_reference_revision, expected_encoder_contract=contract,
        expected_landmarks=landmarks, expected_feature_dim=d + 1152,
        expected_input_mode=cfg.rdm_input_mode, expected_pool=contract['pool'], expected_rows_per_video=1)
    receipt = dict(reference_metadata, selected_token_count=cfg.rdm_token_count,
                   reference_bundle_sha256=loaded.sha256, text_table_sha256=cfg.rdm_joint_text_table_sha256,
                   prompt_rows_sha256=prompt_sha, feature_rows=str(rows_dir.relative_to(output)))
    temp = output / 'receipt.json.tmp'
    temp.write_text(json.dumps(receipt, indent=2) + '\n'); temp.replace(output / 'receipt.json')
    print(f'Completed reference: {reference_path}', flush=True)


if __name__ == '__main__':
    main()
