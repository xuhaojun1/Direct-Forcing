"""VideoMAE + DiT token supervision, sharing one Wan generator and rollout.

Each MMD keeps its original pinned reference, text scale, and self-normalization.
Separate encoder input VJPs meet at the latent, before a single generator replay.
"""

from contextlib import contextmanager
import hashlib
import json
import time

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
import wandb

from trainer.joint import DirectForcingTrainer
from trainer.rdm import RDMTrainer
from utils.transport import exact_ot, squared_cost
from trainer.losses.ensemble import combine_rdm_branches
from model.encoders.config import feature_runtime_config


class MultiEncoderTrainer(DirectForcingTrainer):
    branch_names = ('dit', 'videomae')
    runtime_keys = ('config', 'rdm_encoder', 'latent_feature_encoder',
                    'rdm_reference', 'joint_text_beta')
    probe_keys = ('gt_cache', 'probe_gt', 'gt_variance', 'probe_sigma2', 'null_ot')

    @staticmethod
    def _validate_static_config(config):
        DirectForcingTrainer._validate_static_config(config)
        secondary = config.joint_secondary_backend
        DirectForcingTrainer._validate_static_config(secondary)
        if (config.joint_rdm_encoder != 'videomae_v2_dit'
                or config.rdm_encoder_id != 'wan21_14b_block8_bid_clean_tau300_pooled'
                or secondary.rdm_encoder_id != 'videomae_v2_vitl_l15_5x16_224'
                or config.joint_encoder_combination != 'independent_cca_and_mean_self_normalized_mmd_v1'
                or config.get('joint_cca_backend') is None
                or config.rdm_dit_timestep != 300):
            raise ValueError('The main recipe uses an independent CCA encoder and V16 + D9 RDM')
        if config.get('joint_tertiary_backend') is not None:
            raise ValueError('This release supports two RDM encoders')
        for key in ('rdm_joint_text_table_sha256', 'rdm_joint_prompt_rows_sha256',
                    'rdm_expected_reference_rows', 'clean_data_path', 'joint_cca_batch',
                    'rdm_expected_global_rows', 'rdm_taew_checkpoint_sha256'):
            if config[key] != secondary[key]:
                raise ValueError(f'Encoder branches disagree on {key}')


    def __init__(self, config):
        self.branch_names = ('dit', 'videomae')
        super().__init__(config)


    @contextmanager
    def _feature_branch(self, name):
        settings = self.feature_branches[name]
        saved = {key: getattr(self, key, None) for key in settings}
        try:
            for key, value in settings.items():
                setattr(self, key, value)
            yield
        finally:
            for key, value in saved.items():
                setattr(self, key, value)

    def _prepare_training_state(self, config):
        # Called after the primary encoder, decoder and generator have loaded,
        # before the checkpoint's complete training contract is validated.
        primary_name = self.branch_names[0]
        self.feature_branches = {primary_name: {key: getattr(self, key) for key in self.runtime_keys}}
        sources = [config.joint_secondary_backend]
        for name, source in zip(self.branch_names[1:], sources):
            runtime = feature_runtime_config(config, source)
            encoder, is_latent, reference, beta = self._load_feature_backend(runtime)
            self.feature_branches[name] = dict(config=runtime, rdm_encoder=encoder,
                latent_feature_encoder=is_latent, rdm_reference=reference, joint_text_beta=beta)
        super()._prepare_training_state(config)
        self.feature_branches[primary_name].update({key: getattr(self, key) for key in self.probe_keys})
        for name in self.branch_names[1:]:
            secondary_config = self.feature_branches[name]['config']
            encoder = self.feature_branches[name]['rdm_encoder']
            cache_contract = dict(self.contract['gt_cache_contract'])
            cache_contract.update(encoder=secondary_config.rdm_encoder_checkpoint_sha256,
                decoder=secondary_config.rdm_taew_checkpoint_sha256,
                encoder_contract=encoder.contract(), input='normalized clean Wan latent; TAEW; frozen video encoder')
            cache_contract.pop('retained_pool', None)
            cache_key = hashlib.sha256(json.dumps(cache_contract, sort_keys=True).encode()).hexdigest()[:20]
            from pathlib import Path
            cache = Path(secondary_config.feature_cache_root)/cache_key
            cache.mkdir(parents=True, exist_ok=True)
            self.feature_branches[name]['gt_cache'] = cache
            if self.is_main_process:
                (cache/'contract.json').write_text(json.dumps(cache_contract, indent=2)+'\n')
            selected = self.control_ids + self.report_ids + self.calibration_ids
            with self._feature_branch(name):
                for index in selected[self.rank::self.world_size]:
                    self._gt_feature(index)
                dist.barrier(group=self.io_group)
                features = {index: self._gt_feature(index) for index in selected}
            calibration = torch.cat([features[i] for i in self.calibration_ids]).double().numpy()
            variance = float(np.square(calibration-calibration.mean(0)).sum()/len(calibration))
            costs = squared_cost(calibration, calibration)
            sigma2 = float(np.median(costs[np.triu_indices(len(calibration), 1)]))
            if variance <= 0 or sigma2 <= 0:
                raise ValueError('Degenerate secondary-encoder GT calibration')
            half = len(calibration)//2
            self.feature_branches[name].update(probe_gt=features, gt_variance=variance,
                probe_sigma2=sigma2, null_ot=exact_ot(calibration[:half], calibration[half:])[0])

    def _extend_contract(self, config):
        super()._extend_contract(config)
        self.contract.update(
            dual_feature_version=4,
            mmd_objective='arithmetic mean of independently self-normalized per-encoder MMD',
            feature_encoders={name: branch['rdm_encoder'].contract()
                              for name, branch in self.feature_branches.items()},
            reference_sha256_by_encoder={name: branch['rdm_reference'].sha256
                                         for name, branch in self.feature_branches.items()},
            ot_cost='reported separately in each original feature space; not averaged across encoders',
            cca_stop_step=config.get('joint_cca_stop_step'),
            evaluation_sampling='same fixed prompts/noise, one deterministic evaluation pass per encoder')
        cca_contract = OmegaConf.to_container(config.joint_cca_feature_contract, resolve=True)
        self.contract.update(
            cca_objective=cca_contract['encoder_id']+'; '+cca_contract['loss'],
            cca_encoders=[config.joint_cca_encoder],
            cca_feature_encoder=cca_contract,
            mmd_encoder_weights={name: 1. for name in self.branch_names},
            mmd_encoder_divisor=float(len(self.branch_names)))

    def _branch_weights(self):
        configured = self.config.get('joint_rdm_encoder_weights')
        weights = [float(configured[n]) for n in self.branch_names] if configured else [1.] * len(self.branch_names)
        divisor = float(self.config.get('joint_rdm_encoder_divisor', len(self.branch_names)))
        return weights, divisor

    def _mmd_feature_gradients(self, chunks):
        values, gradients, records = [], [], {}
        for name in self.branch_names:
            with self._feature_branch(name):
                v, g, logs, _ = RDMTrainer._mmd_feature_gradients(self, chunks)
            values.append(v)
            gradients.append(g)
            records[name] = logs
        weights, divisor = self._branch_weights()
        packed_values, packed_gradients = combine_rdm_branches(values, gradients, weights, divisor)
        # The branch coefficient has already been applied to each feature VJP.
        logs = dict(records[self.branch_names[0]])
        for key in ('raw_mmd', 'normalized_mmd', 'k_gg', 'k_gr_nystrom', 'k_rr'):
            logs[key] = sum(w*records[name][key] for name, w in zip(self.branch_names, weights))/divisor
        # Branches execute sequentially: time is summed, never loss-weighted.
        for key in ('timing/reference_h2d_seconds', 'timing/mmd_kernel_backward_seconds'):
            logs[key] = sum(records[name][key] for name in self.branch_names)
        for name in self.branch_names:
            logs.update({f'{key}/{name}': value for key, value in records[name].items()})
        return packed_values, packed_gradients, logs, None

    def _feature_vjp_to_latent(self, chunk, feature_gradient, pass1_features):
        dimensions = [int(self.feature_branches[n]['rdm_reference'].feature_dim) for n in self.branch_names]
        gradients = feature_gradient.split(dimensions, dim=-1)
        cached = pass1_features.split(dimensions, dim=-1)
        combined, maximum_parity = None, 0.
        for name, gradient, features in zip(self.branch_names, gradients, cached):
            with self._feature_branch(name):
                latent_gradient, parity = RDMTrainer._feature_vjp_to_latent(self, chunk, gradient, features)
            combined = latent_gradient if combined is None else combined + latent_gradient
            maximum_parity = max(maximum_parity, parity)
        return combined, maximum_parity

    def _assert_frozen_parameter_grads(self):
        for name in self.branch_names:
            with self._feature_branch(name):
                super()._assert_frozen_parameter_grads()

    @torch.no_grad()
    def _evaluate(self, role):
        evaluation_start = time.perf_counter()
        # Reuse the validated single-space evaluator. Seeds/prompts are identical;
        # generation is repeated, not trained, between the two feature passes.
        run_path, disabled = self.run_path, self.disable_wandb
        metrics_by_encoder, logs = {}, {}
        try:
            self.disable_wandb = True
            for name in self.branch_names:
                self.run_path = run_path/'feature_eval'/name
                self.run_path.mkdir(parents=True, exist_ok=True)
                with self._feature_branch(name):
                    super()._evaluate(role)
                if self.is_main_process:
                    path = self.run_path/'evaluation'/f'{role}_step{self.step:06d}.json'
                    metrics_by_encoder[name] = json.loads(path.read_text())
        finally:
            self.run_path, self.disable_wandb = run_path, disabled
        if self.is_main_process:
            for name, metrics in metrics_by_encoder.items():
                prefix = 'eval' if role == 'report' else 'control'
                logs[f'{prefix}/{name}_ot'] = metrics['empirical_w2_squared']
                logs[f'{prefix}/{name}_mmd'] = metrics['official_joint_nystrom_mmd']['raw_biased_mmd2']
                if role == 'report':
                    logs[f'eval/{name}_temporal_ot'] = metrics['temporal_ot']
            if role == 'report':
                primary = metrics_by_encoder[self.branch_names[0]]
                weights, divisor = self._branch_weights()
                logs.update({
                    'eval/mmd': sum(w*logs[f'eval/{n}_mmd'] for n, w in zip(self.branch_names, weights))/divisor,
                    'eval/chunk_jump_ratio': primary['boundary']['rgb']['relative_jump_ratio'],
                    'eval/interior_motion_ratio': primary['boundary']['rgb']['interior_motion_ratio'],
                })
                media = primary.get('visualizations', [])
                if media:
                    logs['eval/cca_cos'] = float(np.mean([m['cca_cos'] for m in media]))
                    if not self.disable_wandb:
                        for position, item in enumerate(media, 1):
                            logs[f'samples/prompt_{position:02d}'] = wandb.Video(item['path'],
                                caption=f"Left: GT | Middle: CCA reconstruction | Right: free AR ({self._inference_step_label()}). step={self.step}. {item['prompt']}")
            logs[f'timing/eval_{role}_seconds'] = time.perf_counter()-evaluation_start
            numeric = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            target = run_path/'evaluation'/f'{role}_step{self.step:06d}.json'
            target.parent.mkdir(exist_ok=True)
            target.write_text(json.dumps({'by_encoder': metrics_by_encoder, 'metrics': numeric}, indent=2,
                                         allow_nan=False)+'\n')
            with (run_path/'evaluation.jsonl').open('a') as stream:
                stream.write(json.dumps({'step': self.step, **numeric}, allow_nan=False)+'\n')
            if not self.disable_wandb:
                wandb.log(logs, step=self.step)
            self._event('evaluation_complete', role=role, step=self.step, **numeric)
        dist.barrier(group=self.io_group)
