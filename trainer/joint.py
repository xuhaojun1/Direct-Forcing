"""Joint CCA + RDM updates with gradient norm matching and residual weighting."""

from contextlib import nullcontext
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import time

# Shared data, evaluation, and checkpoint runtime.
from trainer.runtime import TrainingRuntime, atomic_save
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
import wandb

from trainer.controller import (
    GNMREWController, FixedCCAController, GNMController, ScheduledCCAController,
    GNM_REW_KIND, FIXED_CCA_KIND, GNM_KIND)
from trainer.losses.cca import CCAFeatureLoss, token_cosine
from utils.training_schedule import training_batch
from utils.wan_boundary_metrics import boundary_statistics
from utils.transport import exact_ot
from utils.logging_metrics import timing_metrics


def sample_cca_chunk_timesteps(levels, *, first_chunk_levels=None, batch_size=1,
                             num_chunks=7, frames_per_chunk=3):
    """Use the generator's warped training grids, sharing time within each chunk."""
    choices = torch.randint(len(levels), (batch_size, num_chunks), device=levels.device)
    sampled = levels[choices]
    if first_chunk_levels is not None and len(first_chunk_levels):
        first_chunk_levels = first_chunk_levels.to(levels.device)
        first_choices = torch.randint(len(first_chunk_levels), (batch_size,), device=levels.device)
        sampled[:, 0] = first_chunk_levels[first_choices]
    return sampled.repeat_interleave(frames_per_chunk, dim=1)


class DirectForcingTrainer(TrainingRuntime):
    """Direct-Forcing: one causal generator jointly updated by CCA and RDM.

    RDM's staged backward is inherited from RDMTrainer. This class computes
    CCA gradients, applies GNM and REW, and combines both before one update.
    """
    @staticmethod
    def _validate_static_config(config):
        TrainingRuntime._validate_static_config(config)
        if config.rdm_pairing_mode != 'none' or config.sharding_strategy != 'full':
            raise ValueError('Joint training requires plain MMD and FULL_SHARD')
        if config.joint_cca_mode != 'teacher_forcing':
            raise ValueError('Only teacher forcing performs pairing')
        if config.get('joint_cca_aux_backend') is not None or config.get('joint_cca_tertiary_backend') is not None:
            raise ValueError('This release uses one CCA encoder')
        objective = config.get('joint_cca_objective', 'feature')
        if objective != 'feature':
            raise ValueError('Unsupported CCA objective')
        stop = config.get('joint_cca_stop_step')
        if stop is not None and (config.joint_cca_mode != 'teacher_forcing' or not 1 <= int(stop) <= int(config.max_steps)):
            raise ValueError('CCA stop step must be within the CCA training schedule')
        if config.get('joint_controller_mode') not in {GNM_REW_KIND, FIXED_CCA_KIND, GNM_KIND}:
            raise ValueError('Use GNM + REW, GNM, or fixed CCA weighting')
        if config.joint_controller_mode == FIXED_CCA_KIND:
            FixedCCAController(float(config.joint_cca_fixed_weight))
        elif config.joint_controller_mode == GNM_KIND:
            GNMController(target_ratio=float(config.get('joint_fixed_gradient_ratio', 1.)))
        else:
            GNMREWController(ratio_start=float(config.joint_ratio_start),
                           ratio_min=float(config.joint_ratio_min))
        world = int(config.rdm_expected_world_size)
        if int(config.joint_cca_batch) % world or int(config.joint_cca_batch) < world:
            raise ValueError('CCA batch must divide world size')
        if int(config.joint_cca_batch) > int(config.rdm_expected_global_rows):
            raise ValueError('Pairing uses a subset of this update\'s SF prompts')
        if config.independent_first_frame or int(config.num_frame_per_block) not in (1, 3):
            raise ValueError('Full CCA requires one- or three-latent-frame blocks')
        if not config.gradient_checkpointing:
            raise ValueError('Enable student activation checkpointing for full CCA')
        visual = int(config.joint_visual_count)
        if visual < world or visual % world or visual > int(config.report_probe_count):
            raise ValueError('Visual prompt count must divide world size and fit report set')

    def __init__(self, config):
        super().__init__(config)
        self.cca_objective = self._build_cca_objective(config)
        self.cca_accum = int(config.joint_cca_batch)//self.world_size
        self.model.generator.enable_gradient_checkpointing()

    def _build_cca_objective(self, config):
        source = config.get('joint_cca_backend')
        if source is not None:
            return self._cca_backend_objective(source)
        return self._cca_encoder_objective(self.rdm_encoder, self.latent_feature_encoder)

    def _cca_encoder_objective(self, encoder, latent):
        encoder.eval().requires_grad_(False)
        if latent:
            from trainer.losses.cca_dit import DiTCCALoss
            return DiTCCALoss(encoder)
        return CCAFeatureLoss(encoder, self.taew_decoder)

    def _cca_backend_objective(self, source):
        # Reuse a frozen encoder when CCA and RDM select the same feature space.
        latent = source.rdm_input_mode == 'normalized_wan_latent'
        primary = getattr(self, 'config', None)
        if (primary is not None and primary.rdm_encoder_id == source.rdm_encoder_id
                and primary.rdm_encoder_checkpoint_sha256 == source.rdm_encoder_checkpoint_sha256):
            return self._cca_encoder_objective(self.rdm_encoder, latent)
        for branch in getattr(self, 'feature_branches', {}).values():
            resident = branch['config']
            if (resident.rdm_encoder_id == source.rdm_encoder_id
                    and resident.rdm_encoder_checkpoint_sha256 == source.rdm_encoder_checkpoint_sha256):
                return self._cca_encoder_objective(branch['rdm_encoder'], latent)
        from model.encoders.factory import build_video_encoder
        from utils.hashing import file_sha256
        if not latent and file_sha256(source.rdm_encoder_checkpoint_path) != source.rdm_encoder_checkpoint_sha256:
            raise ValueError('Independent CCA encoder checkpoint checksum mismatch')
        encoder = build_video_encoder(source, self.device)
        if latent and encoder.checkpoint_sha256 != source.rdm_encoder_checkpoint_sha256:
            raise ValueError('Independent CCA encoder checkpoint checksum mismatch')
        return self._cca_encoder_objective(encoder, latent)

    def _inference_step_label(self):
        later = len(self.config.denoising_step_list)
        first = len(self.config.get('first_frame_denoising_step_list', [])) or later
        return f'{later}-step' if first == later else f'first block {first}-step, later blocks {later}-step'

    def _cca_feature_kwargs(self, conditional):
        return ({'embedding': conditional['prompt_embeds']}
                if getattr(self.cca_objective, 'requires_embedding', False) else {})

    def _cca_target_caches(self):
        """Lazy setup also covers objectives replaced by dual trainers."""
        if not self.config.get('cca_target_cache', False) or self.cca_objective is None:
            return []
        if getattr(self, '_cca_cache_objective', None) is self.cca_objective:
            return self._cca_cache_instances
        from trainer.references.cca_cache import CCATargetTokenCache
        from trainer.losses.cca_dit import DiTCCALoss
        config = OmegaConf.to_container(self.config, resolve=True)
        caches = []
        for module in self.cca_objective.modules():
            if not isinstance(module, (CCAFeatureLoss, DiTCCALoss)):
                continue
            if module.target_cache is None:
                sources = [type(module), type(module.encoder)]
                backbone = getattr(module.encoder, 'backbone', module.encoder)
                sources.append(type(backbone))
                if isinstance(module, CCAFeatureLoss):
                    sources.append(type(module.decoder))
                    sources.append(type(module.decoder.decoder))
                encoder_config = getattr(module.encoder, 'config', self.config)
                contract = dict(encoder=backbone.contract(),
                    encoder_weights=encoder_config.rdm_encoder_checkpoint_sha256,
                    decoder_weights=config['rdm_taew_checkpoint_sha256']
                        if isinstance(module, CCAFeatureLoss) else None,
                    tokens='all_paired_tokens_no_pooling_no_quantization', compute=str(self.dtype),
                    torch_version=str(torch.__version__),
                    implementation={str(cls): hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
                                    for cls in sources})
                module.target_cache = CCATargetTokenCache(
                    config.get('cca_target_cache_root') or Path(config['feature_cache_root']).parent/'cca_full_tokens', contract)
            caches.append(module.target_cache)
        self._cca_cache_objective, self._cca_cache_instances = self.cca_objective, caches
        return caches

    def _make_controller(self, config):
        if config.joint_controller_mode == FIXED_CCA_KIND:
            controller = FixedCCAController(float(config.joint_cca_fixed_weight))
        elif config.joint_controller_mode == GNM_KIND:
            controller = GNMController(target_ratio=float(config.get('joint_fixed_gradient_ratio', 1.)))
        else:
            controller = GNMREWController(ratio_start=float(config.joint_ratio_start), ratio_min=float(config.joint_ratio_min))
        stop = config.get('joint_cca_stop_step')
        return ScheduledCCAController(controller, int(stop)) if stop is not None else controller

    def _controller_from_state_dict(self, state):
        from utils.legacy import canonical_metadata
        state = canonical_metadata(state)
        stop = self.config.get('joint_cca_stop_step')
        if stop is not None:
            controller = ScheduledCCAController.from_state_dict(state)
            if controller.stop_step != int(stop) or controller.last_step != self.step:
                raise ValueError('Stopped-CCA schedule/step changed across resume')
            if controller.base.state_dict()['kind'] != self.config.joint_controller_mode:
                raise ValueError('Controller kind changed across resume')
            if self.config.joint_controller_mode == FIXED_CCA_KIND and (
                    controller.base.weight != float(self.config.joint_cca_fixed_weight)):
                raise ValueError('Fixed CCA coefficient changed across resume')
            if self.config.joint_controller_mode == GNM_KIND and (
                    controller.base.target_ratio != float(self.config.get('joint_fixed_gradient_ratio', 1.))):
                raise ValueError('Fixed gradient ratio changed across resume')
            return controller
        if state.get('kind') != self.config.joint_controller_mode:
            raise ValueError('Controller kind changed across resume')
        if self.config.joint_controller_mode == FIXED_CCA_KIND:
            controller = FixedCCAController.from_state_dict(state)
            if controller.weight != float(self.config.joint_cca_fixed_weight):
                raise ValueError('Fixed CCA coefficient changed across resume')
        elif self.config.joint_controller_mode == GNM_KIND:
            controller = GNMController.from_state_dict(state)
            if controller.target_ratio != float(self.config.get('joint_fixed_gradient_ratio', 1.)):
                raise ValueError('Fixed gradient ratio changed across resume')
        else:
            controller = GNMREWController.from_state_dict(state)
        if controller.last_step != self.step:
            raise ValueError('Weighting state and optimizer checkpoint steps differ')
        return controller

    def _select_probe_ids(self, config, sizes):
        if config.smoke_test:
            return super()._select_probe_ids(config, sizes)
        path = Path(config.joint_probe_manifest)
        manifest = json.loads(path.read_text())
        self.probe_manifest_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        roles = [manifest[key] for key in ('control_ids', 'report_ids', 'calibration_ids')]
        all_ids = sum(roles, [])
        if [len(ids) for ids in roles] != sizes or len(set(all_ids)) != len(all_ids):
            raise ValueError('Invalid/overlapping fixed probe splits')
        if manifest['contract']['text_sha256'] != config.rdm_joint_text_table_sha256:
            raise ValueError('Probe manifest text table differs from training table')
        for row in manifest['rows']:
            if self.dataset[row['index']]['prompts'] != row['prompt']:
                raise ValueError('Probe manifest caption changed')
        return roles

    def _extend_contract(self, config):
        if len(config.denoising_step_list) != 4:
            schedule_name = ('heterogeneous_chunk_denoising' if list(config.rdm_first_frame_denoising_step_list)
                             else 'uniform_chunk_denoising')
            self.contract[schedule_name] = dict(
                timesteps=list(config.denoising_step_list),
                first_chunk_timesteps=list(config.rdm_first_frame_denoising_step_list),
                warp=bool(config.warp_denoising_step),
                timestep_shift=float(config.model_kwargs.timestep_shift))
        self.contract.update(joint_version=6,
            joint={k: v for k, v in OmegaConf.to_container(config, resolve=True).items()
                   if k.startswith('joint_') and not (config.smoke_test and k == 'joint_probe_manifest')},
            cca_objective='VJEPA final-LN token cosine',
            cca_input=('GT history and per-chunk four-step GT noising' if len(config.denoising_step_list) == 4
                        else f'GT history and per-chunk {len(config.denoising_step_list)}-step GT noising'),
            combination='one Adam update; CCA global mean; MMD coefficient one; current-step gradient calibration',
            control='current training CCA loss / first global training CCA loss sets target gradient ratio; OT is evaluation only',
            weighting=GNM_REW_KIND,
            weighting_input='globally averaged unweighted training CCA loss and global FULL_SHARD parameter gradient norms',
            evaluation_every=int(config.preview_every))
        if config.joint_controller_mode == FIXED_CCA_KIND:
            self.contract.update(
                weighting=FIXED_CCA_KIND,
                weighting_input='constant joint_cca_fixed_weight; no loss or gradient feedback',
                control='fixed CCA coefficient; OT and gradient statistics are evaluation/logging only',
                combination='one Adam update; official self-normalized MMD plus fixed coefficient times CCA global mean',
                mmd_self_normalization='raw / (stopgrad(abs(raw)) + eps); unchanged official RDM')
        elif config.joint_controller_mode == GNM_REW_KIND:
            self.contract.update(
                weighting=GNM_REW_KIND,
                weighting_input='current global mean training CCA loss and current global FULL_SHARD gradient norms; no EMA',
                control='current CCA loss / first CCA loss sets target; coefficient = target * current RDM norm / current CCA norm',
                combination='one Adam update; official self-normalized MMD plus current-step REW/GNM-calibrated CCA mean',
                mmd_self_normalization='raw / (stopgrad(abs(raw)) + eps); unchanged official RDM')
        elif config.joint_controller_mode == GNM_KIND:
            self.contract.update(
                weighting=GNM_KIND,
                weighting_input='current global FULL_SHARD generator parameter gradient norms; no loss feedback or EMA',
                control='fixed target gradient ratio 1; CCA coefficient = current normalized-RDM norm / max(current CCA norm, 1e-12)',
                combination='one Adam update; self-normalized MMD plus gradient-calibrated CCA global mean',
                mmd_self_normalization='raw / (stopgrad(abs(raw)) + eps); unchanged official RDM')
            if float(config.get('joint_fixed_gradient_ratio', 1.)) != 1.:
                self.contract.update(
                    fixed_gradient_ratio=float(config.joint_fixed_gradient_ratio),
                    control='fixed target gradient ratio r; CCA coefficient = r * current normalized-RDM norm / max(current CCA norm, 1e-12)')
        if not config.smoke_test:
            self.contract['probe_manifest_sha256'] = self.probe_manifest_sha256
        if config.rdm_encoder_id == 'videomae_v2_vitl_l15_5x16_224':
            self.contract['cca_objective'] = 'VideoMAE v2 block15 full-token cosine, 5x16 frames after frame0'
        if config.rdm_encoder_id == 'videomae_v2_vitl_l23_5x16_224':
            self.contract['cca_objective'] = 'VideoMAE v2 block23 full-token cosine, 5x16 frames after frame0'
        if self.latent_feature_encoder:
            self.contract['cca_objective'] = ('Frozen bidirectional Wan14B block8 raw-token cosine; clean latent; tau'
                                              +str(config.get('rdm_dit_timestep', 300)))
        self.contract.update(feature_encoder=self.rdm_encoder.contract(),
            mmd_representation='fixed ordered token matrix; no global pooling; no optical-flow loss',
            ot_cost='Frobenius squared / selected token count; raw Frobenius W2^2 = K * reported OT',
            cca_token_selection='all original tokens, unchanged')
        if config.get('rdm_loss_normalization') == 'elastic_raw_v1':
            self.contract.update(
                mmd_self_normalization='raw / (stopgrad(raw) + 1e-4); Elastic',
                combination='one Adam update; Elastic-normalized MMD plus weighted CCA global mean')
        if config.get('joint_cca_feature_contract') is not None:
            cca_contract = OmegaConf.to_container(config.joint_cca_feature_contract, resolve=True)
            self.contract.update(joint_version=7, cca_feature_encoder=cca_contract,
                cca_objective=cca_contract['encoder_id']+'; '+cca_contract['loss'],
                rdm_feature_encoder=self.rdm_encoder.contract())

    def _training_metrics(self, logs):
        names = {'raw_mmd': 'train/rdm_mmd', 'cca/loss': 'train/cca_cos',
            'cca/active': 'mix/cca_active',
            'cca/coefficient': 'mix/cca_coefficient',
            'cca/target_gradient_ratio': 'mix/target_grad_ratio',
            'rew/relative_error': 'rew/relative_error',
            'joint/weighted_gradient_ratio': 'mix/cca_to_rdm_grad_ratio',
            'lr': 'optim/lr'}
        result = {target: logs[source] for source, target in names.items() if source in logs}
        for branch in ('dit', 'videomae', 'videomae_v2'):
            for source, target in ((f'raw_mmd/{branch}', f'train/rdm_mmd_{branch}'),
                                   (f'cca/loss_{branch}', f'train/cca_cos_{branch}')):
                if source in logs:
                    result[target] = logs[source]
        result.update(timing_metrics(logs))
        return result

    def _cca_inputs(self, index, seed):
        row = self.clean_data[index]
        prompt = self.dataset[index]['prompts']
        if row['caption'] != prompt:
            raise ValueError(f'CCA caption mismatch at row {index}')
        gt = row['wan_gt_latents'].unsqueeze(0).transpose(1, 2).to(self.device, self.dtype)
        conditional = self._get_conditioning([prompt])
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed)
            levels = self.model.denoising_step_list.to(self.device)
            block = int(getattr(self.model, 'num_frame_per_block', 3))
            timesteps = sample_cca_chunk_timesteps(levels,
                num_chunks=gt.shape[1]//block, frames_per_chunk=block,
                first_chunk_levels=getattr(self.model, 'rdm_first_frame_denoising_step_list', None))
            scheduler = self.model.scheduler
            nearest = (scheduler.timesteps.to(self.device).double()[None, :]
                -timesteps.flatten().double()[:, None]).abs().argmin(1)
            sigma = scheduler.sigmas.to(self.device)[nearest].reshape(1, 21, 1, 1, 1).to(gt)
            noise = torch.randn_like(gt)
            noisy = (1.-sigma)*gt+sigma*noise
        inputs = (gt, noisy, timesteps, conditional)
        return inputs

    def _cca_forward(self, gt, noisy, timesteps, conditional):
        # clean_x exposes only preceding clean chunks and the current noisy chunk.
        self.model.generator.model.block_mask = None
        with torch.autocast('cuda', dtype=self.dtype):
            _, prediction = self.model.generator(noisy_image_or_video=noisy,
                conditional_dict=conditional, timestep=timesteps, clean_x=gt.detach())
        return prediction

    def _cca_backward_sample(self, inputs):
        return self._cca_backward_sample_with_history(inputs)

    def _cca_backward_sample_with_history(self, inputs):
        self._cca_target_caches()
        context = torch.autograd.graph.save_on_cpu(pin_memory=True) if (
            self.config.generator_activation_cpu_offload) else nullcontext()
        # CCA pairs are independent: retain this graph while the frozen feature
        # objective computes a detached latent VJP, then backpropagate once.
        # Unlike global MMD, CCA need not collect a whole rollout population first.
        with context:
            prediction = self._cca_forward(*inputs)
        value, latent_gradient = self.cca_objective.value_and_gradient(
            prediction.detach().transpose(1, 2), inputs[0].detach().transpose(1, 2),
            **self._cca_feature_kwargs(inputs[3]))
        components = {name: float(value) for name, value in
                      getattr(self.cca_objective, 'last_values', {}).items()}
        torch.autograd.backward(prediction,
            latent_gradient.transpose(1, 2).to(prediction)/float(self.cca_accum))
        # Preserve the helper's tuple API; no CCA replay/parity check is performed.
        return float(value), 0., components

    def _additional_generator_backward(self):
        """Save RDM gradient shards, compute CCA gradients, then combine before one clip/step."""
        start = time.monotonic()
        stop = self.config.get('joint_cca_stop_step')
        if stop is not None and self.step + 1 > int(stop):
            # Keep existing MMD gradients intact. No CCA forward, decoder/encoder
            # VJP, controller observation, or extra optimizer reset after cutoff.
            return self.controller.skip(self.step + 1)
        parameters = list(self.model.generator.parameters())
        sf_gradients = [p.grad.detach().clone() if p.grad is not None else None for p in parameters]
        self.generator_optimizer.zero_grad(set_to_none=True)
        self.model.inference_pipeline = None
        gc.collect()
        torch.cuda.empty_cache()
        ids = training_batch(self.training_ids, step=self.step+1,
            population=int(self.config.rdm_expected_global_rows), seed=int(self.config.seed))
        ids = ids[:int(self.config.joint_cca_batch)][self.rank::self.world_size]
        losses, branch_losses = [], {}
        caches = self._cca_target_caches()
        for cache in caches:
            cache.reset_stats()
        for position, index in enumerate(ids):
            self._progress('cca_backward', accum_index=position+1)
            inputs = self._cca_inputs(index, int(self.config.seed)+7000003*(self.step+1)+index)
            value, _, components = self._cca_backward_sample(inputs)
            for name, component in components.items():
                branch_losses.setdefault(name, []).append(component)
            losses.append(value)
            del inputs
        # FULL_SHARD + use_orig_params: each gradient entry is its local shard.
        statistics = torch.zeros(3, device=self.device, dtype=torch.float64)
        for parameter, sf in zip(parameters, sf_gradients):
            cca = parameter.grad
            if sf is not None:
                statistics[0] += sf.double().square().sum()
            if cca is not None:
                statistics[1] += cca.double().square().sum()
            if sf is not None and cca is not None:
                statistics[2] += (sf.double()*cca.double()).sum()
        dist.all_reduce(statistics)
        if not torch.isfinite(statistics).all():
            raise FloatingPointError('Nonfinite branch gradient statistics')
        sf_norm, cca_norm = (float(x.sqrt()) for x in statistics[:2])
        # The loss baseline and current-step inputs must be identical on every rank.
        loss = torch.tensor(sum(losses)/len(losses), device=self.device, dtype=torch.float64)
        dist.all_reduce(loss)
        global_cca_loss = float(loss)/self.world_size
        weighting_logs = self.controller.update(self.step+1, global_cca_loss, cca_norm, sf_norm)
        component_logs = {}
        for name, values in branch_losses.items():
            average = torch.tensor(sum(values)/len(values), device=self.device, dtype=torch.float64)
            dist.all_reduce(average)
            component_logs[f'cca/loss_{name}'] = float(average)/self.world_size
        cca_coefficient = self.controller.weight
        sf_coefficient = 1.
        for parameter, sf in zip(parameters, sf_gradients):
            if parameter.grad is not None:
                parameter.grad.mul_(cca_coefficient)
                if sf is not None:
                    parameter.grad.add_(sf, alpha=sf_coefficient)
            elif sf is not None:
                parameter.grad = sf.mul_(sf_coefficient)
        cache_logs = {}
        if caches:
            counts = torch.tensor([sum(c.stats[k] for c in caches) for k in ('hits', 'misses')],
                                  device=self.device, dtype=torch.float64)
            dist.all_reduce(counts)
            cache_logs['cache/cca_target_hit_rate'] = float(counts[0]/counts.sum().clamp_min(1))
            for key in ('key_seconds', 'read_seconds', 'compute_seconds', 'write_seconds'):
                cache_logs['timing/cca_target_'+key] = self._global_max_scalar(sum(c.stats[key] for c in caches))
        return {**weighting_logs, **component_logs, **cache_logs, 'cca/loss': global_cca_loss, 'cca/active': 1.,
            'mmd/weight': sf_coefficient, 'cca/gradient_norm': cca_norm,
            ('mmd/gradient_norm'): sf_norm,
            'joint/gradient_cosine': float(statistics[2])/max(sf_norm*cca_norm, 1e-30),
            'joint/weighted_gradient_ratio': cca_coefficient*cca_norm/max(sf_coefficient*sf_norm, 1e-30),
            'cca/global_samples': len(ids)*self.world_size,
            'timing/cca_seconds': time.monotonic()-start}

    def _boundary(self, rgb, latent):
        rgb_delta = (rgb[:, 1:]-rgb[:, :-1]).abs().mean((0, 2, 3, 4)).cpu().tolist()
        latent_delta = (latent[:, 1:].float()-latent[:, :-1].float()).square().mean(
            (0, 2, 3, 4)).sqrt().cpu().tolist()
        block = int(self.config.num_frame_per_block)
        starts = list(range(block, latent.shape[1], block))
        # Wan decodes latent 0 to one frame and each later latent to four.
        result = {'rgb': boundary_statistics(rgb_delta, [4*i-3 for i in starts])}
        # Single-latent blocks have no within-block latent transitions.
        if block > 1:
            result['latent'] = boundary_statistics(latent_delta, starts)
        return result

    @torch.no_grad()
    def _feature_views(self, latent, rgb, index):
        if self.latent_feature_encoder:
            embedding = self._get_conditioning([self.dataset[index]['prompts']])['prompt_embeds']
            return self.rdm_encoder.forward_views(latent.transpose(1, 2), embedding)
        return self.rdm_encoder.forward_views(rgb)

    @torch.no_grad()
    def _gt_details(self, index):
        suffix = '_framewise' if int(self.config.num_frame_per_block) == 1 else ''
        path = self.gt_cache/f'joint_details{suffix}_v1_row{index:04d}.pt'
        if path.exists():
            return torch.load(path, map_location='cpu', weights_only=True)
        latent = self.clean_data[index]['wan_gt_latents'].unsqueeze(0).transpose(1, 2).to(self.device, self.dtype)
        rgb = (self.taew_decoder.decode_value(latent).float()*.5+.5).clamp(0, 1)
        views = self._feature_views(latent, rgb, index)
        temporal = views['mean_spatial_per_tubelet']
        result = {'temporal': temporal.cpu(), 'boundary': self._boundary(rgb, latent)}
        atomic_save(result, path)
        return result

    @torch.no_grad()
    def _save_comparison(self, index, generated_rgb):
        """Save GT, CCA reconstruction and self-rollout side by side."""
        import imageio.v2 as imageio
        inputs = self._cca_inputs(index, int(self.config.eval_seed)*100000+index+50000000)
        prediction = self._cca_forward(*inputs)
        gt_pixels = self.taew_decoder.decode_value(inputs[0])
        cca_pixels = self.taew_decoder.decode_value(prediction)
        if getattr(self.cca_objective, 'requires_embedding', False):
            loss = float(self.cca_objective.value(prediction.transpose(1, 2), inputs[0].transpose(1, 2),
                                                 inputs[3]['prompt_embeds']))
        else:
            gt_tokens = self.cca_objective._tokens(gt_pixels)
            cca_tokens = self.cca_objective._tokens(cca_pixels)
            loss = float(token_cosine(cca_tokens, gt_tokens))
            del gt_tokens, cca_tokens
        columns = []
        for pixels in ((gt_pixels.float()*.5+.5).clamp(0, 1),
                       (cca_pixels.float()*.5+.5).clamp(0, 1), generated_rgb):
            columns.append(F.interpolate(pixels[0], size=(240, 416), mode='bilinear', align_corners=False))
        video = torch.cat(columns, dim=-1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        path = self.run_path/'visualizations'/f'step{self.step:06d}_row{index:04d}.mp4'
        path.parent.mkdir(exist_ok=True)
        imageio.mimwrite(path, video, fps=int(self.config.rdm_rollout_video_fps), macro_block_size=1)
        loss_key = 'cca_cos'
        return {'path': str(path), loss_key: loss, 'index': index,
                'prompt': self.dataset[index]['prompts']}

    @torch.no_grad()
    def _evaluate(self, role):
        evaluation_start = time.perf_counter()
        from pipeline.causal_inference import CausalInferencePipeline
        ids = self.control_ids if role == 'control' else self.report_ids
        self._event('evaluation', role=role, step=self.step, count=len(ids))
        self.model.inference_pipeline = None
        self.model.eval()
        owner = self

        class Text(torch.nn.Module):
            def forward(self, text_prompts):
                return owner._get_conditioning(text_prompts)

        pipeline = CausalInferencePipeline(self.config, self.device,
            generator=self.model.generator, text_encoder=Text(), vae=torch.nn.Identity())
        rows = []
        visual_rgb = {}
        visual_ids = self.report_ids[:int(self.config.joint_visual_count)]
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            for index in ids[self.rank::self.world_size]:
                torch.manual_seed(int(self.config.eval_seed)*100000+index)
                prefix = 0
                noise = torch.randn(1, prefix+21, 16, 60, 104, device=self.device, dtype=self.dtype)
                _, latent = pipeline.inference(noise, [self.dataset[index]['prompts']],
                    return_latents=True, return_video=False)
                latent = latent[:, prefix:]
                rgb = (self.taew_decoder.decode_value(latent).float()*.5+.5).clamp(0, 1)
                views = self._feature_views(latent, rgb, index)
                feature_key = 'mmd_features'
                row = {'index': index, 'global': views[feature_key].cpu().numpy()}
                if role == 'report':
                    row.update(temporal=views['mean_spatial_per_tubelet'].cpu().numpy(),
                        boundary=self._boundary(rgb, latent), gt=self._gt_details(index))
                    if index in visual_ids:
                        visual_rgb[index] = F.interpolate(rgb[0], size=(240, 416),
                            mode='bilinear', align_corners=False).unsqueeze(0).cpu()
                rows.append(row)
            # FSDP forwards must occur on EVERY rank in the same order/count.
            # Do visual CCA forwards in a separate pass with an evenly sharded subset.
            if role == 'report':
                for index in visual_ids[self.rank::self.world_size]:
                    next(row for row in rows if row['index'] == index)['media'] = self._save_comparison(
                        index, visual_rgb.pop(index).to(self.device))
        gathered = [None]*self.world_size
        dist.all_gather_object(gathered, rows, group=self.io_group)
        if self.is_main_process:
            mapping = {row['index']: row for part in gathered for row in part}
            ordered = [mapping[i] for i in ids]
            x = np.concatenate([row['global'] for row in ordered])
            y = torch.cat([self.probe_gt[i] for i in ids]).double().numpy()
            text = None if self.joint_text_table is None else self.joint_text_table[ids].cpu().numpy()
            # compare also probes analytical feature gradients on detached CPU leaves.
            with torch.enable_grad():
                from utils.token_metrics import compare_tokens
                metrics = compare_tokens(x, y, self.rdm_reference, text, self.device)
                if self.config.rdm_representation == 'tokens':
                    metrics['token_count'] = int(self.config.rdm_token_count)
                    metrics['raw_frobenius_w2_squared'] = metrics['empirical_w2_squared']*int(self.config.rdm_token_count)
                else:
                    metrics['representation'] = 'full-token global mean; squared Euclidean distance'
                    metrics['pooled_token_count'] = int(self.config.rdm_token_count)
            logs = {}
            if role == 'control':
                metrics['used_for_weighting'] = False
                logs = {'control/ot': metrics['empirical_w2_squared']}
            else:
                temporal = np.concatenate([r['temporal'] for r in ordered])
                temporal_gt = torch.cat([r['gt']['temporal'] for r in ordered]).numpy()
                n = len(ids)
                temporal_count = temporal.shape[1]
                if temporal_gt.shape != temporal.shape:
                    raise ValueError('Generated/GT temporal feature grids differ')
                metrics['temporal_ot'] = exact_ot(temporal.reshape(n, -1)/math.sqrt(temporal_count),
                    temporal_gt.reshape(n, -1)/math.sqrt(temporal_count))[0]
                boundary = {}
                for name in ordered[0]['boundary']:
                    b, interior, gb, gi = [float(np.mean(values)) for values in (
                        [r['boundary'][name]['boundary_mean'] for r in ordered],
                        [r['boundary'][name]['interior_mean'] for r in ordered],
                        [r['gt']['boundary'][name]['boundary_mean'] for r in ordered],
                        [r['gt']['boundary'][name]['interior_mean'] for r in ordered])]
                    boundary[name] = {'boundary_mean': b, 'interior_mean': interior,
                        'gt_boundary_mean': gb, 'gt_interior_mean': gi,
                        'relative_jump_ratio': (b/max(interior, 1e-12))/(gb/max(gi, 1e-12)),
                        'interior_motion_ratio': interior/max(gi, 1e-12)}
                metrics['boundary'] = boundary
                logs = {'eval/ot': metrics['empirical_w2_squared'],
                    'eval/mmd': metrics['official_joint_nystrom_mmd']['raw_biased_mmd2'],
                    'eval/variance_ratio': metrics['moments']['variance_trace']/float(np.square(y-y.mean(0)).sum()/n),
                    'eval/temporal_ot': metrics['temporal_ot'],
                    'eval/chunk_jump_ratio': boundary['rgb']['relative_jump_ratio'],
                    'eval/interior_motion_ratio': boundary['rgb']['interior_motion_ratio']}
                media = [mapping[i]['media'] for i in visual_ids]
                if media:
                    loss_key = 'cca_cos'
                    logs['eval/'+loss_key] = float(np.mean([m[loss_key] for m in media]))
                    metrics['visualizations'] = media
                    if not self.disable_wandb:
                        for position, item in enumerate(media, 1):
                            logs[f'samples/prompt_{position:02d}'] = wandb.Video(item['path'],
                                caption=f"Left: GT | Middle: CCA reconstruction with GT history | Right: free AR ({self._inference_step_label()}). step={self.step}, row={item['index']}. {item['prompt']}")
            logs[f'timing/eval_{role}_seconds'] = time.perf_counter()-evaluation_start
            metrics['evaluation_seconds'] = logs[f'timing/eval_{role}_seconds']
            target = self.run_path/'evaluation'/f'{role}_step{self.step:06d}.json'
            target.parent.mkdir(exist_ok=True)
            target.write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n')
            numeric = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            with (self.run_path/'evaluation.jsonl').open('a') as stream:
                stream.write(json.dumps({'step': self.step, **numeric}, allow_nan=False)+'\n')
            if not self.disable_wandb:
                wandb.log(logs, step=self.step)
            self._event('evaluation_complete', role=role, step=self.step, **numeric)
        dist.barrier(group=self.io_group)
        del pipeline, rows, gathered, noise, latent, rgb, views
        gc.collect()
        torch.cuda.empty_cache()
