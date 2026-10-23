"""Focused trainer for text-conditioned RDM updates."""

from __future__ import annotations
from dataclasses import dataclass
import gc
import math
import time
import torch
import torch.distributed as dist
import wandb
from model.encoders.text import couple_video_text, load_frozen_text_table, prompt_rows_sha256
from trainer.references.bank import load_reference
from trainer.losses.rdm import mmd_nystrom_with_terms, self_normalize_rdm_loss
from utils.hashing import file_sha256
from trainer.base import BaseTrainer
from utils.distributed import (
    differentiable_all_gather,
)



@dataclass
class _RolloutChunk:
    noise: torch.Tensor
    context_noise: torch.Tensor
    first_frame_exit_index: int
    first_frame_trajectory_noise: torch.Tensor
    block_exit_indices: torch.Tensor
    block_trajectory_noise: torch.Tensor
    conditional: dict
    latent: torch.Tensor
    dataset_index: int
    prompt: str


class RDMTrainer(BaseTrainer):
    """Focused RDM trainer."""

    def __init__(self, config):
        super().__init__(config)
        self.generator_updates = 0
        self.grad_accum = int(config.rdm_grad_accum_steps)
        self.rdm_generation_mode = str(config.rdm_generation_mode)
        self.bidirectional_full_video = (
            self.rdm_generation_mode == "full_video_bidirectional"
        )
        self.bidirectional_random_x0_4step = bool(
            config.rdm_bidirectional_random_x0_4step
        )
        self.all_chunks_random_x0_4step = bool(config.rdm_all_chunks_random_x0_4step)
        self.heterogeneous_chunkwise_schedule = bool(
            self.all_chunks_random_x0_4step
            and list(config.rdm_first_frame_denoising_step_list)
        )
        self.rollout_step_count = len(list(config.denoising_step_list))
        self.log_full_all_chunks_rollout = bool(
            config.rdm_log_rollout_force_all_chunks_full
        )
        expected_rows = self.world_size * self.grad_accum * int(config.batch_size)
        if expected_rows != int(config.rdm_expected_global_rows):
            raise ValueError(
                f"RDM global rows {expected_rows} != {config.rdm_expected_global_rows}"
            )
        (self.rdm_encoder, self.latent_feature_encoder, self.rdm_reference,
         self.joint_text_beta) = self._load_feature_backend(config)
        self.joint_text_table = None if not config.get('rdm_joint_text_enabled', True) else load_frozen_text_table(
            config.rdm_joint_text_table_path,
            expected_sha256=str(config.rdm_joint_text_table_sha256),
            expected_rows=int(config.rdm_expected_reference_rows),
            expected_dim=int(config.rdm_joint_text_feature_dim),
            device=self.device,
        )
        prompt_contract = [None, None]
        if self.is_main_process:
            prompts = [
                str(self.dataset[index]["prompts"])
                for index in range(len(self.dataset))
            ]
            prompt_contract = [len(prompts), prompt_rows_sha256(prompts)]
        dist.broadcast_object_list(prompt_contract, src=0)
        expected_prompt_contract = [
            int(config.rdm_expected_reference_rows),
            str(config.rdm_joint_prompt_rows_sha256),
        ]
        if prompt_contract != expected_prompt_contract:
            raise RuntimeError(
                f"prompt table {prompt_contract} != {expected_prompt_contract}"
            )
        from model.decoders.factory import build_training_decoder
        # Retain this attribute for existing feature/CCA callers. Its actual
        # decoder and gradient rule are explicit in config and resume contracts.
        self.taew_decoder = build_training_decoder(config, self.device, self.dtype)
        if self.is_main_process:
            print(
                f"Loaded RDM: reference_sha256={self.rdm_reference.sha256[:16]} text_scale={self.joint_text_beta}",
                flush=True,
            )

    def _load_feature_backend(self, config):
        from model.encoders.factory import build_video_encoder

        encoder = build_video_encoder(config, self.device)
        latent_encoder = str(config.rdm_input_mode) == 'normalized_wan_latent'
        reference, text_beta = self._load_feature_reference(config, encoder, latent_encoder)
        return encoder, latent_encoder, reference, text_beta

    def _load_feature_reference(self, config, encoder, latent_encoder):
        """Allow temporal reference banks to share one frozen video encoder."""
        checkpoint_sha = (encoder.checkpoint_sha256 if latent_encoder
                          else file_sha256(config.rdm_encoder_checkpoint_path))
        from model.encoders.text import feature_reference_contract
        contract = (encoder.reference_contract() if latent_encoder and config.get('rdm_joint_text_enabled', True)
                    else feature_reference_contract(config, encoder))
        reference = load_reference(
            config.rdm_reference_bundle_path,
            device=self.device,
            cpu_offload=config.get("rdm_reference_residency", "cuda") == "cpu",
            expected_encoder_id=config.rdm_encoder_id,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_height=int(config.rdm_input_height),
            expected_width=int(config.rdm_input_width),
            expected_rows=int(config.rdm_expected_reference_rows),
            expected_reference_revision=str(config.rdm_reference_revision),
            expected_encoder_contract=contract,
            expected_landmarks=int(config.rdm_nystrom_landmarks),
            expected_feature_dim=int(config.rdm_joint_feature_dim),
            expected_input_mode=str(config.rdm_input_mode),
            expected_pool=contract['pool'],
            expected_rows_per_video=1,
        )
        if reference.sha256 != str(
            config.rdm_reference_bundle_sha256
        ):
            raise RuntimeError("RDM reference checksum mismatch")
        if not config.get('rdm_joint_text_enabled', True):
            if reference.beta is not None or reference.metadata.get('joint_enable') is not False:
                raise ValueError('Visual-only training cannot load a text-coupled reference')
            return reference, None
        if reference.beta is None:
            raise ValueError("RDM reference is missing its text scale")
        metadata_beta = float(reference.metadata.get("joint_beta", math.nan))
        if not math.isclose(
            float(reference.beta), metadata_beta, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("RDM text scale does not match reference metadata")
        text_beta = float(reference.beta)
        return reference, text_beta





    def _decode_rdm_value(self, latent: torch.Tensor) -> torch.Tensor:
        if self.taew_decoder is None:
            raise RuntimeError("Training decoder was not initialized")
        return self.taew_decoder.decode_value(latent)

    def _couple_joint_text_feature(
        self,
        visual_features: torch.Tensor,
        chunk: _RolloutChunk,
        *,
        beta: float | None = None,
    ) -> torch.Tensor:
        """Apply the pinned ``[phi | beta*tau]`` feature coupling."""
        if not self.config.get('rdm_joint_text_enabled', True):
            if tuple(visual_features.shape) != (1, int(self.config.rdm_feature_dim)):
                raise RuntimeError('Invalid visual-only feature row')
            return visual_features
        if self.joint_text_table is None or self.joint_text_beta is None:
            raise RuntimeError("joint text artifacts were not initialized")
        if tuple(visual_features.shape) != (1, int(self.config.rdm_feature_dim)):
            raise RuntimeError(
                f"joint feature coupling requires one visual row, got {tuple(visual_features.shape)}"
            )
        index = int(chunk.dataset_index)
        if not 0 <= index < int(self.joint_text_table.shape[0]):
            raise IndexError(f"joint text dataset index {index} is out of range")
        text_rows = self.joint_text_table[index : index + 1]
        return couple_video_text(
            visual_features,
            text_rows,
            self.joint_text_beta if beta is None else float(beta),
        )

    def _rdm_pixel_vjp_to_latent(
        self,
        latent: torch.Tensor,
        pixel_gradient: torch.Tensor,
        *,
        straight_through_clamp: bool = False,
    ) -> torch.Tensor:
        if self.taew_decoder is None:
            raise RuntimeError("Training decoder was not initialized")
        gradient = self.taew_decoder.full_vjp(
            latent, pixel_gradient, straight_through_clamp=straight_through_clamp
        )
        if tuple(gradient.shape) != tuple(latent.shape):
            raise RuntimeError(
                f"latent VJP {tuple(gradient.shape)} != {tuple(latent.shape)}"
            )
        return gradient


    @staticmethod
    def _validate_static_config(config):
        from utils.validation import validate_short_video_runtime
        from model.encoders.factory import ENCODER_IDS, DIT_ENCODER_IDS
        from model.encoders.tokens import TOKEN_POOL
        validate_short_video_runtime(config)
        if config.rdm_encoder_id not in ENCODER_IDS:
            raise ValueError(f'Unsupported RDM encoder: {config.rdm_encoder_id}')
        latent = config.rdm_encoder_id in DIT_ENCODER_IDS
        channels, count = (5120, 256) if latent else (1024, 1280)
        exact = dict(trainer='direct_forcing', distribution_loss='rdm', batch_size=1,
            same_step_across_blocks=True, i2v=False, independent_first_frame=False,
            rdm_input_mode='normalized_wan_latent' if latent else 'native_video',
            rdm_input_height=480, rdm_input_width=832, rdm_video_frames=81,
            rdm_video_padded_frames=82 if latent else 81,
            rdm_video_temporal_pad='replicate_last' if latent else 'skip_first',
            rdm_token_count=count, rdm_pool=TOKEN_POOL, rdm_feature_dim=count*channels,
            rdm_joint_feature_dim=count*channels + (int(config.rdm_joint_text_feature_dim)
                if config.get('rdm_joint_text_enabled', True) else 0),
            rdm_joint_bandwidth_scale=1.,
            rdm_nystrom_landmarks=min(4096, int(config.rdm_expected_reference_rows)),
            rdm_log_rollout_video=True, rdm_attention_backend='flash_attn_2',
            rdm_require_flash_attention=True, rdm_all_chunks_random_x0_4step=True,
            rdm_bidirectional_random_x0_4step=False,
            rdm_replay_first_frame_trajectory_noise=False,
            rdm_log_rollout_force_all_chunks_full=True, resume_ckpt='')
        for key, expected in exact.items():
            if config.get(key) != expected:
                raise ValueError(f'RDM requires {key}={expected!r}, got {config.get(key)!r}')
        if str(config.rdm_flash_attn_version) not in ('2.8.3', '2.8.3.post1'):
            raise ValueError('RDM requires FlashAttention 2.8.3 or 2.8.3.post1')
        world = int(config.rdm_expected_world_size)
        if world not in (1, 4, 8) or int(config.rdm_expected_local_world_size) != world:
            raise ValueError('Supported single-node process counts are 1, 4, and 8')
        if world*int(config.rdm_grad_accum_steps)*int(config.batch_size) != int(config.rdm_expected_global_rows):
            raise ValueError('RDM accumulation does not match the global batch')


    def _next_rdm_batch(self):
        return next(self.dataloader)




    def _make_rollout_chunk(self, batch) -> _RolloutChunk:
        prompts = batch["prompts"]
        if len(prompts) != 1:
            raise ValueError(f"RDM requires batch size 1, got {len(prompts)}")
        conditional = self._get_conditioning(prompts)
        noise = torch.randn(1, 21, 16, 60, 104, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            (
                latent,
                _,
                _,
                _,
                context_noise,
                first_frame_exit_index,
                first_frame_trajectory_noise,
                block_exit_indices,
                block_trajectory_noise,
            ) = self.model.rollout_from_noise(noise, conditional)
        index = batch.get("idx", -1)
        if torch.is_tensor(index):
            index = int(index.flatten()[0])
        return _RolloutChunk(
            noise=noise.detach(),
            context_noise=context_noise.detach(),
            first_frame_exit_index=int(first_frame_exit_index),
            first_frame_trajectory_noise=first_frame_trajectory_noise.detach(),
            block_exit_indices=block_exit_indices.detach(),
            block_trajectory_noise=block_trajectory_noise.detach(),
            conditional={key: value.detach() for (key, value) in conditional.items()},
            latent=latent.detach(),
            dataset_index=int(index),
            prompt=str(prompts[0]),
        )

    @torch.no_grad()
    def _first4_inference_latent(self, chunk: _RolloutChunk) -> torch.Tensor:
        """Inference-only first-block enhancement, isolated from training rollout state."""
        from pipeline.causal_inference import CausalInferencePipeline

        class FixedConditioning(torch.nn.Module):
            def forward(self, text_prompts):
                return chunk.conditional

        self.model.inference_pipeline = None
        pipeline = CausalInferencePipeline(self.config, self.device,
            generator=self.model.generator, text_encoder=FixedConditioning(), vae=torch.nn.Identity())
        devices = [chunk.noise.device] if chunk.noise.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(self.config.seed)*100000 + self.step)
            _, latent = pipeline.inference(chunk.noise, [chunk.prompt],
                return_latents=True, return_video=False)
        return latent

    @torch.no_grad()
    def _full_all_chunks_rollout_video(
        self, chunk: _RolloutChunk
    ) -> torch.Tensor | None:
        """Re-run accumulation zero at the deterministic evaluation schedule."""
        if not self.all_chunks_random_x0_4step:
            raise RuntimeError("full chunkwise rollout log requires its isolated mode")
        if (self.rollout_step_count < 4 and
                list(self.config.get('first_frame_denoising_step_list', [])) == [1000, 750, 500, 250]):
            latent = self._first4_inference_latent(chunk)
            return self._rollout_video_from_latent(latent)
        later_exit = self.rollout_step_count - 1
        num_blocks = chunk.noise.shape[1] // int(self.config.num_frame_per_block)
        fixed_exits = torch.full(
            (num_blocks,), later_exit, device=chunk.noise.device, dtype=torch.long
        )
        first_exit = None
        if self.heterogeneous_chunkwise_schedule:
            fixed_exits[0] = 3
            first_exit = 3
        self._progress(
            "rollout_visualization",
            detail=f"first_chunk_exit_index={int(fixed_exits[0])} later_chunk_exit_index={later_exit}",
        )
        (latent, _, _, _, replay_context_noise, _, _, exit_indices, _) = (
            self.model.rollout_from_noise(
                chunk.noise,
                chunk.conditional,
                context_noise_replay=chunk.context_noise,
                first_frame_exit_index_replay=first_exit,
                block_exit_indices_replay=fixed_exits,
            )
        )
        if tuple(exit_indices.shape) != (num_blocks,) or bool(
            (exit_indices != fixed_exits).any()
        ):
            raise RuntimeError("visualization rollout did not use its fixed schedule")
        if float((replay_context_noise - chunk.context_noise).abs().max()) != 0.0:
            raise RuntimeError("visualization rollout changed replayed context noise")
        return self._rollout_video_from_latent(latent)

    def _rollout_video_from_latent(self, latent):
        if not self.is_main_process:
            return None
        decoded = self._decode_rdm_value(latent.detach())
        return (
            (decoded[0].float() * 0.5 + 0.5)
            .clamp(0, 1)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .cpu()
        )

    @torch.no_grad()
    def _latent_features(
        self, chunk: _RolloutChunk, *, capture_video: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.latent_feature_encoder:
            visual_features = self.rdm_encoder(chunk.latent.transpose(1, 2), chunk.conditional['prompt_embeds'])
            features = self._couple_joint_text_feature(visual_features, chunk)
            video = None
            if capture_video:
                decoded = self._decode_rdm_value(chunk.latent)
                video = (decoded[0].float()*.5+.5).clamp(0, 1).mul(255).round().to(torch.uint8).cpu()
            return features.detach().float(), video
        decoded = self._decode_rdm_value(chunk.latent)
        video = None
        if capture_video:
            video = (
                (decoded[0].float() * 0.5 + 0.5)
                .clamp(0, 1)
                .mul(255.0)
                .round()
                .to(torch.uint8)
                .cpu()
            )
        images = (decoded.float() * 0.5 + 0.5).clamp(0, 1)
        visual_features = self.rdm_encoder(images).detach().float()
        features = self._couple_joint_text_feature(visual_features, chunk)
        expected_shape = (1, self.rdm_reference.feature_dim)
        if tuple(features.shape) != expected_shape:
            raise RuntimeError(
                f"RDM features {tuple(features.shape)} != {expected_shape}"
            )
        return (features.detach().float(), video)

    def _feature_vjp_to_latent(
        self,
        chunk: _RolloutChunk,
        feature_gradient: torch.Tensor,
        pass1_features: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        if self.latent_feature_encoder:
            latent = chunk.latent.detach().requires_grad_(True)
            features = self._couple_joint_text_feature(
                self.rdm_encoder(latent.transpose(1, 2), chunk.conditional['prompt_embeds']), chunk)
            parity = float((features.detach()-pass1_features).abs().max())
            if not math.isfinite(parity) or parity > float(self.config.rdm_feature_replay_atol):
                raise RuntimeError(f'DiT feature replay mismatch: {parity}')
            gradient, = torch.autograd.grad(features, latent, grad_outputs=feature_gradient.to(features))
            if not torch.isfinite(gradient).all():
                raise RuntimeError('DiT encoder produced an invalid latent VJP')
            return gradient.detach().float(), parity
        with torch.no_grad():
            decoded_value = self._decode_rdm_value(chunk.latent)
        decoded = decoded_value.detach().requires_grad_(True)
        images = (decoded.float() * 0.5 + 0.5).clamp(0, 1)
        visual_features = self.rdm_encoder(images)
        replay_features = self._couple_joint_text_feature(visual_features, chunk)
        parity_max = float((replay_features.detach() - pass1_features).abs().max())
        tolerance = float(self.config.rdm_feature_replay_atol)
        if not math.isfinite(parity_max) or parity_max > tolerance:
            raise RuntimeError(
                f"feature replay mismatch {parity_max:.6g} > {tolerance:.6g}"
            )
        torch.autograd.backward(replay_features, feature_gradient.to(replay_features))
        if decoded.grad is None or not torch.isfinite(decoded.grad).all():
            raise RuntimeError("feature encoder produced an invalid pixel VJP")
        pixel_gradient = decoded.grad.detach()
        del replay_features, visual_features, images, decoded, decoded_value
        if self.config.get('rdm_cleanup_policy', 'sample') == 'sample':
            torch.cuda.empty_cache()
        latent_gradient = self._rdm_pixel_vjp_to_latent(chunk.latent, pixel_gradient)
        if not torch.isfinite(latent_gradient).all():
            raise RuntimeError("pixel decoder produced an invalid latent VJP")
        return (latent_gradient.detach().float(), parity_max)

    def _mmd_feature_gradients(
        self, chunks: list[_RolloutChunk]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], dict, torch.Tensor | None]:
        capture_video = bool(
            self.is_main_process
            and (not self.disable_wandb)
            and (not self.config.no_visualize)
            and self.config.rdm_log_rollout_video
        )
        pass1 = []
        rollout_video = None
        for index, chunk in enumerate(chunks):
            self._progress("features", accum_index=index + 1)
            (features, video) = self._latent_features(
                chunk, capture_video=capture_video and index == 0
            )
            pass1.append(features)
            if video is not None:
                rollout_video = video
        current_local = torch.cat(pass1, dim=0).detach().requires_grad_(True)
        expected_local_rows = self.grad_accum * int(self.config.batch_size)
        if int(current_local.shape[0]) != expected_local_rows:
            raise RuntimeError(
                f"local rows {current_local.shape[0]} != {expected_local_rows}"
            )
        self._progress(
            "kernel",
            detail=f"local_rows={current_local.shape[0]} global_rows={current_local.shape[0] * self.world_size}",
        )
        global_features = differentiable_all_gather(current_local)
        expected_global_rows = int(self.config.rdm_expected_global_rows)
        if int(global_features.shape[0]) != expected_global_rows:
            raise RuntimeError(
                f"global rows {global_features.shape[0]} != {expected_global_rows}"
            )
        torch.cuda.synchronize(self.device)
        kernel_start = time.perf_counter()
        kernel_timings = {}
        (raw, terms) = mmd_nystrom_with_terms(
            global_features,
            self.rdm_reference,
            gen_chunk=int(self.config.rdm_kernel_gen_chunk),
            timings=kernel_timings,
        )
        if self.config.get('rdm_loss_normalization', 'rdm_abs_v1') == 'elastic_raw_v1':
            normalized = raw / (raw.detach() + float(self.config.rdm_self_normalize_eps))
        else:
            normalized = self_normalize_rdm_loss(raw, eps=float(self.config.rdm_self_normalize_eps))
        (normalized / float(self.grad_accum)).backward()
        kernel_seconds = self._sync_time(kernel_start)
        if current_local.grad is None or not torch.isfinite(current_local.grad).all():
            raise RuntimeError("RDM feature gradient is missing or non-finite")
        feature_gradients = list(current_local.grad.detach().split(1, dim=0))
        logs = {
            'timing/reference_h2d_seconds': kernel_timings['reference_h2d_seconds'],
            'timing/mmd_kernel_backward_seconds': max(
                0., kernel_seconds-kernel_timings['reference_h2d_seconds']),
            "raw_mmd": float(raw.detach()),
            "normalized_mmd": float(normalized.detach()),
            "k_gg": float(terms["k_gg"].detach()),
            "k_gr_nystrom": float(terms["k_gr_nystrom"].detach()),
            "k_rr": float(terms["k_rr"].detach()),
            "global_rows": float(global_features.shape[0]),
            "global_video_batch": float(self.world_size * len(chunks)),
            "joint_text_beta": float(self.joint_text_beta) if self.joint_text_beta is not None else 0.,
        }
        del global_features
        return (pass1, feature_gradients, logs, rollout_video)

    def _assert_frozen_parameter_grads(self) -> None:
        modules = {
            "encoder": self.rdm_encoder,
            "decoder": self.taew_decoder,
        }
        offenders = []
        for module_name, module in modules.items():
            if module is None:
                continue
            if any(
                (
                    parameter.grad is not None
                    and bool(parameter.grad.detach().abs().max() > 0)
                    for parameter in module.parameters()
                )
            ):
                offenders.append(module_name)
        if offenders:
            raise RuntimeError(f"frozen modules accumulated gradients: {offenders}")

    def _additional_generator_backward(self) -> dict:
        """Optional research branch, before the single clip/optimizer update."""
        return {}

    def _generator_update(self) -> dict:
        cleanup_policy = self.config.get('rdm_cleanup_policy', 'sample')
        if cleanup_policy not in ('sample', 'stage'):
            raise ValueError('rdm_cleanup_policy must be sample or stage')
        phase_start = time.perf_counter()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        self.model.eval()
        self.rdm_encoder.eval()
        self.generator_optimizer.zero_grad(set_to_none=True)
        rollout_start = time.perf_counter()
        self._progress("generator_begin", detail=f"accum_total={self.grad_accum}")
        chunks = []
        for accum_index in range(self.grad_accum):
            self._progress("rollout", accum_index=accum_index + 1)
            chunk = self._make_rollout_chunk(self._next_rdm_batch())
            chunks.append(chunk)
            shared_exit = (
                int(chunk.block_exit_indices[0])
                if chunk.block_exit_indices.numel()
                else chunk.first_frame_exit_index
            )
            self._progress(
                "rollout_complete",
                accum_index=accum_index + 1,
                detail=f"first_exit={chunk.first_frame_exit_index} shared_exit={shared_exit}",
            )
        forced_rollout_video = None
        if (
            self.log_full_all_chunks_rollout
            and (not self.disable_wandb)
            and (not self.config.no_visualize)
            and self.config.rdm_log_rollout_video
        ):
            forced_rollout_video = self._full_all_chunks_rollout_video(chunks[0])
        rollout_seconds = self._sync_time(rollout_start)
        feature_start = time.perf_counter()
        (pass1_features, feature_gradients, logs, rollout_video) = (
            self._mmd_feature_gradients(chunks)
        )
        if forced_rollout_video is not None:
            rollout_video = forced_rollout_video
        logs["timing/feature_and_kernel_seconds"] = self._sync_time(feature_start)
        logs["generation/bidirectional"] = float(self.bidirectional_full_video)
        first_exits = torch.tensor(
            [chunk.first_frame_exit_index for chunk in chunks],
            device=self.device,
            dtype=torch.float32,
        )
        if self.all_chunks_random_x0_4step:
            chunk_exits = torch.stack([chunk.block_exit_indices for chunk in chunks])
            num_blocks = int(self.config.num_training_frames) // int(self.config.num_frame_per_block)
            if tuple(chunk_exits.shape[1:]) != (num_blocks,):
                raise RuntimeError(f'Causal rollout must contain {num_blocks} blocks')
            if self.heterogeneous_chunkwise_schedule:
                if bool((chunk_exits[:, 0].float() != first_exits).any()):
                    raise RuntimeError("first-chunk replay state is inconsistent")
                if bool((chunk_exits[:, 1:] != chunk_exits[:, 1:2]).any()):
                    raise RuntimeError("later chunks did not share one exit")
                later_exits = chunk_exits[:, 1]
                logs["first_chunk_exit_mean"] = float(first_exits.mean())
                logs["later_chunk_exit_mean"] = float(later_exits.float().mean())
                del later_exits
            else:
                if bool((first_exits != -1).any()):
                    raise RuntimeError(
                        "shared-schedule chunkwise rollout used a separate first exit"
                    )
                if bool((chunk_exits != chunk_exits[:, :1]).any()):
                    raise RuntimeError("chunks did not share one exit")
                logs["shared_chunk_exit_mean"] = float(
                    chunk_exits[:, 0].float().mean()
                )
            del chunk_exits
        del first_exits
        if rollout_video is not None:
            logs["rollout_video"] = wandb.Video(
                rollout_video.numpy(),
                fps=int(self.config.rdm_rollout_video_fps),
                format="mp4",
                caption=f"step={self.step}, dataset_index={chunks[0].dataset_index}, prompt={chunks[0].prompt}",
            )
            del rollout_video
        vjp_start = time.perf_counter()
        mmd_gradients = []
        feature_parity_max = 0.0
        for index, (chunk, feature_gradient, cached_features) in enumerate(
            zip(chunks, feature_gradients, pass1_features), start=1
        ):
            self._progress("feature_vjp", accum_index=index)
            if cleanup_policy == 'sample':
                gc.collect()
                torch.cuda.empty_cache()
            (gradient, parity) = self._feature_vjp_to_latent(
                chunk, feature_gradient, cached_features
            )
            mmd_gradients.append(gradient)
            feature_parity_max = max(feature_parity_max, parity)
        logs["feature_replay_max_abs"] = feature_parity_max
        logs["timing/feature_vjp_seconds"] = self._sync_time(vjp_start)
        del feature_gradients, pass1_features
        torch.cuda.empty_cache()
        combined = mmd_gradients
        backward_start = time.perf_counter()
        replay_max = 0.0
        for index, (chunk, total_gradient) in enumerate(zip(chunks, combined), start=1):
            self._progress("generator_replay", accum_index=index)
            (
                recomputed,
                _,
                _,
                _,
                replay_context_noise,
                replay_first_exit,
                replay_first_trajectory,
                replay_block_exits,
                replay_block_trajectory,
            ) = self.model.rollout_from_noise(
                chunk.noise,
                chunk.conditional,
                context_noise_replay=chunk.context_noise,
                first_frame_exit_index_replay=chunk.first_frame_exit_index,
                first_frame_trajectory_noise_replay=chunk.first_frame_trajectory_noise,
                block_exit_indices_replay=chunk.block_exit_indices,
                block_trajectory_noise_replay=chunk.block_trajectory_noise,
            )
            if replay_first_exit != chunk.first_frame_exit_index:
                raise RuntimeError("first exit changed during replay")
            if not torch.equal(replay_block_exits, chunk.block_exit_indices):
                raise RuntimeError("chunk exits changed during replay")
            if float((replay_context_noise - chunk.context_noise).abs().max()) != 0.0:
                raise RuntimeError("context noise changed during replay")
            if (
                chunk.first_frame_trajectory_noise.numel()
                and float(
                    (replay_first_trajectory - chunk.first_frame_trajectory_noise)
                    .abs()
                    .max()
                )
                != 0.0
            ):
                raise RuntimeError("first trajectory noise changed during replay")
            if (
                chunk.block_trajectory_noise.numel()
                and float(
                    (replay_block_trajectory - chunk.block_trajectory_noise).abs().max()
                )
                != 0.0
            ):
                raise RuntimeError("chunk trajectory noise changed during replay")
            parity = float((recomputed.detach() - chunk.latent).abs().max())
            if not math.isfinite(parity) or parity > float(self.config.rdm_rollout_replay_atol):
                raise RuntimeError(f'rollout replay mismatch: {parity}')
            replay_max = max(replay_max, parity)
            torch.autograd.backward(recomputed, total_gradient.to(recomputed))
            del recomputed
        replay_tolerance = float(self.config.rdm_rollout_replay_atol)
        if not math.isfinite(replay_max) or replay_max > replay_tolerance:
            raise RuntimeError(
                f"rollout replay mismatch {replay_max:.6g} > {replay_tolerance:.6g}"
            )
        logs["rollout_replay_max_abs"] = replay_max
        logs["timing/generator_backward_seconds"] = self._sync_time(backward_start)
        del chunks, mmd_gradients, combined
        logs.update(self._additional_generator_backward())
        self._assert_frozen_parameter_grads()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm_generator
        )
        if not torch.isfinite(generator_grad_norm):
            raise RuntimeError("generator gradient norm is non-finite")
        self.generator_optimizer.step()
        self.generator_optimizer.zero_grad(set_to_none=True)
        self.generator_updates += 1
        logs["rdm_objective_value_proxy"] = logs["normalized_mmd"]
        logs.update(
            {
                "generator_grad_norm": float(generator_grad_norm),
                "generator_updates": float(self.generator_updates),
                "timing/rollout_seconds": rollout_seconds,
                "timing/generator_update_seconds": self._sync_time(phase_start),
                "gpu/peak_allocated_gib": self._global_max_scalar(
                    torch.cuda.max_memory_allocated(self.device) / 2**30
                ),
                "gpu/peak_reserved_gib": self._global_max_scalar(
                    torch.cuda.max_memory_reserved(self.device) / 2**30
                ),
            }
        )
        gc.collect()
        torch.cuda.empty_cache()
        self._progress("generator_complete")
        return logs
