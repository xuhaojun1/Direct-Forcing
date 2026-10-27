"""Shared causal Wan video model and replayable self-rollout."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from model.base import ReplayableVideoModel
from pipeline import SelfRolloutPipeline


class CausalVideoModel(ReplayableVideoModel):
    """Wan generator shared by CCA and RDM in Direct-Forcing."""

    def __init__(self, args, device):
        from utils.validation import validate_short_video_runtime
        validate_short_video_runtime(args)
        super().__init__(args, device)
        self.num_frame_per_block = 3
        self.num_training_frames = 21
        self.same_step_across_blocks = True
        self.independent_first_frame = False
        self.rdm_generation_mode = "chunkwise"
        self.rdm_all_chunks_random_x0_4step = True
        # Empty compatibility field for serialized replay metadata.
        self.rdm_first_frame_denoising_step_list = None
        self.generator.model.num_frame_per_block = self.num_frame_per_block
        if bool(getattr(args, "gradient_checkpointing", False)):
            self.generator.enable_gradient_checkpointing()
        self.inference_pipeline: SelfRolloutPipeline | None = None

    def _initialize_inference_pipeline(self):
        """Create the shared self-rollout pipeline with replayable denoising noise."""
        self.inference_pipeline = SelfRolloutPipeline(
            denoising_step_list=self.denoising_step_list,
            first_frame_denoising_step_list=(
                self.rdm_first_frame_denoising_step_list
            ),
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_training_frames,
            context_noise=self.args.context_noise,
            gradient_num_frames=self.num_training_frames,
            gradient_window_position="tail",
            all_chunks_random_x0_4step=(self.rdm_all_chunks_random_x0_4step),
        )

    def rollout_from_noise(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        *,
        context_noise_replay: torch.Tensor | None = None,
        first_frame_exit_index_replay: int | None = None,
        first_frame_trajectory_noise_replay: torch.Tensor | None = None,
        block_exit_indices_replay: torch.Tensor | None = None,
        block_trajectory_noise_replay: torch.Tensor | None = None,
        force_all_chunks_full: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        int,
        int,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Replayable 21-frame RDM rollout from caller-owned noise."""

        expected_tail = (16, 60, 104)
        if noise.ndim != 5 or tuple(noise.shape[2:]) != expected_tail:
            raise ValueError(
                f"noise must be [B,T,{expected_tail}], got {tuple(noise.shape)}"
            )
        num_frames = int(noise.shape[1])
        if num_frames != 21:
            raise ValueError(f"RDM requires 21 latent frames, got {num_frames}")
        denoise_steps = 3 if force_all_chunks_full else None
        simulation_kwargs = dict(
            noise=noise,
            clean_image_or_video=None,
            denoise_steps=denoise_steps,
            context_noise_replay=context_noise_replay,
            return_context_noise_replay=True,
            first_frame_exit_index_replay=first_frame_exit_index_replay,
            first_frame_trajectory_noise_replay=(first_frame_trajectory_noise_replay),
            return_first_frame_replay=True,
            block_exit_indices_replay=block_exit_indices_replay,
            block_trajectory_noise_replay=block_trajectory_noise_replay,
            return_block_replay=self.rdm_all_chunks_random_x0_4step,
            **conditional_dict,
        )
        if getattr(self.args, "generator_activation_cpu_offload", False):
            with torch.autograd.graph.save_on_cpu(pin_memory=False):
                simulation_result = self._consistency_backward_simulation(
                    **simulation_kwargs
                )
        else:
            simulation_result = self._consistency_backward_simulation(
                **simulation_kwargs
            )
        (
            pred,
            denoised_from,
            denoised_to,
            captured_context_noise,
            first_frame_exit_index,
            captured_first_frame_trajectory_noise,
            captured_block_exit_indices,
            captured_block_trajectory_noise,
        ) = simulation_result
        pred = self._slice_generated_output(pred)
        return (
            pred,
            None,
            -1 if denoised_from is None else denoised_from,
            -1 if denoised_to is None else denoised_to,
            captured_context_noise.detach(),
            int(first_frame_exit_index),
            captured_first_frame_trajectory_noise.detach(),
            captured_block_exit_indices.detach(),
            captured_block_trajectory_noise.detach(),
        )
