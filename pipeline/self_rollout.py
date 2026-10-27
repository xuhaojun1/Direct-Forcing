from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model.wan_adapter import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist


class SelfRolloutPipeline:
    def __init__(
        self,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: WanDiffusionWrapper,
        num_frame_per_block=3,
        independent_first_frame: bool = False,
        same_step_across_blocks: bool = True,
        last_step_only: bool = False,
        num_max_frames: int = 21,
        context_noise: int = 0,
        gradient_num_frames: Optional[int] = None,
        gradient_window_position: str = "tail",
        first_frame_denoising_step_list: Optional[List[int]] = None,
        all_chunks_random_x0_4step: bool = True,
    ):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[
                :-1
            ]  # remove the zero timestep for inference
        self.first_frame_denoising_step_list = first_frame_denoising_step_list

        # Wan specific hyperparameters
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length
        self.gradient_num_frames = gradient_num_frames
        self.gradient_window_position = gradient_window_position
        self.all_chunks_random_x0_4step = bool(all_chunks_random_x0_4step)
        if (independent_first_frame or first_frame_denoising_step_list is not None
                or not same_step_across_blocks or not all_chunks_random_x0_4step
                or num_frame_per_block != 3 or len(self.denoising_step_list) != 4
                or num_max_frames != 21 or gradient_window_position != "tail"):
            raise ValueError("Self-rollout requires uniform four-step, three-frame chunks")

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0, high=num_denoising_steps, size=(num_blocks,), device=device
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        if dist.is_initialized():
            dist.broadcast(indices, src=0)
        return indices.tolist()

    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        clean_image_or_video: torch.Tensor = None,  # same shape as noise
        initial_latent: Optional[torch.Tensor] = None,
        return_sim_step: bool = False,
        denoise_steps: Optional[int] = None,
        context_noise_replay: Optional[torch.Tensor] = None,
        return_context_noise_replay: bool = False,
        first_frame_exit_index_replay: Optional[int] = None,
        first_frame_trajectory_noise_replay: Optional[torch.Tensor] = None,
        return_first_frame_replay: bool = False,
        block_exit_indices_replay: Optional[torch.Tensor] = None,
        block_trajectory_noise_replay: Optional[torch.Tensor] = None,
        return_block_replay: bool = False,
        gradient_frame_index: Optional[int] = None,
        first_frame_skip_parameter_grad: bool = False,
        dynamic_kv_cache: bool = False,
        **conditional_dict,
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        if num_frames != 21 or initial_latent is not None or clean_image_or_video is not None:
            raise ValueError("Self-rollout starts from noise and generates 21 latent frames")
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block
        num_input_frames = 0
        num_output_frames = (
            num_frames + num_input_frames
        )  # add the initial latent frames
        if gradient_frame_index is not None:
            if batch_size != 1:
                raise ValueError("gradient_frame_index currently requires batch size 1")
            gradient_frame_index = int(gradient_frame_index)
            if gradient_frame_index < 0 or gradient_frame_index >= num_output_frames:
                raise ValueError(
                    f"gradient_frame_index={gradient_frame_index} is outside "
                    f"[0, {num_output_frames})"
                )
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype,
        )
        if context_noise_replay is not None and tuple(
            context_noise_replay.shape
        ) != tuple(noise.shape):
            raise ValueError(
                "context_noise_replay must match initial noise shape, got "
                f"{tuple(context_noise_replay.shape)} != {tuple(noise.shape)}"
            )
        captured_context_noise = (
            torch.empty_like(noise) if return_context_noise_replay else None
        )
        if first_frame_exit_index_replay not in {None, -1}:
            raise ValueError(
                "first-frame replay state requires first_frame_denoising_step_list"
            )
        if (
            first_frame_trajectory_noise_replay is not None
            and first_frame_trajectory_noise_replay.numel() != 0
        ):
            raise ValueError(
                "first-frame trajectory replay must be empty when the "
                "first-frame schedule is disabled"
            )
        if (
            first_frame_skip_parameter_grad
            and self.first_frame_denoising_step_list is None
        ):
            raise ValueError(
                "first_frame_skip_parameter_grad requires first-frame denoising steps"
            )

        # Release the previous step's caches before allocating new buffers.
        self.kv_cache1 = None
        self.kv_cache2 = None
        self.crossattn_cache = None
        # Keep allocator blocks warm across repeated rollouts to avoid a
        # device synchronization for every generated video.
        if not dynamic_kv_cache:
            torch.cuda.empty_cache()

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            num_frames=num_output_frames if dynamic_kv_cache else None,
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )

        # Step 2: Cache context feature
        current_start_frame = 0
        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(
            len(all_num_frames), num_denoising_steps, device=noise.device
        )
        if denoise_steps is not None:
            denoise_steps = int(denoise_steps)
            if denoise_steps < 0 or denoise_steps >= num_denoising_steps:
                raise ValueError(
                    f"denoise_steps={denoise_steps} is outside [0, {num_denoising_steps})"
                )
            exit_flags = [denoise_steps for _ in exit_flags]
        else:
            exit_flags = [int(exit_flags[0])] * len(exit_flags)
        if block_exit_indices_replay is not None:
            replay_exit_indices = torch.as_tensor(
                block_exit_indices_replay,
                device=noise.device,
                dtype=torch.long,
            )
            if tuple(replay_exit_indices.shape) != (len(all_num_frames),):
                raise ValueError(
                    "block_exit_indices_replay must have shape "
                    f"[{len(all_num_frames)}], got {tuple(replay_exit_indices.shape)}"
                )
            max_replay_steps = len(self.denoising_step_list)
            if bool(
                (
                    (replay_exit_indices < 0)
                    | (replay_exit_indices >= max_replay_steps)
                ).any()
            ):
                raise ValueError(
                    f"block replay exit indices must lie in [0,{max_replay_steps})"
                )
            if bool((replay_exit_indices != replay_exit_indices[0]).any()):
                raise ValueError("all-chunks replay requires one shared exit index")
            exit_flags = [int(value) for value in replay_exit_indices.tolist()]

        captured_block_trajectory_noise = None
        expected_block_trajectory_shape = (
            len(all_num_frames),
            len(self.denoising_step_list) - 1,
            batch_size,
            self.num_frame_per_block,
            num_channels,
            height,
            width,
        )
        if block_trajectory_noise_replay is not None:
            if tuple(block_trajectory_noise_replay.shape) != (
                expected_block_trajectory_shape
            ):
                raise ValueError(
                    "block trajectory-noise replay shape "
                    f"{tuple(block_trajectory_noise_replay.shape)} != "
                    f"{expected_block_trajectory_shape}"
                )
            for block_index, exit_index in enumerate(exit_flags):
                unused = block_trajectory_noise_replay[
                    block_index, int(exit_index) :
                ]
                if unused.numel() and bool(unused.count_nonzero()):
                    raise ValueError(
                        "unused block trajectory-noise replay entries must be zero"
                    )
        if return_block_replay:
            captured_block_trajectory_noise = noise.new_zeros(
                expected_block_trajectory_shape
            )

        # -1 plus an empty trajectory is the explicit replay contract for modes
        # without a separate first-frame denoising trajectory.
        first_frame_exit_index = -1
        captured_first_frame_trajectory_noise = []

        gradient_num_frames = (
            num_output_frames
            if self.gradient_num_frames is None or self.gradient_num_frames <= 0
            else min(self.gradient_num_frames, num_output_frames)
        )
        start_gradient_frame_index = max(0, num_output_frames - gradient_num_frames)
        end_gradient_frame_index = num_output_frames

        for block_index, current_num_frames in enumerate(all_num_frames):

            noisy_input = noise[
                :,
                current_start_frame
                - num_input_frames : current_start_frame
                + current_num_frames
                - num_input_frames,
            ]

            block_denoising_step_list = self.denoising_step_list
            block_exit_index = exit_flags[0]

            # Step 3.1: Spatial denoising loop
            # Such a loop corresponds to the truncated denoising algorithm:
            #    T -> \tau_1 -> \tau_2 ->...-> \tau —— enable grad ——> 0
            for index, current_timestep in enumerate(block_denoising_step_list):
                exit_flag = index == block_exit_index
                timestep = (
                    torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64,
                    )
                    * current_timestep
                )

                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame
                            * self.frame_seq_length,
                        )
                        next_timestep = block_denoising_step_list[index + 1]
                        if self.all_chunks_random_x0_4step and (
                            block_trajectory_noise_replay is not None
                        ):
                            transition_noise = block_trajectory_noise_replay[
                                block_index, index
                            ].to(denoised_pred)
                        else:
                            transition_noise = torch.randn_like(
                                denoised_pred.flatten(0, 1)
                            ).unflatten(0, denoised_pred.shape[:2])
                        if self.all_chunks_random_x0_4step and return_block_replay:
                            captured_block_trajectory_noise[
                                block_index, index
                            ].copy_(transition_noise.detach())
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            transition_noise.flatten(0, 1),
                            next_timestep
                            * torch.ones(
                                [batch_size * current_num_frames],
                                device=noise.device,
                                dtype=torch.long,
                            ),
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    if gradient_frame_index is None:
                        enable_grad = (
                            current_start_frame < end_gradient_frame_index
                            and current_start_frame + current_num_frames
                            > start_gradient_frame_index
                        )
                    else:
                        enable_grad = (
                            current_start_frame
                            <= gradient_frame_index
                            < current_start_frame + current_num_frames
                        )
                    if not enable_grad:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame
                                * self.frame_seq_length,
                            )
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame
                            * self.frame_seq_length,
                        )
                    break

            # Step 3.2: record the model's output
            output[
                :, current_start_frame : current_start_frame + current_num_frames
            ] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            replay_start = current_start_frame - num_input_frames
            replay_stop = replay_start + current_num_frames
            if context_noise_replay is None:
                # Preserve the random draw's flattened shape and layout.
                # byte-for-byte on callers that do not request replay state.
                sampled_context_noise = torch.randn_like(
                    denoised_pred.flatten(0, 1)
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                sampled_context_noise = context_noise_replay[
                    :, replay_start:replay_stop
                ].to(denoised_pred)
            if captured_context_noise is not None:
                captured_context_noise[:, replay_start:replay_stop].copy_(
                    sampled_context_noise.detach()
                )
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                sampled_context_noise.flatten(0, 1),
                context_timestep
                * torch.ones(
                    [batch_size * current_num_frames],
                    device=noise.device,
                    dtype=torch.long,
                ),
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 3.5: Return the denoised timestep
        range_step_list = self.denoising_step_list
        if exit_flags[0] == len(range_step_list) - 1:
            # corner case when \tau is the smallest non-zero timestep
            denoised_timestep_to = 0
            denoised_timestep_from = (
                1000
                - torch.argmin(
                    (
                        self.scheduler.timesteps.to(noise.device)
                        - range_step_list[exit_flags[0]].to(noise.device)
                    ).abs(),
                    dim=0,
                ).item()
            )
        else:
            denoised_timestep_to = (
                1000
                - torch.argmin(
                    (
                        self.scheduler.timesteps.to(noise.device)
                        - range_step_list[exit_flags[0] + 1].to(noise.device)
                    ).abs(),
                    dim=0,
                ).item()
            )
            denoised_timestep_from = (
                1000
                - torch.argmin(
                    (
                        self.scheduler.timesteps.to(noise.device)
                        - range_step_list[exit_flags[0]].to(noise.device)
                    ).abs(),
                    dim=0,
                ).item()
            )

        first_frame_trajectory_noise = None
        if return_first_frame_replay:
            if captured_first_frame_trajectory_noise:
                first_frame_trajectory_noise = torch.stack(
                    captured_first_frame_trajectory_noise, dim=0
                )
            else:
                first_frame_trajectory_noise = noise.new_empty(
                    (
                        0,
                        batch_size,
                        int(all_num_frames[0]),
                        num_channels,
                        height,
                        width,
                    )
                )
        if return_context_noise_replay and return_first_frame_replay:
            if captured_context_noise is None:
                raise RuntimeError("context-noise capture was not initialized")
            result = (
                output,
                denoised_timestep_from,
                denoised_timestep_to,
                captured_context_noise,
                int(first_frame_exit_index),
                first_frame_trajectory_noise,
            )
            if return_block_replay:
                if captured_block_trajectory_noise is None:
                    raise RuntimeError("block trajectory capture was not initialized")
                return result + (
                    torch.tensor(exit_flags, device=noise.device, dtype=torch.long),
                    captured_block_trajectory_noise,
                )
            return result
        if return_context_noise_replay:
            if captured_context_noise is None:
                raise RuntimeError("context-noise capture was not initialized")
            return (
                output,
                denoised_timestep_from,
                denoised_timestep_to,
                captured_context_noise,
            )
        if return_sim_step:  # False
            return (
                output,
                denoised_timestep_from,
                denoised_timestep_to,
                exit_flags[0] + 1,
            )

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(
        self,
        batch_size,
        dtype,
        device,
        num_frames: Optional[int] = None,
    ):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        cache_size = self.kv_cache_size
        if num_frames is not None:
            num_frames = int(num_frames)
            if num_frames <= 0:
                raise ValueError(f"num_frames must be positive, got {num_frames}")
            cache_size = num_frames * self.frame_seq_length

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append(
                {
                    "k": torch.zeros(
                        [batch_size, cache_size, 12, 128], dtype=dtype, device=device
                    ),
                    "v": torch.zeros(
                        [batch_size, cache_size, 12, 128], dtype=dtype, device=device
                    ),
                    "global_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                    "local_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                }
            )

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                {
                    "k": torch.zeros(
                        [batch_size, 512, 12, 128], dtype=dtype, device=device
                    ),
                    "v": torch.zeros(
                        [batch_size, 512, 12, 128], dtype=dtype, device=device
                    ),
                    "is_init": False,
                }
            )
        self.crossattn_cache = crossattn_cache
