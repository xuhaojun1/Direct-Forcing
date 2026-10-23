"""Exercise the real rollout/replay loop with a small causal generator."""
import unittest

import torch
from omegaconf import OmegaConf

from pipeline.self_rollout import SelfRolloutPipeline
from utils.validation import validate_short_video_runtime


class TinyScheduler:
    timesteps = torch.arange(1000, 0, -1)

    def add_noise(self, clean, noise, timestep):
        sigma = timestep.to(clean).reshape(-1, 1, 1, 1) / 1000
        return (1 - sigma) * clean + sigma * noise


class TinyGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.3))
        self.calls = []

    def forward(self, noisy_image_or_video, timestep, kv_cache, current_start, **kwargs):
        x = noisy_image_or_video
        # Detached causal history models the ordinary frozen KV writer.
        prefix = sum(v for k, v in kv_cache.items() if k < current_start)
        prediction = x.tanh() * self.weight + 0.01 * prefix
        kv_cache[current_start] = x.detach().mean().clone()
        self.calls.append((current_start, torch.is_grad_enabled(), int(timestep[0, 0])))
        return x - prediction, prediction


class TinyRollout(SelfRolloutPipeline):
    def _initialize_kv_cache(self, *args, **kwargs):
        self.kv_cache1 = {}

    def _initialize_crossattn_cache(self, *args, **kwargs):
        self.crossattn_cache = {}


def make_pipeline():
    return TinyRollout(torch.tensor([1000, 750, 500, 250]), TinyScheduler(),
                       TinyGenerator(), same_step_across_blocks=True,
                       all_chunks_random_x0_4step=True, gradient_num_frames=21)


CAPTURE = dict(return_context_noise_replay=True, return_first_frame_replay=True,
               return_block_replay=True)


def replay_kwargs(result):
    return dict(context_noise_replay=result[3], first_frame_exit_index_replay=result[4],
                first_frame_trajectory_noise_replay=result[5],
                block_exit_indices_replay=result[6], block_trajectory_noise_replay=result[7])


class RolloutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.noise = torch.randn(1, 21, 2, 2, 2)

    def test_every_exit_replays_values_and_parameter_gradients(self):
        for exit_index in range(4):
            with self.subTest(exit=exit_index):
                pipeline = make_pipeline()
                first = pipeline.inference_with_trajectory(self.noise, denoise_steps=exit_index, **CAPTURE)
                gradient, = torch.autograd.grad(first[0].square().sum(), pipeline.generator.weight)
                torch.manual_seed(999)  # A replay must not depend on new trajectory noise.
                second = pipeline.inference_with_trajectory(self.noise, **replay_kwargs(first), **CAPTURE)
                replay_gradient, = torch.autograd.grad(second[0].square().sum(), pipeline.generator.weight)
                torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
                torch.testing.assert_close(gradient, replay_gradient, rtol=0, atol=0)
                self.assertTrue(torch.equal(first[6], torch.full((7,), exit_index)))
                self.assertEqual(int(first[7][:, exit_index:].count_nonzero()), 0)

    def test_random_exit_is_shared_by_all_seven_chunks(self):
        result = make_pipeline().inference_with_trajectory(self.noise, **CAPTURE)
        self.assertEqual(result[6].unique().numel(), 1)
        self.assertTrue(0 <= int(result[6][0]) < 4)

    def test_only_exit_denoisers_receive_gradients_and_writers_are_frozen(self):
        pipeline = make_pipeline()
        pipeline.inference_with_trajectory(self.noise, denoise_steps=2, **CAPTURE)
        calls = pipeline.generator.calls
        self.assertEqual(len(calls), 7 * 4)
        for block in range(7):
            group = calls[block * 4:(block + 1) * 4]
            self.assertEqual([entry[1] for entry in group], [False, False, True, False])
            self.assertEqual([entry[2] for entry in group], [1000, 750, 500, 0])

    def test_replay_rejects_nonshared_exits_and_unused_noise(self):
        pipeline = make_pipeline()
        result = pipeline.inference_with_trajectory(self.noise, denoise_steps=0, **CAPTURE)
        replay = replay_kwargs(result)
        replay['block_exit_indices_replay'] = torch.arange(7) % 4
        with self.assertRaisesRegex(ValueError, 'shared exit'):
            pipeline.inference_with_trajectory(self.noise, **replay, **CAPTURE)
        replay = replay_kwargs(result)
        replay['block_trajectory_noise_replay'] = torch.ones_like(result[7])
        with self.assertRaisesRegex(ValueError, 'unused block'):
            pipeline.inference_with_trajectory(self.noise, **replay, **CAPTURE)

    def test_retired_recipes_fail_before_loading_weights(self):
        base = OmegaConf.create(dict(denoising_step_list=[1000, 750, 500, 250]))
        validate_short_video_runtime(base)
        for key, value in [('rdm_context_gradient_mode', 'sgf'), ('rdm_representation', 'pooled'),
                           ('rdm_dynamic_reg_enabled', True), ('joint_rolling', {'enabled': True}),
                           ('num_frame_per_block', 1), ('rdm_generation_mode', 'full_video_bidirectional')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_short_video_runtime(OmegaConf.merge(base, {key: value}))
        with self.assertRaises(ValueError):
            validate_short_video_runtime(OmegaConf.merge(base, {'model_kwargs': {'sink_size': 3}}))
