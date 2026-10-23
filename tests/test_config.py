import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
from utils.config import ROOT, load_spec, compile_runtime, inference_runtime, resolve_path
from utils.portable import portable_contract
from utils.wan_resume import validate_resume_contract


class ConfigTests(unittest.TestCase):
    def compile(self, config='configs/chunkwise_4step.yaml', overrides=()):
        with tempfile.TemporaryDirectory() as temporary:
            metadata = Path(temporary) / 'metadata.json'
            metadata.write_text(json.dumps(dict(num_rows=6505,
                text_table_sha256='text-hash', prompt_rows_sha256='prompt-hash')))
            spec = load_spec(config, [f'paths.metadata={metadata}', *overrides])
            return compile_runtime(spec, load_references=False)

    def test_main_and_baseline_keep_independent_feature_spaces(self):
        main = self.compile()
        baseline = self.compile('configs/chunkwise_4step_v16.yaml')
        self.assertEqual(main.joint_cca_backend.rdm_encoder_id, 'videomae_v2_vitl_l23_5x16_224')
        self.assertEqual(main.joint_secondary_backend.rdm_encoder_id, 'videomae_v2_vitl_l15_5x16_224')
        self.assertEqual(main.rdm_input_mode, 'normalized_wan_latent')
        self.assertEqual(main.joint_cca_backend.rdm_input_mode, 'native_video')
        self.assertIsNone(baseline.joint_cca_backend)
        self.assertIsNone(baseline.get('joint_secondary_backend'))
        self.assertEqual(baseline.rdm_encoder_id, 'videomae_v2_vitl_l15_5x16_224')

    def test_controller_switches_and_batch_repartition(self):
        main = self.compile(overrides=['training.world_size=4'])
        self.assertEqual(main.rdm_grad_accum_steps * 4, 128)
        self.assertEqual(main.joint_cca_batch, 32)
        self.assertEqual(main.joint_controller_mode, 'gnm_rew_v1')
        fixed = self.compile(overrides=['controller.rew=false', 'controller.fixed_ratio=0.5'])
        self.assertEqual(fixed.joint_controller_mode, 'gnm_v1')
        self.assertEqual(fixed.joint_fixed_gradient_ratio, .5)
        plain = self.compile(overrides=['controller.rew=false', 'controller.gnm=false', 'controller.fixed_weight=0.3'])
        self.assertEqual(plain.joint_controller_mode, 'fixed_cca_v1')
        self.assertEqual(plain.joint_cca_fixed_weight, .3)
        with self.assertRaisesRegex(ValueError, 'REW'):
            self.compile(overrides=['controller.gnm=false'])

    def test_paths_do_not_depend_on_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = Path.cwd()
            try:
                os.chdir(temporary)
                spec = load_spec('configs/chunkwise_4step_v16.yaml')
                self.assertEqual(Path(spec.paths.wan), ROOT / 'checkpoints/Wan2.1-T2V-1.3B')
                with patch.dict(os.environ, {'DIRECT_FORCING_TEST_ASSETS': temporary}):
                    self.assertEqual(resolve_path('$DIRECT_FORCING_TEST_ASSETS/model'), Path(temporary) / 'model')
            finally:
                os.chdir(previous)

    def test_inference_needs_no_training_cache(self):
        cfg = inference_runtime(load_spec('configs/chunkwise_4step.yaml'))
        self.assertEqual(list(cfg.denoising_step_list), [1000, 750, 500, 250])
        self.assertEqual(cfg.num_frame_per_block, 3)
        self.assertFalse(cfg.model_kwargs.get("sink_fifo_attention", False))

    def test_changed_reference_receipt_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            metadata = directory / 'metadata.json'
            metadata.write_text(json.dumps(dict(num_rows=6505, text_table_sha256='text', prompt_rows_sha256='prompts')))
            reference = directory / 'videomae_v2'
            reference.mkdir()
            (reference / 'receipt.json').write_text('{}')
            spec = load_spec('configs/chunkwise_4step_v16.yaml',
                [f'paths.metadata={metadata}', f'paths.references={directory}'])
            with self.assertRaisesRegex(ValueError, 'Reference contract mismatch'):
                compile_runtime(spec)

    def test_portable_contract_retains_content_and_objective(self):
        first = dict(checkpoint_path='/disk-a/model', checkpoint_sha256='hash',
                     joint={'asset_paths': {'wan': '/disk-a/wan'}, 'loss': 'cosine'})
        second = dict(checkpoint_path='/disk-b/model', checkpoint_sha256='hash',
                      joint={'asset_paths': {'wan': '/disk-b/wan'}, 'loss': 'cosine'})
        self.assertEqual(portable_contract(first), portable_contract(second))
        second['checkpoint_sha256'] = 'different'
        self.assertNotEqual(portable_contract(first), portable_contract(second))

    def test_resume_repartitions_both_feature_branches_but_preserves_objective(self):
        def contract(world, steps):
            cfg = self.compile(overrides=[f'training.world_size={world}', f'training.steps={steps}'])
            keys = ('max_steps', 'rdm_expected_world_size', 'rdm_expected_global_rows', 'lr_schedule')
            result = {key: cfg[key] for key in keys}
            result['joint'] = {key: value for key, value in OmegaConf.to_container(cfg).items()
                               if key.startswith('joint_')}
            return portable_contract(result)
        saved, current = contract(8, 150), contract(4, 225)
        audit = validate_resume_contract(saved, current, 150, fork=True)
        self.assertEqual(audit['target_world_size'], 4)
        current['joint']['joint_secondary_backend']['rdm_encoder_checkpoint_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'objective/data/LR'):
            validate_resume_contract(saved, current, 150, fork=True)


if __name__ == '__main__':
    unittest.main()
