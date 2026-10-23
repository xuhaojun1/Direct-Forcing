"""Behavioral checks against metadata captured before the CCA/RDM rename."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from omegaconf import OmegaConf
import torch

from trainer.controller import GNMController, GNMREWController, FixedCCAController
from utils.config import load_spec, compile_runtime
from utils.legacy import canonical_metadata
from utils.portable import portable_contract
from trainer.references.bank import load_reference
from utils.hashing import file_sha256
from utils.training_schedule import configured_learning_rate
from utils.wan_resume import validate_resume_contract

FIXTURES = Path(__file__).parent / 'fixtures/legacy_names'


def fixture(name):
    return json.loads((FIXTURES/name).read_text())


def runtime(config='configs/chunkwise_4step.yaml', overrides=()):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'metadata.json'
        path.write_text(json.dumps(dict(num_rows=6505, text_table_sha256='text-hash',
                                        prompt_rows_sha256='prompt-hash')))
        spec = load_spec(config, [f'paths.metadata={path}', *overrides])
        return compile_runtime(spec, load_references=False)


class PaperNamingTests(unittest.TestCase):
    def test_main_and_baseline_equal_the_pre_rename_recipes(self):
        for old, config in [('bs128_runtime.json', 'configs/chunkwise_4step.yaml'),
                            ('v16_runtime.json', 'configs/chunkwise_4step_v16.yaml')]:
            expected = canonical_metadata(fixture(old))
            actual = portable_contract(OmegaConf.to_container(runtime(config)))
            self.assertEqual(actual, expected)

    def test_legacy_controller_restart_has_identical_future_coefficients(self):
        for name, cls in [('gnm_rew', GNMREWController), ('gnm', GNMController),
                          ('fixed', FixedCCAController)]:
            source = fixture(name+'_controller.json')
            before = copy.deepcopy(source)
            controller = cls.from_state_dict(source['state'])
            actual = [controller.update(*args) for args in
                      [(4, .2, .5, 1.), (5, 0., .4, .9), (6, .9, 1.5, .2)]]
            self.assertEqual(actual, canonical_metadata(source['expected_future_logs']))
            self.assertEqual(source, before)

    def test_legacy_resume_keeps_encoder_checks_strict(self):
        old = fixture('bs128_runtime.json')
        saved = {key: old[key] for key in ('max_steps', 'virdm_expected_world_size',
                                          'virdm_expected_global_rows', 'pair_lr_schedule')}
        saved['joint'] = {k: v for k, v in old.items() if k.startswith('joint_')}
        current = canonical_metadata(saved)
        self.assertEqual(validate_resume_contract(saved, current, 150)['changes'], {})
        current['joint']['joint_secondary_backend']['rdm_encoder_checkpoint_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'objective/data/LR'):
            validate_resume_contract(saved, current, 150, fork=True)

    def test_legacy_reference_revision_loads_without_rewriting_the_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'reference.pt'
            metadata = dict(encoder_id='test', encoder_checkpoint_sha256='weights',
                            input_mode='native', input_height=2, input_width=2, pool='cls',
                            feature_dim=4, feature_normalize=False, num_rows=2,
                            rows_per_video=1, virdm_reference_revision='test_v1', nystrom_landmarks=2)
            torch.save(dict(Z=torch.ones(2, 4), alpha=torch.ones(2), sigma=2.,
                            k_rr=1., metadata=metadata), path)
            digest = file_sha256(path)
            ref = load_reference(path, device='cpu', expected_encoder_id='test',
                expected_checkpoint_sha256='weights', expected_height=2, expected_width=2,
                expected_rows=2, expected_reference_revision='test_v1', expected_encoder_contract={},
                expected_landmarks=2, expected_feature_dim=4, expected_rows_per_video=1)
            self.assertEqual(ref.metadata['rdm_reference_revision'], 'test_v1')
            self.assertEqual(file_sha256(path), digest)

    def test_conflicting_metadata_aliases_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            canonical_metadata({'virdm_expected_global_rows': 64, 'rdm_expected_global_rows': 128})

    def test_main_250_updates_keep_the_original_fixed_lr_horizon(self):
        cfg = runtime()
        self.assertEqual(cfg.trainer, 'direct_forcing')
        self.assertEqual(cfg.max_steps, 250)
        self.assertEqual(cfg.rdm_expected_global_rows, 128)
        self.assertEqual(cfg.joint_cca_batch, 32)
        self.assertEqual(cfg.rdm_grad_accum_steps, 16)
        self.assertEqual(cfg.lr_cosine_end, 125)
        self.assertGreater(configured_learning_rate(50, cfg, total=250), cfg.lr_floor)
        for step in (125, 150, 200, 225, 250):
            self.assertEqual(configured_learning_rate(step, cfg, total=250), 5e-7)


if __name__ == '__main__':
    unittest.main()
