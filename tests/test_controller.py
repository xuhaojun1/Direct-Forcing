"""Fit-controlled gradient budgets: scale invariance, current-step response and restart."""

import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

path = Path(__file__).resolve().parents[1]/'trainer/controller.py'
spec = importlib.util.spec_from_file_location('joint_schedule_test', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
GNMREWController = module.GNMREWController
FixedCCAController = module.FixedCCAController
GNMController = module.GNMController


class GradientOnlyTests(unittest.TestCase):
    def test_fixed_ratios_match_gradient_budget_independently_of_loss(self):
        for ratio in (.25, .5, .75):
            a, b = GNMController(target_ratio=ratio), GNMController(target_ratio=ratio)
            for step, cca, rdm in ((1, 40., 1.), (2, .1, 2.), (3, 8., .5)):
                x = a.update(step, 1000./step, cca, rdm)
                y = b.update(step, object(), cca, rdm)
                self.assertEqual(x, y)
                self.assertAlmostEqual(a.weight*cca/rdm, ratio)
                self.assertEqual(x['cca/target_gradient_ratio'], ratio)
                self.assertAlmostEqual(x['cca/gradient_scale'], rdm/cca)
                self.assertFalse(any(k.startswith('rew/') for k in x))
            restored = GNMController.from_state_dict(json.loads(json.dumps(a.state_dict())))
            self.assertEqual(restored.target_ratio, ratio)
            self.assertEqual(a.update(4, 1., 3., 7.), restored.update(4, 9999., 3., 7.))

    def test_legacy_checkpoint_defaults_to_unit_ratio_and_invalid_ratios_fail(self):
        a = GNMController()
        a.update(1, 0., 4., 2.)
        legacy = a.state_dict(); legacy.pop('target_ratio')
        restored = GNMController.from_state_dict(legacy)
        self.assertEqual(restored.target_ratio, 1.)
        self.assertEqual(a.weight, restored.weight)
        for ratio in (-.5, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                GNMController(target_ratio=ratio)

    def test_loss_values_and_history_cannot_change_the_coefficient(self):
        a, b = GNMController(), GNMController()
        for step, cca, rdm in ((1, 4., .2), (2, .1, 2.), (3, 8., .5)):
            x = a.update(step, 1000./step, cca, rdm)
            y = b.update(step, object(), cca, rdm)
            self.assertEqual(x, y)
            self.assertEqual(a.state_dict(), b.state_dict())
            self.assertAlmostEqual(a.weight*cca/rdm, 1.)
            self.assertEqual(x['cca/target_gradient_ratio'], 1.)
            self.assertFalse(any(k.startswith('rew/') for k in x))
        self.assertFalse(any('loss' in k or 'ema' in k for k in a.state_dict()))

    def test_norm_changes_and_loss_unit_rescaling_have_expected_effect(self):
        a, b = GNMController(), GNMController()
        a.update(1, 1., 4., 2.)
        b.update(1, 7., 28., 6.)
        self.assertAlmostEqual(a.weight, b.weight*7./3.)
        a.update(2, 0., 2., 4.)
        self.assertEqual(a.weight, 2.)

    def test_resume_and_limited_cca_keep_the_correct_controller_kind(self):
        a = GNMController()
        a.update(1, .5, 2., .1)
        b = GNMController.from_state_dict(json.loads(json.dumps(a.state_dict())))
        self.assertEqual(a.update(2, .1, 3., .4), b.update(2, 100., 3., .4))
        limited = module.ScheduledCCAController(a, 2, last_step=2)
        restored = module.ScheduledCCAController.from_state_dict(limited.state_dict())
        self.assertEqual(restored.base.state_dict()['kind'], module.GNM_KIND)
        self.assertEqual(restored.skip(3)['cca/coefficient'], 0.)
        with self.assertRaises(ValueError):
            GNMController.from_state_dict(GNMREWController().state_dict())
        with self.assertRaises(ValueError):
            GNMREWController.from_state_dict(b.state_dict())

    def test_invalid_norms_do_not_mutate_state_and_zero_norms_remain_finite(self):
        a = GNMController()
        before = a.state_dict()
        for cca, rdm in ((-1., 2.), (1., float('nan')), (float('inf'), 1.)):
            with self.assertRaises(ValueError):
                a.update(1, .5, cca, rdm)
            self.assertEqual(before, a.state_dict())
        a.update(1, .5, 0., 0.)
        self.assertEqual(a.weight, 0.)
        a.update(2, .5, 0., 1.)
        self.assertTrue(math.isfinite(a.weight))


class FixedWeightTests(unittest.TestCase):
    def test_coefficients_ignore_loss_and_gradient_scale_changes(self):
        for coefficient in (1., .3, 0.):
            controller = FixedCCAController(coefficient)
            for step, loss, cca, rdm in [
                    (1, 10., 100., 1e-6), (2, .01, .001, 100.),
                    (50, 20., 0., 0.), (150, 0., 1000., .5)]:
                logs = controller.update(step, loss, cca, rdm)
                self.assertEqual(controller.weight, coefficient)
                self.assertEqual(logs, {'cca/coefficient': coefficient})

    def test_resume_preserves_constant_coefficient_and_step(self):
        controller = FixedCCAController(.3)
        controller.update(1, .2, 4., .1)
        restored = FixedCCAController.from_state_dict(json.loads(json.dumps(controller.state_dict())))
        self.assertEqual(restored.state_dict(), controller.state_dict())
        self.assertEqual(restored.update(2, 5., .001, 100.), {'cca/coefficient': .3})
        self.assertEqual(restored.last_step, 2)

    def test_rejects_invalid_coefficients_and_mixed_weighting_states(self):
        for value in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                FixedCCAController(value)
        with self.assertRaises(ValueError):
            FixedCCAController.from_state_dict(GNMREWController().state_dict())
        with self.assertRaises(ValueError):
            GNMREWController.from_state_dict(FixedCCAController(.3).state_dict())
        controller = FixedCCAController(1.)
        with self.assertRaises(ValueError):
            controller.update(2, .1, 1., 1.)


class GNMREWTests(unittest.TestCase):
    def test_no_ema_tracks_target_after_abrupt_gradient_and_loss_changes(self):
        c = GNMREWController()
        for step, loss, cca, rdm in [(1, 1., 40., 1.), (2, .5, .1, 2.),
                                      (3, .2, 5., .01), (4, 2., .2, .5)]:
            logs = c.update(step, loss, cca, rdm)
            self.assertAlmostEqual(c.weight*cca/rdm, .1+.9*min(loss,1.))
            self.assertEqual(c.current_loss, loss)
            self.assertEqual(c.current_cca_norm, cca)
            self.assertEqual(c.current_rdm_norm, rdm)
            self.assertFalse(any(key.endswith('_ema') for key in logs))
        self.assertEqual(c.initial_loss,1.)

    def test_no_ema_uses_no_intermediate_history_and_resumes_exactly(self):
        a, b = GNMREWController(), GNMREWController()
        for c in (a,b): c.update(1,1.,40.,1.)
        a.update(2,.01,1000.,.0001)
        b.update(2,3.,.0001,1000.)
        self.assertEqual(a.update(3,.4,.2,.1),b.update(3,.4,.2,.1))
        restored = GNMREWController.from_state_dict(json.loads(json.dumps(a.state_dict())))
        self.assertEqual(restored.state_dict()['kind'],module.GNM_REW_KIND)
        self.assertEqual(restored.update(4,.2,.1,.3),a.update(4,.2,.1,.3))
        bad = restored.state_dict()
        bad['kind'] = 'retired_weighting_kind'
        with self.assertRaisesRegex(ValueError,'Incompatible'):
            GNMREWController.from_state_dict(bad)

    def test_first_step_balances_actual_gradients_not_loss_coefficients(self):
        c = GNMREWController()
        result = c.update(1, .46, 3.872, .460)
        self.assertEqual(c.target_ratio, 1.)
        self.assertAlmostEqual(c.weight, .460/3.872)
        self.assertAlmostEqual(c.weight*3.872/.460, 1.)
        self.assertEqual(result['rew/relative_error'], 1.)

    def test_reduced_fitting_error_reduces_target_ratio(self):
        c = GNMREWController()
        c.update(1, 1., 2., 1.)
        c.update(11, .25, 2., 1.)
        self.assertAlmostEqual(c.current_loss, .25)
        self.assertAlmostEqual(c.target_ratio, .325)
        self.assertAlmostEqual(c.weight*2., .325)
        self.assertEqual(c.initial_loss, 1.)

    def test_high_error_plateau_does_not_trigger_a_timed_decay(self):
        c = GNMREWController()
        for step in range(1, 151):
            c.update(step, .6, 2., .4)
        self.assertAlmostEqual(c.target_ratio, 1.)
        self.assertAlmostEqual(c.weight, .2)

    def test_mmd_gradient_scale_changes_coefficient_without_changing_fit_budget(self):
        c = GNMREWController()
        c.update(1, 1., 2., 1.)
        initial = c.weight
        c.update(11, 1., 2., 11.)
        self.assertAlmostEqual(c.current_rdm_norm, 11.)
        self.assertEqual(c.target_ratio, 1.)
        self.assertAlmostEqual(c.weight, 11.*initial)

    def test_minimum_is_a_gradient_ratio_not_a_fixed_loss_coefficient(self):
        c = GNMREWController()
        c.update(1, 1., 2., 1.)
        for step in range(2, 1502):
            c.update(step, 0., .1, 5.)
        self.assertAlmostEqual(c.target_ratio, .1, places=12)
        self.assertAlmostEqual(c.weight, 5., places=10)
        self.assertAlmostEqual(c.weight*.1/5., .1, places=12)

    def test_loss_unit_rescaling_preserves_the_relative_gradient_budget(self):
        c, d = GNMREWController(), GNMREWController()
        values = [(1, .8, 2., 1.), (2, .4, 1.7, .8), (11, .3, .6, 1.3), (21, .2, .4, 1.)]
        for step, loss, cca, rdm in values:
            c.update(step, loss, cca, rdm)
            d.update(step, 7.*loss, 7.*cca, 3.*rdm)
            self.assertAlmostEqual(c.target_ratio, d.target_ratio)
            self.assertAlmostEqual(c.weight, d.weight*7./3.)

    def test_error_increase_can_restore_cca_budget(self):
        c = GNMREWController()
        c.update(1, 1., 2., 1.)
        c.update(11, .1, 2., 1.)
        smaller = c.target_ratio
        c.update(21, 10., 2., 1.)
        self.assertGreater(c.relative_error, 1.)
        self.assertEqual(c.target_ratio, 1.)
        self.assertGreater(c.target_ratio, smaller)

    def test_serialized_restart_preserves_every_future_coefficient(self):
        c = GNMREWController()
        for step, loss in [(1, .7), (2, .5), (3, .4)]:
            c.update(step, loss, 3./step, .4*step)
        d = GNMREWController.from_state_dict(json.loads(json.dumps(c.state_dict())))
        for step in range(4, 51):
            values = (step, .2+.05*math.sin(step), 2./step, .1+.01*step)
            self.assertEqual(c.update(*values), d.update(*values))
            self.assertEqual(c.state_dict(), d.state_dict())
        self.assertEqual(GNMREWController.from_state_dict(GNMREWController().state_dict()).last_step, 0)

    def test_rejects_bad_measurements_without_advancing_state(self):
        c = GNMREWController()
        c.update(1, .6, 2., 1.)
        before = c.state_dict()
        for args in [(1, .4, 2., 1.), (2, float('nan'), 2., 1.),
                     (2, -.1, 2., 1.), (2, .4, -1., 1.), (2, .4, 2., float('inf'))]:
            with self.assertRaises(ValueError):
                c.update(*args)
            self.assertEqual(c.state_dict(), before)

    def test_zero_gradients_and_cosine_roundoff_are_finite(self):
        c = GNMREWController()
        c.update(1, -1e-8, 0., 0.)
        self.assertTrue(math.isfinite(c.weight))
        self.assertEqual(c.weight, 0.)
        self.assertEqual(c.target_ratio, .1)
        c.update(2, 0., 0., 1.)
        self.assertTrue(math.isfinite(c.weight))

    def test_rejects_legacy_or_corrupt_checkpoint_state(self):
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            GNMREWController.from_state_dict({'cca_gradient_scale': .1, 'last_step': 20})
        state = GNMREWController().state_dict()
        state['last_step'] = 5
        with self.assertRaises(ValueError):
            GNMREWController.from_state_dict(state)
        for kwargs in [{'ratio_min': 2.}, {'ratio_start': 0.}, {'epsilon': 0.}]:
            with self.assertRaises(ValueError):
                GNMREWController(**kwargs)


if __name__ == '__main__':
    unittest.main(verbosity=2)
