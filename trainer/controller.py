"""CCA weighting modes: fixed, GNM, or GNM with REW.

Uses only training loss and parameter gradients already computed for the two
branches. OT evaluation never updates this state. All coefficients are detached.
"""

from dataclasses import asdict, dataclass
import math
from utils.legacy import canonical_metadata


GNM_REW_KIND = 'gnm_rew_v1'
FIXED_CCA_KIND = 'fixed_cca_v1'
GNM_KIND = 'gnm_v1'


class ScheduledCCAController:
    """Stop CCA after an inclusive optimizer update, retaining controller state."""
    def __init__(self, base, stop_step, last_step=0):
        if type(stop_step) is not int or stop_step < 1:
            raise ValueError('CCA stop step must be a positive integer')
        if type(last_step) is not int or last_step < 0 or base.last_step != min(last_step, stop_step):
            raise ValueError('Inconsistent stopped CCA/controller state')
        self.base, self.stop_step, self.last_step = base, stop_step, last_step

    @property
    def weight(self):
        return self.base.weight if self.last_step <= self.stop_step else 0.

    def update(self, step, cca_loss, cca_norm, rdm_norm):
        if step > self.stop_step:
            raise ValueError('CCA is disabled; skip without fabricating loss measurements')
        logs = self.base.update(step, cca_loss, cca_norm, rdm_norm)
        self.last_step = step
        return logs

    def skip(self, step):
        if step != self.last_step + 1 or step <= self.stop_step:
            raise ValueError('Invalid stopped CCA step')
        self.last_step = step
        return {'cca/coefficient': 0., 'cca/target_gradient_ratio': 0.,
                'cca/active': 0., 'joint/weighted_gradient_ratio': 0.,
                'cca/gradient_norm': 0., 'cca/global_samples': 0., 'timing/cca_seconds': 0.}

    def state_dict(self):
        return dict(kind='scheduled_cca_v1', stop_step=self.stop_step,
                    last_step=self.last_step, base=self.base.state_dict())

    @classmethod
    def from_state_dict(cls, state):
        state = canonical_metadata(state)
        if state.get('kind') != 'scheduled_cca_v1':
            raise ValueError('Expected a stopped-CCA controller state')
        inner = state['base']
        factories = {FIXED_CCA_KIND: FixedCCAController, GNM_REW_KIND: GNMREWController,
                     GNM_KIND: GNMController}
        if inner.get('kind') not in factories:
            raise ValueError('Incompatible stopped-CCA controller kind')
        factory = factories[inner['kind']]
        return cls(factory.from_state_dict(inner), state['stop_step'], state['last_step'])


@dataclass
class FixedCCAController:
    """Constant CCA loss coefficient; observations never change the weight."""
    coefficient: float
    last_step: int = 0

    def __post_init__(self):
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError('Fixed CCA coefficient must be finite and nonnegative')
        if type(self.last_step) is not int or self.last_step < 0:
            raise ValueError('Invalid fixed-weight step')

    @property
    def weight(self):
        return self.coefficient

    def update(self, step, cca_loss, cca_norm, rdm_norm):
        if type(step) is not int or step <= self.last_step or (self.last_step == 0 and step != 1):
            raise ValueError('Weighting requires increasing optimizer steps, starting at 1')
        self.last_step = step
        return {'cca/coefficient': self.coefficient}

    def state_dict(self):
        return {'kind': FIXED_CCA_KIND, **asdict(self)}

    @classmethod
    def from_state_dict(cls, state):
        data = canonical_metadata(state)
        if data.pop('kind', None) != FIXED_CCA_KIND:
            raise ValueError('Incompatible fixed-weight checkpoint state')
        return cls(**data)


@dataclass
class GNMController:
    """Current generator-gradient norm calibration, with a fixed target ratio.

    The CCA loss is accepted for interface compatibility, but is never read or
    stored. There is no fitting-error baseline, loss decay, or EMA.
    """
    epsilon: float = 1e-12
    last_step: int = 0
    current_cca_norm: float | None = None
    current_rdm_norm: float | None = None
    target_ratio: float = 1.

    def __post_init__(self):
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError('Invalid gradient calibration epsilon')
        if not math.isfinite(self.target_ratio) or self.target_ratio < 0:
            raise ValueError('Fixed gradient ratio must be finite and nonnegative')
        if type(self.last_step) is not int or self.last_step < 0:
            raise ValueError('Invalid gradient calibration step')
        values = (self.current_cca_norm, self.current_rdm_norm)
        if self.last_step == 0:
            if any(v is not None for v in values):
                raise ValueError('Uninitialized gradient calibration contains measurements')
        elif any(v is None or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('Missing or invalid gradient calibration state')

    @property
    def weight(self):
        if self.current_cca_norm is None:
            return self.target_ratio
        return self.target_ratio*self.current_rdm_norm / max(self.current_cca_norm, self.epsilon)

    def update(self, step, cca_loss, cca_norm, rdm_norm):
        if type(step) is not int or step <= self.last_step or (self.last_step == 0 and step != 1):
            raise ValueError('Weighting requires increasing optimizer steps, starting at 1')
        if not all(math.isfinite(x) and x >= 0 for x in (cca_norm, rdm_norm)):
            raise ValueError('Expected finite nonnegative gradient norms')
        gradient_scale = rdm_norm / max(cca_norm, self.epsilon)
        coefficient = self.target_ratio*gradient_scale
        if not math.isfinite(coefficient):
            raise FloatingPointError('Nonfinite gradient calibration coefficient')
        self.current_cca_norm, self.current_rdm_norm = float(cca_norm), float(rdm_norm)
        self.last_step = step
        return {'cca/coefficient': coefficient, 'cca/target_gradient_ratio': self.target_ratio,
                'cca/gradient_scale': gradient_scale}

    def state_dict(self):
        return {'kind': GNM_KIND, **asdict(self)}

    @classmethod
    def from_state_dict(cls, state):
        data = canonical_metadata(state)
        if data.pop('kind', None) != GNM_KIND:
            raise ValueError('Incompatible gradient-only controller state')
        return cls(**data)


@dataclass
class GNMREWController:
    """Gradient norm matching followed by residual error weighting (GNM + REW)."""
    ratio_start: float = 1.
    ratio_min: float = .1
    epsilon: float = 1e-12
    last_step: int = 0
    initial_loss: float | None = None
    current_loss: float | None = None
    current_cca_norm: float | None = None
    current_rdm_norm: float | None = None

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (
                self.ratio_start, self.ratio_min, self.epsilon)) or not (
                0 <= self.ratio_min <= self.ratio_start
                and self.ratio_start > 0 and self.epsilon > 0):
            raise ValueError('Invalid GNM/REW configuration')
        if type(self.last_step) is not int or self.last_step < 0:
            raise ValueError('Invalid weighting step')
        values = (self.initial_loss, self.current_loss, self.current_cca_norm, self.current_rdm_norm)
        if self.last_step == 0:
            if any(x is not None for x in values):
                raise ValueError('Uninitialized weighting state contains measurements')
        elif any(x is None or not math.isfinite(x) or x < 0 for x in values):
            raise ValueError('Missing or invalid current-step weighting state')

    @property
    def relative_error(self):
        if self.initial_loss is None:
            return 1.
        return self.current_loss / max(self.initial_loss, self.epsilon)

    @property
    def target_ratio(self):
        remaining = min(1., max(0., self.relative_error))
        return self.ratio_min + (self.ratio_start-self.ratio_min)*remaining

    @property
    def gradient_scale(self):
        if self.current_cca_norm is None:
            return 1.
        return self.current_rdm_norm / max(self.current_cca_norm, self.epsilon)

    @property
    def weight(self):
        """Actual coefficient multiplying the unweighted CCA gradient."""
        return self.target_ratio*self.gradient_scale

    def update(self, step, cca_loss, cca_norm, rdm_norm):
        if type(step) is not int or step <= self.last_step or (self.last_step == 0 and step != 1):
            raise ValueError('Weighting requires increasing optimizer steps, starting at 1')
        if not all(math.isfinite(x) for x in (cca_loss, cca_norm, rdm_norm)) or (
                cca_loss < -1e-6 or cca_norm < 0 or rdm_norm < 0):
            raise ValueError('Expected finite nonnegative loss and gradient norms')
        # FP32 cosine can exceed one by a few ulps at a perfect match.
        loss = max(float(cca_loss), 0.)
        initial_loss = loss if self.initial_loss is None else self.initial_loss
        current_loss, current_cca, current_rdm = loss, float(cca_norm), float(rdm_norm)
        relative_error = current_loss/max(initial_loss, self.epsilon)
        target = self.ratio_min+(self.ratio_start-self.ratio_min)*min(1., relative_error)
        coefficient = target*current_rdm/max(current_cca, self.epsilon)
        if not math.isfinite(relative_error) or not math.isfinite(coefficient):
            raise FloatingPointError('Nonfinite CCA weighting coefficient')
        self.initial_loss = initial_loss
        self.current_loss, self.current_cca_norm, self.current_rdm_norm = current_loss, current_cca, current_rdm
        self.last_step = step
        logs = {'cca/coefficient': coefficient,
                'cca/target_gradient_ratio': target,
                'cca/gradient_scale': self.gradient_scale,
                'rew/relative_error': relative_error,
                'rew/initial_loss': initial_loss}
        logs.update({'rew/loss_current': current_loss,
                     'rew/cca_gradient_norm_current': current_cca,
                     'rew/rdm_gradient_norm_current': current_rdm})
        return logs

    def state_dict(self):
        return {'kind': GNM_REW_KIND, **asdict(self)}

    @classmethod
    def from_state_dict(cls, state):
        data = canonical_metadata(state)
        if data.pop('kind', None) != GNM_REW_KIND:
            raise ValueError('Incompatible controller state; use the code snapshot associated with the old run')
        return cls(**data)
