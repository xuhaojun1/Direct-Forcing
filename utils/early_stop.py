"""Checkpoint-cadence temporal OT stopping; select the preceding checkpoint."""
import math
import json
from pathlib import Path


def policy_for_encoder(encoder):
    metrics = {'videomae_v2': 'eval/temporal_ot',
               'videomae_v2_dit': 'eval/videomae_temporal_ot'}
    if encoder not in metrics:
        raise ValueError('Temporal OT stopping supports VideoMAE v2 L16 or L16+DiT MMD')
    return dict(version=1, metric=metrics[encoder], checkpoint_every=25,
                consecutive_increases=2, comparison='strict_greater',
                selection='previous_checkpoint', baseline='first_saved_checkpoint')


class CheckpointOTStopper:
    def __init__(self, policy):
        self.policy = dict(policy)
        expected = policy_for_encoder('videomae_v2_dit' if self.policy.get('metric') ==
                                      'eval/videomae_temporal_ot' else 'videomae_v2')
        if self.policy != expected:
            raise ValueError('Unsupported checkpoint OT stopping policy')
        self.history = []

    @property
    def increases(self):
        count = 0
        for previous, current in reversed(list(zip(self.history, self.history[1:]))):
            if current['value'] <= previous['value']:
                break
            count += 1
        return count

    @property
    def stopped(self):
        return self.increases >= self.policy['consecutive_increases']

    def observe(self, step, value, checkpoint):
        interval = self.policy['checkpoint_every']
        expected_step = self.history[-1]['step'] + interval if self.history else interval
        if self.stopped or step != expected_step:
            raise ValueError('OT observations must follow saved checkpoints at 25-step intervals')
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError('Checkpoint temporal OT must be finite and nonnegative')
        self.history.append(dict(step=int(step), value=value, checkpoint=str(checkpoint)))
        return self.stopped

    def state_dict(self):
        return dict(policy=dict(self.policy), history=[dict(row) for row in self.history])

    @classmethod
    def from_state_dict(cls, state):
        result = cls(state['policy'])
        for row in state['history']:
            result.observe(row['step'], row['value'], row['checkpoint'])
        return result

    def selection(self, completed_step, maximum_steps, final_checkpoint):
        if self.stopped:
            if completed_step != self.history[-1]['step']:
                raise ValueError('Stopping step disagrees with the metric history')
            selected = self.history[-2]
            latest = self.history[-1]['checkpoint']
            reason = 'two_consecutive_temporal_ot_increases'
        else:
            if completed_step != maximum_steps:
                raise ValueError('Training neither reached its limit nor triggered OT stopping')
            latest = (self.history[-1]['checkpoint'] if self.history and
                      self.history[-1]['step'] == completed_step else str(final_checkpoint))
            selected = dict(step=completed_step, checkpoint=latest)
            reason = 'max_steps_reached'
        return dict(reason=reason, completed_updates=completed_step, maximum_updates=maximum_steps,
                    selected_step=selected['step'], selected_checkpoint=selected['checkpoint'],
                    last_checkpoint=latest, state=self.state_dict())


def checkpoint_metric(report, metric):
    if 'metrics' in report:
        return report['metrics'][metric]
    if metric != 'eval/temporal_ot':
        raise ValueError('Dual-encoder temporal OT report is missing its named metric')
    return report['temporal_ot']


def load_training_selection(run_dir, maximum_steps, policy):
    record = json.loads((Path(run_dir)/'training_selection.json').read_text())
    stopper = CheckpointOTStopper.from_state_dict(record['state'])
    if stopper.policy != dict(policy) or record['maximum_updates'] != maximum_steps:
        raise ValueError('Selected checkpoint policy or training limit changed')
    completed = record['completed_updates']
    if not 0 < completed <= maximum_steps:
        raise ValueError('Invalid completed training update count')
    expected = stopper.selection(completed, maximum_steps,
                                 Path(run_dir)/f'checkpoint_model_{completed:06d}/model.pt')
    if record != expected:
        raise ValueError('Selected checkpoint disagrees with the recorded OT decisions')
    for key in ('last_checkpoint', 'selected_checkpoint'):
        if not Path(record[key]).is_file():
            raise FileNotFoundError(record[key])
    return record
