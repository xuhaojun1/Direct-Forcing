"""Deterministic data and learning-rate schedules."""

import math
import random


def learning_rate(step, *, total=150, peak=2e-6, floor=5e-7, warmup=10, decay_start=100):
    if not (1 <= step <= total and 0 < floor <= peak and warmup >= 0):
        raise ValueError("Invalid LR schedule")
    warmup, decay_start = min(warmup, total), min(decay_start, total)
    if step <= warmup:
        return peak * step / max(1, warmup)
    if step <= decay_start:
        return peak
    fraction = (step-decay_start)/max(1, total-decay_start)
    return floor + (peak-floor)*0.5*(1+math.cos(math.pi*fraction))


def cosine_window_learning_rate(step, *, peak=2.5e-6, floor=5e-7, start=0, end=125):
    """LR for the update producing checkpoint ``step``, matching Elastic 1.3B.

    The fixed cosine horizon is independent of the run's max_steps. A one-step
    smoke test and a resume use exactly the corresponding full-run LR.
    """
    if (type(step) is not int or step < 0 or type(start) is not int or type(end) is not int
            or not 0 <= start < end or not all(math.isfinite(v) for v in (peak, floor))
            or not 0 < floor <= peak):
        raise ValueError('Invalid cosine-window LR schedule')
    if step <= start:
        return peak
    if step >= end:
        return floor
    fraction = (step-start)/(end-start)
    return floor+(peak-floor)*.5*(1+math.cos(math.pi*fraction))


def configured_learning_rate(step, config, *, total):
    schedule = config.get('lr_schedule', 'warmup_tail')
    if schedule == 'cosine_window_v1':
        return cosine_window_learning_rate(step, peak=float(config.lr),
            floor=float(config.lr_floor), start=int(config.lr_decay_start),
            end=int(config.lr_cosine_end))
    if schedule != 'warmup_tail':
        raise ValueError(f'Unknown training LR schedule: {schedule}')
    return learning_rate(step, total=total, peak=float(config.lr),
        floor=float(config.lr_floor), warmup=int(config.lr_warmup),
        decay_start=int(config.lr_decay_start))


def training_batch(eligible, *, step, population=64, seed=42):
    """Shuffle each epoch, drop its short tail, and resume exactly from update index."""
    batches = len(eligible)//population
    if step < 1 or batches < 1:
        raise ValueError("Not enough distinct training rows for this population")
    epoch, batch = divmod(step-1, batches)
    order = list(eligible)
    random.Random(seed+epoch).shuffle(order)
    return order[batch*population:(batch+1)*population]
