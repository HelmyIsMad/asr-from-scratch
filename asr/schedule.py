"""Learning-rate schedules."""

import math


def lr_at(step, base_lr, warmup_steps, total_steps, schedule="cosine", min_ratio=0.0):
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if schedule == "constant":
        return base_lr
    span = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / span)
    if schedule == "cosine":
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    elif schedule == "linear":
        factor = 1.0 - progress
    else:
        raise ValueError(f"unknown schedule: {schedule}")
    return base_lr * (min_ratio + (1.0 - min_ratio) * factor)
