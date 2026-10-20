"""Learning-rate schedules and config loading."""

import copy
import math
from pathlib import Path

import yaml


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


def _set_nested(cfg, dotted_key, value):
    node = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot descend into non-dict at {part!r}")
    node[parts[-1]] = value


def load_config(path, overrides=None):
    with Path(path).open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, _, raw = item.partition("=")
        _set_nested(cfg, key.strip(), yaml.safe_load(raw))
    return cfg
