"""Configuration loading with dotted CLI overrides."""

import copy
from pathlib import Path

import yaml


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
