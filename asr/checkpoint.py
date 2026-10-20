"""Atomic checkpoints and GitHub log reporting."""

import os
import subprocess
from pathlib import Path

import torch


def save_atomic(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def push_logs(repo_dir, paths, message, timeout=120):
    """Commit the given paths and push. Never raises: log reporting must not kill training."""
    repo_dir = str(repo_dir)
    identity = ["-c", "user.name=asr-bot", "-c", "user.email=asr-bot@users.noreply.github.com"]
    steps = [["git", "add", "--", *paths], ["git", *identity, "commit", "-m", message], ["git", "push"]]
    output = []
    for cmd in steps:
        try:
            res = subprocess.run(
                cmd, cwd=repo_dir, capture_output=True, text=True, timeout=timeout
            )
        except Exception as exc:  # pragma: no cover - network/git failures
            return False, f"{' '.join(cmd)}: {exc}"
        output.append((res.stdout + res.stderr).strip())
        if res.returncode != 0 and "nothing to commit" not in output[-1]:
            return False, "\n".join(output)
    return True, "\n".join(output)
