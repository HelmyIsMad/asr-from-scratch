"""Regenerate colab.ipynb from readable Python source."""

import json
from pathlib import Path

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md(
    """
# From-scratch CTC ASR on LibriSpeech

Trains a QuartzNet-style CTC recogniser with **no pretrained weights**: own front-end, own augmentation, own encoder, own decoder.

Runtime -> Change runtime type -> **T4 GPU**.

Run top to bottom. If the session dies, reconnect and re-run cells 1-7: training resumes from the last checkpoint on Drive.
"""
)

code(
    """
import shutil
import torch

print("torch", torch.__version__)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - enable a GPU runtime")
free = shutil.disk_usage("/content")
print(f"disk free {free.free / 1e9:.1f} GB")
assert torch.cuda.is_available(), "Runtime > Change runtime type > T4 GPU"
assert free.free > 30e9, "not enough disk for LibriSpeech + checkpoints"
"""
)

md(
    """
## 1. Get the code

Either set `REPO_URL` to your fork, or leave it empty and upload `asr.zip` (built locally with `python scripts/pack.py`).
"""
)

code(
    """
import os
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/HelmyIsMad/asr-from-scratch.git"
WORKDIR = Path("/content/asr")

# Optional: lets the private repo clone and lets the run push its logs.
try:
    from google.colab import userdata
    GH_TOKEN = userdata.get("GH_TOKEN")
except Exception:
    GH_TOKEN = None

def sh(cmd, cwd=None, check=True, secret=None, stream=False):
    print(f"$ {cmd.replace(secret, '***') if secret else cmd}", flush=True)
    if stream:  # inherit stdout/stderr so progress bars stay live
        rc, out, err = subprocess.run(cmd, shell=True, cwd=cwd).returncode, "", ""
    else:
        proc = subprocess.run(cmd, shell=True, cwd=cwd, text=True, capture_output=True)
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    if out.strip():
        print(out.rstrip()[-4000:])
    if rc != 0:
        if err.strip():
            print(err.rstrip()[-4000:])
        if check:
            raise RuntimeError(f"command failed with exit {rc}: {cmd}")
    return rc

if not WORKDIR.exists():
    if REPO_URL:
        url = REPO_URL
        if GH_TOKEN:
            url = "https://" + GH_TOKEN + "@" + REPO_URL.split("//", 1)[1]
        try:
            sh(f"git clone --depth 1 {url} {WORKDIR}", secret=GH_TOKEN)
        except subprocess.CalledProcessError:
            raise RuntimeError(
                "git clone failed (exit 128 is usually auth). If the repo is private, add a "
                "Colab secret named GH_TOKEN with 'repo' scope, or make the repo public."
            ) from None
    else:
        from google.colab import files
        print("Upload asr.zip built by `python scripts/pack.py`")
        uploaded = files.upload()
        name = next(iter(uploaded))
        sh(f"unzip -q -o '/content/{name}' -d /content")
        sh(f"rm -f '/content/{name}'")
print(sorted(p.name for p in WORKDIR.iterdir()))
"""
)

md(
    """
## 2. Dependencies and optional log push

Colab already ships a CUDA build of torch - do not reinstall it. If you store a `GH_TOKEN` secret in Colab (key icon in the sidebar), per-epoch metrics get committed to `logs/` in your repo so training can be monitored from outside.
"""
)

code(
    """
import os

sh("pip install -q soundfile pyyaml tqdm")

PUSH_LOGS = bool(GH_TOKEN and REPO_URL)
if PUSH_LOGS:
    authed = "https://" + GH_TOKEN + "@" + REPO_URL.split("//", 1)[1]
    sh(f"git -C {WORKDIR} remote set-url origin {authed}", secret=GH_TOKEN)
    os.environ["ASR_PUSH_LOGS"] = "1"
print("log push:", PUSH_LOGS)
"""
)

md("## 3. Checkpoints on Drive (survives disconnects)")

code(
    """
from pathlib import Path

RUNS = Path("/content/drive/MyDrive/asr-runs")
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as exc:
    print("drive mount failed, falling back to /content/runs:", exc)
    RUNS = Path("/content/runs")
RUNS.mkdir(parents=True, exist_ok=True)
print(RUNS)
"""
)

md(
    """
## 4. LibriSpeech (train-clean-100 + dev-clean + test-clean, ~7 GB)

Already downloaded? This cell skips it.
"""
)

code(
    """
import shutil
from pathlib import Path

DATA = Path("/content/data")
LS = DATA / "LibriSpeech"
SPLITS = ["train-clean-100", "dev-clean", "test-clean"]
DATA.mkdir(parents=True, exist_ok=True)

# openslr throttles per connection, so 16 parallel streams beat a single wget by a lot.
sh("apt-get -qq install -y aria2 > /dev/null 2>&1 || true")
ARIA = shutil.which("aria2c")

def fetch(url, dest):
    if ARIA:
        sh(f"aria2c -x 16 -s 16 -k 1M -c --file-allocation=none -d /content -o {Path(dest).name} '{url}'", stream=True)
    else:
        sh(f"wget -c --progress=dot:giga '{url}' -O {dest}", stream=True)

for split in SPLITS:
    marker = LS / split / ".complete"
    if marker.exists():
        print("have", split)
        continue
    if (LS / split).exists():
        # A dataset downloaded by an older version of this notebook has no marker.
        existing = len(list((LS / split).rglob("*.flac")))
        if existing >= 1000:
            marker.write_text("ok", encoding="utf-8")
            print("have", split, f"({existing} flac)")
            continue
        # Otherwise the previous run died during extraction: start that split over.
        print(f"removing partial extraction of {split} ({existing} flac)")
        shutil.rmtree(LS / split)

    tar = f"{split}.tar.gz"
    tar_path = Path("/content") / tar
    fetch(f"https://www.openslr.org/resources/12/{tar}", str(tar_path))

    # A truncated tarball would otherwise fail inside tar with a cryptic gzip error.
    expect = {"train-clean-100": 5.5e9, "dev-clean": 2.7e8, "test-clean": 2.7e8}[split]
    got = tar_path.stat().st_size
    if got < expect * 0.95:
        tar_path.unlink(missing_ok=True)
        raise RuntimeError(f"{tar} is only {got / 1e9:.2f} GB, expected ~{expect / 1e9:.2f} GB - re-run this cell")

    rc = sh(f"tar -xzf {tar_path} -C {DATA}", check=False)
    if rc != 0:
        tar_path.unlink(missing_ok=True)
        raise RuntimeError(f"extraction of {tar} failed; deleted the tarball, re-run this cell to fetch it again")
    sh(f"rm -f {tar_path}")
    marker.write_text("ok", encoding="utf-8")

for split in SPLITS:
    n = len(list((LS / split).rglob("*.flac")))
    print(f"{split}: {n} flac files")
    assert n > 100, f"{split} looks truncated ({n} files)"
"""
)

md("## 5. Build manifests (transcript normalisation + duration filtering)")

code(
    """
MANIFESTS = Path("/content/manifests")
sh(f"python -m asr.prepare --root {LS} --out-dir {MANIFESTS}", cwd=WORKDIR)
"""
)

md(
    """
## 6. Smoke test (~10 minutes)

One epoch cap of 30 batches. If this passes, the real run is plumbing only.
"""
)

code(
    """
sh(
    f"python -m asr.train --config configs/base.yaml "
    f"--set data.root={LS} data.manifest_dir={MANIFESTS} "
    f"train.out_dir={RUNS}/smoke train.epochs=1 train.limit_batches=30 train.val_limit_batches=5 train.log_every=10",
    cwd=WORKDIR,
)
"""
)

md(
    """
## 7. Train

Resumable: re-run this cell after a disconnect and it continues from `last.pt`.

Rough T4 timings with `configs/base.yaml`: ~10-14 min/epoch, ~50 epochs. Expect **~22-28% WER greedy** on test-clean when finished - this is a from-scratch model on 100 h, not a Whisper replacement. `configs/large.yaml` does better if you can get an A100.
"""
)

code(
    """
sh(
    f"python -m asr.train --config configs/base.yaml "
    f"--set data.root={LS} data.manifest_dir={MANIFESTS} train.out_dir={RUNS}/base",
    cwd=WORKDIR,
)
"""
)

md("## 8. Evaluate (greedy and beam search)")

code(
    """
sh(f"python -m asr.evaluate --checkpoint {RUNS}/base/best.pt --split test --beam-width 0", cwd=WORKDIR)
sh(f"python -m asr.evaluate --checkpoint {RUNS}/base/best.pt --split test --beam-width 16 --limit 200", cwd=WORKDIR)
"""
)

md(
    """
## 9. Augmentation ablation

Each run is a single extra training job. The comparison table is the real deliverable.

| run | command |
|---|---|
| no augmentation | `--set augment.speed_perturb='[1.0]' augment.gain_p=0.0 augment.specaug.p=0.0` |
| + speed perturb | `--set augment.gain_p=0.0 augment.specaug.p=0.0` |
| + SpecAugment | `--set augment.specaug.p=0.6` (default) |
| + noise + RIR | upload MUSAN/RIRs, then `--set augment.noise_dir=/content/data/musan/noise augment.noise_p=0.3 augment.rir_dir=/content/data/rirs augment.rir_p=0.2` |

Keep epochs identical across runs, and compare `best.pt` only.
"""
)

NOTEBOOK = {
    "cells": [
        {
            "cell_type": kind,
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")],
            **({"execution_count": None, "outputs": []} if kind == "code" else {}),
        }
        for kind, text in CELLS
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "colab.ipynb"
    out.write_text(json.dumps(NOTEBOOK, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({len(NOTEBOOK['cells'])} cells)")


if __name__ == "__main__":
    main()
