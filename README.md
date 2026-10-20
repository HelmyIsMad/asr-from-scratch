# asr-from-scratch

A CTC speech recogniser built from zero on LibriSpeech — own log-mel front-end, own augmentation, own QuartzNet-style encoder, own decoder. **No pretrained weights anywhere.**

Training happens on Colab (T4). This repo is code only; nothing is trained locally.

## What to expect

| Setup | test-clean WER |
|---|---|
| `configs/base.yaml`, 50 epochs, 100 h, greedy | ~22–28% |
| `configs/base.yaml` + beam search | ~2–3 points better |
| `configs/large.yaml`, 960 h, A100 | ~12–16% |

Whisper gets 2–4% because it saw 680k h of weakly-labelled audio. These numbers are the honest cost of doing it yourself, and the point of the project is the ablation table, not the leaderboard.

## Pipeline

```
flac ──► decode + mono + resample ──► augment (waveform) ──► log-mel ──► CMVN ──► SpecAugment
                                                                              │
                                                                    QuartzNet (4x subsample)
                                                                              │
                                                                     CTC ──► greedy / beam
```

| Stage | Where | Detail |
|---|---|---|
| Features | `asr/features.py` | 80 mel bins, 25 ms window, 10 ms hop, log, per-utterance CMVN |
| Augmentation | `asr/augment.py` | speed perturb 0.9/1.0/1.1, RIR, noise @ 5–20 dB SNR, gain, SpecAugment |
| Model | `asr/model.py` | 2 stride-2 convs (4x), time-channel separable blocks, residual, CTC head |
| Decoding | `asr/decode.py` | greedy + prefix beam search (no external LM) |
| Training | `asr/train.py` | AdamW, warmup + cosine, AMP, grad accumulation, length bucketing, resume |

## Repo layout

```
asr/            features, augment, dataset, model, decode, metrics, train, evaluate, prepare
configs/        base.yaml (T4) · large.yaml (A100) · smoke.yaml (plumbing only)
scripts/        make_tiny_data.py · pack.py · build_colab_notebook.py
colab.ipynb     the whole run, cell by cell
```

## Training on Colab

1. Open `colab.ipynb` in Colab (`File > Upload notebook`), set **Runtime > T4 GPU**.
2. Run cell 1 to get the code: either leave `REPO_URL` as your fork, or upload `asr.zip` built with `python scripts/pack.py`.
3. Run the cells in order. Cell 4 downloads ~7 GB of LibriSpeech, cell 5 builds manifests, cell 6 is a 10-minute smoke test.
4. Cell 7 trains. **Re-run it after any disconnect** — it resumes from `last.pt` on Drive.

Checkpoints and metrics go to `MyDrive/asr-runs/<run>/` (`last.pt`, `best.pt`, `metrics.csv`, `train.log`).

### Monitoring from outside Colab

Store a GitHub token as a Colab secret named `GH_TOKEN` (key icon in the sidebar) and set `REPO_URL`. Every epoch then commits `logs/<run>/{metrics.csv,tail.log,status.json}` to the repo, so progress is visible without keeping the tab open.

## Ablation (the actual deliverable)

Train four runs with identical epochs, compare `best.pt`:

```bash
# 1. no augmentation
--set augment.speed_perturb='[1.0]' augment.gain_p=0.0 augment.specaug.p=0.0
# 2. + speed perturbation
--set augment.gain_p=0.0 augment.specaug.p=0.0
# 3. + SpecAugment  (this is configs/base.yaml)
# 4. + noise + reverberation
--set augment.noise_dir=/content/data/musan/noise augment.noise_p=0.3 \
      augment.rir_dir=/content/data/rirs augment.rir_p=0.2
```

Noise/RIR sources: [MUSAN](https://www.openslr.org/17/), [OpenSLR RIRs](https://www.openslr.org/28/).

## Local use: inference only

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m asr.evaluate --checkpoint best.pt --split test --beam-width 0
```

Download `best.pt` from Drive first. `evaluate.py` uses the config stored inside the checkpoint, so no local config is needed. On the 4 GB P1000 expect roughly real time for greedy decoding.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | activations scale with batch × channels × T | halve `train.batch_size`, raise `train.grad_accum`; cap `data.max_duration` |
| `loss=inf` early | utterance shorter than its transcript under CTC | `zero_infinity=True` is already on; raise `data.max_duration` filter threshold, check transcript normalisation |
| Epochs far slower than 12 min | `num_workers` too low for FLAC decode | raise to 8 on Colab, `persistent_workers` is already on |
| WER plateaus ~90% | LR too high or warmup too short | try `train.lr=0.005`, `train.warmup_epochs=4` |
| Validation WER rising while loss falls | overfitting on 100 h | stop at `best.pt`, enable noise augmentation |

## License

MIT for the code. LibriSpeech is CC BY 4.0 — cite Panayotov et al. 2015 if you publish results.
