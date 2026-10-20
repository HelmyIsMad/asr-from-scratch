"""Data augmentation.

Waveform domain (applied before the mel front-end, so it behaves physically):
    speed perturbation, RIR reverberation, additive noise at a target SNR, gain.

Feature domain (applied after CMVN):
    SpecAugment frequency/time masking.
"""

import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3"}


def resample(wave, orig_sr, target_sr):
    if orig_sr == target_sr:
        return wave
    new_len = max(1, int(round(wave.numel() * target_sr / orig_sr)))
    out = F.interpolate(wave.view(1, 1, -1), size=new_len, mode="linear", align_corners=False)
    return out.view(-1)


def speed_perturb(wave, factor):
    """Resample so the utterance plays `factor` times faster. factor > 1 -> shorter."""
    if abs(factor - 1.0) < 1e-6:
        return wave
    new_len = max(1, int(round(wave.numel() / factor)))
    out = F.interpolate(wave.view(1, 1, -1), size=new_len, mode="linear", align_corners=False)
    return out.view(-1)


def rms(x, eps=1e-8):
    return x.pow(2).mean().sqrt().clamp_min(eps)


def mix_at_snr(clean, noise, snr_db):
    noise = noise[: clean.numel()]
    if noise.numel() < clean.numel():
        noise = noise.repeat(int(clean.numel() / max(1, noise.numel())) + 1)[: clean.numel()]
    target_noise_rms = rms(clean) / (10.0 ** (snr_db / 20.0))
    noise = noise * (target_noise_rms / rms(noise))
    return clean + noise


def apply_rir(wave, rir):
    """Convolve with a room impulse response, keeping the original length."""
    rir = rir - rir.mean()
    rir = rir / rir.pow(2).sum().sqrt().clamp_min(1e-8)
    wet = F.conv1d(wave.view(1, 1, -1), rir.view(1, 1, -1), padding=rir.numel() - 1)
    return wet.view(-1)[: wave.numel()]


class AudioBank:
    """Randomly cropped excerpts from a directory of noise / RIR recordings."""

    def __init__(self, root):
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"augmentation source not found: {root}")
        self.files = sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXT)
        if not self.files:
            raise FileNotFoundError(f"no audio files under {root}")

    def sample(self, n_samples, sample_rate, max_tries=8):
        for _ in range(max_tries):
            path = random.choice(self.files)
            try:
                info = sf.info(str(path))
            except Exception:
                continue
            if info.frames < 1:
                continue
            take = min(n_samples, info.frames)
            start = random.randint(0, info.frames - take)
            data, file_sr = sf.read(
                str(path), start=start, frames=take, dtype="float32", always_2d=False
            )
            if data.ndim > 1:
                data = data.mean(axis=1)
            if data.size == 0:
                continue
            wave = torch.from_numpy(np.ascontiguousarray(data))
            wave = resample(wave, file_sr, sample_rate)
            if wave.numel() < n_samples:
                wave = wave.repeat(int(n_samples / wave.numel()) + 1)[:n_samples]
            return wave[:n_samples]
        return torch.zeros(n_samples)


class WaveformAugment:
    def __init__(self, cfg, sample_rate=16000):
        self.cfg = cfg or {}
        self.sample_rate = sample_rate
        speeds = self.cfg.get("speed_perturb") or [1.0]
        self.speeds = [float(s) for s in speeds]
        self.speed_p = float(self.cfg.get("speed_perturb_p", 0.5))
        self.gain_db = self.cfg.get("gain_db")
        self.gain_p = float(self.cfg.get("gain_p", 0.0))
        self.noise_p = float(self.cfg.get("noise_p", 0.0))
        self.noise_snr = self.cfg.get("noise_snr_db") or [5.0, 20.0]
        self.rir_p = float(self.cfg.get("rir_p", 0.0))
        self._noise = None
        self._rir = None
        if self.noise_p > 0 and self.cfg.get("noise_dir"):
            self._noise = AudioBank(self.cfg["noise_dir"])
        if self.rir_p > 0 and self.cfg.get("rir_dir"):
            self._rir = AudioBank(self.cfg["rir_dir"])

    def __call__(self, wave):
        if len(self.speeds) > 1 and random.random() < self.speed_p:
            factor = random.choice(self.speeds)
            wave = speed_perturb(wave, factor)
        if self._rir is not None and random.random() < self.rir_p:
            wave = apply_rir(wave, self._rir.sample(wave.numel(), self.sample_rate))
            wave = wave / wave.abs().max().clamp_min(1e-8) * 0.5
        if self._noise is not None and random.random() < self.noise_p:
            snr = random.uniform(float(self.noise_snr[0]), float(self.noise_snr[1]))
            wave = mix_at_snr(wave, self._noise.sample(wave.numel(), self.sample_rate), snr)
            peak = wave.abs().max()
            if peak > 1.0:  # protect against clipping without changing the SNR
                wave = wave / peak
        if self.gain_db and random.random() < self.gain_p:
            db = random.uniform(float(self.gain_db[0]), float(self.gain_db[1]))
            wave = wave * (10.0 ** (db / 20.0))
        return wave


def spec_augment(mel, freq_masks=0, freq_width=27, time_masks=0, time_width_frac=0.05, p=1.0, fill=0.0):
    """Mask bands of the (n_mels, T) log-mel tensor in place. Assumes CMVN, so fill=0 == mean."""
    if p < 1.0 and random.random() >= p:
        return mel
    n_mels, n_frames = mel.shape
    for _ in range(int(freq_masks)):
        width = random.randint(0, int(freq_width))
        if width <= 0:
            continue
        start = random.randint(0, max(0, n_mels - width))
        mel[start : start + width, :] = fill
    max_width = max(1, int(round(n_frames * float(time_width_frac))))
    for _ in range(int(time_masks)):
        width = random.randint(0, max_width)
        if width <= 0:
            continue
        start = random.randint(0, max(0, n_frames - width))
        mel[:, start : start + width] = fill
    return mel
