"""Log-mel front-end and feature utilities. Pure torch, no external audio models."""

import math

import torch


def hz_to_mel(freq):
    """HTK mel scale."""
    return 2595.0 * math.log10(1.0 + float(freq) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (float(mel) / 2595.0) - 1.0)


def mel_filterbank(sample_rate=16000, n_fft=400, n_mels=80, f_min=0.0, f_max=None):
    """Triangular mel filters, shape (n_mels, n_fft // 2 + 1)."""
    f_max = float(f_max) if f_max is not None else sample_rate / 2.0
    n_freqs = n_fft // 2 + 1
    fft_freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs)

    edges = torch.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz = torch.tensor([mel_to_hz(m) for m in edges.tolist()], dtype=torch.float32)

    fb = torch.zeros(n_mels, n_freqs)
    for i in range(n_mels):
        left, centre, right = hz[i], hz[i + 1], hz[i + 2]
        rising = (fft_freqs - left) / max(float(centre - left), 1e-8)
        falling = (right - fft_freqs) / max(float(right - centre), 1e-8)
        fb[i] = torch.clamp(torch.minimum(rising, falling), min=0.0)
    return fb


class LogMel:
    """Waveform (T,) -> log-mel spectrogram (n_mels, T')."""

    def __init__(
        self,
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        win_length=400,
        n_mels=80,
        f_min=0.0,
        f_max=None,
        log_offset=1e-10,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.log_offset = log_offset
        self.register = mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)
        self.window = torch.hann_window(win_length)

    def n_frames(self, n_samples):
        """Frame count for a waveform of n_samples (center=True)."""
        return 1 + n_samples // self.hop_length

    def __call__(self, wave):
        if wave.dim() != 1:
            raise ValueError(f"expected 1-D waveform, got shape {tuple(wave.shape)}")
        spec = torch.stft(
            wave,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = spec.abs().pow(2)  # (freq, time)
        mel = self.register @ power
        return torch.log(mel.clamp_min(self.log_offset))


def cmvn(mel, eps=1e-5):
    """Per-utterance mean/variance normalisation over time, per mel bin."""
    mean = mel.mean(dim=1, keepdim=True)
    std = mel.std(dim=1, keepdim=True, unbiased=False)
    return (mel - mean) / (std + eps)


def peak_normalize(wave, eps=1e-8):
    peak = wave.abs().max()
    return wave / peak.clamp_min(eps)


def subsampled_length(n_frames, factor=4):
    """Output frame count after log2(factor) stride-2 convolutions."""
    n = n_frames
    for _ in range(int(math.log2(factor))):
        n = (n + 1) // 2
    return n
