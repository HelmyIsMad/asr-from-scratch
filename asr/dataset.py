"""Torch dataset, collation and length-bucketed batching."""

import random

import soundfile as sf
import torch
import numpy as np
from torch.utils.data import Dataset, Sampler

from .features import LogMel, cmvn, peak_normalize, subsampled_length
from .augment import WaveformAugment, resample, spec_augment


class ASRDataset(Dataset):
    def __init__(self, entries, vocab, feat_cfg, augment_cfg=None, sample_rate=16000, is_train=False):
        self.vocab = vocab
        self.is_train = is_train
        self.sample_rate = sample_rate
        self.log_mel = LogMel(
            sample_rate=sample_rate,
            n_fft=feat_cfg.get("n_fft", 400),
            hop_length=feat_cfg.get("hop_length", 160),
            win_length=feat_cfg.get("win_length", 400),
            n_mels=feat_cfg.get("n_mels", 80),
            f_min=feat_cfg.get("f_min", 0.0),
            f_max=feat_cfg.get("f_max"),
        )
        self.do_cmvn = bool(feat_cfg.get("cmvn", True))
        aug_cfg = augment_cfg if is_train else None
        self.augment = WaveformAugment(aug_cfg, sample_rate) if aug_cfg else None
        self.specaug = (aug_cfg or {}).get("specaug") or {}

        # Drop utterances the encoder cannot emit enough frames for under CTC.
        kept, dropped = [], 0
        for entry in entries:
            tokens = vocab.encode(entry["text"])
            n_frames = self.log_mel.n_frames(int(entry["duration"] * sample_rate))
            if tokens and subsampled_length(n_frames) > len(tokens) + 2:
                kept.append(entry)
            else:
                dropped += 1
        self.entries = kept
        self.dropped = dropped
        if not self.entries:
            raise ValueError("no usable utterances after length filtering")

    def __len__(self):
        return len(self.entries)

    def durations(self):
        return torch.tensor([e["duration"] for e in self.entries], dtype=torch.float32)

    def _load(self, path):
        wave, sr = sf.read(path, dtype="float32", always_2d=False)
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        wave = torch.from_numpy(np.ascontiguousarray(wave))
        if sr != self.sample_rate:
            wave = resample(wave, sr, self.sample_rate)
        return wave

    def __getitem__(self, index):
        entry = self.entries[index]
        wave = self._load(entry["audio"])
        if self.augment is not None:
            wave = self.augment(wave)
        if wave.numel() < self.log_mel.n_fft:
            wave = torch.nn.functional.pad(wave, (0, self.log_mel.n_fft - wave.numel()))
        wave = peak_normalize(wave, eps=1e-5)

        mel = self.log_mel(wave)
        if self.do_cmvn:
            mel = cmvn(mel)
        if self.is_train and self.specaug:
            mel = spec_augment(
                mel,
                freq_masks=self.specaug.get("freq_masks", 0),
                freq_width=self.specaug.get("freq_width", 27),
                time_masks=self.specaug.get("time_masks", 0),
                time_width_frac=self.specaug.get("time_width_frac", 0.05),
                p=self.specaug.get("p", 1.0),
            )

        return {
            "utt_id": entry["id"],
            "mel": mel.contiguous(),
            "tokens": torch.tensor(self.vocab.encode(entry["text"]), dtype=torch.long),
            "text": entry["text"],
        }

    @staticmethod
    def collate(batch):
        mels = [b["mel"] for b in batch]
        lengths = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)
        n_mels = mels[0].shape[0]
        max_t = int(lengths.max())
        padded = torch.zeros(len(batch), n_mels, max_t)
        for i, mel in enumerate(mels):
            padded[i, :, : mel.shape[1]] = mel
        tokens = torch.cat([b["tokens"] for b in batch]) if batch[0]["tokens"].numel() else torch.zeros(0, dtype=torch.long)
        token_lengths = torch.tensor([b["tokens"].numel() for b in batch], dtype=torch.long)
        return {
            "mel": padded.unsqueeze(1),  # (B, 1, n_mels, T)
            "mel_lens": lengths,
            "tokens": tokens,
            "token_lens": token_lengths,
            "utt_ids": [b["utt_id"] for b in batch],
            "texts": [b["text"] for b in batch],
        }


class BucketBatchSampler(Sampler):
    """Sort by length inside large buckets for cheap padding, then shuffle batch order."""

    def __init__(self, durations, batch_size, shuffle=True, bucket_multiplier=100, drop_last=True, seed=0):
        self.durations = list(durations) if not torch.is_tensor(durations) else durations.tolist()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = max(batch_size, batch_size * bucket_multiplier)
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _batches(self, epoch):
        rng = random.Random(self.seed + epoch)
        order = list(range(len(self.durations)))
        if self.shuffle:
            rng.shuffle(order)
        order.sort(key=lambda i: self.durations[i])  # stable: random tie-break within equal lengths
        batches = []
        for start in range(0, len(order), self.bucket_size):
            chunk = order[start : start + self.bucket_size]
            if self.shuffle:
                rng.shuffle(chunk)
            for s in range(0, len(chunk), self.batch_size):
                batch = chunk[s : s + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches(self.epoch))

    def __len__(self):
        return len(self._batches(0))
