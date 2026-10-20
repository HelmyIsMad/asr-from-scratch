"""QuartzNet-style CTC encoder, built from scratch."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def subsampled_lengths(lengths, factor=4):
    out = lengths
    for _ in range(int(round(math.log2(factor)))):
        out = (out + 1) // 2
    return out


class SepConvBlock(nn.Module):
    """Time-channel separable convolution block with optional residual connection."""

    def __init__(self, channels, kernel_size=33, residual=False, dropout=0.0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, (kernel_size, 1), padding=(kernel_size // 2, 0), groups=channels
        )
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.bn = nn.BatchNorm2d(channels)
        self.residual = residual
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.pointwise(self.depthwise(x))
        y = self.dropout(F.relu(self.bn(y)))
        return y + x if self.residual else y


class QuartzNet(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_mels=80,
        channels=256,
        block_repeats=(2, 2, 2, 2, 2),
        kernel_size=33,
        dropout=0.1,
        rnn_layers=0,
        rnn_hidden=256,
    ):
        super().__init__()
        self.subsample_factor = 4
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, (5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, (5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        blocks = []
        for repeat in block_repeats:
            for i in range(int(repeat)):
                blocks.append(SepConvBlock(channels, kernel_size, residual=i > 0, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)

        self.rnn = None
        if rnn_layers:
            self.rnn = nn.GRU(
                channels,
                rnn_hidden,
                num_layers=int(rnn_layers),
                batch_first=True,
                bidirectional=True,
                dropout=dropout if rnn_layers > 1 else 0.0,
            )
            self.head = nn.Linear(rnn_hidden * 2, vocab_size)
        else:
            self.head = nn.Conv2d(channels, vocab_size, 1)

    def forward(self, x, lengths=None):
        """x: (B, 1, n_mels, T) -> log-probs (B, T', V), out_lengths."""
        out_lengths = subsampled_lengths(lengths, self.subsample_factor) if lengths is not None else None
        y = self.stem(x)
        y = self.blocks(y)  # (B, C, n_mels, T')

        if self.rnn is not None:
            y = y.mean(dim=2).transpose(1, 2)  # (B, T', C)
            if out_lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    y, out_lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                y, _ = self.rnn(packed)
                y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, total_length=int(out_lengths.max()))
            else:
                y, _ = self.rnn(y)
            y = self.head(y)  # (B, T', V)
        else:
            y = self.head(y)  # (B, V, n_mels, T')
            y = y.mean(dim=2).transpose(1, 2)  # (B, T', V)

        return y.log_softmax(dim=-1), out_lengths

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg, vocab_size):
    return QuartzNet(
        vocab_size=vocab_size,
        n_mels=cfg.get("n_mels", 80),
        channels=cfg.get("channels", 256),
        block_repeats=cfg.get("block_repeats", (2, 2, 2, 2, 2)),
        kernel_size=cfg.get("kernel_size", 33),
        dropout=cfg.get("dropout", 0.1),
        rnn_layers=cfg.get("rnn_layers", 0),
        rnn_hidden=cfg.get("rnn_hidden", 256),
    )
