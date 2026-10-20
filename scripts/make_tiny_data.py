"""Generate a tiny LibriSpeech-shaped dataset so the pipeline can be smoke-tested offline.

Not speech: each "word" is a tone burst, so WER is meaningless. The point is to exercise
manifest building, length filtering, CTC feasibility, augmentation and checkpointing.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf

WORDS = {"one": 220.0, "two": 330.0, "three": 440.0, "four": 550.0}
SR = 16000


def synth(words, rng):
    chunks = []
    for word in words:
        freq = WORDS[word] * rng.uniform(0.97, 1.03)
        n = int(0.28 * SR)
        t = np.arange(n) / SR
        tone = 0.4 * np.sin(2 * np.pi * freq * t) * np.hanning(n)
        chunks.append(tone.astype(np.float32))
        chunks.append(np.zeros(int(0.05 * SR), dtype=np.float32))
    audio = np.concatenate(chunks) if chunks else np.zeros(SR // 2, dtype=np.float32)
    audio += 0.002 * rng.standard_normal(len(audio)).astype(np.float32)
    return audio


def write_split(root, split, n_speakers, utts_per_speaker, rng):
    for s in range(n_speakers):
        speaker = str(100 + s)
        chapter = str((s + 1) * 10)
        folder = root / split / speaker / chapter
        folder.mkdir(parents=True, exist_ok=True)
        lines = []
        for u in range(utts_per_speaker):
            utt_id = f"{speaker}-{chapter}-{u:04d}"
            words = [rng.choice(list(WORDS)) for _ in range(rng.randint(1, 3))]
            sf.write(folder / f"{utt_id}.flac", synth(words, rng), SR, format="FLAC")
            lines.append(f"{utt_id} {' '.join(words).upper()}")
        (folder / f"{speaker}-{chapter}.trans.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/tiny")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    root = Path(args.root)
    write_split(root, "train-clean-100", 4, 16, rng)
    write_split(root, "dev-clean", 2, 6, rng)
    write_split(root, "test-clean", 2, 6, rng)
    print(f"tiny dataset written to {root}")


if __name__ == "__main__":
    main()
