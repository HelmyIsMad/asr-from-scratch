"""Build JSONL manifests from LibriSpeech (or any <split>/<speaker>/<utt>.flac tree)."""

import argparse
import json
from pathlib import Path

from .manifest import build_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build train/dev/test manifests")
    parser.add_argument("--root", required=True, help="dataset root containing the split directories")
    parser.add_argument("--out-dir", default="data/manifests")
    parser.add_argument("--train-splits", nargs="+", default=["train-clean-100"])
    parser.add_argument("--val-split", default="dev-clean")
    parser.add_argument("--test-split", default="test-clean")
    parser.add_argument("--min-duration", type=float, default=0.3)
    parser.add_argument("--max-duration", type=float, default=20.0)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)

    meta = {}
    meta["train"] = build_manifest(root, args.train_splits, out_dir / "train.jsonl", args.min_duration, args.max_duration)
    meta["dev"] = build_manifest(root, [args.val_split], out_dir / "dev.jsonl", args.min_duration, args.max_duration)
    meta["test"] = build_manifest(root, [args.test_split], out_dir / "test.jsonl", args.min_duration, args.max_duration)

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for split, info in meta.items():
        print(f"{split:>5}: {info['utterances']:>7} utts  {info['hours']:>7.2f} h  (skipped {info['skipped']})")
    return meta


if __name__ == "__main__":
    main()
