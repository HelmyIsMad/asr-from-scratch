"""Evaluate a checkpoint on the test split: greedy and optional beam search."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import load
from .dataset import ASRDataset, BucketBatchSampler
from .decode import greedy_decode, prefix_beam_decode
from .manifest import load_manifest
from .metrics import corpus_cer, corpus_wer, wer
from .model import build_model
from .train import pick_device
from .vocab import Vocab


def evaluate(cfg, checkpoint, split="test", beam_width=0, limit=0, batch_size=8, workers=2):
    device = pick_device()
    state = load(checkpoint, map_location=device)
    vocab = Vocab(state["vocab"])
    model = build_model(state["config"]["model"], len(vocab)).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    root = cfg["data"]["root"]
    entries = load_manifest(Path(cfg["data"]["manifest_dir"]) / f"{split}.jsonl", root)
    if limit:
        entries = entries[:limit]
    ds = ASRDataset(entries, vocab, cfg["data"]["features"], None, cfg["data"]["sample_rate"], is_train=False)
    sampler = BucketBatchSampler(ds.durations(), batch_size, shuffle=False, drop_last=False)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=ASRDataset.collate, num_workers=workers)

    refs, hyps = [], []
    with torch.no_grad():
        for batch in loader:
            mel = batch["mel"].to(device)
            log_probs, out_lens = model(mel, batch["mel_lens"])
            log_probs = log_probs.float()
            if beam_width > 1:
                preds = [
                    prefix_beam_decode(log_probs[b, : int(out_lens[b])], beam_width=beam_width)
                    for b in range(log_probs.shape[0])
                ]
            else:
                preds = greedy_decode(log_probs, out_lens)
            hyps.extend(vocab.decode(p) for p in preds)
            refs.extend(batch["texts"])

    w, errors, words = corpus_wer(refs, hyps)
    c, _, _ = corpus_cer(refs, hyps)
    per_utt = sorted(((wer(r, h), r, h) for r, h in zip(refs, hyps)), reverse=True)
    return {
        "split": split,
        "checkpoint": str(checkpoint),
        "beam_width": beam_width,
        "utterances": len(refs),
        "wer": round(w, 5),
        "cer": round(c, 5),
        "errors": errors,
        "words": words,
        "worst": [{"wer": e, "ref": r, "hyp": h} for e, r, h in per_utt[:5]],
        "best": [{"wer": e, "ref": r, "hyp": h} for e, r, h in per_utt[-3:]],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--beam-width", type=int, default=0, help="0 = greedy, >1 = prefix beam search")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    state = load(args.checkpoint, map_location="cpu")
    report = evaluate(
        state["config"], args.checkpoint, args.split, args.beam_width, args.limit, args.batch_size, args.workers
    )
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / f"report_{args.split}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"{report['split']}: {report['utterances']} utts  "
        f"WER {report['wer'] * 100:.2f}%  CER {report['cer'] * 100:.2f}%  beam={args.beam_width}"
    )
    for item in report["worst"][:3]:
        print(f"  worst | ref: {item['ref']}\n        | hyp: {item['hyp']}")
    print(f"report -> {out}")
    return report


if __name__ == "__main__":
    main()
