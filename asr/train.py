"""Train a from-scratch CTC recogniser."""

import argparse
import csv
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoint import load, push_logs, save_atomic
from .config import load_config
from .dataset import ASRDataset, BucketBatchSampler
from .decode import greedy_decode
from .manifest import load_manifest
from .metrics import corpus_cer, corpus_wer
from .model import build_model, subsampled_lengths
from .schedule import lr_at
from .vocab import Vocab

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RunLogger:
    def __init__(self, out_dir, resume=False):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "train.log"
        self.csv_path = self.out_dir / "metrics.csv"
        self.log_path.write_text("", encoding="utf-8") if not resume else None
        self._fh = self.log_path.open("a", encoding="utf-8")
        self._csv = self.csv_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._csv, fieldnames=["epoch", "step", "lr", "train_loss", "val_loss", "val_wer", "val_cer", "seconds"]
        )
        if self.csv_path.stat().st_size == 0:
            self._writer.writeheader()
            self._csv.flush()

    def log(self, message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def metric(self, row):
        self._writer.writerow(row)
        self._csv.flush()

    def tail(self, n=400):
        lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        (self.out_dir / "tail.log").write_text("\n".join(lines[-n:]), encoding="utf-8")

    def close(self):
        self._fh.close()
        self._csv.close()


def build_loaders(cfg, vocab):
    data_cfg = cfg["data"]
    root = data_cfg["root"]
    manifest_dir = Path(data_cfg["manifest_dir"])
    feature_cfg = data_cfg["features"]
    aug_cfg = cfg.get("augment") or {}
    train_cfg = cfg["train"]

    train_entries = load_manifest(manifest_dir / "train.jsonl", root)
    val_entries = load_manifest(manifest_dir / "dev.jsonl", root)
    val_cap = int(data_cfg.get("val_max_utts", 0) or 0)
    if val_cap and len(val_entries) > val_cap:
        stride = len(val_entries) / val_cap
        val_entries = [val_entries[int(i * stride)] for i in range(val_cap)]

    train_ds = ASRDataset(train_entries, vocab, feature_cfg, aug_cfg, data_cfg["sample_rate"], is_train=True)
    val_ds = ASRDataset(val_entries, vocab, feature_cfg, None, data_cfg["sample_rate"], is_train=False)

    workers = int(train_cfg.get("num_workers", 4))
    train_sampler = BucketBatchSampler(
        train_ds.durations(), int(train_cfg["batch_size"]), shuffle=True, drop_last=True, seed=int(cfg.get("seed", 0))
    )
    val_sampler = BucketBatchSampler(
        val_ds.durations(), int(train_cfg.get("eval_batch_size", 8)), shuffle=False, drop_last=False
    )

    def seed_worker(worker_id):
        worker_seed = (torch.initial_seed() + worker_id) % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    common = dict(
        num_workers=workers,
        collate_fn=ASRDataset.collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
    )
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **common)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, **common)
    return train_ds, val_ds, train_loader, val_loader, train_sampler


@torch.no_grad()
def validate(model, loader, vocab, criterion, device, amp, limit_batches=0):
    model.eval()
    refs, hyps = [], []
    total_loss, total_frames = 0.0, 0
    for i, batch in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        mel = batch["mel"].to(device, non_blocking=True)
        tokens = batch["tokens"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            log_probs, out_lens = model(mel, batch["mel_lens"])
            loss = criterion(log_probs.transpose(0, 1), tokens, out_lens, batch["token_lens"])
        total_loss += float(loss) * tokens.numel()
        total_frames += int(tokens.numel())
        preds = greedy_decode(log_probs.float(), out_lens)
        hyps.extend(vocab.decode(p) for p in preds)
        refs.extend(batch["texts"])
    val_loss = total_loss / max(1, total_frames)
    wer, _, _ = corpus_wer(refs, hyps)
    cer, _, _ = corpus_cer(refs, hyps)
    return val_loss, wer, cer, refs, hyps


def train(cfg):
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = pick_device()
    train_cfg = cfg["train"]
    out_dir = Path(train_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    resume = bool(train_cfg.get("resume", True)) and (out_dir / "last.pt").exists()
    logger = RunLogger(out_dir, resume=resume)
    logger.log(f"device={device}  out_dir={out_dir}  resume={resume}")

    manifest_dir = Path(cfg["data"]["manifest_dir"])
    train_texts = [e["text"] for e in load_manifest(manifest_dir / "train.jsonl", cfg["data"]["root"])]
    vocab_path = out_dir / "vocab.json"
    if vocab_path.exists():
        vocab = Vocab.load(vocab_path)
    else:
        vocab = Vocab.build(train_texts, min_count=int(cfg["data"].get("min_char_count", 1)))
        vocab.save(vocab_path)
    logger.log(f"vocab size={len(vocab)} tokens={''.join(vocab.tokens)!r}")

    train_ds, val_ds, train_loader, val_loader, train_sampler = build_loaders(cfg, vocab)
    logger.log(
        f"train utts={len(train_ds)} ({sum(e['duration'] for e in train_ds.entries) / 3600:.2f} h, "
        f"dropped {train_ds.dropped})  val utts={len(val_ds)}  batches/epoch={len(train_loader)}"
    )

    model = build_model(cfg["model"], len(vocab)).to(device)
    logger.log(f"model params={model.num_parameters() / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-3)),
        betas=(0.9, 0.98),
    )
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    epochs = int(train_cfg["epochs"])
    grad_accum = max(1, int(train_cfg.get("grad_accum", 1)))
    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(steps_per_epoch * float(train_cfg.get("warmup_epochs", 0)))

    start_epoch, global_step, best_wer = 0, 0, float("inf")
    if resume:
        state = load(out_dir / "last.pt", map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_wer = float(state.get("best_wer", float("inf")))
        logger.log(f"resumed from epoch {start_epoch} (best val wer {best_wer:.4f})")

    limit_batches = int(train_cfg.get("limit_batches", 0) or 0)
    log_every = int(train_cfg.get("log_every", 50))
    repo_dir = train_cfg.get("repo_dir")
    push = os.environ.get("ASR_PUSH_LOGS", "0") == "1" and repo_dir

    for epoch in range(start_epoch, epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        t0 = time.time()
        running_loss, seen_tokens = 0.0, 0
        lr = float(train_cfg["lr"])
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            if limit_batches and i >= limit_batches:
                break
            mel = batch["mel"].to(device, non_blocking=True)
            tokens = batch["tokens"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                log_probs, out_lens = model(mel, batch["mel_lens"])
                loss = criterion(log_probs.transpose(0, 1), tokens, out_lens, batch["token_lens"])
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            running_loss += float(loss) * grad_accum * tokens.numel()
            seen_tokens += int(tokens.numel())

            done = (i + 1) % grad_accum == 0 or (i + 1) == len(train_loader)
            if done:
                lr = lr_at(
                    global_step,
                    float(train_cfg["lr"]),
                    warmup_steps,
                    total_steps,
                    schedule=train_cfg.get("schedule", "cosine"),
                    min_ratio=float(train_cfg.get("min_lr_ratio", 0.0)),
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 5.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if log_every and (i + 1) % log_every == 0:
                logger.log(
                    f"epoch {epoch} step {i + 1}/{len(train_loader)} "
                    f"loss {running_loss / max(1, seen_tokens):.4f} lr {lr:.2e}"
                )

        train_loss = running_loss / max(1, seen_tokens)
        val_loss, val_wer, val_cer, refs, hyps = validate(
            model, val_loader, vocab, criterion, device, amp,
            limit_batches=int(train_cfg.get("val_limit_batches", 0) or 0),
        )
        elapsed = time.time() - t0
        logger.metric(
            dict(
                epoch=epoch, step=global_step, lr=round(lr, 8), train_loss=round(train_loss, 5),
                val_loss=round(val_loss, 5), val_wer=round(val_wer, 5), val_cer=round(val_cer, 5),
                seconds=round(elapsed, 1),
            )
        )
        logger.log(
            f"epoch {epoch} done in {elapsed / 60:.1f} min | train_loss {train_loss:.4f} "
            f"val_loss {val_loss:.4f} | WER {val_wer * 100:.2f}% CER {val_cer * 100:.2f}%"
        )
        logger.log(f"  ref: {refs[0]}")
        logger.log(f"  hyp: {hyps[0]}")

        state = dict(
            model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            epoch=epoch, global_step=global_step, best_wer=best_wer, config=cfg, vocab=vocab.tokens,
        )
        if val_wer < best_wer:
            best_wer = val_wer
            state["best_wer"] = best_wer
            save_atomic(state, out_dir / "best.pt")
        save_atomic(state, out_dir / "last.pt")
        (out_dir / "status.json").write_text(
            json.dumps(
                dict(epoch=epoch, epochs=epochs, val_wer=val_wer, val_cer=val_cer, best_wer=best_wer, complete=epoch == epochs - 1),
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.tail()

        if push:
            # Logs live on Drive; mirror just the small text files into the repo before pushing.
            dest = Path(repo_dir) / "logs" / out_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for name in ("metrics.csv", "tail.log", "status.json"):
                shutil.copy2(out_dir / name, dest / name)
            ok, msg = push_logs(repo_dir, [str(dest)], f"epoch {epoch}: wer {val_wer * 100:.2f}%")
            if not ok:
                logger.log(f"log push failed: {msg.splitlines()[-1] if msg else 'unknown'}")

    logger.log("training complete")
    logger.close()
    return best_wer


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a CTC recogniser from scratch")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.lr=0.01")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)
    train(cfg)


if __name__ == "__main__":
    main()
