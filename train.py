"""
Training loop.

Usage:
    python train.py
    python train.py --config config.yaml
    python train.py --config config.yaml --resume checkpoints/checkpoint_epoch0_final.pt
"""

import argparse
import csv
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config, load_config
from data import TokenDataset
from model import GPT

# checkpoints embed a Config dataclass; PyTorch 2.6 defaults torch.load to
# weights_only=True, which refuses to unpickle it unless allowlisted (safe
# here since we only ever load our own files)
torch.serialization.add_safe_globals([Config])


def save_checkpoint(model, optimizer, epoch, step, config, path):
    torch.save({
        "epoch": epoch,
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
    }, path)
    print(f"saved checkpoint -> {path}")


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"], ckpt["step"]


@torch.no_grad()
def evaluate(model, val_loader, device, vocab_size):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        logits = logits.float()  # cast to fp32 for the loss, whatever autocast dtype forward ran in
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = config.device if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    torch.manual_seed(config.seed)

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # append if resuming (keep prior run's history), otherwise start fresh
    log_mode = "a" if args.resume and os.path.exists(config.log_file) else "w"
    log_file = open(config.log_file, log_mode, newline="")
    log_writer = csv.writer(log_file)
    if log_mode == "w":
        log_writer.writerow(["split", "epoch", "step", "global_step", "loss"])

    train_ds = TokenDataset(config.train_bin, config.block_size)
    val_ds = TokenDataset(config.val_bin, config.block_size)
    # num_workers=0: memmap handles don't pickle reliably across the
    # multiprocessing "spawn" start method Windows DataLoader workers use.
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                               drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                             drop_last=True, num_workers=0)
    print(f"train: {len(train_ds)} windows ({len(train_loader)} steps/epoch)")
    print(f"val:   {len(val_ds)} windows ({len(val_loader)} steps)")

    model = GPT(vocab_size=config.vocab_size, block_size=config.block_size,
                n_embd=config.n_embd, n_head=config.n_head, n_layer=config.n_layer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch += 1
        print(f"resumed from {args.resume}, starting at epoch {start_epoch}")

    amp_dtype = torch.bfloat16  # RTX 40-series native tensor cores, no GradScaler needed

    for epoch in range(start_epoch, config.epochs):
        model.train()
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
                logits, _ = model(x)
            logits = logits.float()
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            global_step = epoch * len(train_loader) + step

            if step % config.log_every == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
                log_writer.writerow(["train", epoch, step, global_step, loss.item()])
                log_file.flush()

            # skip step 0 so we don't immediately overwrite with a checkpoint
            # of the untrained model
            if step > 0 and step % config.checkpoint_every == 0:
                ckpt_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch{epoch}_step{step}.pt")
                save_checkpoint(model, optimizer, epoch, step, config, ckpt_path)

        val_loss = evaluate(model, val_loader, device, config.vocab_size)
        perplexity = math.exp(val_loss)
        print(f"Epoch {epoch} | Val Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f}")
        log_writer.writerow(["val", epoch, len(train_loader) - 1, (epoch + 1) * len(train_loader) - 1, val_loss])
        log_file.flush()

        ckpt_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch{epoch}_final.pt")
        save_checkpoint(model, optimizer, epoch, step, config, ckpt_path)

    log_file.close()


if __name__ == "__main__":
    main()
