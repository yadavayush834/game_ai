#!/usr/bin/env python3
"""
train_new.py — Train the Game-Bot model on recorded sessions.

Usage:
    python train_new.py                        # Train with defaults
    python train_new.py --epochs 200           # Custom epoch count
    python train_new.py --lr 0.0005            # Custom learning rate
    python train_new.py --batch 32             # Custom batch size
    python train_new.py --resume               # Resume from latest checkpoint
"""

import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from core.model import GameBotModel
from core.dataset import get_train_val_datasets
from core.utils import ACTIONS


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_ds, val_ds = get_train_val_datasets(data_dir=args.data_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = GameBotModel(num_actions=8).to(device)

    # Multi-label classification → BCEWithLogitsLoss
    # Use pos_weight to handle class imbalance (keys are mostly NOT pressed)
    pos_weight = torch.tensor([3.0] * 8, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 0
    best_val_loss = float("inf")

    # ── Resume ────────────────────────────────────────────────────────────
    ckpt_dir = os.path.join("Data", "Checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.resume:
        ckpt_path = os.path.join(ckpt_dir, "latest.pt")
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            print(f"Resumed from epoch {start_epoch}")
        else:
            print("No checkpoint found, starting fresh.")

    # ── Training Loop ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  GAME-BOT TRAINER")
    print(f"  Epochs: {args.epochs}  |  Batch: {args.batch}  |  LR: {args.lr}")
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        # ── Train ─────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        t0 = time.perf_counter()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # ── Validate ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                # Per-action accuracy (threshold at 0.5)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.numel()

        val_loss /= len(val_loader)
        accuracy = correct / total if total > 0 else 0.0
        elapsed = time.perf_counter() - t0

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:4d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {accuracy:.4f} | "
            f"LR: {lr_now:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # ── Checkpoints ──────────────────────────────────────────────
        ckpt_data = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": min(best_val_loss, val_loss),
        }
        torch.save(ckpt_data, os.path.join(ckpt_dir, "latest.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_data, os.path.join(ckpt_dir, "best.pt"))
            print(f"  ★ New best model saved (val_loss={val_loss:.4f})")

    # ── Final export ──────────────────────────────────────────────────────
    model_dir = os.path.join("Data", "Model")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "game_bot.pt"))
    print(f"\nTraining complete. Final model saved to {model_dir}/game_bot.pt")


def main():
    parser = argparse.ArgumentParser(description="Game-Bot Trainer")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", type=str, default="Data/Recordings")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
