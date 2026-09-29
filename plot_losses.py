"""
Plot train/val loss curves from training_log.csv (written by train.py).

Usage:
    python plot_losses.py
    python plot_losses.py --log training_log.csv --out loss_curve.png
"""

import argparse
import csv

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="training_log.csv")
    parser.add_argument("--out", type=str, default="loss_curve.png")
    args = parser.parse_args()

    train_steps, train_losses = [], []
    val_steps, val_losses = [], []

    with open(args.log, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "train":
                train_steps.append(int(row["global_step"]))
                train_losses.append(float(row["loss"]))
            else:
                val_steps.append(int(row["global_step"]))
                val_losses.append(float(row["loss"]))

    plt.figure(figsize=(9, 5))
    plt.plot(train_steps, train_losses, label="train loss", color="tab:blue", linewidth=1, alpha=0.8)
    plt.plot(val_steps, val_losses, label="val loss", color="tab:orange", linewidth=2, marker="o")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
