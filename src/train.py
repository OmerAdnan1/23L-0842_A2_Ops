"""
Task 2.3 / 2.4 - Train a VisionGuard model with fixed seeds and standard callbacks.

Callbacks: EarlyStopping (val_loss, restores best weights), ModelCheckpoint
(best val_loss only), ReduceLROnPlateau, CSVLogger.

baseline: one stage.
transfer: stage 1 trains the head on a frozen MobileNetV2, stage 2 unfreezes
the last block and continues with a much smaller learning rate.

Outputs per stage: models/<name>.keras, reports/<name>_history.csv,
reports/figures/<name>_curves.png; for transfer also
reports/<name>_trainable.txt.

Usage:
    python src/train.py --model baseline --epochs 40 --lr 1e-3
    python src/train.py --model transfer --epochs 20 --lr 1e-3 --ft-epochs 20 --ft-lr 1e-5
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

from data import SEED, get_datasets
from models import (build_baseline_cnn, build_transfer, compile_model,
                    trainable_layers, unfreeze_from)


def make_callbacks(ckpt_path, log_path, patience=8):
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss",
                                        save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                          patience=3, min_lr=1e-7, verbose=1),
        keras.callbacks.CSVLogger(log_path),
    ]


def plot_curves(hist, title, out_path):
    best = int(np.argmin(hist["val_loss"]))
    epochs = np.arange(1, len(hist) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(epochs, hist[metric], label=f"train {metric}")
        ax.plot(epochs, hist[f"val_{metric}"], label=f"val {metric}")
        ax.axvline(best + 1, color="red", ls="--", label=f"best epoch {best + 1}")
        ax.scatter(best + 1, hist[f"val_{metric}"][best], color="red", zorder=3)
        ax.set_xlabel("epoch")
        ax.set_title(metric)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return best + 1


def run_stage(model, train, val, name, lr, epochs):
    compile_model(model, lr)
    model.fit(train, validation_data=val, epochs=epochs,
              callbacks=make_callbacks(f"models/{name}.keras",
                                       f"reports/{name}_history.csv"))

    hist = pd.read_csv(f"reports/{name}_history.csv")
    best = plot_curves(hist, f"{name} (lr={lr})", f"reports/figures/{name}_curves.png")
    row = hist.iloc[best - 1]
    print(f"\n[{name}] best epoch: {best}/{len(hist)} | val_loss {row.val_loss:.4f} | "
          f"val_accuracy {row.val_accuracy:.4f} | val_auc {row.val_auc:.4f}\n")


def save_trainable(model, name):
    names = trainable_layers(model)
    Path(f"reports/{name}_trainable.txt").write_text("\n".join(names) + "\n")
    print(f"[{name}] {len(names)} trainable layers, "
          f"{sum(np.prod(w.shape) for w in model.trainable_weights):,} trainable params")
    print("  " + "\n  ".join(names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "transfer"], default="baseline")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--ft-epochs", type=int, default=20)
    ap.add_argument("--ft-lr", type=float, default=1e-5)
    ap.add_argument("--unfreeze-from", default="block_16_expand")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--name", default=None, help="run name, defaults to --model")
    args = ap.parse_args()

    name = args.name or args.model
    keras.utils.set_random_seed(SEED)
    train, val, _ = get_datasets(args.splits, args.root)
    Path("models").mkdir(exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    kw = {} if args.dropout is None else {"dropout": args.dropout}

    if args.model == "baseline":
        model = build_baseline_cnn(**kw)
        run_stage(model, train, val, name, args.lr, args.epochs)
        return

    model, base = build_transfer(**kw)
    save_trainable(model, f"{name}_stage1")
    run_stage(model, train, val, f"{name}_stage1", args.lr, args.epochs)

    unfreeze_from(base, args.unfreeze_from)
    save_trainable(model, f"{name}_stage2")
    run_stage(model, train, val, f"{name}_stage2", args.ft_lr, args.ft_epochs)


if __name__ == "__main__":
    main()
