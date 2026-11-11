"""
Task 2.3 / 2.4 / 3.3-3.5 - Train a VisionGuard model with fixed seeds and
standard callbacks, optionally tracked with MLflow.

Callbacks: EarlyStopping (val_loss, restores best weights), ModelCheckpoint
(best val_loss only), ReduceLROnPlateau, CSVLogger.

baseline: one stage.
transfer: stage 1 trains the head on a frozen MobileNetV2, stage 2 unfreezes
the last --unfreeze-blocks blocks and continues with a much smaller learning rate.

--track manual|autolog logs the run to MLflow (see src/tracking.py) and, at the
end, picks a threshold on validation and logs test metrics and artifacts.

Usage:
    python src/train.py --model baseline --epochs 40 --lr 1e-3
    python src/train.py --model transfer --epochs 40 --ft-epochs 40
    python src/train.py --model transfer --track manual --name tl_ref
"""
import argparse
from contextlib import nullcontext
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

from data import IMG_SIZE, SEED, get_datasets
from models import (build_baseline_cnn, build_transfer, compile_model,
                    trainable_layers, unfreeze_last_blocks)


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


def run_stage(model, train, val, name, lr, epochs, extra_callbacks=()):
    compile_model(model, lr)
    model.fit(train, validation_data=val, epochs=epochs,
              callbacks=make_callbacks(f"models/{name}.keras",
                                       f"reports/{name}_history.csv") + list(extra_callbacks))

    hist = pd.read_csv(f"reports/{name}_history.csv")
    best = plot_curves(hist, f"{name} (lr={lr})", f"reports/figures/{name}_curves.png")
    row = hist.iloc[best - 1]
    print(f"\n[{name}] best epoch: {best}/{len(hist)} | val_loss {row.val_loss:.4f} | "
          f"val_accuracy {row.val_accuracy:.4f} | val_auc {row.val_auc:.4f}\n")
    return hist, best


def save_trainable(model, name):
    names = trainable_layers(model)
    Path(f"reports/{name}_trainable.txt").write_text("\n".join(names) + "\n")
    print(f"[{name}] {len(names)} trainable layers, "
          f"{sum(np.prod(w.shape) for w in model.trainable_weights):,} trainable params")
    print("  " + "\n  ".join(names))


def build_stages(args, name):
    """The model plus a list of (stage_name, lr, epochs, prepare_fn)."""
    kw = {"img_size": args.img_size}
    if args.dropout is not None:
        kw["dropout"] = args.dropout
    if args.model == "baseline":
        return build_baseline_cnn(**kw), [(name, args.lr, args.epochs, lambda: None)]
    model, base = build_transfer(**kw)
    return model, [
        (f"{name}_stage1", args.lr, args.epochs, lambda: None),
        (f"{name}_stage2", args.ft_lr, args.ft_epochs,
         lambda: unfreeze_last_blocks(base, args.unfreeze_blocks)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "transfer"], default="baseline")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--ft-epochs", type=int, default=40)
    ap.add_argument("--ft-lr", type=float, default=1e-5)
    ap.add_argument("--unfreeze-blocks", type=int, default=1)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--name", default=None, help="run name, defaults to --model")
    ap.add_argument("--track", choices=["none", "manual", "autolog"], default="none")
    ap.add_argument("--max-fpr", type=float, default=0.10)
    args = ap.parse_args()

    name = args.name or args.model
    keras.utils.set_random_seed(SEED)
    train, val, _ = get_datasets(args.splits, args.root, args.img_size,
                                 augment=not args.no_augment)
    Path("models").mkdir(exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    model, stages = build_stages(args, name)

    tracker = None
    if args.track != "none":
        from tracking import Tracker
        tracker = Tracker(args.track, name)

    with tracker.run(vars(args), SEED) if tracker else nullcontext():
        for stage, lr, epochs, prepare in stages:
            prepare()
            save_trainable(model, stage)
            with tracker.stage(stage) if tracker else nullcontext():
                hist, best = run_stage(model, train, val, stage, lr, epochs,
                                       tracker.callbacks() if tracker else ())
            if tracker:
                tracker.after_stage(stage, hist, best)

        if tracker:
            from evaluate import predict
            splits = Path(args.splits)
            tracker.finish(model, f"models/{stages[-1][0]}.keras",
                           predict(model, splits / "val.csv", args.root, args.img_size),
                           predict(model, splits / "test.csv", args.root, args.img_size),
                           hist.iloc[best - 1], args.max_fpr)


if __name__ == "__main__":
    main()
