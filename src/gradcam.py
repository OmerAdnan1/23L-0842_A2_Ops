"""
Task 2 bonus - Grad-CAM for 4 correct and 4 wrong test predictions.

The heat map shows which regions of the last MobileNetV2 feature map (7x7)
pushed the model towards the class it predicted.

Usage:
    python src/gradcam.py --model transfer_stage2
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from data import decode_resize


def gradcam(model, x, predicted_ai):
    pre = model.get_layer("mobilenet_preprocess")
    base = next(l for l in model.layers if isinstance(l, keras.Model))
    gap, head = model.get_layer("gap"), model.get_layer("p_ai")
    with tf.GradientTape() as tape:
        feats = base(pre(x), training=False)
        tape.watch(feats)
        p = head(gap(feats))[:, 0]
        target = p if predicted_ai else 1.0 - p
    grads = tape.gradient(target, feats)[0]
    cam = tf.nn.relu(tf.reduce_sum(feats[0] * tf.reduce_mean(grads, axis=(0, 1)), -1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return tf.image.resize(cam[..., tf.newaxis], x.shape[1:3]).numpy()[..., 0], float(p[0])


def pick(df, n_correct=4, n_wrong=4):
    df = df.assign(confidence=(df["prob"] - 0.5).abs())
    ok, wrong = df[df["pred"] == df["label"]], df[df["pred"] != df["label"]]
    correct = pd.concat([ok[ok["label"] == c].nlargest(n_correct // 2, "confidence")
                         for c in (0, 1)])
    return pd.concat([correct, wrong.nlargest(n_wrong, "confidence")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="transfer_stage2")
    ap.add_argument("--root", default="data/raw")
    args = ap.parse_args()

    model = keras.models.load_model(f"models/{args.model}.keras")
    size = model.input_shape[1]
    rows = pick(pd.read_csv(f"reports/predictions/{args.model}_test.csv"))

    fig, axes = plt.subplots(2, 8, figsize=(24, 6.5))
    for col, r in enumerate(rows.itertuples()):
        raw = tf.io.read_file(str(Path(args.root) / r.path))
        img = decode_resize(raw, size)
        cam, p = gradcam(model, tf.cast(img, tf.float32)[tf.newaxis], r.pred == 1)
        kind = "correct" if r.pred == r.label else "WRONG"
        title = (f"{kind}: true {'AI' if r.label else 'Real'}, "
                 f"pred {'AI' if r.pred else 'Real'}\nP(AI)={p:.2f}")
        axes[0, col].imshow(img.numpy())
        axes[0, col].set_title(title, fontsize=9, color="black" if kind == "correct" else "red")
        axes[1, col].imshow(img.numpy())
        axes[1, col].imshow(cam, cmap="jet", alpha=0.45)
        for ax in axes[:, col]:
            ax.axis("off")
    fig.suptitle(f"Grad-CAM ({args.model}): top = input as the model sees it, "
                 "bottom = regions supporting the predicted class")
    fig.tight_layout()
    out = f"reports/figures/gradcam_{args.model}.png"
    fig.savefig(out, dpi=120)
    print(rows[["path", "label", "pred", "prob"]].to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
