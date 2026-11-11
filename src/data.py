"""
Task 2.1 - tf.data input pipeline for VisionGuard.

Stages (per split):
  read CSV -> decode (force 3-channel RGB) -> resize to IMG_SIZE x IMG_SIZE
  -> cache as uint8 -> [train only: shuffle] -> batch -> cast to float32 (0-255)
  -> [train only: random augmentation] -> prefetch

The pipeline outputs raw pixel values in 0-255. Each model applies its OWN
preprocessing (Rescaling for the CNN, preprocess_input for MobileNetV2) as a
layer inside the model, so training and serving cannot drift apart.

Running this file saves a figure of one augmented training batch.
"""
import argparse
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow import keras

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def make_augmenter(seed=SEED):
    """Mild augmentation; only ever applied to the training split."""
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=seed),
        keras.layers.RandomRotation(0.05, seed=seed),
        keras.layers.RandomZoom(0.1, seed=seed),
        keras.layers.RandomContrast(0.1, seed=seed),
    ], name="augmentation")


def decode_resize(raw, img_size=IMG_SIZE):
    """Encoded bytes -> uint8 RGB (img_size, img_size, 3). Mirrored by app/inference.py."""
    # channels=3: RGBA drops alpha, grayscale/palette become RGB
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    # plain resize (squashes aspect ratio); antialias avoids aliasing on big images
    img = tf.image.resize(img, [img_size, img_size], antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)


def _load(path, label, root, img_size):
    raw = tf.io.read_file(tf.strings.join([root, path], "/"))
    return decode_resize(raw, img_size), tf.cast(label, tf.float32)


def build_dataset(csv_path, root, training, img_size=IMG_SIZE,
                  batch_size=BATCH_SIZE, seed=SEED, augment=None):
    augment = training if augment is None else augment
    df = pd.read_csv(csv_path)
    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values))
    ds = ds.map(lambda p, y: _load(p, y, root, img_size), num_parallel_calls=AUTOTUNE)
    ds = ds.cache()  # decode + resize once, then reuse from RAM (uint8)
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    if augment:
        aug = make_augmenter(seed)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def get_datasets(splits_dir="splits", root="data/raw", img_size=IMG_SIZE,
                 batch_size=BATCH_SIZE, augment=True):
    d = Path(splits_dir)
    return (build_dataset(d / "train.csv", root, True, img_size, batch_size, augment=augment),
            build_dataset(d / "val.csv", root, False, img_size, batch_size),
            build_dataset(d / "test.csv", root, False, img_size, batch_size))


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--out", default="reports/figures/augmented_batch.png")
    args = ap.parse_args()

    keras.utils.set_random_seed(SEED)
    train, val, test = get_datasets(args.splits, args.root)
    x, y = next(iter(train))
    print("train batch:", x.shape, x.dtype, "| value range:",
          float(tf.reduce_min(x)), "-", float(tf.reduce_max(x)))

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, img, lab in zip(axes.ravel(), x.numpy(), y.numpy()):
        ax.imshow(img.astype("uint8"))
        ax.set_title("AI" if lab == 1 else "Real", fontsize=9)
        ax.axis("off")
    fig.suptitle("One augmented training batch (224x224)")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("Saved", args.out)
