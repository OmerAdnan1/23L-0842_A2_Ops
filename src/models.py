"""
Task 2.2 - Model definitions for VisionGuard.

All models take float32 images of shape (IMG_SIZE, IMG_SIZE, 3) in 0-255, as
produced by src/data.py, and do their own preprocessing as the first layer.
Output is a single sigmoid unit: P(AI-generated).

Running this file prints the baseline summary and saves a per-layer table.
"""
import argparse
from pathlib import Path

import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from data import IMG_SIZE


def conv_block(x, filters, name):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    return layers.MaxPooling2D(2, name=f"{name}_pool")(x)


def build_baseline_cnn(img_size=IMG_SIZE, filters=(32, 64, 128, 256), dropout=0.4):
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)
    for i, f in enumerate(filters, start=1):
        x = conv_block(x, f, f"block{i}")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def build_transfer(img_size=IMG_SIZE, dropout=0.3):
    """MobileNetV2 (ImageNet) without its classifier, frozen, plus our own head.

    Rescaling(1/127.5, offset=-1) is exactly mobilenet_v2.preprocess_input
    (0-255 -> -1..1), but serialisable. The base is called with training=False
    so its BatchNorm layers stay in inference mode, also when fine-tuning.
    """
    base = keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3),
                                          include_top=False, weights="imagenet")
    base.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return keras.Model(inputs, outputs, name="mobilenetv2_transfer"), base


def unfreeze_from(base, start="block_16_expand"):
    """Unfreeze `start` and every later layer of the base, BatchNorm excluded."""
    base.trainable = True
    names = [l.name for l in base.layers]
    cut = names.index(start)
    for i, l in enumerate(base.layers):
        l.trainable = i >= cut and not isinstance(l, layers.BatchNormalization)


def unfreeze_last_blocks(base, n):
    """Unfreeze the last n inverted-residual blocks (16 is the last) plus Conv_1."""
    if n == 0:
        base.trainable = False
    else:
        unfreeze_from(base, f"block_{17 - n}_expand")


def trainable_layers(model):
    """Names of every layer (nested ones included) that has trainable weights."""
    out = []
    for l in model.layers:
        sub = getattr(l, "layers", None)
        if sub:
            out += [f"{l.name}/{s.name}" for s in sub if s.trainable_weights]
        elif l.trainable_weights:
            out.append(l.name)
    return out


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )
    return model


def layer_table(model):
    return pd.DataFrame([{
        "layer": l.name,
        "type": type(l).__name__,
        "output_shape": str(tuple(l.output.shape)),
        "params": l.count_params(),
    } for l in model.layers])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/baseline_cnn_layers")
    args = ap.parse_args()

    model = compile_model(build_baseline_cnn(), lr=1e-3)
    model.summary()

    df = layer_table(model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    out.with_suffix(".md").write_text(df.to_markdown(index=False))
    print(f"Saved {out.with_suffix('.csv')} and {out.with_suffix('.md')}")
