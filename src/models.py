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
