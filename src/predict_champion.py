"""
Task 3.6 - Load the registered model by alias and classify one image.

Usage:
    python src/predict_champion.py data/raw/RealArt/<image>.jpg --alias champion
"""
import argparse

import mlflow
import tensorflow as tf
from mlflow import MlflowClient

from data import decode_resize
from tracking import MODEL_NAME, setup_tracking


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--name", default=MODEL_NAME)
    ap.add_argument("--alias", default="champion")
    args = ap.parse_args()

    setup_tracking(None)
    mv = MlflowClient().get_model_version_by_alias(args.name, args.alias)
    threshold = MlflowClient().get_run(mv.run_id).data.metrics.get("threshold", 0.5)
    model = mlflow.keras.load_model(f"models:/{args.name}@{args.alias}")

    img_size = model.input_shape[1]
    x = tf.cast(decode_resize(tf.io.read_file(args.image), img_size), tf.float32)[tf.newaxis]
    p = float(model(x, training=False)[0, 0])
    label = "AI-generated" if p >= threshold else "Real"
    print(f"{args.name}@{args.alias} (version {mv.version}, run {mv.run_id[:8]})")
    print(f"{args.image}: P(AI) = {p:.4f}, threshold {threshold:.3f} -> {label}")


if __name__ == "__main__":
    main()
