"""
Task 4.1 - Load the champion Keras model once and classify raw image bytes.

preprocess() mirrors src/data.py decode_resize(): decode to 3-channel RGB,
antialiased resize to the model's input size, round to uint8. The model
rescales internally, so it receives float32 0-255 exactly as in training
(tests/test_preprocess.py checks the two match).

Usage (console demo):
    python -m app.inference --model models/champion.keras --threshold 0.80 img1.jpg img2.png
"""
import argparse
import time
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from app.verdict import decide


def preprocess(raw, img_size):
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [img_size, img_size], antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return tf.cast(img, tf.float32)[tf.newaxis]


class Predictor:
    def __init__(self, model_path, threshold, name=None):
        self.model = keras.models.load_model(model_path, compile=False)
        self.img_size = self.model.input_shape[1]
        self.threshold = threshold
        self.name = name or Path(model_path).stem
        self.model(tf.zeros([1, self.img_size, self.img_size, 3]), training=False)

    def predict(self, raw, filename=None):
        x = preprocess(raw, self.img_size)
        return decide(float(self.model(x, training=False)[0, 0]), self.threshold)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default="models/champion.keras")
    ap.add_argument("--threshold", type=float, default=0.80)
    args = ap.parse_args()

    t0 = time.perf_counter()
    predictor = Predictor(args.model, args.threshold)
    print(f"Loaded {args.model} in {time.perf_counter() - t0:.2f}s "
          f"(input {predictor.img_size}x{predictor.img_size}, threshold {args.threshold})")
    for path in args.images:
        t0 = time.perf_counter()
        v = predictor.predict(Path(path).read_bytes())
        print(f"{path}: {v.label} | P(AI)={v.p_ai:.4f} | confidence {v.confidence:.2%} | "
              f"{(time.perf_counter() - t0) * 1000:.0f} ms")
