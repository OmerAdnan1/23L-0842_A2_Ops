"""Serving preprocessing must equal training preprocessing (skipped without TensorFlow)."""
import sys
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_serving_matches_training(png_bytes):
    from app.inference import preprocess
    from data import decode_resize

    train_img = tf.cast(decode_resize(tf.constant(png_bytes), 224), tf.float32).numpy()
    serve_img = preprocess(png_bytes, 224)[0].numpy()
    assert np.array_equal(train_img, serve_img)
