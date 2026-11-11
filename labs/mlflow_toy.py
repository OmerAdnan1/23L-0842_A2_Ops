"""
Task 3.1 / 3.2 - MLflow tool lab on Fashion-MNIST.

Logs parameters, per-epoch metrics, a loss plot, tags and the model.
--local tracks to ./mlflow_toy.db (view with `mlflow ui --backend-store-uri
sqlite:///mlflow_toy.db`); otherwise MLFLOW_TRACKING_URI from .env (DagsHub).

Usage:
    python labs/mlflow_toy.py --lr 1e-3 --local
    python labs/mlflow_toy.py --lr 1e-2
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from dotenv import load_dotenv
from tensorflow import keras

SEED = 42


class EpochLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


def build(units, lr):
    model = keras.Sequential([
        keras.Input(shape=(28, 28)),
        keras.layers.Rescaling(1.0 / 255),
        keras.layers.Flatten(),
        keras.layers.Dense(units, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--units", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--local", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    uri = "sqlite:///mlflow_toy.db" if args.local else os.getenv(
        "MLFLOW_TRACKING_URI", "sqlite:///mlflow_toy.db")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("toy-fashion-mnist")

    keras.utils.set_random_seed(SEED)
    (x, y), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    model = build(args.units, args.lr)

    with mlflow.start_run(run_name=f"lr={args.lr}"):
        mlflow.log_params({"lr": args.lr, "units": args.units, "epochs": args.epochs,
                           "optimizer": "adam", "seed": SEED})
        mlflow.set_tags({"lab": "3.1", "dataset": "fashion_mnist",
                         "store": "local" if args.local else "dagshub"})
        hist = model.fit(x, y, epochs=args.epochs, validation_split=0.1,
                         callbacks=[EpochLogger()], verbose=2)
        _, test_acc = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metric("test_accuracy", test_acc)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(hist.history["loss"], label="train loss")
        ax.plot(hist.history["val_loss"], label="val loss")
        ax.set_xlabel("epoch")
        ax.set_title(f"Fashion-MNIST, lr={args.lr}")
        ax.legend()
        mlflow.log_figure(fig, "loss_curve.png")
        plt.close(fig)

        try:
            mlflow.keras.log_model(model, name="model")
        except TypeError:
            mlflow.keras.log_model(model, artifact_path="model")

    print(f"Tracking URI: {uri} | lr={args.lr} | test accuracy {test_acc:.4f}")


if __name__ == "__main__":
    main()
