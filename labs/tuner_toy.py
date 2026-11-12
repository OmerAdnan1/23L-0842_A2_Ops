"""
Task 2.7 (toy) - Keras Tuner random search on Fashion-MNIST.

Searches the number of hidden units and the learning rate of a one-hidden-layer
network, then prints and saves every trial.

Usage:
    python labs/tuner_toy.py --trials 10 --epochs 5
"""
import argparse
from pathlib import Path

import keras_tuner as kt
import pandas as pd
from tensorflow import keras

SEED = 42


def build(hp):
    model = keras.Sequential([
        keras.Input(shape=(28, 28)),
        keras.layers.Rescaling(1.0 / 255),
        keras.layers.Flatten(),
        keras.layers.Dense(hp.Int("units", 32, 512, step=32), activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def trials_table(tuner):
    rows = [{"trial": t.trial_id, **t.hyperparameters.values, "val_accuracy": t.score}
            for t in tuner.oracle.trials.values()]
    return pd.DataFrame(rows).sort_values("val_accuracy", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--out", default="reports/tuner_toy_results")
    args = ap.parse_args()

    keras.utils.set_random_seed(SEED)
    (x, y), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=args.trials,
                            seed=SEED, directory="tuner_runs", project_name="fashion_toy",
                            overwrite=True)
    tuner.search(x, y, epochs=args.epochs, validation_split=0.2, verbose=1)
    tuner.results_summary(num_trials=3)

    df = trials_table(tuner)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    out.with_suffix(".md").write_text(df.round(5).to_markdown(index=False))
    print("\n" + df.to_string(index=False))

    best_hp = tuner.get_best_hyperparameters(1)[0].values
    best = tuner.get_best_models(1)[0]
    _, test_acc = best.evaluate(x_test, y_test, verbose=0)
    print(f"\nBest: {best_hp} | test accuracy {test_acc:.4f}")


if __name__ == "__main__":
    main()
