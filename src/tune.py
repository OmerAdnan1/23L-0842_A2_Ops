"""
Task 2.7 - Keras Tuner random search for the MobileNetV2 fine-tuning stage.

Search space: fine-tune learning rate, head dropout, number of unfrozen
MobileNetV2 blocks (0-4, counted from the top). Every trial starts from the
trained stage-1 head (models/transfer_stage1.keras), so only stage 2 is
searched. Objective: validation AUC. The test set is not used.

Usage:
    python src/tune.py --trials 10 --epochs 15
"""
import argparse
from pathlib import Path

import keras_tuner as kt
import pandas as pd
from tensorflow import keras

from data import SEED, get_datasets
from models import build_transfer, compile_model, trainable_layers, unfreeze_last_blocks


def make_builder(stage1_path):
    def build(hp):
        model, base = build_transfer(dropout=hp.Float("dropout", 0.1, 0.5, step=0.1))
        model.load_weights(stage1_path)
        unfreeze_last_blocks(base, hp.Int("unfrozen_blocks", 0, 4))
        return compile_model(model, hp.Float("lr", 1e-6, 1e-4, sampling="log"))
    return build


def layers_per_setting(max_blocks=4):
    model, base = build_transfer()
    out = {}
    for n in range(max_blocks + 1):
        unfreeze_last_blocks(base, n)
        out[n] = len(trainable_layers(model)) - 1  # minus the Dense head
    return out


def trials_table(tuner):
    n_layers = layers_per_setting()
    rows = []
    for t in tuner.oracle.trials.values():
        hp = t.hyperparameters.values
        rows.append({"trial": t.trial_id, **hp,
                     "unfrozen_conv_layers": n_layers[hp["unfrozen_blocks"]],
                     "val_auc": t.score,
                     "best_epoch": t.best_step + 1 if t.best_step is not None else None})
    return pd.DataFrame(rows).sort_values("val_auc", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--stage1", default="models/transfer_stage1.keras")
    ap.add_argument("--out", default="reports/tuner_visionguard")
    args = ap.parse_args()

    keras.utils.set_random_seed(SEED)
    train, val, _ = get_datasets()

    tuner = kt.RandomSearch(make_builder(args.stage1),
                            objective=kt.Objective("val_auc", direction="max"),
                            max_trials=args.trials, seed=SEED, directory="tuner_runs",
                            project_name="visionguard", overwrite=True)
    tuner.search_space_summary()
    tuner.search(train, validation_data=val, epochs=args.epochs, verbose=2,
                 callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                                          patience=4)])
    tuner.results_summary(num_trials=3)

    df = trials_table(tuner)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    out.with_suffix(".md").write_text(df.round(6).to_markdown(index=False))
    print("\n" + df.to_string(index=False))
    print(f"\nBest: {tuner.get_best_hyperparameters(1)[0].values}")
    print(f"Saved {out.with_suffix('.csv')} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
