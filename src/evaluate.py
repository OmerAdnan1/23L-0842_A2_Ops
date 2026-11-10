"""
Task 2.5 - Evaluate trained models on the test set, once.

The decision threshold is chosen on validation data only: the highest recall
on AI images (TPR) whose false-positive rate (real photos called AI) stays
<= --max-fpr. Youden's J and 0.5 are reported for comparison.

Outputs:
    reports/test_metrics.{csv,md}        one row per model
    reports/thresholds.json              chosen threshold per model (serving)
    reports/predictions/<name>_test.csv  per-image probabilities (Task 2.6)
    reports/figures/roc_test.png, reports/figures/cm_<name>.png

Usage:
    python src/evaluate.py --models baseline transfer_stage2 --max-fpr 0.10
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from tensorflow import keras

from data import build_dataset


def predict(model, csv_path, root):
    df = pd.read_csv(csv_path)
    ds = build_dataset(csv_path, root, training=False)
    df["prob"] = model.predict(ds, verbose=0).ravel()
    return df


def pick_threshold(y, p, max_fpr):
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(fpr <= max_fpr)[0]
    i = ok[np.argmax(tpr[ok])]
    j = np.argmax(tpr - fpr)
    return {"threshold": float(min(thr[i], 1.0)), "val_fpr": float(fpr[i]),
            "val_tpr": float(tpr[i]), "youden_threshold": float(min(thr[j], 1.0))}


def scores(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": round(t, 4),
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred),
            "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, p),
            "fpr_real_called_ai": fp / (fp + tn),
            "fnr_ai_called_real": fn / (fn + tp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["baseline", "transfer_stage2"])
    ap.add_argument("--max-fpr", type=float, default=0.10)
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--root", default="data/raw")
    args = ap.parse_args()

    splits = Path(args.splits)
    Path("reports/predictions").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    rows, thresholds = [], {}
    fig_roc, ax_roc = plt.subplots(figsize=(6, 6))

    for name in args.models:
        model = keras.models.load_model(f"models/{name}.keras")
        val = predict(model, splits / "val.csv", args.root)
        test = predict(model, splits / "test.csv", args.root)

        th = pick_threshold(val["label"], val["prob"], args.max_fpr)
        thresholds[name] = th
        t = th["threshold"]
        print(f"[{name}] val threshold {t:.3f} (val FPR {th['val_fpr']:.3f}, "
              f"TPR {th['val_tpr']:.3f}) | Youden {th['youden_threshold']:.3f}")

        y, p = test["label"].values, test["prob"].values
        rows.append({"model": name, "rule": f"val FPR<={args.max_fpr}", **scores(y, p, t)})
        rows.append({"model": name, "rule": "fixed 0.5", **scores(y, p, 0.5)})

        test["pred"] = (p >= t).astype(int)
        test.to_csv(f"reports/predictions/{name}_test.csv", index=False)

        fpr, tpr, _ = roc_curve(y, p)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")

        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y, test["pred"], labels=[0, 1]),
                               display_labels=["Real", "AI"]).plot(ax=ax_cm, colorbar=False)
        ax_cm.set_title(f"{name} - test, threshold {t:.2f}")
        fig_cm.tight_layout()
        fig_cm.savefig(f"reports/figures/cm_{name}.png", dpi=150)
        plt.close(fig_cm)

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_xlabel("False positive rate (real called AI)")
    ax_roc.set_ylabel("True positive rate (AI caught)")
    ax_roc.set_title("ROC - test set")
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig("reports/figures/roc_test.png", dpi=150)

    df = pd.DataFrame(rows).round(4)
    df.to_csv("reports/test_metrics.csv", index=False)
    Path("reports/test_metrics.md").write_text(df.to_markdown(index=False))
    Path("reports/thresholds.json").write_text(json.dumps(thresholds, indent=2))
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
