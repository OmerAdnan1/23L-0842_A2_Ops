"""
Task 3.5 - Compare the tracked VisionGuard runs in one table and one chart.

Runs are ranked by best validation AUC (the selection metric); test metrics
are shown for reference only.

Usage:
    python src/compare_runs.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

from tracking import EXPERIMENT, setup_tracking

COLUMNS = {
    "tags.mlflow.runName": "run",
    "run_id": "run_id",
    "params.model": "model",
    "params.lr": "lr",
    "params.ft_lr": "ft_lr",
    "params.unfreeze_blocks": "unfreeze_blocks",
    "params.img_size": "img_size",
    "params.no_augment": "no_augment",
    "tags.tracking": "tracking",
    "metrics.best_val_loss": "val_loss",
    "metrics.best_val_auc": "val_auc",
    "metrics.threshold": "threshold",
    "metrics.test_roc_auc": "test_auc",
    "metrics.test_f1": "test_f1",
    "metrics.test_recall": "test_recall",
    "metrics.test_fpr_real_called_ai": "test_fpr",
}


def main():
    setup_tracking(None)
    df = mlflow.search_runs(experiment_names=[EXPERIMENT],
                            filter_string="attributes.status = 'FINISHED'")
    if "tags.mlflow.parentRunId" in df:
        df = df[df["tags.mlflow.parentRunId"].isna()]
    df = df[[c for c in COLUMNS if c in df]].rename(columns=COLUMNS)
    df = df.dropna(subset=["val_auc"]).sort_values("val_auc", ascending=False)

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    df.to_csv("reports/run_comparison.csv", index=False)
    Path("reports/run_comparison.md").write_text(df.round(4).to_markdown(index=False))
    print(df.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(df))
    ax.bar([i - 0.2 for i in x], df["val_auc"], width=0.4, label="best val AUC")
    ax.bar([i + 0.2 for i in x], df["test_auc"], width=0.4, label="test AUC")
    ax.set_xticks(list(x), df["run"], rotation=30, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("VisionGuard runs (sorted by val AUC)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("reports/figures/run_comparison.png", dpi=150)
    print("Saved reports/run_comparison.{csv,md} and reports/figures/run_comparison.png")


if __name__ == "__main__":
    main()
