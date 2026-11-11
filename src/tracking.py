"""
Task 3 - MLflow tracking helpers shared by train.py, compare_runs.py and the
registry scripts.

The tracking address and credentials come from .env (MLFLOW_TRACKING_URI,
MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD); without them runs go to a
local SQLite store.

Modes:
    manual   our Keras callback logs every epoch; one run, stages share a step axis
    autolog  mlflow.tensorflow.autolog(); one nested child run per training stage
"""
import os
import subprocess
from contextlib import contextmanager, nullcontext
from importlib.metadata import distributions
from pathlib import Path

import mlflow
import yaml
from dotenv import load_dotenv
from tensorflow import keras

EXPERIMENT = "VisionGuard-23L-0842"
MODEL_NAME = "VisionGuard-23L-0842"


def setup_tracking(experiment=EXPERIMENT):
    load_dotenv()
    uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(uri)
    if experiment:
        mlflow.set_experiment(experiment)
    return uri


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_tags(seed, dvc_file="data/raw.dvc"):
    out = yaml.safe_load(Path(dvc_file).read_text())["outs"][0]
    return {"git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": str(bool(_git("status", "--porcelain"))),
            "dvc_data_md5": out["md5"],
            "dvc_data_nfiles": str(out.get("nfiles", "")),
            "seed": str(seed)}


def package_list():
    return "\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions()))


def log_keras_model(model, name="model"):
    """Log as an MLflow model; MLflow 3 uses `name`, MLflow 2 `artifact_path`."""
    try:
        return mlflow.keras.log_model(model, name=name)
    except TypeError:
        return mlflow.keras.log_model(model, artifact_path=name)


class EpochLogger(keras.callbacks.Callback):
    def __init__(self, step_offset=0):
        super().__init__()
        self.step_offset = step_offset

    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()},
                           step=self.step_offset + epoch)


class Tracker:
    def __init__(self, mode, run_name, experiment=EXPERIMENT):
        self.mode, self.run_name, self.experiment = mode, run_name, experiment
        self.step = 0

    @contextmanager
    def run(self, params, seed):
        setup_tracking(self.experiment)
        if self.mode == "autolog":
            mlflow.tensorflow.autolog()
        with mlflow.start_run(run_name=self.run_name) as run:
            mlflow.log_params(params)
            mlflow.set_tags({**run_tags(seed), "tracking": self.mode})
            yield run

    def stage(self, name):
        if self.mode == "autolog":
            return mlflow.start_run(run_name=name, nested=True)
        return nullcontext()

    def callbacks(self):
        return [EpochLogger(self.step)] if self.mode == "manual" else []

    def after_stage(self, name, hist, best_epoch):
        self.step += len(hist)
        mlflow.log_metric(f"{name}_best_epoch", best_epoch)
        for f in [f"reports/{name}_history.csv", f"reports/{name}_trainable.txt",
                  f"reports/figures/{name}_curves.png"]:
            if Path(f).exists():
                mlflow.log_artifact(f, "training")

    def finish(self, model, ckpt_path, val_df, test_df, best_row, max_fpr):
        from evaluate import pick_threshold, plot_cm, plot_roc, scores

        th = pick_threshold(val_df["label"], val_df["prob"], max_fpr)
        t = th["threshold"]
        y, p = test_df["label"].values, test_df["prob"].values
        test = scores(y, p, t)
        mlflow.log_metrics({
            "threshold": t, "val_fpr_at_threshold": th["val_fpr"],
            "best_val_loss": float(best_row["val_loss"]),
            "best_val_accuracy": float(best_row["val_accuracy"]),
            "best_val_auc": float(best_row["val_auc"]),
            **{f"test_{k}": float(v) for k, v in test.items() if k != "threshold"},
        })

        out = Path("reports/mlflow") / self.run_name
        out.mkdir(parents=True, exist_ok=True)
        plot_cm(y, (p >= t).astype(int), f"{self.run_name} - test, t={t:.2f}",
                out / "confusion_matrix.png")
        plot_roc([(self.run_name, y, p)], out / "roc_curve.png")
        test_df.assign(pred=(p >= t).astype(int)).to_csv(out / "test_predictions.csv",
                                                         index=False)
        mlflow.log_artifacts(str(out), "evaluation")
        mlflow.log_artifacts("splits", "splits")
        mlflow.log_text(package_list(), "environment/requirements.txt")
        if Path("uv.lock").exists():
            mlflow.log_artifact("uv.lock", "environment")

        mlflow.log_artifact(ckpt_path, "model_file")
        info = log_keras_model(model)
        mlflow.set_tag("model_uri", info.model_uri)
        print(f"Test @ t={t:.3f}: " + ", ".join(f"{k}={v:.3f}" for k, v in test.items()
                                               if k != "threshold"))
