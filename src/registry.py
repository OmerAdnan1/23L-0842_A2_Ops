"""
Task 3.6 - Model registry: register runs, set the champion alias, and export the
champion .keras file for serving (so the app never calls DagsHub per request).

Usage:
    python src/registry.py register --run-id <RUN_ID>
    python src/registry.py alias --version 2 --alias champion
    python src/registry.py export --alias champion --out models/champion.keras
"""
import argparse
import json
import shutil
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from tracking import MODEL_NAME, setup_tracking


def register(client, run_id, name):
    uri = client.get_run(run_id).data.tags.get("model_uri", f"runs:/{run_id}/model")
    mv = mlflow.register_model(uri, name)
    print(f"Registered {name} version {mv.version} from run {run_id}")


def set_alias(client, name, version, alias):
    client.set_registered_model_alias(name, alias, version)
    print(f"{name}@{alias} -> version {version}")


def export(client, name, alias, out):
    mv = client.get_model_version_by_alias(name, alias)
    run = client.get_run(mv.run_id)
    local = Path(mlflow.artifacts.download_artifacts(run_id=mv.run_id,
                                                     artifact_path="model_file"))
    src = next(local.glob("*.keras"))
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, out)
    meta = {"model_name": name, "version": mv.version, "alias": alias,
            "run_id": mv.run_id, "threshold": run.data.metrics.get("threshold")}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"Exported {name}@{alias} (v{mv.version}) to {out}\n{json.dumps(meta, indent=2)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["register", "alias", "export"])
    ap.add_argument("--name", default=MODEL_NAME)
    ap.add_argument("--run-id")
    ap.add_argument("--version")
    ap.add_argument("--alias", default="champion")
    ap.add_argument("--out", type=Path, default=Path("models/champion.keras"))
    args = ap.parse_args()

    setup_tracking(None)
    client = MlflowClient()
    if args.action == "register":
        register(client, args.run_id, args.name)
    elif args.action == "alias":
        set_alias(client, args.name, args.version, args.alias)
    else:
        export(client, args.name, args.alias, args.out)


if __name__ == "__main__":
    main()
