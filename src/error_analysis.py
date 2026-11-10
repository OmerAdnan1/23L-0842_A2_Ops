"""
Task 2.6 - Grid of misclassified test images for error analysis.

Takes the per-image test predictions written by evaluate.py, keeps the
misclassified ones, and shows the most confident mistakes of each kind
(real called AI, AI called real) with their audit metadata. Images are shown
at their original aspect ratio, not as the model saw them.

Usage:
    python src/error_analysis.py --model transfer_stage2 --n 10
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def pick_errors(df, n):
    wrong = df[df["pred"] != df["label"]].copy()
    wrong["confidence"] = (wrong["prob"] - 0.5).abs()
    fp = wrong[wrong["label"] == 0].nlargest(n // 2, "confidence")
    fn = wrong[wrong["label"] == 1].nlargest(n - len(fp), "confidence")
    if len(fp) + len(fn) < n:
        fp = wrong[wrong["label"] == 0].nlargest(n - len(fn), "confidence")
    return pd.concat([fp, fn])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="transfer_stage2")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--audit", default="reports/audit_table.csv")
    ap.add_argument("--root", default="data/raw")
    args = ap.parse_args()

    preds = pd.read_csv(f"reports/predictions/{args.model}_test.csv")
    audit = pd.read_csv(args.audit)[["path", "width", "height", "file_size_kb", "mode", "format"]]
    errors = pick_errors(preds, args.n).merge(audit, on="path", how="left")

    n_wrong = (preds["pred"] != preds["label"]).sum()
    print(f"[{args.model}] {n_wrong}/{len(preds)} test images misclassified, showing {len(errors)}")

    cols = 5
    rows = -(-len(errors) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.6 * rows))
    for i, (ax, r) in enumerate(zip(axes.ravel(), errors.itertuples()), start=1):
        ax.imshow(Image.open(Path(args.root) / r.path).convert("RGB"))
        kind = "Real -> AI" if r.label == 0 else "AI -> Real"
        ax.set_title(f"#{i} {kind}  P(AI)={r.prob:.2f}\n"
                     f"{r.width}x{r.height} {r.format} {r.mode} {r.file_size_kb:.0f}KB",
                     fontsize=9, color="firebrick" if r.label == 0 else "navy")
        ax.axis("off")
    for ax in axes.ravel()[len(errors):]:
        ax.axis("off")
    fig.suptitle(f"{args.model}: most confident test errors", fontsize=13)
    fig.tight_layout()

    out = Path(f"reports/figures/errors_{args.model}.png")
    fig.savefig(out, dpi=130)
    errors.insert(0, "id", range(1, len(errors) + 1))
    errors["hypothesis"] = ""
    errors.drop(columns=["sha256", "confidence"], errors="ignore").to_csv(
        f"reports/errors_{args.model}.csv", index=False)
    print(errors[["id", "path", "label", "prob", "width", "height", "format", "mode"]]
          .to_string(index=False))
    print(f"Saved {out} and reports/errors_{args.model}.csv")


if __name__ == "__main__":
    main()
