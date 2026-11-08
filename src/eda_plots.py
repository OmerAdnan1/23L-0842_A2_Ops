"""
Task 1.3 - Visual EDA for VisionGuard.

Reads reports/audit_table.csv (made by audit_dataset.py) and produces:
  - one sample grid per class
  - histograms of width, height and file size per class
  - (extra) aspect-ratio histogram, useful for the resize discussion in 2.1

Usage:
    python src/eda_plots.py --root data/raw --table reports/audit_table.csv --out reports/figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display in WSL; write files only
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

SEED = 42


def sample_grid(df, root, cls, out_dir, n_rows=3, n_cols=4):
    sub = df[df["class"] == cls]
    n = min(n_rows * n_cols, len(sub))
    picks = sub.sample(n=n, random_state=SEED)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, row) in zip(axes.ravel(), picks.iterrows()):
        with Image.open(Path(root) / row["path"]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f'{row["width"]}x{row["height"]}', fontsize=8)
    fig.suptitle(f"Random samples - class: {cls}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"samples_{cls}.png", dpi=150)
    plt.close(fig)


def hist_by_class(df, column, xlabel, title, fname, out_dir, log_x=False, bins=40):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = df[column].dropna()
    if log_x:
        import numpy as np
        edges = np.logspace(np.log10(max(data.min(), 1e-3)), np.log10(data.max()), bins)
    else:
        edges = bins
    for cls, g in df.groupby("class"):
        ax.hist(g[column].dropna(), bins=edges, alpha=0.55, label=cls)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of images")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--table", default="reports/audit_table.csv")
    ap.add_argument("--out", default="reports/figures")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.table)
    df = df[df["readable"]].copy()
    df["aspect"] = df["width"] / df["height"]

    for cls in sorted(df["class"].unique()):
        sample_grid(df, args.root, cls, out_dir)

    hist_by_class(df, "width", "Width (pixels)",
                  "Image width per class", "hist_width.png", out_dir)
    hist_by_class(df, "height", "Height (pixels)",
                  "Image height per class", "hist_height.png", out_dir)
    hist_by_class(df, "file_size_kb", "File size (KB, log scale)",
                  "File size per class", "hist_filesize.png", out_dir, log_x=True)
    hist_by_class(df, "aspect", "Aspect ratio (width / height)",
                  "Aspect ratio per class", "hist_aspect.png", out_dir)

    print("Saved figures to", out_dir)
    for p in sorted(out_dir.glob("*.png")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
