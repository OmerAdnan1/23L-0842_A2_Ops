"""
Task 1.2 - Dataset audit for VisionGuard.

Usage:
    python src/audit_dataset.py --root data/raw --out reports
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Hash the raw bytes of the file (so only byte-identical files match)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def inspect_image(path: Path) -> dict:
    """Try to fully decode the image; record metadata or the error."""
    rec = {"readable": True, "error": "", "width": None, "height": None,
           "mode": None, "format": None}
    try:
        # verify() checks file integrity but invalidates the handle,
        # so we reopen and call load() to force a real pixel decode.
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            rec.update(width=im.width, height=im.height,
                       mode=im.mode, format=im.format)
    except Exception as e:  # any failure = corrupt / unreadable
        rec["readable"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def build_table(root: Path) -> pd.DataFrame:
    rows, skipped = [], 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXT:
            skipped += 1
            continue
        rel = p.relative_to(root)
        # class = first folder under root (the label is the folder name)
        cls = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        row = {"path": str(rel), "class": cls,
               "file_size_kb": round(p.stat().st_size / 1024, 2),
               "sha256": sha256_of(p)}
        row.update(inspect_image(p))
        rows.append(row)
    print(f"Skipped {skipped} non-image files.")
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> str:
    out = []
    ok = df[df["readable"]]

    out.append("=== Images per class (all files found) ===")
    out.append(df["class"].value_counts().to_string())

    out.append("\n=== Min / median / max per class (readable images) ===")
    stats = ok.groupby("class")[["width", "height", "file_size_kb"]].agg(
        ["min", "median", "max"])
    out.append(stats.to_string())

    out.append("\n=== Same stats, whole dataset ===")
    out.append(ok[["width", "height", "file_size_kb"]]
               .agg(["min", "median", "max"]).to_string())

    out.append("\n=== Colour modes per class ===")
    out.append(pd.crosstab(ok["class"], ok["mode"]).to_string())

    out.append("\n=== Formats per class ===")
    out.append(pd.crosstab(ok["class"], ok["format"]).to_string())

    bad = df[~df["readable"]]
    out.append(f"\n=== Corrupt / unreadable files: {len(bad)} ===")
    if len(bad):
        out.append(bad[["path", "error"]].to_string(index=False))

    # Duplicates: only consider readable files
    dup_mask = ok.duplicated("sha256", keep="first")
    groups = ok[ok.duplicated("sha256", keep=False)]
    n_groups = groups["sha256"].nunique()
    out.append(f"\n=== Exact duplicates ===")
    out.append(f"Duplicate groups: {n_groups}")
    out.append(f"Redundant extra copies (removable): {int(dup_mask.sum())}")
    if n_groups:
        cross = groups.groupby("sha256")["class"].nunique()
        out.append(f"Groups spanning BOTH classes (label conflict): "
                   f"{int((cross > 1).sum())}")
        out.append("Redundant copies per class:")
        out.append(ok[dup_mask]["class"].value_counts().to_string())
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root (contains class folders)")
    ap.add_argument("--out", default="reports", help="output folder")
    args = ap.parse_args()

    root, out_dir = Path(args.root), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_table(root)
    df.to_csv(out_dir / "audit_table.csv", index=False)

    report = summarise(df)
    (out_dir / "audit_summary.txt").write_text(report)
    print(report)
    print(f"\nSaved: {out_dir/'audit_table.csv'} and {out_dir/'audit_summary.txt'}")


if __name__ == "__main__":
    main()
