"""
Task 2 - Build train/val/test split lists.

Steps: drop unreadable files -> drop exact duplicates (by SHA-256, keep first)
-> stratified 70/15/15 split with a fixed seed -> write CSVs.

Label convention: 1 = AI-generated (AiArtData), 0 = real (RealArt).
The sigmoid output of every model is therefore P(AI-generated).

Usage:
    python src/make_splits.py --table reports/audit_table.csv --out splits
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
LABELS = {"RealArt": 0, "AiArtData": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="reports/audit_table.csv")
    ap.add_argument("--out", default="splits")
    args = ap.parse_args()

    df = pd.read_csv(args.table)
    n_all = len(df)

    df = df[df["readable"]]
    n_corrupt = n_all - len(df)

    n_before = len(df)
    df = df.drop_duplicates("sha256", keep="first")
    n_dups = n_before - len(df)

    df = df.assign(label=df["class"].map(LABELS))
    assert df["label"].notna().all(), "unexpected class folder name"
    df = df[["path", "class", "label", "sha256"]].reset_index(drop=True)

    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"],
                                   random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"],
                                 random_state=SEED)

    # leakage guard: no hash may appear in two splits
    sets = {n: set(s["sha256"]) for n, s in
            [("train", train), ("val", val), ("test", test)]}
    assert not (sets["train"] & sets["val"]), "train/val overlap"
    assert not (sets["train"] & sets["test"]), "train/test overlap"
    assert not (sets["val"] & sets["test"]), "val/test overlap"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(out / f"{name}.csv", index=False)

    print(f"Audited files: {n_all} | corrupt dropped: {n_corrupt} | "
          f"duplicates dropped: {n_dups} | kept: {len(df)}")
    summary = pd.DataFrame({
        n: s["class"].value_counts() for n, s in
        [("train", train), ("val", val), ("test", test)]
    }).fillna(0).astype(int)
    summary["total"] = summary.sum(axis=1)
    print(summary.to_string())
    print("Totals:", {n: len(s) for n, s in
                      [("train", train), ("val", val), ("test", test)]})
    print(f"Saved split lists to {out}/ (seed={SEED})")


if __name__ == "__main__":
    main()
