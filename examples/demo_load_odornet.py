"""Minimal OdorNet loading example.

Run from the repository root:

    python examples/demo_load_odornet.py
"""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odornet.datasets import LABEL_COLUMNS, label_frame, load_odornet


MISSING_LABEL_POLICIES = ("drop", "union", "intersection")


def labels_for_policy(df: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive model targets and valid-entry masks for one missing-label policy."""
    labels = label_frame(df)
    observed_mask = labels.notna()

    if policy == "drop":
        return labels, observed_mask
    if policy == "union":
        return labels.fillna(1.0), pd.DataFrame(True, index=labels.index, columns=labels.columns)
    if policy == "intersection":
        return labels.fillna(0.0), pd.DataFrame(True, index=labels.index, columns=labels.columns)
    raise ValueError(f"Unknown policy {policy!r}; expected one of {MISSING_LABEL_POLICIES}")


def summarize_dataset(name: str, df: pd.DataFrame) -> None:
    labels = label_frame(df)

    print(f"\n{name}")
    print(f"shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"unique SMILES: {df['SMILES'].nunique()}")
    if "Source" in df.columns:
        print(f"rows with Source records: {df['Source'].notna().sum()}")
        first_source = df.loc[df["Source"].notna(), "Source"].iloc[0]
        print(f"first molecule source records: {len(first_source)}")
        print(f"first source record keys: {', '.join(first_source[0])}")
    print("label columns:")
    print(", ".join(LABEL_COLUMNS))

    positive_counts = labels.sum().sort_values(ascending=False)
    missing_counts = labels.isna().sum().sort_values(ascending=False)

    print("\npositive label counts:")
    print(positive_counts.to_string())

    print("\nmissing label counts:")
    print(missing_counts.to_string())

    print("\nlong-tail labels by positive count:")
    print(positive_counts.sort_values(ascending=True).head(10).to_string())


def summarize_missing_label_policy(df: pd.DataFrame, policy: str) -> None:
    targets, valid_mask = labels_for_policy(df, policy)
    total_entries = targets.shape[0] * targets.shape[1]
    valid_entries = int(valid_mask.to_numpy().sum())
    positives = targets.sum().sort_values(ascending=False)

    print(f"\nPolicy: {policy}")
    print(f"target matrix: {targets.shape[0]} molecules x {targets.shape[1]} labels")
    print(f"valid entries for loss/metrics: {valid_entries} / {total_entries}")
    print("top positive counts after policy conversion:")
    print(positives.head(5).to_string())


def main() -> None:
    datasets = {
        "full": load_odornet("full", root=ROOT),
        "train": load_odornet("train", root=ROOT),
        "validation": load_odornet("test", root=ROOT),
    }

    for name, df in datasets.items():
        summarize_dataset(name, df)

    print("\nMissing-label policy demonstration on the training split")
    for policy in MISSING_LABEL_POLICIES:
        summarize_missing_label_policy(datasets["train"], policy)


if __name__ == "__main__":
    main()
