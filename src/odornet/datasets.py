"""Dataset loading and source-metadata utilities for OdorNet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


LABEL_COLUMNS = [
    "animalic&ambery",
    "sweety&gourmand",
    "floral",
    "fruity&vegetable",
    "pungent&disagreeable",
    "green&herbal",
    "nutty",
    "woody&mossy",
    "resinous&balsamic",
    "cooked",
    "odorless",
    "spice",
]


DEFAULT_PROCESSED_DIR = Path("data") / "processed"
DEFAULT_RAW_SOURCE_PATH = Path("data") / "raw" / "merged_8892_cleaned_251230.pkl"
DEFAULT_FULL_PATH = DEFAULT_PROCESSED_DIR / "full_dataset.csv"
DEFAULT_TRAIN_PATH = DEFAULT_PROCESSED_DIR / "dataset_train_aligned.csv"
DEFAULT_VAL_PATH = DEFAULT_PROCESSED_DIR / "dataset_val_aligned.csv"
DEFAULT_TEST_PATH = DEFAULT_PROCESSED_DIR / "dataset_test_aligned.csv"


def project_root(start: Path | None = None) -> Path:
    """Return the repository root by walking upward until `pyproject` or `.git`."""
    current = (start or Path.cwd()).resolve()
    for path in [current, *current.parents]:
        if (path / ".git").exists() or (path / "README.md").exists():
            return path
    return current


def _resolve(root: Path | str | None, path: Path | str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root(Path(root) if root is not None else None) / path


def _json_dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value):
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return json.loads(stripped)
    return value


def validate_label_columns(columns: Iterable[str]) -> None:
    missing = [col for col in LABEL_COLUMNS if col not in columns]
    if missing:
        raise ValueError(f"Missing expected OdorNet label columns: {missing}")


def parse_source_column(
    df: pd.DataFrame,
    source_column: str = "Source",
) -> pd.DataFrame:
    """Parse a JSON-encoded source provenance column into Python objects."""
    if source_column not in df.columns:
        return df
    df = df.copy()
    df[source_column] = df[source_column].apply(_json_loads)
    return df


def load_odornet(split: str = "full", root: Path | str | None = None) -> pd.DataFrame:
    """Load one processed OdorNet split.

    Parameters
    ----------
    split:
        One of `full`, `train`, `val`/`validation`, or `test`.
    root:
        Repository root. If omitted, it is inferred from the current directory.
    """
    split_to_path = {
        "full": DEFAULT_FULL_PATH,
        "train": DEFAULT_TRAIN_PATH,
        "val": DEFAULT_VAL_PATH,
        "validation": DEFAULT_VAL_PATH,
        "test": DEFAULT_TEST_PATH,
    }
    if split not in split_to_path:
        raise ValueError(f"Unknown split {split!r}; expected one of {sorted(split_to_path)}")
    path = _resolve(root, split_to_path[split])
    df = pd.read_csv(path)
    df = parse_source_column(df)
    validate_label_columns(df.columns)
    return df


def label_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric label columns, preserving blank cells as NaN."""
    validate_label_columns(df.columns)
    return df[LABEL_COLUMNS].apply(pd.to_numeric, errors="coerce")


def load_source_metadata(
    root: Path | str | None = None,
    metadata_path: Path | str = DEFAULT_RAW_SOURCE_PATH,
) -> pd.DataFrame:
    """Load the raw molecule table that contains source-level annotations."""
    path = _resolve(root, metadata_path)
    df = pd.read_pickle(path)
    required = {"SMILES", "Source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Source metadata is missing required columns: {sorted(missing)}")
    if df["SMILES"].duplicated().any():
        raise ValueError("Source metadata contains duplicate SMILES values.")
    return df


def merge_source_into_full(
    root: Path | str | None = None,
    full_path: Path | str = DEFAULT_FULL_PATH,
    metadata_path: Path | str = DEFAULT_RAW_SOURCE_PATH,
    output_path: Path | str | None = DEFAULT_FULL_PATH,
) -> pd.DataFrame:
    """Merge the source-level annotation column into the processed full table.

    The merge is keyed by `SMILES` because the source pkl and processed CSV have
    the same molecule set but not the same row order.
    """
    full_csv = _resolve(root, full_path)
    metadata_pkl = _resolve(root, metadata_path)
    output_csv = _resolve(root, output_path) if output_path is not None else None

    full_df = pd.read_csv(full_csv)
    validate_label_columns(full_df.columns)
    if full_df["SMILES"].duplicated().any():
        raise ValueError("Full dataset contains duplicate SMILES values.")

    source_df = pd.read_pickle(metadata_pkl)
    required = {"SMILES", "Source"}
    missing = required - set(source_df.columns)
    if missing:
        raise ValueError(f"Source metadata is missing required columns: {sorted(missing)}")
    if source_df["SMILES"].duplicated().any():
        raise ValueError("Source metadata contains duplicate SMILES values.")

    full_smiles = set(full_df["SMILES"])
    source_smiles = set(source_df["SMILES"])
    if full_smiles != source_smiles:
        raise ValueError(
            "SMILES mismatch between full dataset and source metadata: "
            f"{len(full_smiles - source_smiles)} only in full, "
            f"{len(source_smiles - full_smiles)} only in source metadata."
        )

    source_lookup = source_df[["SMILES", "Source"]].copy()
    source_lookup["Source"] = source_lookup["Source"].map(_json_dumps)
    merged = full_df.drop(columns=["Source"], errors="ignore").merge(
        source_lookup, on="SMILES", how="left", validate="one_to_one"
    )

    ordered_cols = ["SMILES", "Source", *LABEL_COLUMNS]
    merged = merged[ordered_cols]

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_csv, index=False, na_rep="")

    return merged
