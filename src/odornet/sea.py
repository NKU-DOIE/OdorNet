"""Clean SEA taxonomy processing helpers for OdorNet.

SEA denotes Statistical co-occurrence, Expert correction, and AI alignment.
This module implements the deterministic data-processing steps that can be
executed from the public repository without calling external AI services.
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import LABEL_COLUMNS


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING_PATH = REPO_ROOT / "data" / "metadata" / "olfactory_classification_strong_weak.json"
DEFAULT_SPECIALIST_MAPPING_PATH = REPO_ROOT / "data" / "metadata" / "final_specialist_label_mapping.json"
DEFAULT_SEMANTIC_DESCRIPTION_PATH = REPO_ROOT / "data" / "metadata" / "olfactory_data_translated.json"


def load_main_label_mapping(
    mapping_path: Path | str = DEFAULT_MAPPING_PATH,
    specialist_mapping_path: Path | str | None = DEFAULT_SPECIALIST_MAPPING_PATH,
) -> dict[str, list[str]]:
    """Load and merge SEA descriptor-to-primary-category mapping files.

    The statistical/AI-aligned mapping is loaded first. If the specialist
    correction mapping is present, its descriptors are appended to matching
    primary categories. Duplicates are removed while preserving sorted output
    for deterministic downstream processing.
    """
    with Path(mapping_path).open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    mapping = {key: list(values) for key, values in mapping.items()}
    mapping.setdefault("odorless", ["odorless"])

    if specialist_mapping_path is not None and Path(specialist_mapping_path).exists():
        with Path(specialist_mapping_path).open("r", encoding="utf-8") as handle:
            specialist_mapping = json.load(handle)
        for main_label, descriptors in specialist_mapping.items():
            mapping.setdefault(main_label, [])
            mapping[main_label].extend(descriptors)

    mapping = {key: sorted(set(values)) for key, values in mapping.items()}
    return mapping


def build_reverse_mapping(mapping: dict[str, list[str]]) -> dict[str, list[str]]:
    """Build a descriptor -> primary category lookup."""
    reverse: dict[str, list[str]] = defaultdict(list)
    for main_label, descriptors in mapping.items():
        for descriptor in descriptors:
            reverse[descriptor].append(main_label)
    return dict(reverse)


def load_semantic_descriptions(
    description_path: Path | str = DEFAULT_SEMANTIC_DESCRIPTION_PATH,
) -> dict[str, dict[str, object]]:
    """Load descriptor-level semantic descriptions used for AI-assisted alignment."""
    path = Path(description_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_mapping_evidence_table(
    statistical_mapping_path: Path | str = DEFAULT_MAPPING_PATH,
    specialist_mapping_path: Path | str | None = DEFAULT_SPECIALIST_MAPPING_PATH,
    semantic_description_path: Path | str | None = DEFAULT_SEMANTIC_DESCRIPTION_PATH,
    term_counts: pd.Series | None = None,
    cooccurrence_probability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Connect S/E/A evidence for each descriptor-category mapping.

    Columns beginning with `S_` summarize statistical evidence, `E_` records
    expert correction membership, and `A_` records semantic-description
    evidence used by AI-assisted alignment.
    """
    with Path(statistical_mapping_path).open("r", encoding="utf-8") as handle:
        statistical_mapping = json.load(handle)

    specialist_mapping = {}
    if specialist_mapping_path is not None and Path(specialist_mapping_path).exists():
        with Path(specialist_mapping_path).open("r", encoding="utf-8") as handle:
            specialist_mapping = json.load(handle)

    semantic_descriptions = (
        load_semantic_descriptions(semantic_description_path)
        if semantic_description_path is not None
        else {}
    )

    pairs: dict[tuple[str, str], dict[str, object]] = {}

    def add_pairs(mapping: dict[str, list[str]], source: str) -> None:
        for primary_category, descriptors in mapping.items():
            for descriptor in descriptors:
                key = (primary_category, descriptor)
                row = pairs.setdefault(
                    key,
                    {
                        "primary_category": primary_category,
                        "descriptor": descriptor,
                        "S_in_statistical_ai_mapping": False,
                        "E_in_specialist_mapping": False,
                    },
                )
                if source == "S/A":
                    row["S_in_statistical_ai_mapping"] = True
                elif source == "E":
                    row["E_in_specialist_mapping"] = True

    add_pairs(statistical_mapping, "S/A")
    add_pairs(specialist_mapping, "E")

    rows = []
    for (_, descriptor), row in pairs.items():
        descriptor_info = semantic_descriptions.get(descriptor, {})
        count = int(term_counts.get(descriptor, 0)) if term_counts is not None else np.nan

        top_links = ""
        if cooccurrence_probability is not None and descriptor in cooccurrence_probability.index:
            links = (
                cooccurrence_probability.loc[descriptor]
                .drop(labels=[descriptor], errors="ignore")
                .sort_values(ascending=False)
                .head(5)
            )
            top_links = "; ".join(f"{idx}:{val:.2f}" for idx, val in links.items() if val > 0)

        row = dict(row)
        row.update(
            {
                "S_descriptor_count": count,
                "S_top_conditional_links": top_links,
                "A_has_semantic_description": bool(descriptor_info),
                "A_description_source": descriptor_info.get("source", ""),
                "A_description_type": descriptor_info.get("type", ""),
                "SEA_evidence": "+".join(
                    part
                    for part, present in [
                        ("S/A", row["S_in_statistical_ai_mapping"]),
                        ("E", row["E_in_specialist_mapping"]),
                        ("A_desc", bool(descriptor_info)),
                    ]
                    if present
                ),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["primary_category", "descriptor"], ignore_index=True
    )


def summarize_mapping_evidence(evidence_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate descriptor-level evidence by primary category."""
    grouped = evidence_df.groupby("primary_category", dropna=False)
    return (
        grouped.agg(
            descriptor_count=("descriptor", "nunique"),
            S_or_A_mapping_count=("S_in_statistical_ai_mapping", "sum"),
            E_specialist_count=("E_in_specialist_mapping", "sum"),
            A_description_count=("A_has_semantic_description", "sum"),
            total_descriptor_observations=("S_descriptor_count", "sum"),
        )
        .reset_index()
        .sort_values("descriptor_count", ascending=False)
    )


def map_descriptors_to_main_labels(
    descriptors: list[str],
    reverse_mapping: dict[str, list[str]],
) -> list[str]:
    """Map source-level processed descriptors to primary labels."""
    labels: set[str] = set()
    for descriptor in descriptors:
        labels.update(reverse_mapping.get(descriptor, []))
    labels.discard("not any type")
    return sorted(labels)


def attach_main_labels(
    source_df: pd.DataFrame,
    reverse_mapping: dict[str, list[str]],
    source_column: str = "Source",
) -> pd.DataFrame:
    """Return a copy of the source table with `Main_label` in each source record."""
    result = source_df.copy(deep=True)
    updated_sources = []
    for sources in result[source_column]:
        records = []
        for record in copy.deepcopy(sources):
            processed = record.get("Processed_Labels", [])
            record["Main_label"] = map_descriptors_to_main_labels(processed, reverse_mapping)
            records.append(record)
        updated_sources.append(records)
    result[source_column] = updated_sources
    return result


def build_double_drop_matrix(
    source_df: pd.DataFrame,
    target_labels: list[str] | None = None,
    smiles_column: str = "SMILES",
    source_column: str = "Source",
) -> pd.DataFrame:
    """Build the molecule-level SEA label matrix with unresolved labels as NaN.

    For each primary label:
    - `1.0`: all source records support the strong label.
    - `0.0`: all source records are absent for that label and no weak label appears.
    - `NaN`: conflicting or weak evidence remains unresolved.

    If `odorless` conflicts with any positive odor label, the conflicting
    labels are set to `NaN`, matching the reference processing script.
    """
    target_labels = target_labels or LABEL_COLUMNS
    rows = []
    odor_labels = [label for label in target_labels if label != "odorless"]

    for _, source_row in source_df.iterrows():
        sources = source_row[source_column]
        row = {"SMILES": source_row[smiles_column]}

        for label in target_labels:
            main_labels = [record.get("Main_label", []) for record in sources]
            has_weak = any(f"{label}_weak" in labels for labels in main_labels)
            is_all_strong = all(label in labels for labels in main_labels)
            is_all_absent = all(label not in labels for labels in main_labels)

            if is_all_strong:
                row[label] = 1.0
            elif is_all_absent and not has_weak:
                row[label] = 0.0
            else:
                row[label] = np.nan

        if row.get("odorless") == 1.0:
            conflict = False
            for odor_label in odor_labels:
                if row.get(odor_label) == 1.0:
                    row[odor_label] = np.nan
                    conflict = True
            if conflict:
                row["odorless"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def split_perfect_stratified(
    matrix_df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 959,
    target_labels: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Split complete-label molecules into train/test and keep incomplete rows in train.

    `iterative-stratification` is used if installed. Otherwise, the function
    falls back to `sklearn.model_selection.train_test_split`.
    """
    target_labels = target_labels or LABEL_COLUMNS
    complete_mask = matrix_df[target_labels].notna().all(axis=1)
    complete_df = matrix_df[complete_mask].copy()
    incomplete_df = matrix_df[~complete_mask].copy()

    method = "random"
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed
        )
        train_idx, test_idx = next(
            splitter.split(complete_df["SMILES"].values, complete_df[target_labels].values)
        )
        method = "multilabel_stratified"
    except ImportError:
        from sklearn.model_selection import train_test_split

        train_idx, test_idx = train_test_split(
            list(range(len(complete_df))), test_size=test_size, random_state=seed
        )

    test_df = complete_df.iloc[test_idx].copy()
    train_complete = complete_df.iloc[train_idx].copy()
    train_df = pd.concat([train_complete, incomplete_df], axis=0).sample(
        frac=1, random_state=seed
    )
    metadata = {
        "method": method,
        "complete_rows": int(len(complete_df)),
        "incomplete_rows": int(len(incomplete_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }
    return train_df, test_df, metadata


def compare_label_matrices(
    left: pd.DataFrame,
    right: pd.DataFrame,
    target_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Compare two label matrices by SMILES and summarize per-label differences."""
    target_labels = target_labels or LABEL_COLUMNS
    merged = left[["SMILES", *target_labels]].merge(
        right[["SMILES", *target_labels]],
        on="SMILES",
        how="inner",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    rows = []
    for label in target_labels:
        a = pd.to_numeric(merged[f"{label}_left"], errors="coerce")
        b = pd.to_numeric(merged[f"{label}_right"], errors="coerce")
        equal_or_both_nan = (a == b) | (a.isna() & b.isna())
        rows.append({"label": label, "different_rows": int((~equal_or_both_nan).sum())})
    return pd.DataFrame(rows)
