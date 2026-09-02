"""Raw multi-dataset preparation utilities for reviewer comparisons."""

from __future__ import annotations

import ast
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .reviewer_evaluations import canonical_structure_keys


def normalize_label(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return " ".join(text.split())


def parse_label_value(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        text = str(value).strip()
        values = None
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, set)):
                    values = list(parsed)
            except (SyntaxError, ValueError):
                values = None
        if values is None:
            values = re.split(r"[;,|]", text)
    return sorted(
        {
            normalized
            for normalized in (normalize_label(item) for item in values)
            if normalized
        }
    )


def load_raw_source(
    path: Path | str,
    smiles_column: str,
    label_column: str = "Labels",
) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {smiles_column, label_column}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return pd.DataFrame(
        {
            "SMILES": raw[smiles_column].astype(str),
            "_raw_labels": raw[label_column].map(parse_label_value),
        }
    )


def clean_source_labels(
    frame: pd.DataFrame,
    dataset_name: str,
    rare_fraction: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not 0 <= rare_fraction < 1:
        raise ValueError("rare_fraction must be in [0, 1).")
    molecule_count = len(frame)
    counts = Counter(label for labels in frame["_raw_labels"] for label in set(labels))
    minimum_count = int(math.ceil(molecule_count * rare_fraction))
    kept_labels = sorted(label for label, count in counts.items() if count >= minimum_count)
    removed_labels = sorted(label for label, count in counts.items() if count < minimum_count)

    label_matrix = {
        label: frame["_raw_labels"].map(lambda values, item=label: int(item in set(values)))
        for label in kept_labels
    }
    cleaned = pd.concat(
        [frame[["SMILES"]].copy(), pd.DataFrame(label_matrix, index=frame.index)], axis=1
    )

    key_frame = cleaned["SMILES"].map(canonical_structure_keys).apply(pd.Series)
    key_frame["row_id"] = np.arange(len(cleaned), dtype=int)
    invalid_mask = key_frame["parse_status"].ne("valid")
    if invalid_mask.any():
        key_frame.loc[invalid_mask, "connectivity_smiles"] = [
            f"invalid::{idx}" for idx in key_frame.index[invalid_mask]
        ]
    cleaned = pd.concat(
        [cleaned.reset_index(drop=True), key_frame.reset_index(drop=True)], axis=1
    )
    summary = {
        "dataset": dataset_name,
        "molecule_count": int(molecule_count),
        "raw_label_count": int(len(counts)),
        "minimum_count_for_retention": minimum_count,
        "rare_label_fraction": float(rare_fraction),
        "kept_label_count": int(len(kept_labels)),
        "removed_label_count": int(len(removed_labels)),
        "kept_labels": kept_labels,
        "removed_labels": removed_labels,
        "parse_failures": int(invalid_mask.sum()),
        "raw_label_counts": {label: int(counts[label]) for label in sorted(counts)},
    }
    return cleaned, summary


def _build_group_table(frame: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows = []
    for connectivity, group in frame.groupby("connectivity_smiles", sort=True):
        rows.append(
            {
                "connectivity_smiles": connectivity,
                "group_size": int(len(group)),
                "_positive_counts": [int(group[label].sum()) for label in labels],
            }
        )
    return pd.DataFrame(rows)


def _select_groups(
    group_table: pd.DataFrame,
    target_rows: int,
    target_rates: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    if target_rows <= 0 or group_table.empty:
        return np.array([], dtype=int), 0
    sizes = group_table["group_size"].to_numpy(dtype=int)
    counts = np.asarray(group_table["_positive_counts"].tolist(), dtype=float)
    order = np.arange(len(group_table), dtype=int)
    rng.shuffle(order)
    used = np.zeros(len(group_table), dtype=bool)
    selected: list[int] = []
    current_counts = np.zeros(len(target_rates), dtype=float)
    current_rows = 0

    while current_rows < target_rows:
        feasible = order[(~used[order]) & (sizes[order] <= target_rows - current_rows)]
        if len(feasible) == 0:
            break
        candidate_counts = current_counts[None, :] + counts[feasible]
        candidate_rows = current_rows + sizes[feasible]
        candidate_rates = candidate_counts / candidate_rows[:, None]
        rate_loss = np.mean((candidate_rates - target_rates[None, :]) ** 2, axis=1)
        size_loss = np.abs(target_rows - candidate_rows) / max(target_rows, 1)
        losses = rate_loss + 0.02 * size_loss
        best_loss = float(losses.min())
        ties = np.flatnonzero(np.isclose(losses, best_loss, rtol=1e-12, atol=1e-12))
        chosen = int(feasible[int(ties[rng.integers(len(ties))])])
        used[chosen] = True
        selected.append(chosen)
        current_counts += counts[chosen]
        current_rows += int(sizes[chosen])
    return np.asarray(selected, dtype=int), int(current_rows)


def _split_score(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    labels: list[str],
    target_sizes: tuple[int, int, int],
) -> tuple[int, float, float]:
    frames = (train, val, test)
    rates = [frame[labels].mean().to_numpy(dtype=float) for frame in frames]
    pairwise = float(
        np.mean(
            [
                np.mean(np.abs(rates[i] - rates[j]))
                for i, j in ((0, 1), (0, 2), (1, 2))
            ]
        )
    )
    size_error = int(sum(abs(len(frame) - target) for frame, target in zip(frames, target_sizes)))
    max_gap = float(
        max(np.max(np.abs(rates[i] - rates[j])) for i, j in ((0, 1), (0, 2), (1, 2)))
    )
    return size_error, pairwise, max_gap


def split_connectivity_balanced(
    frame: pd.DataFrame,
    labels: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed_candidates=range(16),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to one.")
    if frame.empty:
        raise ValueError("Cannot split an empty frame.")
    group_table = _build_group_table(frame, labels)
    target_train = int(round(len(frame) * train_ratio))
    target_val = int(round(len(frame) * val_ratio))
    target_test = len(frame) - target_train - target_val
    target_sizes = (target_train, target_val, target_test)
    overall_rates = frame[labels].mean().to_numpy(dtype=float)
    best = None

    for seed in seed_candidates:
        rng = np.random.default_rng(int(seed))
        train_idx, _ = _select_groups(group_table, target_train, overall_rates, rng)
        remaining = group_table.drop(index=train_idx).reset_index(drop=True)
        test_idx, _ = _select_groups(remaining, target_test, overall_rates, rng)
        train_groups = set(group_table.iloc[train_idx]["connectivity_smiles"])
        test_groups = set(remaining.iloc[test_idx]["connectivity_smiles"])
        val_groups = set(group_table["connectivity_smiles"]) - train_groups - test_groups
        train = frame[frame["connectivity_smiles"].isin(train_groups)].copy()
        val = frame[frame["connectivity_smiles"].isin(val_groups)].copy()
        test = frame[frame["connectivity_smiles"].isin(test_groups)].copy()
        score = _split_score(train, val, test, labels, target_sizes)
        if best is None or score < best[0]:
            best = (score, int(seed), train, val, test)

    if best is None:
        raise RuntimeError("No valid connectivity-aware split was generated.")
    score, selected_seed, train, val, test = best
    connectivity_sets = [
        set(train["connectivity_smiles"]),
        set(val["connectivity_smiles"]),
        set(test["connectivity_smiles"]),
    ]
    if any(connectivity_sets[i] & connectivity_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("Connectivity leakage detected.")
    if len(train) + len(val) + len(test) != len(frame):
        raise RuntimeError("Split does not cover all source rows.")

    balance_rows = []
    for label in labels:
        rates = {
            name: float(part[label].mean())
            for name, part in (("train", train), ("val", val), ("test", test))
        }
        balance_rows.append(
            {
                "label": label,
                "train_positive_rate": rates["train"],
                "val_positive_rate": rates["val"],
                "test_positive_rate": rates["test"],
                "train_val_abs_diff": abs(rates["train"] - rates["val"]),
                "train_test_abs_diff": abs(rates["train"] - rates["test"]),
            }
        )
    balance_df = pd.DataFrame(balance_rows)
    metadata = {
        "strategy": "connectivity_group_greedy_multilabel_balance",
        "selected_seed": selected_seed,
        "target_rows": {"train": target_train, "val": target_val, "test": target_test},
        "actual_rows": {"train": len(train), "val": len(val), "test": len(test)},
        "size_error": score[0],
        "pairwise_mean_abs_rate_difference": score[1],
        "max_pairwise_abs_rate_difference": score[2],
        "train_val_mean_abs_rate_difference": float(balance_df["train_val_abs_diff"].mean()),
        "train_test_mean_abs_rate_difference": float(balance_df["train_test_abs_diff"].mean()),
        "connectivity_groups": int(len(group_table)),
        "shared_connectivity_count": 0,
        "nan_cells": {
            "train": int(train[labels].isna().sum().sum()),
            "val": int(val[labels].isna().sum().sum()),
            "test": int(test[labels].isna().sum().sum()),
        },
    }
    return train, val, test, metadata, balance_df
